"""EBM training, exact local-explanation extraction, and joblib bundle I/O.

Glassbox math reference (verified against interpret-core==0.7.8 at
implementation time, see tests/test_engine_additivity.py):

  binary:     P(y=1|x) = sigmoid(intercept_[0] + sum(term contribution scores))
  multiclass: P(y=k|x) = softmax(intercept_ + sum(term contribution score vectors))[k]

Interaction terms are identified via ``model.term_features_`` (index tuples
into ``feature_names_in_``), never by parsing ``term_names_`` strings — the
join delimiter is an implementation detail of the library, not a stable
contract.
"""
from __future__ import annotations

import json
import threading
import time
import warnings
from datetime import datetime, timezone
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Callable, Optional

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from glin.hyperparameters import HYPERPARAMETERS, resolve_hyperparameters
from glin.preprocessor import EBMTabularPreprocessor
from glin.validation import MIN_ROWS, validate_dataset

BUNDLE_FILENAME = "bundle.joblib"
METADATA_FILENAME = "metadata.json"
DEFAULT_MODELS_ROOT = Path.home() / ".glin" / "models"

# The holdout split is seeded independently of the model's random_state so
# two trials that differ only in hyperparameters (or only in random_state)
# are always scored on the exact same held-out rows -- otherwise a metric
# delta could just be a different draw of the split.
HOLDOUT_SPLIT_SEED = 0
MIN_HOLDOUT_ROWS = 10


def _to_native(value: Any) -> Any:
    """numpy scalars aren't JSON-serializable; metadata.json needs plain types."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _evaluate(
    model: ExplainableBoostingClassifier,
    X: pd.DataFrame,
    y: np.ndarray,
    n_classes: int,
    prior: np.ndarray,
) -> dict[str, Any]:
    """Out-of-sample quality of ``model`` on (X, y). ``log_loss`` is the
    headline number (it's what the EBM optimizes and it scores the
    probabilities, not just the argmax); ``baseline_log_loss`` is what
    always predicting the training class frequencies would score, so
    ``log_loss`` below it means the model learned something. AUC is None
    when the holdout is missing a class and AUC is undefined."""
    proba = model.predict_proba(X)
    labels = list(range(n_classes))
    pred = np.argmax(proba, axis=1)

    try:
        auc = (
            roc_auc_score(y, proba[:, 1])
            if n_classes == 2
            else roc_auc_score(y, proba, multi_class="ovr", labels=labels)
        )
    except ValueError:
        auc = None

    return {
        "log_loss": float(log_loss(y, proba, labels=labels)),
        "baseline_log_loss": float(log_loss(y, np.tile(prior, (len(y), 1)), labels=labels)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": None if auc is None else float(auc),
    }


def train_model(
    df: pd.DataFrame,
    target_column: str,
    *,
    model_name: str,
    hyperparameters: Optional[dict[str, Any]] = None,
    holdout_fraction: float = 0.0,
    on_progress: Optional[Callable[[float, float], None]] = None,
    progress_interval: float = 1.0,
) -> dict[str, Any]:
    """Fits EBMTabularPreprocessor on the feature columns and an
    ExplainableBoostingClassifier on the transformed output. Works for
    binary or multiclass targets (interaction terms are fitted for both
    in interpret-core 0.7.8). Raises ValueError if the
    dataset fails a hard validation rule (see glin.validation); returns any
    soft warnings alongside the trained bundle.

    ``hyperparameters`` overrides any subset of glin.hyperparameters.HYPERPARAMETERS
    (validated; unspecified ones keep their defaults).

    If ``holdout_fraction`` > 0, that share of rows (stratified by class
    when possible) is set aside BEFORE the preprocessor is fitted, the
    model is trained only on the rest, and the returned bundle carries
    ``evaluation``: out-of-sample metrics for exactly the model that gets
    saved. This is what makes hyperparameter tuning possible -- EBM's own
    internal validation split is used for early stopping and so isn't an
    unbiased score. With 0 the model trains on every row and
    ``evaluation`` is None.

    ``on_progress``, if given, is invoked roughly every ``progress_interval``
    seconds as ``on_progress(best_log_loss_so_far, elapsed_seconds)``. EBM
    trains its bagged ensemble across parallel worker processes, each with
    its own independent memory, so getting one coherent "best across all
    bags" number requires a ``multiprocessing.Manager`` value the workers
    all update -- a plain closure variable would only ever see its own
    process's bag. ``on_progress`` itself always runs back in this process
    (from a watcher thread), so it's free to print, log, or touch anything
    that isn't fork-safe.
    """
    validation = validate_dataset(df, target_column)
    if not validation.is_valid:
        raise ValueError("; ".join(issue.message for issue in validation.errors))

    params = resolve_hyperparameters(hyperparameters)
    if not 0.0 <= holdout_fraction <= 0.5:
        raise ValueError(f"holdout_fraction must be between 0 and 0.5, got {holdout_fraction}")

    df = df.dropna(subset=[target_column])
    X_raw = df.drop(columns=[target_column])
    y_raw = df[target_column]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    target_classes = [_to_native(c) for c in label_encoder.classes_]
    n_classes = len(target_classes)

    X_holdout_raw = y_holdout = None
    if holdout_fraction > 0:
        n_holdout = int(round(len(df) * holdout_fraction))
        if n_holdout < MIN_HOLDOUT_ROWS or len(df) - n_holdout < MIN_ROWS:
            raise ValueError(
                f"holdout_fraction={holdout_fraction} on {len(df)} rows leaves "
                f"{n_holdout} holdout / {len(df) - n_holdout} training rows; need at "
                f"least {MIN_HOLDOUT_ROWS} and {MIN_ROWS} respectively -- use a "
                "different fraction or pass holdout_fraction=0 to train on all rows"
            )
        # Stratify only when every class can appear on both sides of the split.
        can_stratify = np.bincount(y).min() >= 2 and n_holdout >= n_classes
        X_raw, X_holdout_raw, y, y_holdout = train_test_split(
            X_raw,
            y,
            test_size=holdout_fraction,
            random_state=HOLDOUT_SPLIT_SEED,
            stratify=y if can_stratify else None,
        )

    preprocessor = EBMTabularPreprocessor()
    X = preprocessor.fit_transform(X_raw)

    if X.shape[1] == 0:
        raise ValueError(
            "every feature column was dropped (ID-like, constant, or too "
            "high-cardinality) — nothing left to train on"
        )

    ebm_callback = None
    manager = None
    watcher_thread = None
    stop_watching = None

    if on_progress is not None:
        manager = Manager()
        best_so_far = manager.Value("d", float("inf"))
        lock = manager.Lock()

        def ebm_callback(bag_idx: int, step_idx: int, progress_made: bool, metric: float) -> bool:
            with lock:
                if metric < best_so_far.value:
                    best_so_far.value = metric
            return False  # never request early termination

        stop_watching = threading.Event()
        start_time = time.monotonic()

        def _watch() -> None:
            while not stop_watching.wait(progress_interval):
                current = best_so_far.value
                if current != float("inf"):
                    on_progress(current, time.monotonic() - start_time)

        watcher_thread = threading.Thread(target=_watch, daemon=True)
        watcher_thread.start()

    model = ExplainableBoostingClassifier(
        feature_names=list(X.columns),
        feature_types=preprocessor.get_feature_types(),
        callback=ebm_callback,
        **params,
    )
    fit_started = time.monotonic()
    try:
        with warnings.catch_warnings():
            # interpret-core warns that missing values won't render in its own
            # interactive dashboard, which glin never uses — irrelevant noise
            # for CLI/MCP callers who only ever see explain_record()'s output.
            warnings.filterwarnings("ignore", category=UserWarning, module=r"interpret\..*")
            model.fit(X, y)
    finally:
        if watcher_thread is not None:
            stop_watching.set()
            watcher_thread.join()
        if on_progress is not None and best_so_far.value != float("inf"):
            on_progress(best_so_far.value, time.monotonic() - start_time)
        if manager is not None:
            manager.shutdown()
    training_seconds = time.monotonic() - fit_started

    evaluation = None
    if y_holdout is not None:
        prior = np.bincount(y, minlength=n_classes) / len(y)
        evaluation = _evaluate(
            model, preprocessor.transform(X_holdout_raw), y_holdout, n_classes, prior
        )
        evaluation["train_log_loss"] = float(
            log_loss(y, model.predict_proba(X), labels=list(range(n_classes)))
        )
        evaluation["holdout_rows"] = len(y_holdout)

    return {
        "model_name": model_name,
        "target_column": target_column,
        "preprocessor": preprocessor,
        "model": model,
        "target_classes": target_classes,
        "hyperparameters": params,
        "holdout_fraction": holdout_fraction,
        "evaluation": evaluation,
        "training_rows": len(y),
        "training_seconds": training_seconds,
        "warnings": [issue.message for issue in validation.warnings]
        + preprocessor.warnings_,
    }


def save_bundle(bundle: dict[str, Any], models_root: Path) -> Path:
    model_dir = Path(models_root) / bundle["model_name"]
    model_dir.mkdir(parents=True, exist_ok=True)

    # train_model()'s on_progress= wires a closure into the fitted model's
    # .callback attribute; joblib.dump uses plain pickle (not cloudpickle),
    # which can't serialize closures. It's a training-time-only hook --
    # nothing at inference time reads model.callback -- so drop it before
    # persisting rather than widening the pickler.
    bundle["model"].callback = None

    joblib.dump(
        {
            "preprocessor": bundle["preprocessor"],
            "model": bundle["model"],
            "target_classes": bundle["target_classes"],
        },
        model_dir / BUNDLE_FILENAME,
    )

    preprocessor: EBMTabularPreprocessor = bundle["preprocessor"]
    model = bundle["model"]
    metadata = {
        "model_name": bundle["model_name"],
        "target_column": bundle["target_column"],
        "target_classes": bundle["target_classes"],
        "features": {
            "numeric": preprocessor.numeric_columns_,
            "categorical": preprocessor.categorical_columns_,
        },
        "dropped_columns": preprocessor.dropped_columns_,
        "hyperparameters": {
            name: _to_native(getattr(model, name)) for name in HYPERPARAMETERS
        },
        "holdout_fraction": bundle.get("holdout_fraction", 0.0),
        "evaluation": bundle.get("evaluation"),
        "training_rows": bundle.get("training_rows"),
        "training_seconds": bundle.get("training_seconds"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (model_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    return model_dir


def load_bundle(model_dir: Path) -> dict[str, Any]:
    return joblib.load(Path(model_dir) / BUNDLE_FILENAME)


def load_metadata(model_dir: Path) -> dict[str, Any]:
    return json.loads((Path(model_dir) / METADATA_FILENAME).read_text())


def predict_record(bundle: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    X = bundle["preprocessor"].transform(features)
    proba = bundle["model"].predict_proba(X)[0]
    target_classes = bundle["target_classes"]
    predicted_idx = int(np.argmax(proba))

    return {
        "predicted_class": target_classes[predicted_idx],
        "probabilities": {
            str(target_classes[i]): float(proba[i]) for i in range(len(target_classes))
        },
    }


def _term_contributions(
    model: ExplainableBoostingClassifier, data: dict, features: dict[str, Any]
) -> list[dict[str, Any]]:
    feature_names_in = list(model.feature_names_in_)
    contributions = []

    for i, term_idx_tuple in enumerate(model.term_features_):
        is_interaction = len(term_idx_tuple) > 1
        involved = [feature_names_in[idx] for idx in term_idx_tuple]
        raw_value = (
            [features.get(f) for f in involved] if is_interaction else features.get(involved[0])
        )
        contributions.append(
            {
                "feature": " & ".join(involved),
                "is_interaction": is_interaction,
                "value": raw_value,
                "score": np.asarray(data["scores"][i], dtype=float),
            }
        )
    return contributions


def explain_record(
    bundle: dict[str, Any], features: dict[str, Any], top_n: int = 10
) -> dict[str, Any]:
    """Full glassbox audit: intercept, per-term contributions (sorted by
    magnitude, main effects and interactions together), and an explicit
    additivity check against the model's own predict_proba. Binary targets
    use the PRD's exact sigmoid formula; targets with more than two classes
    generalize it via softmax over per-class score vectors."""
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    target_classes = bundle["target_classes"]
    n_classes = len(target_classes)

    X = preprocessor.transform(features)
    data = model.explain_local(X).data(0)
    contributions = _term_contributions(model, data, features)
    proba = model.predict_proba(X)[0]
    intercept = np.asarray(model.intercept_, dtype=float)
    predicted_class = target_classes[int(np.argmax(proba))]

    if n_classes == 2:
        base_rate = float(intercept[0])
        total_logit = base_rate + sum(float(c["score"]) for c in contributions)

        for c in contributions:
            c["score"] = float(c["score"])
            c["direction"] = (
                f"favors {target_classes[1]}" if c["score"] > 0 else f"favors {target_classes[0]}"
            )

        contributions.sort(key=lambda c: abs(c["score"]), reverse=True)

        reconstructed = float(_sigmoid(total_logit))
        model_probability = float(proba[1])

        return {
            "predicted_class": predicted_class,
            "base_rate": base_rate,
            "probabilities": {
                str(target_classes[i]): float(proba[i]) for i in range(n_classes)
            },
            "contributions": contributions[:top_n],
            "reconstructed_probability": reconstructed,
            "model_probability": model_probability,
            "additivity_verified": bool(abs(reconstructed - model_probability) < 1e-3),
            **preprocessor.check_inputs(features),
        }

    # Multiclass: each term's score is an (n_classes,)-shaped vector; the
    # additive link is softmax rather than sigmoid.
    total_logits = intercept + sum(c["score"] for c in contributions)

    for c in contributions:
        top_class_idx = int(np.argmax(c["score"]))
        c["score_by_class"] = {
            str(target_classes[k]): float(c["score"][k]) for k in range(n_classes)
        }
        c["direction"] = f"favors {target_classes[top_class_idx]}"
        c["score"] = float(np.max(np.abs(c["score"])))

    contributions.sort(key=lambda c: abs(c["score"]), reverse=True)

    reconstructed_vec = _softmax(total_logits)
    reconstructed = {str(target_classes[k]): float(reconstructed_vec[k]) for k in range(n_classes)}
    model_probability = {str(target_classes[k]): float(proba[k]) for k in range(n_classes)}

    return {
        "predicted_class": predicted_class,
        "base_rate": {str(target_classes[k]): float(intercept[k]) for k in range(n_classes)},
        "probabilities": model_probability,
        "contributions": contributions[:top_n],
        "reconstructed_probability": reconstructed,
        "model_probability": model_probability,
        "additivity_verified": bool(np.all(np.abs(reconstructed_vec - proba) < 1e-3)),
        **preprocessor.check_inputs(features),
    }


def _shape_summary(global_data: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    """Compact description of one main-effect curve: for a categorical, the
    categories pushing hardest toward and away from the positive class; for a
    continuous feature, where the curve peaks and bottoms out. Scores are
    log-odds contributions (the same units ``predict`` reports)."""
    scores = np.asarray(global_data["scores"], dtype=float)
    names = list(global_data["names"])
    if len(scores) == len(names):  # categorical: one score per category
        order = np.argsort(scores)
        pick = lambda idx: [{"category": str(names[i]), "score": round(float(scores[i]), 4)} for i in idx]
        return {
            "kind": "categorical",
            "highest": pick(order[::-1][:top_k]),
            "lowest": pick(order[:top_k]),
        }
    # continuous: len(names) == len(scores) + 1 bin edges
    lo = global_data.get("lower_bounds", names[:-1])
    hi = global_data.get("upper_bounds", names[1:])
    imax, imin = int(np.argmax(scores)), int(np.argmin(scores))
    at = lambda i: f"[{float(lo[i]):.4g}, {float(hi[i]):.4g}]"
    return {
        "kind": "continuous",
        "highest": {"range": at(imax), "score": round(float(scores[imax]), 4)},
        "lowest": {"range": at(imin), "score": round(float(scores[imin]), 4)},
    }


def global_importance(
    bundle: dict[str, Any], top_n: int = 15, include_shapes: bool = True
) -> dict[str, Any]:
    """Model-wide feature importance, straight from the fitted additive
    structure (no sampling, no approximation): each term's importance is the
    mean absolute contribution it makes to the log-odds across the training
    data (interpret-core's ``term_importances``). Reported per term (main
    effects and pairwise interactions) and rolled up per feature, with each
    feature's share of the total. For binary targets, main effects also get
    a short shape summary showing which values push the prediction up or down."""
    model = bundle["model"]
    names_in = list(model.feature_names_in_)
    importances = np.asarray(model.term_importances(), dtype=float)
    total = float(importances.sum()) or 1.0

    glob = model.explain_global() if include_shapes and len(bundle["target_classes"]) == 2 else None

    terms = []
    per_feature: dict[str, dict[str, float]] = {n: {"main": 0.0, "interaction": 0.0} for n in names_in}
    for i, (idx_tuple, imp) in enumerate(zip(model.term_features_, importances)):
        involved = [names_in[j] for j in idx_tuple]
        is_interaction = len(idx_tuple) > 1
        for f in involved:
            per_feature[f]["interaction" if is_interaction else "main"] += float(imp)
        term: dict[str, Any] = {
            "term": " & ".join(involved),
            "is_interaction": is_interaction,
            "importance": round(float(imp), 4),
            "share": round(float(imp) / total, 4),
        }
        if glob is not None and not is_interaction:
            term["shape"] = _shape_summary(glob.data(i))
        terms.append(term)
    terms.sort(key=lambda t: t["importance"], reverse=True)

    features = [
        {
            "feature": f,
            "total_importance": round(v["main"] + v["interaction"], 4),
            "main_effect": round(v["main"], 4),
            "via_interactions": round(v["interaction"], 4),
        }
        for f, v in per_feature.items()
    ]
    features.sort(key=lambda r: r["total_importance"], reverse=True)

    binary = len(bundle["target_classes"]) == 2
    return {
        **(
            {"shape_score_meaning": f"positive score pushes toward '{bundle['target_classes'][1]}', negative toward '{bundle['target_classes'][0]}'"}
            if glob is not None and binary
            else {}
        ),
        "importance_definition": (
            "mean absolute contribution to the log-odds across the training data; "
            "units are comparable within this model only"
        ),
        "features": features,
        "terms": terms[:top_n],
        "n_terms": len(terms),
        "n_interaction_terms": sum(t["is_interaction"] for t in terms),
    }
