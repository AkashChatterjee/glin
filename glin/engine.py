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
from sklearn.preprocessing import LabelEncoder

from glin.preprocessor import EBMTabularPreprocessor
from glin.validation import validate_dataset

BUNDLE_FILENAME = "bundle.joblib"
METADATA_FILENAME = "metadata.json"
DEFAULT_MODELS_ROOT = Path.home() / ".glin" / "models"


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


def train_model(
    df: pd.DataFrame,
    target_column: str,
    *,
    model_name: str,
    max_bins: int = 256,
    interactions: int = 10,
    random_state: int = 42,
    on_progress: Optional[Callable[[float, float], None]] = None,
    progress_interval: float = 1.0,
) -> dict[str, Any]:
    """Fits EBMTabularPreprocessor on the feature columns and an
    ExplainableBoostingClassifier on the transformed output. Works for
    binary or multiclass targets (interpret-core automatically strips
    interaction terms for multiclass models). Raises ValueError if the
    dataset fails a hard validation rule (see glin.validation); returns any
    soft warnings alongside the trained bundle.

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

    df = df.dropna(subset=[target_column])
    X_raw = df.drop(columns=[target_column])
    y_raw = df[target_column]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    target_classes = [_to_native(c) for c in label_encoder.classes_]

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
        max_bins=max_bins,
        interactions=interactions,
        random_state=random_state,
        callback=ebm_callback,
    )
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

    return {
        "model_name": model_name,
        "target_column": target_column,
        "preprocessor": preprocessor,
        "model": model,
        "target_classes": target_classes,
        "warnings": [issue.message for issue in validation.warnings],
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
            "max_bins": _to_native(model.max_bins),
            "interactions": _to_native(model.interactions),
            "random_state": _to_native(model.random_state),
        },
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
    }
