"""Derived-target leakage check.

Flags features that, on their own, (near-)perfectly separate the target: the
target is probably a threshold or simple function of that column, so the
model would just read it back. It runs on the preprocessed training features,
so coerced numerics (``44.98%``, ``$1,234``) and derived date terms
(``signup_date__year``) are judged as the model sees them.

Each feature gets a single-feature stump: a decision tree with at most
``n_classes + 1`` leaves for numerics (thresholds or a band), or a
category -> majority-class map for categoricals and for numerics with at most
``MAX_CODE_VALUES`` distinct values (integer-coded categories). It is scored by
balanced accuracy and flagged at ``LEAKAGE_THRESHOLD``. Chance and
memorisation are guarded against by:

* a minimum leaf / category support, so one-row "categories" can't separate;
* categoricals must have >= ``MIN_COVERAGE`` of rows in categories with at
  least ``MIN_LEAF_FLOOR`` rows, so a near-unique column scores nothing;
* classes with fewer than ``MIN_CLASS_ROWS`` rows are left out of the check,
  and it is skipped below ``MIN_CHECK_ROWS`` rows.

It only ever produces warnings: a strong legitimate predictor is not an error.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

LEAKAGE_THRESHOLD = 0.99
MIN_CHECK_ROWS = 30
MIN_CLASS_ROWS = 10
MIN_LEAF_FLOOR = 5
MIN_COVERAGE = 0.90
MAX_CODE_VALUES = 20
MAX_CHECK_ROWS = 20_000
MAX_REPORTED = 10
MAX_MAPPING_ITEMS = 6


def _balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _numeric_stump(
    values: np.ndarray, y: np.ndarray, min_leaf: int, class_names: Sequence[Any]
) -> tuple[float, str] | None:
    missing = np.isnan(values)
    if missing.all():
        return None
    lowest = float(np.nanmin(values))
    x = np.where(missing, lowest - 1.0, values).reshape(-1, 1)

    n_classes = len(np.unique(y))
    tree = DecisionTreeClassifier(
        max_leaf_nodes=n_classes + 1,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=0,
    ).fit(x, y)
    score = _balanced_accuracy(y, tree.predict(x))
    if score < LEAKAGE_THRESHOLD:
        return None

    t = tree.tree_
    cuts = np.sort(t.threshold[t.children_left != -1])
    cuts = cuts[cuts >= lowest]
    parts: list[str] = []
    if len(cuts) == 0:
        parts.append(f"present -> {class_names[int(tree.predict([[lowest]])[0])]!r}")
    else:
        probes = np.concatenate([[cuts[0] - 1], (cuts[:-1] + cuts[1:]) / 2, [cuts[-1] + 1]])
        labels = [class_names[int(c)] for c in tree.predict(probes.reshape(-1, 1))]
        start = 0
        for i in range(1, len(labels) + 1):
            if i < len(labels) and labels[i] == labels[start]:
                continue
            lo = None if start == 0 else cuts[start - 1]
            hi = None if i == len(labels) else cuts[i - 1]
            if lo is None and hi is None:
                cond = "present"
            elif lo is None:
                cond = f"<= {_fmt(hi)}"
            elif hi is None:
                cond = f"> {_fmt(lo)}"
            else:
                cond = f"in ({_fmt(lo)}, {_fmt(hi)}]"
            parts.append(f"{cond} -> {labels[start]!r}")
            start = i
    if missing.any():
        parts.append(f"missing -> {class_names[int(tree.predict([[lowest - 1.0]])[0])]!r}")
    return score, "; ".join(parts)


def _categorical_map(
    codes: np.ndarray, categories: Sequence[str], y: np.ndarray, class_names: Sequence[Any]
) -> tuple[float, str] | None:
    n_classes = len(class_names)
    counts = np.zeros((len(categories), n_classes))
    np.add.at(counts, (codes, y), 1)

    supported = counts.sum(axis=1) >= MIN_LEAF_FLOOR
    keep = supported[codes]
    if keep.mean() < MIN_COVERAGE or len(np.unique(y[keep])) < 2:
        return None

    class_totals = np.maximum(counts[supported].sum(axis=0), 1)
    majority = np.argmax(counts / class_totals, axis=1)
    score = _balanced_accuracy(y[keep], majority[codes[keep]])
    if score < LEAKAGE_THRESHOLD:
        return None

    parts = []
    for k in np.unique(majority[supported]):
        members = [str(categories[i]) for i in np.flatnonzero(supported & (majority == k))]
        shown = ", ".join(members[:MAX_MAPPING_ITEMS])
        more = f", ... ({len(members)} total)" if len(members) > MAX_MAPPING_ITEMS else ""
        parts.append(f"{{{shown}{more}}} -> {class_names[int(k)]!r}")
    return score, "; ".join(parts)


def _numeric_codes(values: np.ndarray) -> tuple[np.ndarray, list[str]] | None:
    """Codes for a numeric column with few distinct values (likely category
    codes), with missing as its own category; None for a continuous column."""
    limit = MAX_CODE_VALUES + 1  # + missing
    if len(np.unique(values[:1000])) > limit:
        return None
    codes, uniques = pd.factorize(values, sort=True, use_na_sentinel=False)
    if len(uniques) > MAX_CODE_VALUES + int(np.isnan(uniques).any()):
        return None
    return codes, ["missing" if np.isnan(u) else _fmt(u) for u in uniques]


def find_leaky_features(
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: Sequence[Any],
    *,
    categorical: Sequence[str],
    source_columns: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Single features of ``X`` that (near-)perfectly separate ``y`` (integer
    codes indexing ``class_names``), best first, one entry per source column.

    ``source_columns`` maps derived feature names (``signup_date__year``) to
    the raw column that produced them; ``categorical`` names the nominal
    features. Returns dicts with ``feature``, ``column``, ``score`` and
    ``rule``."""
    y = np.asarray(y)
    counts = np.bincount(y, minlength=len(class_names))
    eligible = np.flatnonzero(counts >= MIN_CLASS_ROWS)
    rows = np.flatnonzero(np.isin(y, eligible))
    if len(eligible) < 2 or len(rows) < MIN_CHECK_ROWS:
        return []

    if len(rows) > MAX_CHECK_ROWS:
        rows = np.sort(np.random.RandomState(0).choice(rows, MAX_CHECK_ROWS, replace=False))
    y_sub = y[rows]
    # Re-code to the eligible classes only.
    remap = {int(c): i for i, c in enumerate(eligible)}
    y_sub = np.array([remap[int(c)] for c in y_sub])
    names = [class_names[int(c)] for c in eligible]

    sub_counts = np.bincount(y_sub)
    min_leaf = max(MIN_LEAF_FLOOR, min(int(0.01 * len(rows)), int(sub_counts.min()) // 2))
    X_sub = X.iloc[rows]
    categorical = set(categorical)
    source_columns = source_columns or {}

    found: dict[str, dict[str, Any]] = {}
    for feature in X_sub.columns:
        if feature in categorical:
            result = _categorical_map(*pd.factorize(X_sub[feature].astype(str)), y_sub, names)
        else:
            values = pd.to_numeric(X_sub[feature], errors="coerce").to_numpy(float)
            result = _numeric_stump(values, y_sub, min_leaf, names)
            coded = _numeric_codes(values) if result is None else None
            if coded is not None:
                result = _categorical_map(*coded, y_sub, names)
        if result is None:
            continue
        score, rule = result
        column = source_columns.get(feature, feature)
        if column not in found or score > found[column]["score"]:
            found[column] = {"feature": feature, "column": column, "score": round(score, 4), "rule": rule}

    return sorted(found.values(), key=lambda r: (-r["score"], r["column"]))


def leakage_warnings(suspects: Sequence[dict[str, Any]], target_column: str) -> list[str]:
    """One human-readable warning per suspect (capped), each naming the column,
    the evidence and the suggested action."""
    messages = []
    for s in suspects[:MAX_REPORTED]:
        via = f" (via derived term '{s['feature']}')" if s["feature"] != s["column"] else ""
        messages.append(
            f"Possible target leakage: '{s['column']}'{via} separates '{target_column}' "
            f"almost perfectly on its own (balanced accuracy {s['score']:.3f}): {s['rule']}. "
            "The model may just be reading the target back from this column. If it would "
            "not be known when making a real prediction, drop it and retrain; keep it "
            "only if the relationship is legitimate."
        )
    if len(suspects) > MAX_REPORTED:
        messages.append(
            f"{len(suspects) - MAX_REPORTED} more columns also separate '{target_column}' "
            "near-perfectly; review the columns above first."
        )
    return messages
