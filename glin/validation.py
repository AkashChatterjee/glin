"""Pre-training dataset validation.

Each rule is a plain function of (df, target_column) -> Issue | None,
registered in RULES below. This is deliberately a flat list rather than a
class hierarchy: adding a rule for a newly-supported (or newly-unsupported)
data shape is a one-function, one-line-in-RULES change, and nothing else in
glin needs to know the rule exists. An "error" blocks training (train_model
raises); a "warning" is informational and training proceeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

MIN_ROWS = 20
MAX_REASONABLE_TARGET_CLASSES = 20


@dataclass
class Issue:
    severity: str  # "error" or "warning"
    message: str


@dataclass
class ValidationResult:
    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors


Rule = Callable[[pd.DataFrame, str], Optional[Issue]]


def _rule_target_column_exists(df: pd.DataFrame, target_column: str) -> Optional[Issue]:
    if target_column not in df.columns:
        return Issue("error", f"target column '{target_column}' not found in CSV")
    return None


def _rule_min_rows(df: pd.DataFrame, target_column: str) -> Optional[Issue]:
    if len(df) < MIN_ROWS:
        return Issue(
            "error", f"only {len(df)} rows — need at least {MIN_ROWS} to train a model"
        )
    return None


def _rule_target_has_at_least_two_classes(
    df: pd.DataFrame, target_column: str
) -> Optional[Issue]:
    if target_column not in df.columns:
        return None  # already reported by _rule_target_column_exists
    n_classes = df[target_column].dropna().nunique()
    if n_classes < 2:
        return Issue(
            "error",
            f"target column '{target_column}' has {n_classes} distinct value(s) — "
            "need at least 2 to train a classifier",
        )
    return None


def _rule_target_missing_values(df: pd.DataFrame, target_column: str) -> Optional[Issue]:
    if target_column not in df.columns:
        return None
    n_missing = int(df[target_column].isna().sum())
    if n_missing > 0:
        return Issue(
            "warning",
            f"{n_missing} row(s) have a missing target value and will be dropped "
            "before training",
        )
    return None


_BAND_ADVICE = (
    "Workarounds: band the target into a few meaningful classes (quantiles or "
    "domain thresholds) and train on the band, or pick a different target. "
    "Full regression is planned separately."
)


def _rule_target_looks_continuous(df: pd.DataFrame, target_column: str) -> Optional[Issue]:
    if target_column not in df.columns:
        return None
    series = df[target_column].dropna()
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return None
    n_unique = series.nunique()
    if n_unique > MAX_REASONABLE_TARGET_CLASSES:
        return Issue(
            "warning",
            f"target column '{target_column}' has {n_unique} distinct numeric "
            "values — glin trains classifiers, not regressors, so each value would "
            f"be treated as its own unrelated class. {_BAND_ADVICE}",
        )
    if n_unique >= 3:
        return Issue(
            "warning",
            f"target column '{target_column}' is numeric with {n_unique} distinct "
            "values — glin will train a multiclass classifier that ignores their "
            "order (predicting 4 when the truth is 5 is as wrong as predicting 1). "
            "If the values are ordered levels or a score, consider banding them "
            f"into fewer classes or a binary threshold (e.g. high vs low). {_BAND_ADVICE}",
        )
    return None


def _rule_list_or_dict_cells(df: pd.DataFrame, target_column: str) -> Optional[Issue]:
    flagged = [
        col
        for col in df.columns
        if col != target_column
        and df[col].dropna().head(50).apply(lambda v: isinstance(v, (list, dict))).any()
    ]
    if flagged:
        return Issue(
            "warning",
            f"column(s) {flagged} contain list/dict values — glin only supports flat "
            "numeric and categorical columns; these will be treated as opaque text",
        )
    return None


RULES: list[Rule] = [
    _rule_target_column_exists,
    _rule_min_rows,
    _rule_target_has_at_least_two_classes,
    _rule_target_missing_values,
    _rule_target_looks_continuous,
    _rule_list_or_dict_cells,
]


def validate_dataset(df: pd.DataFrame, target_column: str) -> ValidationResult:
    issues = [issue for rule in RULES if (issue := rule(df, target_column)) is not None]
    return ValidationResult(issues=issues)
