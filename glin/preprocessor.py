"""Heuristic tabular schema detection and sanitization for EBM training/inference.

Column handling order matters and is deliberately NOT the naive top-to-bottom
reading of a drop-rule list: the ID-cardinality and free-text-cardinality
drop rules only apply once a column has failed to qualify as numeric. EBM
bins continuous values regardless of how many distinct values they take, so
a dirty numeric column stored as ``object`` dtype (e.g. a currency-like
total with a few blank rows for new customers) is exempted from cardinality
-based dropping the moment it's recognized as mostly-numeric, no matter how
unique its values are. A genuine identifier stored as a *native* numeric
dtype (a sequential passenger index, read by pandas as a clean int64 column)
is still caught by the unique-ratio rule, since it never needed coercion in
the first place. See tests/test_preprocessor.py for the exact per-column
-type cases this ordering is designed to satisfy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

CONTINUOUS = "continuous"
NOMINAL = "nominal"


class EBMTabularPreprocessor(BaseEstimator, TransformerMixin):
    """Sanitizes arbitrary tabular feature columns for an EBM.

    Learned state (fitted on training data only) is replayed identically on
    every future ``transform`` call, whether given a batch DataFrame or a
    single-record inference dict — nothing is re-learned from inference-time
    data, so unseen categories and structurally missing columns degrade
    gracefully instead of raising.
    """

    def __init__(
        self,
        id_column_threshold: float = 0.95,
        max_categorical_cardinality: int = 250,
        numeric_coerce_threshold: float = 0.80,
        missing_token: str = "__missing__",
        min_rows_for_id_check: int = 50,
    ):
        self.id_column_threshold = id_column_threshold
        self.max_categorical_cardinality = max_categorical_cardinality
        self.numeric_coerce_threshold = numeric_coerce_threshold
        self.missing_token = missing_token
        self.min_rows_for_id_check = min_rows_for_id_check

    def fit(self, X: pd.DataFrame, y=None) -> "EBMTabularPreprocessor":
        X = pd.DataFrame(X)
        n_rows = len(X)

        dropped: list[str] = []
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []

        for col in X.columns:
            series = X[col]
            n_unique = series.nunique(dropna=True)

            if n_unique <= 1:
                dropped.append(col)
                continue

            is_id_like = (
                n_rows > self.min_rows_for_id_check
                and (n_unique / n_rows) > self.id_column_threshold
            )

            if pd.api.types.is_numeric_dtype(series):
                # Already a clean numeric dtype: a high unique-ratio here means
                # a genuine sequential/identifier column, not a dirty numeric
                # feature (those only reach this branch via object dtype below).
                if is_id_like:
                    dropped.append(col)
                    continue
                numeric_cols.append(col)
                continue

            if self._is_mostly_numeric(series):
                # A numeric-looking object column (dirty formatting, blank
                # rows) is exempt from cardinality-based dropping: EBM bins
                # continuous values however many distinct values they take.
                numeric_cols.append(col)
                continue

            if is_id_like or n_unique > self.max_categorical_cardinality:
                dropped.append(col)
                continue

            categorical_cols.append(col)

        self.dropped_columns_ = dropped
        self.numeric_columns_ = numeric_cols
        self.categorical_columns_ = categorical_cols
        self.output_columns_ = numeric_cols + categorical_cols

        self.categorical_vocab_ = {
            col: set(self._clean_categorical(X[col]).unique()) | {self.missing_token}
            for col in categorical_cols
        }

        self.numeric_ranges_ = {}
        for col in numeric_cols:
            values = pd.to_numeric(self._strip_numeric(X[col]), errors="coerce").dropna()
            if len(values):
                self.numeric_ranges_[col] = (float(values.min()), float(values.max()))

        self.feature_types_ = {col: CONTINUOUS for col in numeric_cols}
        self.feature_types_.update({col: NOMINAL for col in categorical_cols})

        return self

    def transform(self, X: pd.DataFrame | dict) -> pd.DataFrame:
        check_is_fitted(self, "output_columns_")

        if isinstance(X, dict):
            X = pd.DataFrame([X])
        else:
            X = pd.DataFrame(X)

        out = pd.DataFrame(index=X.index)

        for col in self.numeric_columns_:
            if col in X.columns:
                out[col] = pd.to_numeric(self._strip_numeric(X[col]), errors="coerce")
            else:
                out[col] = np.nan

        for col in self.categorical_columns_:
            if col in X.columns:
                cleaned = self._clean_categorical(X[col])
                vocab = self.categorical_vocab_[col]
                out[col] = cleaned.where(cleaned.isin(vocab), self.missing_token)
            else:
                out[col] = self.missing_token

        return out[self.output_columns_]

    def check_inputs(self, record: dict) -> dict[str, list]:
        """Flag a single inference record's inputs the model has not seen.

        ``warnings`` lists unseen categories and numerics outside the training
        range; ``missing_features`` lists absent/blank inputs, which are valid
        (the model bins them) and so reported separately.
        """
        check_is_fitted(self, "output_columns_")
        warnings: list[dict] = []
        missing: list[str] = []
        frame = pd.DataFrame([record])

        for col in self.numeric_columns_:
            if col not in frame.columns:
                missing.append(col)
                continue
            value = pd.to_numeric(self._strip_numeric(frame[col]), errors="coerce").iloc[0]
            if pd.isna(value):
                missing.append(col)
                continue
            low, high = getattr(self, "numeric_ranges_", {}).get(col, (-np.inf, np.inf))
            if value < low or value > high:
                warnings.append(
                    {
                        "feature": col,
                        "value": record[col],
                        "issue": "out_of_range",
                        "training_range": [low, high],
                    }
                )

        for col in self.categorical_columns_:
            if col not in frame.columns:
                missing.append(col)
                continue
            cleaned = self._clean_categorical(frame[col]).iloc[0]
            if cleaned == self.missing_token:
                missing.append(col)
            elif cleaned not in self.categorical_vocab_[col]:
                warnings.append(
                    {"feature": col, "value": record[col], "issue": "unseen_category"}
                )

        return {"warnings": warnings, "missing_features": missing}

    def get_feature_types(self) -> list[str]:
        """Ordered feature_types list matching output_columns_, for EBM's
        feature_types= kwarg (avoids EBM's own autodetection heuristics)."""
        check_is_fitted(self, "output_columns_")
        return [self.feature_types_[col] for col in self.output_columns_]

    def _is_mostly_numeric(self, series: pd.Series) -> bool:
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        coerced = pd.to_numeric(self._strip_numeric(non_null), errors="coerce")
        return coerced.notna().mean() >= self.numeric_coerce_threshold

    @staticmethod
    def _strip_numeric(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series
        stripped = series.astype(str).str.strip()
        return stripped.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    def _clean_categorical(self, series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.strip().str.lower()
        is_missing = series.isna() | cleaned.isin(["nan", "none", ""])
        return cleaned.where(~is_missing, self.missing_token)
