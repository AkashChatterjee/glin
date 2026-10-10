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

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

CONTINUOUS = "continuous"
NOMINAL = "nominal"

_CURRENCY_RE = re.compile(r"[$€£¥₹₩₽₺₪฿\s\u00a0]")
_PAREN_NEG_RE = re.compile(r"^\((.*)\)$")
_EU_ONLY_COMMA_RE = re.compile(r"^[-+]?\d*,\d{1,2}$|^[-+]?\d*,\d{4,}$")
_EU_GROUPED_DOTS_RE = re.compile(r"^[-+]?\d{1,3}(\.\d{3}){2,}(,\d+)?$")
_EU_BOTH_RE = re.compile(r"^[-+]?\d{1,3}(\.\d{3})+,\d+$")
_US_BOTH_RE = re.compile(r"^[-+]?\d{1,3}(,\d{3})+\.\d+$")
_US_GROUPED_COMMAS_RE = re.compile(r"^[-+]?\d{1,3}(,\d{3}){2,}$")


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
        self.decimal_comma_columns_: list[str] = []
        self.warnings_: list[str] = []
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

            decimal_comma = self._detect_decimal_comma(col, series)
            if self._is_mostly_numeric(series, decimal_comma):
                if decimal_comma:
                    self.decimal_comma_columns_.append(col)
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
                out[col] = pd.to_numeric(
                    self._strip_numeric(X[col], col in self.decimal_comma_columns_),
                    errors="coerce",
                )
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

    def get_feature_types(self) -> list[str]:
        """Ordered feature_types list matching output_columns_, for EBM's
        feature_types= kwarg (avoids EBM's own autodetection heuristics)."""
        check_is_fitted(self, "output_columns_")
        return [self.feature_types_[col] for col in self.output_columns_]

    def _detect_decimal_comma(self, col: str, series: pd.Series) -> bool:
        """True if the column uses the European convention (``1.234,56``).

        Only unambiguous evidence counts; a bare ``1,234`` is read as a US
        thousands group. Mixed conventions fall back to US with a warning.
        """
        if pd.api.types.is_numeric_dtype(series):
            return False
        cleaned = self._strip_symbols(series.dropna())
        eu = cleaned.str.match(_EU_BOTH_RE) | cleaned.str.match(_EU_GROUPED_DOTS_RE)
        eu |= cleaned.str.match(_EU_ONLY_COMMA_RE)
        us = cleaned.str.match(_US_BOTH_RE) | cleaned.str.match(_US_GROUPED_COMMAS_RE)
        if eu.any() and us.any():
            self.warnings_.append(
                f"column '{col}' mixes US (1,234.56) and European (1.234,56) number "
                "formats; parsing as US, so European-style values may become missing"
            )
            return False
        return bool(eu.any())

    def _is_mostly_numeric(self, series: pd.Series, decimal_comma: bool = False) -> bool:
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        coerced = pd.to_numeric(
            self._strip_numeric(non_null, decimal_comma), errors="coerce"
        )
        return coerced.notna().mean() >= self.numeric_coerce_threshold

    @staticmethod
    def _strip_symbols(series: pd.Series) -> pd.Series:
        return series.astype(str).str.replace(_CURRENCY_RE, "", regex=True)

    @classmethod
    def _strip_numeric(cls, series: pd.Series, decimal_comma: bool = False) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series
        cleaned = cls._strip_symbols(series)
        cleaned = cleaned.str.replace(_PAREN_NEG_RE, r"-\1", regex=True)
        if decimal_comma:
            cleaned = cleaned.str.replace(".", "", regex=False).str.replace(
                ",", ".", regex=False
            )
        else:
            cleaned = cleaned.str.replace(",", "", regex=False)
        return cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    def _clean_categorical(self, series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.strip().str.lower()
        is_missing = series.isna() | cleaned.isin(["nan", "none", ""])
        return cleaned.where(~is_missing, self.missing_token)
