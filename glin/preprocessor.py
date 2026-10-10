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
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

CONTINUOUS = "continuous"
NOMINAL = "nominal"

DEFAULT_SUPPRESSION_CODES = ("SUPP", "NE", "NP", "z")

_SYMBOL_RE = re.compile(r"[$€£¥₹₩₽₺₪฿\s\u00a0]|%(?=\)?\s*$)")
_PAREN_NEG_RE = re.compile(r"^\((.*)\)$")
_EU_ONLY_COMMA_RE = re.compile(r"^[-+]?\d*,\d{1,2}$|^[-+]?\d*,\d{4,}$")
_EU_GROUPED_DOTS_RE = re.compile(r"^[-+]?\d{1,3}(\.\d{3}){2,}(,\d+)?$")
_EU_BOTH_RE = re.compile(r"^[-+]?\d{1,3}(\.\d{3})+,\d+$")
_US_BOTH_RE = re.compile(r"^[-+]?\d{1,3}(,\d{3})+\.\d+$")
_US_GROUPED_COMMAS_RE = re.compile(r"^[-+]?\d{1,3}(,\d{3}){2,}$")

_MAX_UNKNOWN_CODE_TOKENS = 5
_MAX_UNKNOWN_CODE_LENGTH = 8
_MIN_UNKNOWN_CODE_REPEATS = 2


class EBMTabularPreprocessor(BaseEstimator, TransformerMixin):
    """Sanitizes arbitrary tabular feature columns for an EBM.

    Learned state (fitted on training data only) is replayed identically on
    every future ``transform`` call, whether given a batch DataFrame or a
    single-record inference dict — nothing is re-learned from inference-time
    data, so unseen categories and structurally missing columns degrade
    gracefully instead of raising.

    In numeric columns, a trailing ``%`` is stripped (the value stays in
    percentage points: ``44.98%`` -> 44.98) and ``suppression_codes`` (matched
    case-insensitively, ignoring surrounding whitespace) become missing.
    Categorical columns are left untouched, so a code like ``NE`` is only
    treated as missing where the rest of the column is numeric.
    """

    def __init__(
        self,
        id_column_threshold: float = 0.95,
        max_categorical_cardinality: int = 250,
        numeric_coerce_threshold: float = 0.80,
        missing_token: str = "__missing__",
        min_rows_for_id_check: int = 50,
        suppression_codes: Sequence[str] = DEFAULT_SUPPRESSION_CODES,
    ):
        self.id_column_threshold = id_column_threshold
        self.max_categorical_cardinality = max_categorical_cardinality
        self.numeric_coerce_threshold = numeric_coerce_threshold
        self.missing_token = missing_token
        self.min_rows_for_id_check = min_rows_for_id_check
        self.suppression_codes = suppression_codes

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
                self._warn_on_unrecognized_codes(col, series, decimal_comma)
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
        """True if enough non-null, non-code values parse as numbers."""
        non_null = series.dropna()
        non_null = non_null[~self._is_suppression_code(non_null)]
        if len(non_null) == 0:
            return False
        coerced = pd.to_numeric(
            self._strip_numeric(non_null, decimal_comma), errors="coerce"
        )
        return coerced.notna().mean() >= self.numeric_coerce_threshold

    def _warn_on_unrecognized_codes(
        self, col: str, series: pd.Series, decimal_comma: bool
    ) -> None:
        """Flag a numeric column whose unparseable leftovers look like codes.

        They become missing either way; the warning lets the caller confirm
        that, or add them to ``suppression_codes`` explicitly.
        """
        non_null = series.dropna()
        stripped = self._strip_numeric(non_null, decimal_comma)
        unparsed = non_null[pd.to_numeric(stripped, errors="coerce").isna() & stripped.notna()]
        counts = unparsed.astype(str).str.strip().value_counts()
        if counts.empty or len(counts) > _MAX_UNKNOWN_CODE_TOKENS:
            return
        if any(
            len(token) > _MAX_UNKNOWN_CODE_LENGTH or n < _MIN_UNKNOWN_CODE_REPEATS
            for token, n in counts.items()
        ):
            return
        tokens = ", ".join(f"'{t}' x{n}" for t, n in counts.items())
        confirmed = [*self._known_codes(), *counts.index]
        self.warnings_.append(
            f"column '{col}' is numeric but has repeated non-numeric values ({tokens}) "
            "that look like missing-value codes; they are treated as missing. If that "
            f"is right, retrain with suppression_codes={confirmed} to confirm; if they "
            "carry meaning, recode them before training"
        )

    def _known_codes(self) -> list[str]:
        codes = getattr(self, "suppression_codes", DEFAULT_SUPPRESSION_CODES)
        return [codes] if isinstance(codes, str) else [str(c) for c in codes]

    def _code_set(self) -> set[str]:
        return {c.strip().lower() for c in self._known_codes()}

    def _is_suppression_code(self, series: pd.Series) -> pd.Series:
        return self._strip_symbols(series).str.lower().isin(self._code_set())

    @staticmethod
    def _strip_symbols(series: pd.Series) -> pd.Series:
        return series.astype(str).str.replace(_SYMBOL_RE, "", regex=True)

    def _strip_numeric(self, series: pd.Series, decimal_comma: bool = False) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series
        cleaned = self._strip_symbols(series)
        cleaned = cleaned.mask(cleaned.str.lower().isin(self._code_set()))
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
