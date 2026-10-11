"""Heuristic tabular schema detection and sanitization for EBM training/inference.

Column handling order matters and is deliberately NOT the naive top-to-bottom
reading of a drop-rule list: the ID-cardinality and free-text-cardinality
drop rules only apply once a column has failed to qualify as numeric. EBM
bins continuous values regardless of how many distinct values they take, so
a dirty numeric column stored as ``object`` dtype (e.g. a currency-like
total with a few blank rows for new customers) is exempted from cardinality
-based dropping the moment it's recognized as mostly-numeric, unless it is an
integer identifier as below. A genuine identifier stored as a *native* numeric
dtype (a sequential passenger index, read by pandas as a clean int64 column)
is caught when the column is highly unique, every value is a distinct integer,
and the values fill a constant-step range or the name looks like an ID; the
same rule applies to mostly-numeric object columns. Repeated values, fractions
or large uneven gaps (salaries, epoch seconds) mark a measurement. Date
columns (string columns of full calendar dates) are recognized before any of
those checks, since a date column is usually unique per row and would otherwise
be dropped as an identifier. Every dropped column adds a reason to ``warnings_``.
See tests/test_preprocessor.py for the per-column-type cases.
"""
from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from glin.dates import (
    DATE_PARTS,
    date_shaped,
    derive_parts,
    derived_name,
    detect_dayfirst,
    numeric_order,
    order_issue,
    parse_dates,
)

CONTINUOUS = "continuous"
NOMINAL = "nominal"

DEFAULT_SUPPRESSION_CODES = ("SUPP", "NE", "NP", "z", "*", "..")

KEEP_HINT = (
    "If it is a real measurement, round or bin it so values repeat and it is kept."
)

_CURRENCY_CODES = (
    "USD|EUR|GBP|JPY|CNY|RMB|INR|PKR|LKR|NPR|BDT|AUD|CAD|NZD|SGD|HKD|CHF|SEK|NOK|"
    "DKK|ZAR|BRL|MXN|KRW|RUB|TRY|PLN|ILS|THB|AED|SAR"
)
_SYMBOL_RE = re.compile(r"[$€£¥₹₩₽₺₪฿\s\u00a0]|%(?=\)?\s*$)")
# Applied after _SYMBOL_RE, so spaces are already gone: "USD 100" is "USD100".
_CURRENCY_CODE_RE = re.compile(
    rf"(?i)^([-+(]?)(?:{_CURRENCY_CODES}|rs\.?)(?=[-+(.,\d])"
    rf"|(?<=[\d.])(?:{_CURRENCY_CODES})(?=\)?$)"
)
_LETTER_RE = re.compile(r"[A-Za-z]")
_PAREN_NEG_RE = re.compile(r"^\((.*)\)$")
_TRAILING_NEG_RE = re.compile(r"^(\d[\d.,]*)-$")
_EU_ONLY_COMMA_RE = re.compile(r"^[-+]?\d*,\d{1,2}$|^[-+]?\d*,\d{4,}$")
_EU_GROUPED_DOTS_RE = re.compile(r"^[-+]?\d{1,3}(\.\d{3}){2,}(,\d+)?$")
_EU_BOTH_RE = re.compile(r"^[-+]?\d{1,3}(\.\d{3})+,\d+$")
_US_BOTH_RE = re.compile(r"^[-+]?\d{1,3}(,\d{3})+\.\d+$")
_US_GROUPED_COMMAS_RE = re.compile(r"^[-+]?\d{1,3}(,\d{3}){2,}$")
_EU_ONE_DOT_GROUP_RE = re.compile(r"^[-+]?[1-9]\d{0,2}\.\d{3}$")
_DOT_GROUPS_RE = re.compile(r"^[-+]?\d{1,3}(\.\d{3})+$")

# A distinct-integer column is an ID when its values fill at least this share
# of the constant-step range they span. Random unique measurements fill far
# less (no repeats in n draws needs a range of about n**2 slots).
_MIN_ID_RANGE_FILL = 0.2
_ID_NAME_TOKENS = {"id", "ids", "idx", "uuid", "guid", "key", "pk"}
_ID_NAMES = {"index", "rowid", "rownum", "rownumber", "rowindex", "recordid"}
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_MAX_UNKNOWN_CODE_TOKENS = 5
_MAX_UNKNOWN_CODE_LENGTH = 8
_MIN_UNKNOWN_CODE_REPEATS = 2
_MIN_NUMERIC_SHARE_FOR_CODE_HINT = 0.5


def _is_native_number(value) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(
        value, (bool, np.bool_)
    )


def _json_safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _finite(values: pd.Series) -> pd.Series:
    infinite = values.isin([np.inf, -np.inf])
    return values.mask(infinite) if infinite.any() else values


def _normalize_number(text, decimal_comma: bool):
    """Move a ``(1,234)`` or ``1234-`` sign to the front, then normalize separators."""
    if not isinstance(text, str):
        return text
    if text.startswith("(") or text.endswith("-"):
        negative = _PAREN_NEG_RE.match(text) or _TRAILING_NEG_RE.match(text)
        if negative:
            text = "-" + negative.group(1)
    return _normalize_separators(text, decimal_comma)


def _normalize_separators(text, decimal_comma: bool):
    """Rewrite ``text`` with a ``.`` decimal mark and no thousands separators.

    A value with both separators uses its last one as the decimal mark, and a
    repeated separator is a thousands separator. Only a value with one
    separator uses the column convention. Other shapes, such as ``1.2.3``,
    stay unparseable.
    """
    if not isinstance(text, str):
        return text
    commas, dots = text.count(","), text.count(".")
    if commas and dots:
        mark = "," if text.rfind(",") > text.rfind(".") else "."
    elif commas:
        mark = "," if decimal_comma and commas == 1 else "."
    elif (dots > 1 and _DOT_GROUPS_RE.match(text)) or (
        decimal_comma and _EU_ONE_DOT_GROUP_RE.match(text)
    ):
        mark = ","
    else:
        mark = "."
    if mark == ",":
        return text.replace(".", "").replace(",", ".")
    return text.replace(",", "")


def _id_like_name(col) -> bool:
    words = re.split(r"[^A-Za-z0-9]+", _CAMEL_RE.sub("_", str(col)))
    tokens = [w.lower() for w in words if w]
    return "".join(tokens) in _ID_NAMES or any(t in _ID_NAME_TOKENS for t in tokens)


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

    String columns whose values are full calendar dates (``2020-01-31``,
    ``01/31/2020``, ``31 Jan 2020``, optionally with a time) become the named
    features ``<col>__year``, ``__month``, ``__day_of_week``, ``__hour`` and
    ``__days_since_start`` (whole days since the earliest training date);
    parts that are constant in training are dropped. Numeric columns are never
    read as dates, so years, IDs and epoch timestamps stay numbers. Day/month
    order is inferred from the values (a field above 12); with no evidence it
    is assumed month-first and a warning is recorded.
    """

    def __init__(
        self,
        id_column_threshold: float = 0.95,
        max_categorical_cardinality: int = 250,
        numeric_coerce_threshold: float = 0.80,
        missing_token: str = "__missing__",
        min_rows_for_id_check: int = 50,
        suppression_codes: Sequence[str] = DEFAULT_SUPPRESSION_CODES,
        date_parse_threshold: float = 0.80,
    ):
        self.id_column_threshold = id_column_threshold
        self.max_categorical_cardinality = max_categorical_cardinality
        self.numeric_coerce_threshold = numeric_coerce_threshold
        self.missing_token = missing_token
        self.min_rows_for_id_check = min_rows_for_id_check
        self.suppression_codes = suppression_codes
        self.date_parse_threshold = date_parse_threshold

    def fit(self, X: pd.DataFrame, y=None) -> "EBMTabularPreprocessor":
        X = pd.DataFrame(X)
        n_rows = len(X)

        dropped: list[str] = []
        self.decimal_comma_columns_: list[str] = []
        self.warnings_: list[str] = []
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        self.date_columns_: list[str] = []
        self.date_dayfirst_: dict[str, bool] = {}
        self.date_orders_: dict[str, str | None] = {}
        self.date_references_: dict[str, pd.Timestamp] = {}
        self.date_ranges_: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        self.date_feature_names_: dict[str, list[str]] = {}
        date_vocab: dict[str, set[str]] = {}
        date_types: dict[str, str] = {}

        for col in X.columns:
            series = X[col]
            n_unique = series.nunique(dropna=True)

            if n_unique <= 1:
                self._drop(dropped, col, "it is constant (at most one distinct value)")
                continue

            # Before the numeric/ID checks: a date column is typically unique
            # per row and must not be dropped as an identifier.
            if not pd.api.types.is_numeric_dtype(series):
                dayfirst = self._detect_date_column(col, series)
                if dayfirst is not None:
                    if self._fit_date_column(col, series, dayfirst, date_vocab, date_types):
                        self.date_columns_.append(col)
                    else:
                        self._drop(
                            dropped,
                            col,
                            "it is a date column whose derived parts (year, month, "
                            "day of week, hour, days since start) never vary",
                        )
                    continue

            is_id_like = (
                n_rows > self.min_rows_for_id_check
                and (n_unique / n_rows) > self.id_column_threshold
            )

            if pd.api.types.is_numeric_dtype(series):
                if is_id_like and self._is_integer_id(col, series):
                    self._drop_integer_id(dropped, col)
                    continue
                numeric_cols.append(col)
                continue

            decimal_comma = self._detect_decimal_comma(col, series)
            if self._is_mostly_numeric(series, decimal_comma):
                parsed = pd.to_numeric(
                    self._strip_numeric(series, decimal_comma), errors="coerce"
                )
                if is_id_like and self._is_integer_id(col, parsed):
                    self._drop_integer_id(dropped, col)
                    continue
                if decimal_comma:
                    self.decimal_comma_columns_.append(col)
                self._warn_on_unrecognized_codes(col, series, decimal_comma)
                # A numeric-looking object column (dirty formatting, blank
                # rows) is exempt from cardinality-based dropping: EBM bins
                # continuous values however many distinct values they take.
                numeric_cols.append(col)
                continue
            self._warn_on_codes_blocking_numeric(col, series, decimal_comma)

            if is_id_like:
                self._drop(
                    dropped,
                    col,
                    f"it looks like an identifier (>{self.id_column_threshold:.0%} "
                    "of rows have a unique value)",
                )
                continue
            if n_unique > self.max_categorical_cardinality:
                self._drop(
                    dropped,
                    col,
                    f"it has {n_unique} distinct values, above the "
                    f"{self.max_categorical_cardinality} categorical limit",
                )
                continue

            categorical_cols.append(col)

        self.dropped_columns_ = dropped
        self.numeric_columns_ = numeric_cols
        self.categorical_columns_ = categorical_cols
        date_cols = [n for names in self.date_feature_names_.values() for n in names]
        self.output_columns_ = numeric_cols + categorical_cols + date_cols

        self.categorical_vocab_ = {
            col: set(self._clean_categorical(X[col]).unique()) | {self.missing_token}
            for col in categorical_cols
        }
        self.categorical_vocab_.update(date_vocab)

        self.numeric_ranges_ = {}
        for col in numeric_cols:
            values = pd.to_numeric(
                self._strip_numeric(X[col], col in self.decimal_comma_columns_),
                errors="coerce",
            )
            n_infinite = int(values.isin([np.inf, -np.inf]).sum())
            if n_infinite:
                self.warnings_.append(
                    f"column '{col}' has {n_infinite} infinite value(s); they are "
                    "treated as missing"
                )
            values = _finite(values).dropna()
            if len(values):
                self.numeric_ranges_[col] = (float(values.min()), float(values.max()))

        self.feature_types_ = {col: CONTINUOUS for col in numeric_cols}
        self.feature_types_.update({col: NOMINAL for col in categorical_cols})
        self.feature_types_.update(date_types)

        return self

    def _drop(
        self, dropped: list[str], col: str, reason: str, hint: str = ""
    ) -> None:
        dropped.append(col)
        message = f"Dropped column '{col}': {reason}."
        self.warnings_.append(f"{message} {hint}" if hint else message)

    def _drop_integer_id(self, dropped: list[str], col: str) -> None:
        self._drop(
            dropped,
            col,
            "it looks like a row identifier (distinct whole numbers that fill a "
            "range, or an ID-like name)",
            KEEP_HINT,
        )

    @staticmethod
    def _is_integer_id(col, values: pd.Series) -> bool:
        """Distinct integers that fill a constant-step range, or that have an
        ID-like name. Unique integers with large uneven gaps (salaries, epoch
        seconds) are measurements."""
        values = values.dropna()
        if len(values) < 2 or values.nunique() != len(values):
            return False
        if not bool((values == np.floor(values)).all()):
            return False
        if _id_like_name(col):
            return True
        ints = np.sort(values.to_numpy(dtype=np.float64))
        if np.abs(ints).max() >= 2**62:
            return False
        ints = ints.astype(np.int64)
        step = np.gcd.reduce(np.diff(ints))
        slots = (ints[-1] - ints[0]) // step + 1
        return len(ints) / slots >= _MIN_ID_RANGE_FILL

    def transform(self, X: pd.DataFrame | dict) -> pd.DataFrame:
        check_is_fitted(self, "output_columns_")

        if isinstance(X, dict):
            X = pd.DataFrame([X])
        else:
            X = pd.DataFrame(X)

        out: dict[str, pd.Series] = {}

        for col in self.numeric_columns_:
            if col in X.columns:
                out[col] = _finite(
                    pd.to_numeric(
                        self._strip_numeric(X[col], col in self.decimal_comma_columns_),
                        errors="coerce",
                    )
                )
            else:
                out[col] = pd.Series(np.nan, index=X.index)

        for col in self.categorical_columns_:
            if col in X.columns:
                cleaned = self._clean_categorical(X[col])
                vocab = self.categorical_vocab_[col]
                out[col] = cleaned.where(cleaned.isin(vocab), self.missing_token)
            else:
                out[col] = pd.Series(self.missing_token, index=X.index, dtype=object)

        for col in getattr(self, "date_columns_", []):
            out.update(self._date_parts(X[col] if col in X.columns else None, col, X.index))

        # One concat: adding columns one by one fragments wide frames.
        if not self.output_columns_:
            return pd.DataFrame(index=X.index)[self.output_columns_]
        return pd.concat(
            [out[name].rename(name) for name in self.output_columns_], axis=1
        )

    def check_inputs(self, record: dict) -> dict[str, list]:
        """Flag a single inference record's inputs the model has not seen.

        ``warnings`` lists unseen categories, numerics outside the training
        range and inputs that do not parse; ``missing_features`` lists
        absent/blank inputs, which are valid (the model bins them) and so
        reported separately.
        """
        check_is_fitted(self, "output_columns_")
        warnings: list[dict] = []
        missing: list[str] = []
        frame = pd.DataFrame([record])

        for col in self.numeric_columns_:
            if col not in frame.columns:
                missing.append(col)
                continue
            decimal_comma = col in getattr(self, "decimal_comma_columns_", [])
            value = pd.to_numeric(
                self._strip_numeric(frame[col], decimal_comma), errors="coerce"
            ).iloc[0]
            if pd.isna(value) or np.isinf(value):
                missing.append(col)
                if not self._is_missing_marker(frame[col]):
                    warnings.append(
                        {"feature": col, "value": record[col], "issue": "unparseable_number"}
                    )
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

        for col in getattr(self, "date_columns_", []):
            self._check_date_input(col, record, frame, warnings, missing)

        for warning in warnings:
            warning["value"] = _json_safe(warning["value"])
        return {"warnings": warnings, "missing_features": missing}

    def _check_date_input(
        self, col: str, record: dict, frame: pd.DataFrame, warnings: list, missing: list
    ) -> None:
        if col not in frame.columns:
            missing.append(col)
            return
        date = parse_dates(frame[col], self.date_dayfirst_[col]).iloc[0]
        if pd.isna(date):
            missing.append(col)
            if not self._is_blank(frame[col].iloc[0]):
                warnings.append(
                    {"feature": col, "value": record[col], "issue": "unparseable_date"}
                )
            return
        orders = getattr(self, "date_orders_", {})
        issue = order_issue(frame[col].iloc[0], orders[col]) if col in orders else None
        if issue:
            warnings.append(
                {
                    "feature": col,
                    "value": record[col],
                    "issue": issue,
                    "parsed_as": date.date().isoformat(),
                }
            )
        low, high = self.date_ranges_[col]
        if date.normalize() < low or date.normalize() > high:
            warnings.append(
                {
                    "feature": col,
                    "value": record[col],
                    "issue": "out_of_range",
                    "training_range": [low.date().isoformat(), high.date().isoformat()],
                }
            )
        parts = self._date_parts(frame[col], col, frame.index)
        for name, values in parts.items():
            if name in self.categorical_vocab_ and values.iloc[0] == self.missing_token:
                warnings.append(
                    {"feature": name, "value": record[col], "issue": "unseen_category"}
                )

    def _is_blank(self, value) -> bool:
        return bool(pd.isna(value)) or str(value).strip().lower() in ("", "nan", "none")

    def _is_missing_marker(self, series: pd.Series) -> bool:
        return self._is_blank(series.iloc[0]) or bool(self._is_suppression_code(series).iloc[0])

    def get_feature_types(self) -> list[str]:
        """Ordered feature_types list matching output_columns_, for EBM's
        feature_types= kwarg (avoids EBM's own autodetection heuristics)."""
        check_is_fitted(self, "output_columns_")
        return [self.feature_types_[col] for col in self.output_columns_]

    def _detect_date_column(self, col: str, series: pd.Series) -> bool | None:
        """Day-first flag if ``col`` is a date column, else None.

        Requires >= ``date_parse_threshold`` of the non-null values to both
        have a full-date shape and parse; a few stragglers become missing.
        """
        if pd.api.types.is_datetime64_any_dtype(series):
            return False
        non_null = series.dropna()
        if non_null.empty or date_shaped(non_null).mean() < self.date_parse_threshold:
            return None
        dayfirst, problem = detect_dayfirst(non_null)
        parsed = parse_dates(non_null, dayfirst)
        if parsed.notna().mean() < self.date_parse_threshold:
            return None
        if problem:
            self.warnings_.append(
                f"date column '{col}' {problem}; parsing as month/day. Use ISO "
                "dates (YYYY-MM-DD) to remove the ambiguity"
            )
        n_bad = int(parsed.isna().sum())
        if n_bad:
            self.warnings_.append(
                f"date column '{col}' has {n_bad} value(s) that are not parseable "
                "dates; they are treated as missing"
            )
        return dayfirst

    def _fit_date_column(
        self,
        col: str,
        series: pd.Series,
        dayfirst: bool,
        vocab: dict[str, set[str]],
        types: dict[str, str],
    ) -> bool:
        """Learn the reference date and which derived parts vary. False if none do."""
        dates = parse_dates(series, dayfirst)
        valid = dates.dropna()
        if valid.empty:
            return False
        reference = valid.min().normalize()
        parts = derive_parts(dates, reference)
        names: list[str] = []
        for part in DATE_PARTS:
            values = parts[part.suffix]
            if values.nunique(dropna=True) <= 1:
                continue
            name = derived_name(col, part.suffix)
            names.append(name)
            types[name] = NOMINAL if part.nominal else CONTINUOUS
            if part.nominal:
                vocab[name] = set(values.dropna()) | {self.missing_token}
        if not names:
            return False
        self.date_dayfirst_[col] = dayfirst
        self.date_orders_[col] = numeric_order(series, dayfirst)
        self.date_references_[col] = reference
        self.date_ranges_[col] = (reference, valid.max().normalize())
        self.date_feature_names_[col] = names
        return True

    def _date_parts(
        self, series: pd.Series | None, col: str, index: pd.Index
    ) -> dict[str, pd.Series]:
        """Derived feature columns for ``col`` (all missing if the input lacks it)."""
        if series is None:
            dates = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
        else:
            dates = parse_dates(series, self.date_dayfirst_[col])
        parts = derive_parts(dates, self.date_references_[col])
        out: dict[str, pd.Series] = {}
        for name in self.date_feature_names_[col]:
            values = parts[name[len(col) + 2 :]]
            if name in self.categorical_vocab_:
                values = values.fillna(self.missing_token)
                values = values.where(values.isin(self.categorical_vocab_[name]), self.missing_token)
            out[name] = values
        return out

    def _detect_decimal_comma(self, col: str, series: pd.Series) -> bool:
        """True if the column uses the European convention (``1.234,56``).

        Only unambiguous evidence counts; a bare ``1,234`` is read as a US
        thousands group. Mixed conventions fall back to US with a warning.
        """
        if pd.api.types.is_numeric_dtype(series):
            return False
        cleaned = self._strip_symbols(series[~series.map(_is_native_number)].dropna())
        eu = cleaned.str.match(_EU_BOTH_RE) | cleaned.str.match(_EU_GROUPED_DOTS_RE)
        eu |= cleaned.str.match(_EU_ONLY_COMMA_RE)
        us = cleaned.str.match(_US_BOTH_RE) | cleaned.str.match(_US_GROUPED_COMMAS_RE)
        if eu.any() and us.any():
            self.warnings_.append(
                f"column '{col}' mixes US (1,234.56) and European (1.234,56) number "
                "formats; glin reads a value with both separators by its last separator "
                "and reads every other comma as a thousands separator, so a European "
                "value such as '12,5' becomes 125"
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

    def _code_like_leftovers(
        self, series: pd.Series, decimal_comma: bool
    ) -> tuple[pd.Series, float] | None:
        """Counts of the unparseable values and the share of values that parse,
        or None unless the unparseable values are few, short and repeated."""
        non_null = series.dropna()
        stripped = self._strip_numeric(non_null, decimal_comma)
        parsed = pd.to_numeric(stripped, errors="coerce").notna()
        counts = non_null[~parsed & stripped.notna()].astype(str).str.strip().value_counts()
        if counts.empty or len(counts) > _MAX_UNKNOWN_CODE_TOKENS:
            return None
        if any(
            len(token) > _MAX_UNKNOWN_CODE_LENGTH or n < _MIN_UNKNOWN_CODE_REPEATS
            for token, n in counts.items()
        ):
            return None
        n_parsed = int(parsed.sum())
        return counts, n_parsed / (n_parsed + int(counts.sum()))

    def _code_hint(self, counts: pd.Series) -> tuple[str, list[str]]:
        tokens = ", ".join(f"'{t}' x{n}" for t, n in counts.items())
        return tokens, [*self._known_codes(), *counts.index]

    def _warn_on_unrecognized_codes(
        self, col: str, series: pd.Series, decimal_comma: bool
    ) -> None:
        """Flag a numeric column whose unparseable leftovers look like codes.

        They become missing either way; the warning lets the caller confirm
        that, or add them to ``suppression_codes`` explicitly.
        """
        leftovers = self._code_like_leftovers(series, decimal_comma)
        if leftovers is None:
            return
        tokens, confirmed = self._code_hint(leftovers[0])
        self.warnings_.append(
            f"column '{col}' is numeric but has repeated non-numeric values ({tokens}) "
            "that look like missing-value codes; they are treated as missing. If that "
            f"is right, retrain with suppression_codes={confirmed} to confirm; if they "
            "carry meaning, recode them before training"
        )

    def _warn_on_codes_blocking_numeric(
        self, col: str, series: pd.Series, decimal_comma: bool
    ) -> None:
        """Flag a mostly-numeric column that code-like values keep from being numeric."""
        if pd.api.types.is_numeric_dtype(series):
            return
        leftovers = self._code_like_leftovers(series, decimal_comma)
        if leftovers is None or leftovers[1] < _MIN_NUMERIC_SHARE_FOR_CODE_HINT:
            return
        tokens, confirmed = self._code_hint(leftovers[0])
        self.warnings_.append(
            f"column '{col}' is {leftovers[1]:.0%} numbers, below the "
            f"{self.numeric_coerce_threshold:.0%} numeric threshold, because of repeated "
            f"non-numeric values ({tokens}) that look like missing-value codes; it is "
            "not used as a numeric column. If they are codes, retrain with "
            f"suppression_codes={confirmed} to read it as numbers"
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
        cleaned = series.astype(str).str.replace(_SYMBOL_RE, "", regex=True)
        cleaned = cleaned.str.replace("\u2212", "-", regex=False)
        has_letter = cleaned.str.contains(_LETTER_RE).to_numpy(dtype=bool)
        if has_letter.any():
            cleaned.iloc[np.flatnonzero(has_letter)] = (
                cleaned[has_letter].str.replace(_CURRENCY_CODE_RE, r"\1", regex=True).to_numpy()
            )
        return cleaned

    def _strip_numeric(self, series: pd.Series, decimal_comma: bool = False) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            return series
        native = series.map(_is_native_number)
        cleaned = self._strip_symbols(series)
        cleaned = cleaned.mask(cleaned.str.lower().isin(self._code_set()))
        cleaned = cleaned.map(lambda text: _normalize_number(text, decimal_comma))
        cleaned = cleaned.mask(cleaned.isin(["", "nan", "None"]))
        return cleaned.mask(native, series)

    def _clean_categorical(self, series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.strip().str.lower()
        is_missing = series.isna() | cleaned.isin(["nan", "none", ""])
        return cleaned.where(~is_missing, self.missing_token)
