"""Date/timestamp parsing and feature derivation for the tabular preprocessor.

Only string columns that are unmistakably dates are parsed: each value must
look like a full calendar date (``2020-01-31``, ``01/31/2020``,
``31 Jan 2020``, ``Jan 31, 2020``), optionally followed by a time. Bare
numbers (years, epoch seconds, ``20200131``) never qualify, so a plain integer
column such as a year or an ID is never silently turned into a date.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTH_NAMES = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

_DATE_SHAPE_RE = re.compile(
    r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    r"|\d{1,2}\.\d{1,2}\.\d{4}"
    r"|\d{1,2}[ -][A-Za-z]{3,9}\.?,?[ -]\d{2,4}"
    r"|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}"
    r")(?:[T ,].*)?$"
)
_NUMERIC_DMY_RE = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}(?![\d])")


@dataclass(frozen=True)
class DatePart:
    suffix: str
    nominal: bool


# Derived features in output order. Month and day of week are categorical
# (readable names, and an EBM learns any per-category pattern, so no sin/cos
# encoding is needed); the rest are ordered numerics.
DATE_PARTS = (
    DatePart("year", nominal=False),
    DatePart("month", nominal=True),
    DatePart("day_of_week", nominal=True),
    DatePart("hour", nominal=False),
    DatePart("days_since_start", nominal=False),
)


def derived_name(col: str, suffix: str) -> str:
    return f"{col}__{suffix}"


def _text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().reset_index(drop=True)


def date_shaped(series: pd.Series) -> pd.Series:
    """Boolean mask (positional) of values that look like full calendar dates."""
    return _text(series).str.match(_DATE_SHAPE_RE, na=False)


def detect_dayfirst(series: pd.Series) -> tuple[bool, str | None]:
    """Resolve ``dd/mm`` vs ``mm/dd`` from the values themselves.

    Returns ``(dayfirst, problem)``. A field above 12 settles the order; with
    no such evidence the order is ambiguous and month-first is assumed. Values
    pointing both ways are inconsistent and also parsed month-first.
    """
    fields = _text(series).str.extract(_NUMERIC_DMY_RE).dropna().astype(int)
    if fields.empty:
        return False, None
    first, second = fields[0], fields[1]
    day_first_evidence = bool((first > 12).any())
    month_first_evidence = bool((second > 12).any())
    if day_first_evidence and month_first_evidence:
        return False, "mixes day/month and month/day order"
    if day_first_evidence:
        return True, None
    if month_first_evidence:
        return False, None
    if ((first != second) & (first <= 12) & (second <= 12)).any():
        return False, "has ambiguous day/month order (every value could be either)"
    return False, None


def _to_utc(part: pd.Series, dayfirst: bool) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(part, format="ISO8601", utc=True)
        except (ValueError, TypeError):
            return pd.to_datetime(
                part, format="mixed", dayfirst=dayfirst, errors="coerce", utc=True
            )


def parse_dates(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    """Parse to tz-naive datetimes (UTC wall-clock); anything that is not a
    recognizable date becomes ``NaT``."""
    index = series.index
    if pd.api.types.is_datetime64_any_dtype(series):
        values = series
        if getattr(series.dt, "tz", None) is not None:
            values = series.dt.tz_convert("UTC").dt.tz_localize(None)
        return pd.Series(values.to_numpy(dtype="datetime64[ns]"), index=index)

    text = _text(series)
    shaped = text.str.match(_DATE_SHAPE_RE, na=False)
    year_first = text.str.match(r"\d{4}", na=False)
    parsed = pd.Series(pd.NaT, index=range(len(text)), dtype="datetime64[ns]")
    # dayfirst must not touch year-first strings (dateutil would swap them).
    for mask, order in ((shaped & year_first, False), (shaped & ~year_first, dayfirst)):
        if mask.any():
            parsed[mask] = _to_utc(text[mask], order).dt.tz_localize(None)
    parsed.index = index
    return parsed


def derive_parts(
    dates: pd.Series, reference: pd.Timestamp
) -> dict[str, pd.Series | np.ndarray]:
    """All candidate derived features, keyed by ``DatePart.suffix``. Nominal
    parts are lowercase labels (``NaN`` where the date is missing)."""
    month = dates.dt.month
    dow = dates.dt.dayofweek
    labels = lambda codes, names, base: codes.map(lambda v: np.nan if pd.isna(v) else names[int(v) - base])
    return {
        "year": dates.dt.year.astype("float64"),
        "month": labels(month, MONTH_NAMES, 1),
        "day_of_week": labels(dow, DAY_NAMES, 0),
        "hour": dates.dt.hour.astype("float64"),
        "days_since_start": (dates.dt.normalize() - reference).dt.days.astype("float64"),
    }
