"""Read CSV and Excel (.xlsx / .xls) files into a DataFrame.

Excel support is optional (`pip install 'glin-ml[excel]'`): openpyxl reads
.xlsx and xlrd reads legacy .xls. Row numbers in this module's options and
messages are 1-based sheet rows, i.e. the numbers shown in Excel.
"""
from __future__ import annotations

import datetime
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

EXCEL_SUFFIXES = (".xlsx", ".xls")
EXCEL_INSTALL_HINT = (
    "pip install 'glin-ml[excel]' (with uvx: uvx --from 'glin-ml[excel]' glin serve)"
)

HeaderRows = Union[int, str, Sequence[int], None]
SheetRef = Union[int, str, None]

# Header detection reads at most this many leading rows into its decision.
_HEADER_SCAN_ROWS = 100
_HEADER_FILL_RATIO = 0.6
_LABEL_RATIO = 0.8
_YEAR_RANGE = (1800, 2100)
_HEADER_SEPARATOR = " - "
# A trailing row is a footnote if its text starts with one of these or is a
# sentence of at least _NOTE_MIN_WORDS words.
_NOTE_START = re.compile(r"(sources?|notes?|footnotes?)\b|[*†‡©]", re.IGNORECASE)
_NOTE_MIN_WORDS = 6


@dataclass(frozen=True)
class OptionNames:
    """How messages name the Excel options: MCP arguments or CLI flags."""

    sheet: str = "sheet"
    header_rows: str = "header_rows"


MCP_OPTIONS = OptionNames()
CLI_OPTIONS = OptionNames("--sheet", "--header-rows")


@dataclass
class LoadedTable:
    df: pd.DataFrame
    sheet: Optional[str] = None
    header_rows: Optional[list[int]] = None
    warnings: list[str] = field(default_factory=list)


def is_excel_path(path: Union[str, Path]) -> bool:
    return Path(path).suffix.lower() in EXCEL_SUFFIXES


def load_table(
    path: Union[str, Path],
    sheet: SheetRef = None,
    header_rows: HeaderRows = None,
    options: OptionNames = MCP_OPTIONS,
) -> LoadedTable:
    """Load a table. `.xlsx` / `.xls` go through the Excel reader; anything
    else is read as a CSV. `sheet` and `header_rows` apply to Excel only.
    `options` sets how error messages name those two options."""
    path = Path(path)
    if not is_excel_path(path):
        if sheet is not None or header_rows is not None:
            raise ValueError(
                f"{options.sheet} and {options.header_rows} only apply to "
                ".xlsx / .xls files"
            )
        return LoadedTable(pd.read_csv(path))
    return _load_excel(path, sheet, header_rows, options)


def parse_header_rows(
    spec: HeaderRows, options: OptionNames = MCP_OPTIONS
) -> Optional[list[int]]:
    """Normalise `header_rows` to a sorted list of consecutive 1-based rows.
    Accepts 4, "4", "3-4", "3,4" or [3, 4]."""
    name = options.header_rows
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None
    if isinstance(spec, bool):
        raise ValueError(f"{name} must be a row number, a range like '3-4', or a list")
    if isinstance(spec, int):
        rows = [spec]
    elif isinstance(spec, str):
        rows = _parse_header_rows_text(spec, name)
    else:
        try:
            rows = [_as_row_number(r, name) for r in spec]
        except TypeError:
            raise ValueError(
                f"{name} must be a row number, a range like '3-4', or a list"
            ) from None
        if not rows:
            raise ValueError(
                f"{name} is empty; give at least one row number, or omit it to "
                "detect the header"
            )
    if any(r < 1 for r in rows):
        raise ValueError(f"{name} are 1-based sheet row numbers (as shown in Excel)")
    rows = sorted(set(rows))
    if rows != list(range(rows[0], rows[-1] + 1)):
        raise ValueError(f"{name} must be consecutive rows, got {rows}")
    return rows


def _as_row_number(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} entries must be integers, got {value!r}")
    return int(value)


def _parse_header_rows_text(text: str, name: str) -> list[int]:
    text = text.strip()
    try:
        if "-" in text:
            first, last = (int(p) for p in text.split("-", 1))
            if last < first:
                raise ValueError
            return list(range(first, last + 1))
        return [int(p) for p in text.split(",")]
    except ValueError:
        raise ValueError(
            f"could not parse {name} {text!r}; use a row number, a range "
            "like '3-4', or a comma list like '3,4' (1-based sheet rows)"
        ) from None


def _load_excel(
    path: Path, sheet: SheetRef, header_rows: HeaderRows, options: OptionNames
) -> LoadedTable:
    explicit = parse_header_rows(header_rows, options)
    try:
        workbook = pd.ExcelFile(path)
    except ImportError as exc:
        raise ValueError(_missing_dependency_message(path, exc)) from None
    except Exception as exc:
        raise ValueError(
            f"could not read {path.name} as an Excel workbook ({type(exc).__name__}); "
            "it may be corrupt, password-protected, or not really "
            f"{path.suffix.lower()}"
        ) from None
    with workbook:
        sheet_name = _resolve_sheet(workbook.sheet_names, sheet, options.sheet)
        try:
            raw = workbook.parse(sheet_name, header=None, dtype=object)
        except ImportError as exc:
            raise ValueError(_missing_dependency_message(path, exc)) from None
    return _extract_table(raw, explicit, sheet_name, options)


def _missing_dependency_message(path: Path, exc: ImportError) -> str:
    library = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    if library not in str(exc).lower():
        library = "openpyxl or xlrd"
    return (
        f"Reading {path.suffix.lower()} files needs {library}, which is not installed. "
        f"Install glin's Excel extra: {EXCEL_INSTALL_HINT}"
    )


def _resolve_sheet(names: list[str], sheet: SheetRef, option: str) -> str:
    listing = "; ".join(f"{i}. {name!r}" for i, name in enumerate(names, 1))
    if sheet is None or (isinstance(sheet, str) and not sheet.strip()):
        if len(names) == 1:
            return names[0]
        raise ValueError(
            f"workbook has {len(names)} sheets; choose one with {option} (a name "
            f"or 1-based position). Available: {listing}"
        )
    if isinstance(sheet, bool):
        raise ValueError(f"{option} must be a sheet name or a 1-based position")
    if isinstance(sheet, int):
        if 1 <= sheet <= len(names):
            return names[sheet - 1]
        raise ValueError(f"sheet {sheet} is out of range. Available: {listing}")
    if sheet in names:
        return sheet
    folded = [n for n in names if n.strip().casefold() == sheet.strip().casefold()]
    if len(folded) == 1:
        return folded[0]
    if folded:
        matches = ", ".join(repr(n) for n in folded)
        raise ValueError(
            f"sheet name {sheet!r} is ambiguous: it matches {matches} when case and "
            f"outer spaces are ignored. Give {option} the exact name or a 1-based "
            f"position. Available: {listing}"
        )
    if sheet.strip().isdigit():
        return _resolve_sheet(names, int(sheet), option)
    raise ValueError(f"no sheet named {sheet!r}. Available: {listing}")


def _is_label(value: Any) -> bool:
    """Text, dates, or year-like integers (2023) read as header labels."""
    if isinstance(value, (str, datetime.date)):
        return True
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and float(value).is_integer()
        and _YEAR_RANGE[0] <= value <= _YEAR_RANGE[1]
    )


def _clean_cells(raw: pd.DataFrame) -> pd.DataFrame:
    def clean(value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return value if value else np.nan
        return np.nan if pd.isna(value) else value

    cleaned = raw.apply(lambda col: col.map(clean)).astype(object)
    return cleaned.dropna(axis=1, how="all")


def _extract_table(
    raw: pd.DataFrame,
    explicit: Optional[list[int]],
    sheet_name: str,
    options: OptionNames,
) -> LoadedTable:
    cells = _clean_cells(raw)
    counts = cells.notna().sum(axis=1)
    if cells.empty or not counts.any():
        raise ValueError(f"sheet {sheet_name!r} is empty")

    if explicit is None:
        explicit = [_detect_header_row(cells, counts, sheet_name, options) + 1]
    elif explicit[-1] > len(cells):
        raise ValueError(
            f"{options.header_rows} {explicit} is beyond the last row ({len(cells)}) of "
            f"sheet {sheet_name!r}"
        )

    first, last = explicit[0] - 1, explicit[-1] - 1
    names = _flatten_headers(cells.iloc[first : last + 1])
    body = cells.iloc[last + 1 :]
    body, footnotes = _split_footnotes(body)
    body = body.dropna(how="all")
    if body.empty:
        raise ValueError(
            f"no data rows below header row(s) {explicit} in sheet {sheet_name!r}"
        )
    warnings = []
    if footnotes:
        rows = ", ".join(str(r) for r in footnotes)
        warnings.append(
            f"dropped {len(footnotes)} footnote row(s) at the end of sheet "
            f"{sheet_name!r} (sheet row(s) {rows}); they hold only note text, such "
            "as 'Source: ...'"
        )
    body = body.copy()
    body.columns = names
    df = body.reset_index(drop=True).infer_objects()
    return LoadedTable(df, sheet_name, explicit, warnings)


def _split_footnotes(body: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Strip trailing footnote rows (see _is_footnote); blank rows between
    them are skipped. Returns the rest and the dropped 1-based sheet rows."""
    keep = len(body)
    dropped: list[int] = []
    while keep > 0:
        row = body.iloc[keep - 1]
        if row.notna().any():
            if not _is_footnote(row):
                break
            dropped.append(int(body.index[keep - 1]) + 1)
        keep -= 1
    return body.iloc[:keep], sorted(dropped)


def _is_footnote(row: pd.Series) -> bool:
    """Mostly empty (at most half the columns, rounded up), all text, and the
    first filled cell reads like a note."""
    values = row.dropna()
    if len(values) > math.ceil(row.size / 2):
        return False
    if not all(isinstance(v, str) for v in values):
        return False
    first = values.iloc[0]
    return bool(_NOTE_START.match(first)) or len(first.split()) >= _NOTE_MIN_WORDS


def _detect_header_row(
    cells: pd.DataFrame, counts: pd.Series, sheet_name: str, options: OptionNames
) -> int:
    """0-based index of the single header row, or ValueError when unclear."""
    window = counts.iloc[:_HEADER_SCAN_ROWS]
    populated = counts[counts > 0]
    modal = populated.value_counts()
    width = int(modal[modal == modal.max()].index.max())
    threshold = max(2, math.ceil(_HEADER_FILL_RATIO * width))
    hint = (
        f"Set {options.header_rows} to the header row number or a range such as "
        "'3-4' (1-based, as numbered in Excel)."
    )

    candidates = window[window >= threshold]
    if candidates.empty:
        raise ValueError(
            f"could not find a header row in sheet {sheet_name!r}: no early row has "
            f"at least {threshold} filled cells. {hint}"
        )
    idx = int(candidates.index[0])
    row = cells.iloc[idx]
    values = row.dropna()
    if sum(_is_label(v) for v in values) < _LABEL_RATIO * len(values):
        start = idx
        while start > 0 and _is_label_row(cells.iloc[start - 1]):
            start -= 1
        if start < idx:
            raise ValueError(
                f"header is ambiguous in sheet {sheet_name!r}: rows {start + 1}-{idx} "
                f"hold labels ({_preview(cells.iloc[start].dropna())}) but are not "
                f"full-width, and row {idx + 1} looks like data. This may be a "
                f"multi-row header. {hint}"
            )
        raise ValueError(
            f"the first full-width row ({idx + 1}) of sheet {sheet_name!r} looks like "
            f"data, not a header: {_preview(values)}. {hint}"
        )

    if idx > 0:
        above = cells.iloc[idx - 1].dropna()
        if len(above) >= 2 and all(_is_label(v) for v in above):
            raise ValueError(
                f"header is ambiguous in sheet {sheet_name!r}: row {idx + 1} "
                f"({_preview(values)}) looks like a header, but row {idx} directly "
                f"above it has labels too ({_preview(above)}), which may be a "
                f"multi-row header or a banner. {hint}"
            )
    if idx + 1 < len(cells):
        below = cells.iloc[idx + 1]
        sub_labels = below.notna() & row.isna()
        if row.isna().any() and sub_labels.any() and _is_label_row(below):
            raise ValueError(
                f"header is ambiguous in sheet {sheet_name!r}: row {idx + 1} "
                f"({_preview(values)}) has blank header cells and row {idx + 2} "
                f"below it holds labels ({_preview(below.dropna())}), which may be "
                f"a multi-row header. {hint}"
            )
    if idx + 1 >= len(cells):
        raise ValueError(
            f"no data rows below the detected header (row {idx + 1}) in sheet "
            f"{sheet_name!r}. {hint}"
        )
    return idx


def _is_label_row(row: pd.Series) -> bool:
    values = row.dropna()
    return len(values) >= 2 and all(_is_label(v) for v in values)


def _preview(values: pd.Series, limit: int = 4) -> str:
    shown = ", ".join(repr(str(v)[:24]) for v in values.iloc[:limit])
    return shown + (", ..." if len(values) > limit else "")


def _header_text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    elif isinstance(value, datetime.datetime):
        if value.time() == datetime.time():
            value = value.date().isoformat()
        else:
            value = value.isoformat(sep=" ")
    elif isinstance(value, datetime.date):
        value = value.isoformat()
    return " ".join(str(value).split())


def _flatten_headers(header_block: pd.DataFrame) -> list[str]:
    """One name per column. Blank cells in upper rows continue the label to
    their left (merged / centred group headers); parts are joined, dropping
    blanks and immediate repeats."""
    rows = [
        [None if pd.isna(v) else _header_text(v) for v in header_block.iloc[r]]
        for r in range(len(header_block))
    ]
    for r in range(len(rows) - 1):
        carried = None
        for c, value in enumerate(rows[r]):
            if value is None:
                rows[r][c] = carried
            else:
                carried = value

    names: list[str] = []
    for c in range(header_block.shape[1]):
        parts: list[str] = []
        for r in range(len(rows)):
            part = rows[r][c]
            if part and (not parts or parts[-1] != part):
                parts.append(part)
        names.append(_HEADER_SEPARATOR.join(parts) or f"column_{c + 1}")
    return _make_unique(names)


def _make_unique(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        candidate, n = name, 1
        while candidate in seen:
            n += 1
            candidate = f"{name}_{n}"
        seen.add(candidate)
        result.append(candidate)
    return result
