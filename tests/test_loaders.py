import datetime
import sys

import pandas as pd
import pytest

from glin import loaders
from glin.loaders import load_table, parse_header_rows
from tests.excel_helpers import make_workbook

openpyxl = pytest.importorskip("openpyxl")


def _data_rows(n=30):
    return [[f"r{i % 3}", i, 1.5 * i, "Yes" if i % 2 else "No"] for i in range(n)]


def banner_sheet():
    rows = [
        ["Quarterly Report 2023"],
        ["Source: Ministry of Data"],
        [],
        ["region", "count", "score", "outcome"],
        *_data_rows(),
        [],
        ["Source: some footnote"],
    ]
    return rows, []


def test_parse_header_rows_forms():
    assert parse_header_rows(None) is None
    assert parse_header_rows(4) == [4]
    assert parse_header_rows("4") == [4]
    assert parse_header_rows("3-5") == [3, 4, 5]
    assert parse_header_rows("3,4") == [3, 4]
    assert parse_header_rows([4, 3]) == [3, 4]
    assert parse_header_rows(range(2, 4)) == [2, 3]


@pytest.mark.parametrize("bad", [0, "abc", "5-3", [1, 3], [1.5], True, "3-", []])
def test_parse_header_rows_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_header_rows(bad)


def test_empty_header_rows_list_has_clear_message():
    with pytest.raises(ValueError, match="header_rows is empty"):
        parse_header_rows([])


def test_single_sheet_clean_header_is_autodetected(tmp_path):
    path = make_workbook(
        tmp_path / "clean.xlsx", {"Data": ([["a", "b", "c", "d"], *_data_rows()], [])}
    )
    table = load_table(path)
    assert list(table.df.columns) == ["a", "b", "c", "d"]
    assert len(table.df) == 30
    assert table.sheet == "Data"
    assert table.header_rows == [1]
    assert pd.api.types.is_integer_dtype(table.df["b"])
    assert pd.api.types.is_float_dtype(table.df["c"])


def test_banner_rows_and_footnotes_are_skipped(tmp_path):
    path = make_workbook(tmp_path / "banner.xlsx", {"Data": banner_sheet()})
    table = load_table(path)
    assert list(table.df.columns) == ["region", "count", "score", "outcome"]
    assert len(table.df) == 30
    assert table.header_rows == [4]
    assert table.df["count"].iloc[-1] == 29


def test_banner_footnote_warning_gives_sheet_row(tmp_path):
    path = make_workbook(tmp_path / "banner.xlsx", {"Data": banner_sheet()})
    (warning,) = load_table(path).warnings
    assert "1 footnote row(s)" in warning and "sheet row(s) 36" in warning


def test_two_cell_footnote_is_dropped(tmp_path):
    rows = [["region", "count", "score", "outcome"], *_data_rows(), ["Source:", "Dept of Stuff"]]
    path = make_workbook(tmp_path / "f.xlsx", {"S": (rows, [])})
    table = load_table(path)
    assert len(table.df) == 30
    assert "Dept of Stuff" not in table.df["count"].tolist()
    assert "sheet row(s) 32" in table.warnings[0]


def test_footnote_is_dropped_on_two_column_sheet(tmp_path):
    rows = [["x", "target"], *[[i, "y" if i % 2 else "n"] for i in range(10)]]
    rows += [[], ["Source: thing"], ["Figures may not add up because of rounding"]]
    path = make_workbook(tmp_path / "f.xlsx", {"S": (rows, [])})
    table = load_table(path)
    assert len(table.df) == 10
    assert table.df["x"].dtype.kind == "i"
    assert "sheet row(s) 13, 14" in table.warnings[0]


@pytest.mark.parametrize(
    "last", [[99, None, None, None], ["r1", None, None, None], [None, "Notebook"]]
)
def test_sparse_final_data_row_is_kept(tmp_path, last):
    rows = [["region", "count", "score", "outcome"], *_data_rows(), last]
    path = make_workbook(tmp_path / "f.xlsx", {"S": (rows, [])})
    table = load_table(path)
    assert len(table.df) == 31
    assert table.warnings == []


def test_date_header_cells_become_iso_dates(tmp_path):
    jan, feb = datetime.date(2024, 1, 1), datetime.datetime(2024, 2, 1, 9, 30)
    rows = [["name", jan, feb], *[[f"k{i}", i, i] for i in range(5)]]
    path = make_workbook(tmp_path / "d.xlsx", {"S": (rows, [])})
    assert load_table(path).df.columns.tolist() == ["name", "2024-01-01", "2024-02-01 09:30:00"]


def test_missing_cells_in_data_are_kept_as_nan(tmp_path):
    rows = [["title"], ["a", "b", "c"], [1, 2, 3], [4, None, 6], [7, 8, 9], [1, 2, 3]]
    path = make_workbook(tmp_path / "gaps.xlsx", {"S": (rows, [])})
    table = load_table(path)
    assert table.header_rows == [2]
    assert len(table.df) == 4
    assert table.df["b"].isna().sum() == 1


def test_multi_row_merged_header_with_explicit_header_rows(tmp_path):
    rows = [
        ["Report title"],
        [],
        ["region", "2023", None, "2024", None, "label"],
        [None, "sales", "cost", "sales", "cost", None],
        *[[f"r{i % 2}", i, i * 2, i + 1, i * 3, "Y" if i % 2 else "N"] for i in range(20)],
    ]
    merges = ["B3:C3", "D3:E3", "A3:A4", "F3:F4"]
    path = make_workbook(tmp_path / "multi.xlsx", {"S": (rows, merges)})

    table = load_table(path, header_rows="3-4")
    assert list(table.df.columns) == [
        "region",
        "2023 - sales",
        "2023 - cost",
        "2024 - sales",
        "2024 - cost",
        "label",
    ]
    assert len(table.df) == 20
    assert table.header_rows == [3, 4]
    assert load_table(path, header_rows=[3, 4]).df.columns.tolist() == table.df.columns.tolist()


def test_multi_row_header_without_option_is_ambiguous_with_actionable_error(tmp_path):
    rows = [
        ["region", "2023", None, "2024", None, "label"],
        [None, "sales", "cost", "sales", "cost", None],
        *[[f"r{i % 2}", i, i * 2, i + 1, i * 3, "Y"] for i in range(20)],
    ]
    path = make_workbook(tmp_path / "multi.xlsx", {"S": (rows, [])})
    with pytest.raises(ValueError, match="ambiguous") as exc:
        load_table(path)
    assert "header_rows" in str(exc.value)


def test_duplicate_and_blank_header_names_are_made_unique(tmp_path):
    rows = [["x", "x", None, "y\nz"], *[[i, i, i, i] for i in range(5)]]
    path = make_workbook(tmp_path / "dups.xlsx", {"S": (rows, [])})
    assert load_table(path).df.columns.tolist() == ["x", "x_2", "column_3", "y z"]


def test_numeric_header_labels_become_clean_strings(tmp_path):
    rows = [["id", 2022, 2023], *[[f"k{i}", i, i] for i in range(5)]]
    path = make_workbook(tmp_path / "years.xlsx", {"S": (rows, [])})
    assert load_table(path).df.columns.tolist() == ["id", "2022", "2023"]


def test_header_row_that_looks_like_data_is_an_error(tmp_path):
    rows = [[i, i * 2, i * 3] for i in range(10)]
    path = make_workbook(tmp_path / "nohdr.xlsx", {"S": (rows, [])})
    with pytest.raises(ValueError, match="looks like data") as exc:
        load_table(path)
    assert "header_rows" in str(exc.value)


def test_explicit_header_rows_beyond_sheet_and_without_data(tmp_path):
    path = make_workbook(tmp_path / "s.xlsx", {"S": ([["a", "b"], [1, 2]], [])})
    with pytest.raises(ValueError, match="beyond the last row"):
        load_table(path, header_rows=9)
    with pytest.raises(ValueError, match="no data rows"):
        load_table(path, header_rows=2)


def test_empty_sheet_is_an_error(tmp_path):
    path = make_workbook(tmp_path / "e.xlsx", {"S": ([], [])})
    with pytest.raises(ValueError, match="empty"):
        load_table(path)


@pytest.fixture
def two_sheets(tmp_path):
    return make_workbook(
        tmp_path / "two.xlsx",
        {
            "Notes": ([["read me"], ["nothing here"]], []),
            "Data 2023": banner_sheet(),
        },
    )


def test_multiple_sheets_without_choice_lists_available(two_sheets):
    with pytest.raises(ValueError, match="2 sheets") as exc:
        load_table(two_sheets)
    assert "'Notes'" in str(exc.value) and "'Data 2023'" in str(exc.value)


def test_sheet_by_name_position_and_numeric_string(two_sheets):
    for ref in ("Data 2023", "data 2023", 2, "2"):
        table = load_table(two_sheets, sheet=ref)
        assert table.sheet == "Data 2023"
        assert len(table.df) == 30


def test_unknown_sheet_lists_available(two_sheets):
    with pytest.raises(ValueError, match="no sheet named 'Nope'") as exc:
        load_table(two_sheets, sheet="Nope")
    assert "Available" in str(exc.value)
    with pytest.raises(ValueError, match="out of range"):
        load_table(two_sheets, sheet=3)


def test_sheet_name_that_matches_several_sheets_is_ambiguous(tmp_path):
    sheets = {"Notes": ([["x"]], []), "Data": banner_sheet(), "data ": banner_sheet()}
    path = make_workbook(tmp_path / "s.xlsx", sheets)
    with pytest.raises(ValueError, match="'data' is ambiguous") as exc:
        load_table(path, sheet="data")
    assert "'Data', 'data '" in str(exc.value)
    assert load_table(path, sheet="Data").sheet == "Data"


def test_csv_is_unchanged_and_rejects_excel_options(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    table = load_table(path)
    assert table.df.equals(pd.read_csv(path))
    assert table.sheet is None and table.header_rows is None
    with pytest.raises(ValueError, match="only apply"):
        load_table(path, sheet="x")
    with pytest.raises(ValueError, match="only apply"):
        load_table(path, header_rows=1)


def test_missing_openpyxl_gives_install_hint(tmp_path, monkeypatch):
    path = make_workbook(tmp_path / "x.xlsx", {"S": ([["a", "b"], [1, 2]], [])})
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(ValueError, match=r"glin-ml\[excel\]") as exc:
        load_table(path)
    assert "openpyxl" in str(exc.value)
    assert "pip install 'glin-ml[excel]'" in str(exc.value)
    assert "uvx --from 'glin-ml[excel]' glin serve" in str(exc.value)


def test_missing_xlrd_gives_install_hint(tmp_path, monkeypatch):
    xlwt = pytest.importorskip("xlwt")
    path = tmp_path / "x.xls"
    book = xlwt.Workbook()
    sheet = book.add_sheet("S")
    for r, row in enumerate([["a", "b"], [1, 2]]):
        for c, value in enumerate(row):
            sheet.write(r, c, value)
    book.save(str(path))
    monkeypatch.setitem(sys.modules, "xlrd", None)
    with pytest.raises(ValueError, match=r"glin-ml\[excel\]") as exc:
        load_table(path)
    assert "xlrd" in str(exc.value)


def test_corrupt_workbook_error_does_not_echo_contents(tmp_path):
    path = tmp_path / "bad.xlsx"
    path.write_text("TOPSECRET not a zip")
    with pytest.raises(ValueError, match="could not read") as exc:
        load_table(path)
    assert "TOPSECRET" not in str(exc.value)


def test_legacy_xls_with_banner_and_multi_row_header(tmp_path):
    xlwt = pytest.importorskip("xlwt")
    rows = [
        ["Annual figures"],
        [],
        ["region", "2023", None, "label"],
        [None, "sales", "cost", None],
        *[[f"r{i % 2}", i, i * 2, "Y" if i % 2 else "N"] for i in range(12)],
    ]
    path = tmp_path / "legacy.xls"
    book = xlwt.Workbook()
    sheet = book.add_sheet("Data")
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if value is not None:
                sheet.write(r, c, value)
    book.save(str(path))

    table = load_table(path, header_rows="3-4")
    assert table.df.columns.tolist() == ["region", "2023 - sales", "2023 - cost", "label"]
    assert len(table.df) == 12
    assert table.df["2023 - cost"].iloc[3] == 6
    assert loaders.is_excel_path(path)
