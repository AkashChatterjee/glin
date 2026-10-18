import sys

import pytest

from glin import engine, server
from tests.excel_helpers import churn_rows, make_workbook, messy_churn_sheet
from tests.mcp_compat import call_structured, input_schema

pytest.importorskip("openpyxl")

ToolError = server.ToolError


def _train(models_root, allowed, path, **extra):
    mcp = server.build_server(models_root=models_root, allowed_dirs=allowed)
    return call_structured(
        mcp,
        "train_model",
        {"csv_path": str(path), "target_column": "churn", "model_name": "xl", **extra},
    )


@pytest.fixture
def messy_xlsx(tmp_path):
    return make_workbook(tmp_path / "messy.xlsx", {"Data": messy_churn_sheet()})


@pytest.mark.asyncio
async def test_train_model_reads_messy_xlsx_via_csv_path(tmp_path, messy_xlsx):
    result = await _train(tmp_path / "models", [tmp_path], messy_xlsx)
    assert result["rows_read"] == 200
    assert result["sheet"] == "Data"
    assert result["header_rows"] == [4]
    assert set(result["target_classes"]) == {"Yes", "No"}


@pytest.mark.asyncio
async def test_train_model_sheet_and_multi_row_header_options(tmp_path):
    rows = [
        ["Report"],
        ["Customer", None, "churn"],
        ["age", "plan", None],
        *churn_rows(),
    ]
    sheets = {"Notes": ([["hello"]], []), "Data": (rows, ["A2:B2", "C2:C3"])}
    path = make_workbook(tmp_path / "multi.xlsx", sheets)
    models = tmp_path / "models"

    with pytest.raises(ToolError, match="2 sheets"):
        await _train(models, [tmp_path], path)

    result = await _train(
        models, [tmp_path], path, sheet=2, header_rows="2-3", overwrite=True
    )
    assert result["sheet"] == "Data"
    assert result["header_rows"] == [2, 3]
    assert result["rows_read"] == 200
    assert result["features_kept"] == ["Customer - age", "Customer - plan"]

    result = await _train(
        models, [tmp_path], path, sheet="Data", header_rows=[2, 3], overwrite=True
    )
    assert result["header_rows"] == [2, 3]


@pytest.mark.asyncio
async def test_footnote_rows_are_dropped_and_reported(tmp_path):
    rows = [["age", "plan", "churn"], *churn_rows(), ["Source:", "Dept of Stuff"]]
    path = make_workbook(tmp_path / "f.xlsx", {"Data": (rows, [])})
    result = await _train(tmp_path / "models", [tmp_path], path, holdout_fraction=0)
    assert result["rows_read"] == 200
    assert set(result["target_classes"]) == {"Yes", "No"}
    assert any("sheet row(s) 202" in w for w in result["warnings"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra, message",
    [
        ({"sheet": True}, "sheet must not be true or false"),
        ({"header_rows": True}, "header_rows must not be true or false"),
        ({"header_rows": [True]}, "header_rows must not be true or false"),
        ({"header_rows": []}, "header_rows is empty"),
    ],
)
async def test_train_model_rejects_bool_and_empty_excel_options(
    tmp_path, messy_xlsx, extra, message
):
    with pytest.raises(ToolError, match=message):
        await _train(tmp_path / "models", [tmp_path], messy_xlsx, **extra)


@pytest.mark.asyncio
async def test_excel_path_obeys_allowed_dirs_and_symlink_rules(tmp_path, messy_xlsx):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    models = tmp_path / "models"
    with pytest.raises(ToolError, match="not found or outside"):
        await _train(models, [allowed], messy_xlsx)
    with pytest.raises(ToolError, match="not found or outside"):
        await _train(models, [allowed], allowed / ".." / "messy.xlsx")

    link = allowed / "link.xlsx"
    link.symlink_to(messy_xlsx)
    with pytest.raises(ToolError, match="not found or outside"):
        await _train(models, [allowed], link)

    inside = allowed / "inside.xlsx"
    inside.write_bytes(messy_xlsx.read_bytes())
    result = await _train(models, [allowed], inside)
    assert result["rows_read"] == 200


@pytest.mark.asyncio
async def test_excel_path_rejects_other_extensions_and_directories(tmp_path):
    (tmp_path / "x.txt").write_text("TOPSECRET")
    with pytest.raises(ToolError, match=r"\.csv, \.xlsx or \.xls") as exc:
        await _train(tmp_path / "m", [tmp_path], tmp_path / "x.txt")
    assert "TOPSECRET" not in str(exc.value)
    (tmp_path / "dir.xlsx").mkdir()
    with pytest.raises(ToolError, match="regular .xlsx file"):
        await _train(tmp_path / "m", [tmp_path], tmp_path / "dir.xlsx")


@pytest.mark.asyncio
async def test_excel_options_rejected_for_csv_content_and_csv_path(tmp_path):
    mcp = server.build_server(models_root=tmp_path / "m", allowed_dirs=[tmp_path])
    csv_text = "a,b\n" + "\n".join(f"{i},{i % 2}" for i in range(30))
    with pytest.raises(ToolError, match="only apply"):
        await mcp.call_tool(
            "train_model",
            {"csv_content": csv_text, "target_column": "b", "model_name": "m", "sheet": "x"},
        )
    csv = tmp_path / "d.csv"
    csv.write_text(csv_text)
    with pytest.raises(ToolError, match="only apply"):
        await _train(tmp_path / "m", [tmp_path], csv, header_rows=1)


@pytest.mark.asyncio
async def test_missing_excel_extra_surfaces_install_hint(tmp_path, messy_xlsx, monkeypatch):
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(ToolError, match=r"glin-ml\[excel\]"):
        await _train(tmp_path / "models", [tmp_path], messy_xlsx)


@pytest.mark.asyncio
async def test_train_model_schema_documents_excel_options(tmp_path):
    mcp = server.build_server(models_root=tmp_path, allowed_dirs=[tmp_path])
    tool = next(t for t in await mcp.list_tools() if t.name == "train_model")
    props = input_schema(tool)["properties"]
    assert "sheet" in props and "header_rows" in props
    assert "csv_path" in props


@pytest.mark.asyncio
async def test_train_model_saves_messy_excel_and_reports_footnote(tmp_path, messy_xlsx):
    result = await _train(tmp_path / "models", [tmp_path], messy_xlsx)
    assert result["rows_read"] == 200
    assert len(result["features_kept"]) + len(result["features_dropped"]) == 2
    assert any("dropped 1 footnote row(s)" in w for w in result["warnings"])
    assert (tmp_path / "models" / "xl" / engine.METADATA_FILENAME).exists()


@pytest.mark.asyncio
async def test_train_model_sheet_choice_names_the_sheet_argument(tmp_path):
    rows = [["Customer", None, "churn"], ["age", "plan", None], *churn_rows()]
    path = make_workbook(tmp_path / "m.xlsx", {"A": ([["x"]], []), "B": (rows, [])})
    models = tmp_path / "models"

    with pytest.raises(ToolError, match="2 sheets; choose one with sheet") as exc:
        await _train(models, [tmp_path], path)
    assert "--sheet" not in str(exc.value)

    result = await _train(models, [tmp_path], path, sheet="B", header_rows="1-2")
    assert result["header_rows"] == [1, 2]


@pytest.mark.asyncio
async def test_train_model_ambiguous_header_is_a_clean_error(tmp_path):
    rows = [["Customer", None, "churn"], ["age", "plan", None], *churn_rows()]
    path = make_workbook(tmp_path / "m.xlsx", {"B": (rows, [])})
    with pytest.raises(ToolError, match="ambiguous") as exc:
        await _train(tmp_path / "models", [tmp_path], path)
    message = str(exc.value)
    assert "header_rows" in message and "--header-rows" not in message
    assert "Traceback" not in message
    assert not (tmp_path / "models" / "xl").exists()
