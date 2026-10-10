import sys

import pytest
from click.testing import CliRunner
from mcp.server.fastmcp.exceptions import ToolError

from glin import cli, engine, server
from tests.excel_helpers import churn_rows, make_workbook, messy_churn_sheet

pytest.importorskip("openpyxl")


def _train(models_root, allowed, path, **extra):
    mcp = server.build_server(models_root=models_root, allowed_dirs=allowed)
    return mcp.call_tool(
        "train_model",
        {"csv_path": str(path), "target_column": "churn", "model_name": "xl", **extra},
    )


@pytest.fixture
def messy_xlsx(tmp_path):
    return make_workbook(tmp_path / "messy.xlsx", {"Data": messy_churn_sheet()})


@pytest.mark.asyncio
async def test_train_model_reads_messy_xlsx_via_csv_path(tmp_path, messy_xlsx):
    _, result = await _train(tmp_path / "models", [tmp_path], messy_xlsx)
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

    _, result = await _train(
        models, [tmp_path], path, sheet=2, header_rows="2-3", overwrite=True
    )
    assert result["sheet"] == "Data"
    assert result["header_rows"] == [2, 3]
    assert result["rows_read"] == 200
    assert result["features_kept"] == ["Customer - age", "Customer - plan"]

    _, result = await _train(
        models, [tmp_path], path, sheet="Data", header_rows=[2, 3], overwrite=True
    )
    assert result["header_rows"] == [2, 3]


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
    _, result = await _train(models, [allowed], inside)
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
    props = tool.inputSchema["properties"]
    assert "sheet" in props and "header_rows" in props
    assert "csv_path" in props


def test_cli_train_reads_messy_excel(tmp_path, monkeypatch, messy_xlsx):
    monkeypatch.setattr(cli, "DEFAULT_MODELS_ROOT", tmp_path / "models")
    result = CliRunner().invoke(
        cli.main, ["train", str(messy_xlsx), "--target", "churn", "--name", "xl", "--quiet"]
    )
    assert result.exit_code == 0, result.output
    assert "sheet 'Data', header row(s): 4" in result.output
    assert "200 rows, 3 columns" in result.output
    assert (tmp_path / "models" / "xl" / engine.METADATA_FILENAME).exists()


def test_cli_train_sheet_and_header_rows_options(tmp_path, monkeypatch):
    rows = [["Customer", None, "churn"], ["age", "plan", None], *churn_rows()]
    path = make_workbook(tmp_path / "m.xlsx", {"A": ([["x"]], []), "B": (rows, [])})
    monkeypatch.setattr(cli, "DEFAULT_MODELS_ROOT", tmp_path / "models")

    args = ["train", str(path), "--target", "churn", "--name", "xl", "--quiet"]
    missing = CliRunner().invoke(cli.main, args)
    assert missing.exit_code != 0
    assert "2 sheets" in missing.output

    ok = CliRunner().invoke(cli.main, [*args, "--sheet", "B", "--header-rows", "1-2"])
    assert ok.exit_code == 0, ok.output
    assert "header row(s): 1, 2" in ok.output


def test_cli_train_ambiguous_header_is_a_clean_error(tmp_path, monkeypatch):
    rows = [["Customer", None, "churn"], ["age", "plan", None], *churn_rows()]
    path = make_workbook(tmp_path / "m.xlsx", {"B": (rows, [])})
    monkeypatch.setattr(cli, "DEFAULT_MODELS_ROOT", tmp_path / "models")
    result = CliRunner().invoke(
        cli.main, ["train", str(path), "--target", "churn", "--name", "xl", "--quiet"]
    )
    assert result.exit_code != 0
    assert "ambiguous" in result.output and "header_rows" in result.output
    assert "Traceback" not in result.output
