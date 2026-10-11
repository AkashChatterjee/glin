import json
import shutil

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from glin import cli, engine, server
from tests.mcp_compat import call_structured

VALID_NAMES = ["m", "churn_v1", "report-v1.2", "A9", "a..b", "x" * 64]


def _path_names(tmp_path):
    """Names that are not a single folder name: rejected on load and delete too."""
    return ["../escape", "..", ".", "", "a/b", "a\\b", str(tmp_path / "escape")]


# Single folder names that only the stricter rule for new models rejects.
STRICT_ONLY_BAD = [".hidden", "x" * 65, "has space", "line\n"]


def _bad_names(tmp_path):
    return [*_path_names(tmp_path), *STRICT_ONLY_BAD]


def _df(n=200):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"age": rng.randint(18, 80, size=n).astype(float)})
    df["churn"] = np.where(df["age"] < 40, "Yes", "No")
    return df


@pytest.fixture(scope="module")
def trained_bundle():
    return engine.train_model(_df(), "churn", model_name="good")


@pytest.fixture
def roots(tmp_path, trained_bundle):
    """models root with model 'good', plus a decoy model just outside it."""
    models_root = tmp_path / "models"
    good = engine.save_bundle(trained_bundle, models_root)
    shutil.copytree(good, tmp_path / "escape")
    return models_root


@pytest.mark.parametrize("name", VALID_NAMES)
def test_valid_names_pass(tmp_path, name):
    assert engine.validate_model_name(name) == name
    assert engine.model_dir_for(tmp_path, name) == tmp_path / name


def test_invalid_names_are_rejected_with_the_rule(tmp_path):
    for name in [*_bad_names(tmp_path), None, 5]:
        with pytest.raises(ValueError, match="1-64 characters") as exc:
            engine.model_dir_for(tmp_path, name)
        assert "must not start with '.'" in str(exc.value)


def test_existing_names_only_need_to_be_one_folder(tmp_path):
    for name in STRICT_ONLY_BAD:
        assert engine.model_dir_for(tmp_path, name, existing=True) == tmp_path / name
    for name in [*_path_names(tmp_path), "nul\0", None]:
        with pytest.raises(ValueError, match="single folder name"):
            engine.model_dir_for(tmp_path, name, existing=True)


@pytest.mark.parametrize("existing", [False, True])
def test_symlinked_model_dir_outside_root_is_rejected(roots, tmp_path, existing):
    (roots / "linked").symlink_to(tmp_path / "escape", target_is_directory=True)
    with pytest.raises(ValueError, match="invalid model name"):
        engine.model_dir_for(roots, "linked", existing=existing)


def test_save_bundle_rejects_bad_name(tmp_path, trained_bundle):
    bundle = {**trained_bundle, "model_name": "../escape"}
    with pytest.raises(ValueError, match="invalid model name"):
        engine.save_bundle(bundle, tmp_path / "models")
    assert not (tmp_path / "escape").exists()


@pytest.mark.asyncio
async def test_tools_reject_bad_names(roots, tmp_path):
    mcp = server.build_server(models_root=roots)
    csv = _df(50).to_csv(index=False)
    for name in _bad_names(tmp_path):
        with pytest.raises(server.ToolError, match="1-64 characters"):
            await mcp.call_tool(
                "train_model",
                {"model_name": name, "csv_content": csv, "target_column": "churn",
                 "overwrite": True},
            )
    loaders = {"predict": {"features": {"age": 30}}, "explain_model": {}, "inspect_model": {}}
    for tool, args in loaders.items():
        for name in _path_names(tmp_path):
            with pytest.raises(server.ToolError, match="single folder name"):
                await mcp.call_tool(tool, {"model_name": name, **args})
        for name in STRICT_ONLY_BAD:
            with pytest.raises(server.ToolError, match="not found"):
                await mcp.call_tool(tool, {"model_name": name, **args})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["escape", "models"]
    assert sorted(p.name for p in roots.iterdir()) == ["good"]


def _legacy_model(roots, name="old model"):
    """A model saved by an older glin that allowed any name."""
    legacy = roots / name
    shutil.copytree(roots / "good", legacy)
    meta = engine.load_metadata(legacy)
    meta["model_name"] = name
    (legacy / engine.METADATA_FILENAME).write_text(json.dumps(meta))
    return legacy


@pytest.mark.asyncio
async def test_legacy_name_with_space_still_loads(roots):
    _legacy_model(roots)
    mcp = server.build_server(models_root=roots)
    pred = await call_structured(
        mcp, "predict", {"model_name": "old model", "features": {"age": 30}}
    )
    assert pred["predicted_class"] == "Yes"
    meta = await call_structured(mcp, "inspect_model", {"model_name": "old model"})
    assert meta["model_name"] == "old model"


def test_cli_delete_removes_legacy_name_with_space(roots, monkeypatch):
    legacy = _legacy_model(roots)
    monkeypatch.setattr(cli, "DEFAULT_MODELS_ROOT", roots)
    result = CliRunner().invoke(cli.main, ["delete", "old model", "--yes"])
    assert result.exit_code == 0, result.output
    assert not legacy.exists()
    assert (roots / "good").exists()


@pytest.mark.asyncio
async def test_tools_accept_valid_name(roots):
    mcp = server.build_server(models_root=roots)
    meta = await call_structured(mcp, "inspect_model", {"model_name": "good"})
    assert meta["model_name"] == "good"
    pred = await call_structured(
        mcp, "predict", {"model_name": "good", "features": {"age": 30}}
    )
    assert pred["predicted_class"] == "Yes"


def test_cli_delete_rejects_bad_names(roots, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_MODELS_ROOT", roots)
    runner = CliRunner()
    for name in _path_names(tmp_path):
        result = runner.invoke(cli.main, ["delete", name, "--yes"])
        assert result.exit_code != 0
        assert "single folder name" in result.output
    for name in STRICT_ONLY_BAD:
        result = runner.invoke(cli.main, ["delete", name, "--yes"])
        assert result.exit_code != 0
        assert "not found" in result.output
    assert (tmp_path / "escape" / engine.METADATA_FILENAME).exists()
    assert (roots / "good").exists()


def test_cli_train_rejects_bad_name_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_MODELS_ROOT", tmp_path / "models")
    monkeypatch.setattr(cli.engine, "train_model", lambda *a, **k: pytest.fail("trained"))
    csv_path = tmp_path / "data.csv"
    _df().to_csv(csv_path, index=False)
    result = CliRunner().invoke(
        cli.main, ["train", str(csv_path), "--target", "churn", "--name", "../escape"]
    )
    assert result.exit_code != 0
    assert "1-64 characters" in result.output
    assert not (tmp_path / "escape").exists()
