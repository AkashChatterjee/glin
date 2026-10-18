import numpy as np
import pandas as pd
import pytest

import glin
from glin import cli, engine, server
from tests.mcp_compat import call_structured


def _train_fixture_model(models_root, model_name="test_model"):
    rng = np.random.RandomState(0)
    n = 100
    df = pd.DataFrame(
        {
            "age": rng.randint(18, 80, size=n).astype(float),
            "plan": rng.choice(["basic", "premium"], size=n),
        }
    )
    df["churn"] = np.where(df["age"] < 40, "Yes", "No")
    bundle = engine.train_model(df, "churn", model_name=model_name)
    engine.save_bundle(bundle, models_root)


@pytest.mark.asyncio
async def test_delete_model_removes_model_directory(tmp_path):
    _train_fixture_model(tmp_path)
    _train_fixture_model(tmp_path, "other")
    mcp = server.build_server(models_root=tmp_path)

    result = await call_structured(mcp, "delete_model", {"model_name": "test_model"})

    assert result["deleted"] is True
    assert result["model_name"] == "test_model"
    assert not (tmp_path / "test_model").exists()
    assert (tmp_path / "other").exists()
    listed = await mcp.call_tool("list_models", {})
    assert "test_model" not in str(listed)


@pytest.mark.asyncio
async def test_delete_model_unknown_model_errors(tmp_path):
    mcp = server.build_server(models_root=tmp_path)
    with pytest.raises(server.ToolError, match="model 'does_not_exist' not found"):
        await mcp.call_tool("delete_model", {"model_name": "does_not_exist"})


@pytest.mark.asyncio
async def test_train_model_reports_date_derived_features(tmp_path):
    rng = np.random.RandomState(0)
    days = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.randint(0, 700, 200), unit="D")
    csv = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "signup": days.strftime("%d/%m/%Y"),
            "churn": np.where(days.dayofweek >= 5, "Yes", "No"),
        }
    ).to_csv(csv, index=False)
    mcp = server.build_server(models_root=tmp_path / "models", allowed_dirs=[tmp_path])

    result = await call_structured(
        mcp, "train_model", {"csv_path": str(csv), "target_column": "churn", "model_name": "m"}
    )

    assert result["features_dropped"] == []
    derived = result["date_columns"]["signup"]
    assert derived and all(name.startswith("signup__") for name in derived)
    assert set(derived) <= set(result["features_kept"])
    meta = engine.load_metadata(tmp_path / "models" / "m")
    assert meta["features"]["date"]["signup"]["features"] == derived
    inspected = await call_structured(mcp, "inspect_model", {"model_name": "m"})
    assert inspected["features"]["date"]["signup"]["features"] == derived


@pytest.mark.asyncio
async def test_train_model_reports_leakage_warnings_and_still_succeeds(tmp_path):
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {"noise": rng.normal(size=200).round(1), "tox": rng.uniform(0, 10, 200).round(2)}
    )
    df["unsafe"] = np.where(df["tox"] > 6, "yes", "no")
    csv = tmp_path / "leak.csv"
    df.to_csv(csv, index=False)
    mcp = server.build_server(models_root=tmp_path / "models", allowed_dirs=[tmp_path])

    result = await call_structured(
        mcp, "train_model", {"csv_path": str(csv), "target_column": "unsafe", "model_name": "m"}
    )

    assert any("Possible target leakage: 'tox'" in w for w in result["warnings"])
    assert (tmp_path / "models" / "m" / engine.BUNDLE_FILENAME).exists()


def _run(capsys, monkeypatch, argv):
    calls = []
    monkeypatch.setattr(cli.server, "run_stdio", lambda **kw: calls.append(kw))
    try:
        cli.main(argv)
        code = 0
    except SystemExit as exc:
        code = exc.code
    out, err = capsys.readouterr()
    return code, out, err, calls


def test_version_option_prints_package_version(capsys, monkeypatch):
    code, out, _, _ = _run(capsys, monkeypatch, ["--version"])

    assert code == 0
    assert glin.__version__ in out


@pytest.mark.parametrize("command", ["train", "list", "delete"])
def test_removed_commands_do_not_exist(capsys, monkeypatch, command):
    code, out, err, calls = _run(capsys, monkeypatch, [command])

    assert code == 2
    assert calls == []
    assert "invalid choice" in err


def test_serve_accepts_deprecated_mode_stdio_with_notice_on_stderr_only(capsys, monkeypatch):
    code, out, err, calls = _run(capsys, monkeypatch, ["serve", "--mode", "stdio"])

    assert code == 0
    assert calls == [{"allowed_dirs": None, "max_file_mb": 200}]
    assert out == ""
    assert err.count("deprecated") == 1


def test_serve_without_mode_writes_nothing(capsys, monkeypatch):
    code, out, err, calls = _run(capsys, monkeypatch, ["serve"])

    assert code == 0
    assert len(calls) == 1
    assert out == "" and err == ""


@pytest.mark.parametrize("mode", ["http", "sse"])
def test_serve_http_mode_explains_removal(capsys, monkeypatch, mode):
    code, out, err, calls = _run(capsys, monkeypatch, ["serve", "--mode", mode])

    assert code == 2
    assert calls == []
    assert out == ""
    assert (
        "HTTP mode was removed in v0.1.4; v0.1.3 is the last release with it. "
        "glin runs over stdio only."
    ) in err


def test_serve_passes_allow_dir_and_max_file_mb(capsys, monkeypatch, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    code, _, err, calls = _run(
        capsys,
        monkeypatch,
        ["serve", "--allow-dir", str(tmp_path), "--allow-dir", str(other), "--max-file-mb", "50"],
    )

    assert code == 0, err
    assert calls == [{"allowed_dirs": [tmp_path, other], "max_file_mb": 50}]


def test_serve_rejects_missing_allow_dir(capsys, monkeypatch, tmp_path):
    code, _, _, calls = _run(
        capsys, monkeypatch, ["serve", "--allow-dir", str(tmp_path / "nope")]
    )

    assert code == 2
    assert calls == []


@pytest.mark.parametrize("value", ["0", "-1", "abc", "nan"])
def test_serve_rejects_non_positive_max_file_mb(capsys, monkeypatch, value):
    code, _, _, calls = _run(capsys, monkeypatch, ["serve", "--max-file-mb", value])

    assert code == 2
    assert calls == []


def test_serve_rejects_unknown_mode(capsys, monkeypatch):
    code, _, _, calls = _run(capsys, monkeypatch, ["serve", "--mode", "ftp"])

    assert code == 2
    assert calls == []
