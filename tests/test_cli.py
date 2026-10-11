import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import glin
from glin import cli, engine


def _train_fixture_model(models_root, model_name="cli_test_model"):
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


def test_delete_removes_model_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_MODELS_ROOT", tmp_path)
    _train_fixture_model(tmp_path)
    assert (tmp_path / "cli_test_model").exists()

    runner = CliRunner()
    result = runner.invoke(cli.main, ["delete", "cli_test_model", "--yes"])

    assert result.exit_code == 0
    assert "Deleted model 'cli_test_model'" in result.output
    assert not (tmp_path / "cli_test_model").exists()


def test_delete_unknown_model_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_MODELS_ROOT", tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["delete", "does_not_exist", "--yes"])

    assert result.exit_code != 0
    assert "not found" in result.output


def test_delete_without_yes_prompts_and_aborts_on_no(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_MODELS_ROOT", tmp_path)
    _train_fixture_model(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["delete", "cli_test_model"], input="n\n")

    assert result.exit_code != 0
    assert (tmp_path / "cli_test_model").exists()  # not deleted


def test_delete_without_yes_prompts_and_confirms_on_yes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_MODELS_ROOT", tmp_path)
    _train_fixture_model(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["delete", "cli_test_model"], input="y\n")

    assert result.exit_code == 0
    assert not (tmp_path / "cli_test_model").exists()


def test_train_and_list_report_date_derived_features(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_MODELS_ROOT", tmp_path / "models")
    rng = np.random.RandomState(0)
    days = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.randint(0, 700, 200), unit="D")
    csv = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "signup": days.strftime("%d/%m/%Y"),
            "churn": np.where(days.dayofweek >= 5, "Yes", "No"),
        }
    ).to_csv(csv, index=False)

    runner = CliRunner()
    result = runner.invoke(
        cli.main, ["train", str(csv), "--target", "churn", "--name", "m", "--quiet"]
    )
    assert result.exit_code == 0, result.output
    assert "from dates" in result.output
    assert "Features dropped: none" in result.output

    listed = runner.invoke(cli.main, ["list"])
    assert listed.exit_code == 0
    meta = engine.load_metadata(tmp_path / "models" / "m")
    assert str(len(meta["features"]["date"]["signup"]["features"])) in listed.output


def test_train_prints_leakage_warnings_and_still_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DEFAULT_MODELS_ROOT", tmp_path / "models")
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"noise": rng.normal(size=200).round(1), "tox": rng.uniform(0, 10, 200).round(2)})
    df["unsafe"] = np.where(df["tox"] > 6, "yes", "no")
    csv = tmp_path / "leak.csv"
    df.to_csv(csv, index=False)

    result = CliRunner().invoke(
        cli.main, ["train", str(csv), "--target", "unsafe", "--name", "m", "--quiet"]
    )
    assert result.exit_code == 0, result.output
    assert "Leakage warnings:" in result.output
    assert "Possible target leakage: 'tox'" in result.output
    assert (tmp_path / "models" / "m" / engine.BUNDLE_FILENAME).exists()


def _runner() -> CliRunner:
    try:  # click < 8.2 mixes stderr into stdout unless told otherwise
        return CliRunner(mix_stderr=False)
    except TypeError:
        return CliRunner()


def test_version_option_prints_package_version():
    result = CliRunner().invoke(cli.main, ["--version"])

    assert result.exit_code == 0
    assert glin.__version__ in result.output


def test_serve_accepts_deprecated_mode_stdio_with_notice_on_stderr_only(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.server, "run_stdio", lambda **kw: calls.append(kw))

    result = _runner().invoke(cli.main, ["serve", "--mode", "stdio"])

    assert result.exit_code == 0
    assert calls == [{"allowed_dirs": None, "max_file_mb": 200}]
    assert result.stdout == ""
    assert result.stderr.count("deprecated") == 1


def test_serve_without_mode_writes_nothing(monkeypatch):
    monkeypatch.setattr(cli.server, "run_stdio", lambda **kw: None)

    result = _runner().invoke(cli.main, ["serve"])

    assert result.exit_code == 0
    assert result.stdout == "" and result.stderr == ""


@pytest.mark.parametrize("mode", ["http", "sse"])
def test_serve_http_mode_explains_removal(monkeypatch, mode):
    calls = []
    monkeypatch.setattr(cli.server, "run_stdio", lambda **kw: calls.append(kw))

    result = _runner().invoke(cli.main, ["serve", "--mode", mode])

    assert result.exit_code == 2
    assert calls == []
    assert (
        "HTTP mode was removed in v0.1.4; v0.1.3 is the last release with it. "
        "glin runs over stdio only."
    ) in result.stderr


def test_serve_passes_allow_dir_and_max_file_mb(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli.server, "run_stdio", lambda **kw: calls.append(kw))

    result = _runner().invoke(
        cli.main, ["serve", "--allow-dir", str(tmp_path), "--max-file-mb", "50"]
    )

    assert result.exit_code == 0, result.output
    assert calls == [{"allowed_dirs": [tmp_path], "max_file_mb": 50}]


def test_serve_rejects_non_positive_max_file_mb():
    result = _runner().invoke(cli.main, ["serve", "--max-file-mb", "0"])

    assert result.exit_code != 0


def test_serve_rejects_unknown_mode():
    result = CliRunner().invoke(cli.main, ["serve", "--mode", "ftp"])

    assert result.exit_code == 2
