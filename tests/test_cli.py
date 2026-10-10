import numpy as np
import pandas as pd
from click.testing import CliRunner

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
