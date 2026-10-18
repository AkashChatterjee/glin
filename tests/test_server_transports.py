import asyncio
import json
import logging
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import glin
from glin import engine, server
from tests.mcp_compat import call_structured

ToolError = server.ToolError


def _train_fixture_model(models_root, model_name="server_test_model"):
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame(
        {
            "age": rng.randint(18, 80, size=n).astype(float),
            "plan": rng.choice(["basic", "premium"], size=n),
        }
    )
    df["churn"] = np.where(df["age"] < 40, "Yes", "No")
    bundle = engine.train_model(df, "churn", model_name=model_name)
    engine.save_bundle(bundle, models_root)


_call = call_structured


def _hint(annotations, camel_name):
    """Read a ToolAnnotations field: camelCase in mcp 1.x, snake_case in 2.x."""
    if hasattr(annotations, camel_name):
        return getattr(annotations, camel_name)
    return getattr(annotations, re.sub(r"[A-Z]", lambda m: "_" + m.group().lower(), camel_name))


@pytest.fixture
def models_root(tmp_path):
    _train_fixture_model(tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_all_tools_registered(models_root):
    mcp = server.build_server(models_root=models_root)
    tools = await mcp.list_tools()
    assert {t.name for t in tools} == {
        "list_models",
        "inspect_model",
        "predict",
        "train_model",
        "list_hyperparameters",
        "explain_model",
        "delete_model",
    }


@pytest.mark.asyncio
async def test_list_models_reads_metadata_without_unpickling(models_root):
    mcp = server.build_server(models_root=models_root)
    structured = await _call(mcp, "list_models", {})
    models = structured["result"]
    assert len(models) == 1
    assert models[0]["model_name"] == "server_test_model"
    assert models[0]["target_classes"] == ["No", "Yes"]


@pytest.mark.asyncio
async def test_inspect_model_returns_feature_schema(models_root):
    mcp = server.build_server(models_root=models_root)
    meta = await _call(mcp, "inspect_model", {"model_name": "server_test_model"})
    assert meta["features"]["numeric"] == ["age"]
    assert meta["features"]["categorical"] == ["plan"]


@pytest.mark.asyncio
async def test_inspect_model_unknown_name_raises(models_root):
    mcp = server.build_server(models_root=models_root)
    with pytest.raises(ToolError, match="does_not_exist"):
        await mcp.call_tool("inspect_model", {"model_name": "does_not_exist"})


@pytest.mark.asyncio
async def test_predict_returns_prediction_and_verified_additivity(models_root):
    mcp = server.build_server(models_root=models_root)
    result = await _call(
        mcp,
        "predict",
        {"model_name": "server_test_model", "features": {"age": 25.0, "plan": "basic"}},
    )
    assert result["predicted_class"] in ("Yes", "No")
    assert "probabilities" in result
    assert result["additivity_verified"] is True
    assert "base_logit" in result and "base_rate" not in result
    assert result["score_class"] in result["probabilities"]
    assert set(result["model_probability"]) == set(result["probabilities"])
    assert isinstance(result["contributions"], list)


@pytest.mark.asyncio
async def test_predict_top_n_truncates_contributions(models_root):
    mcp = server.build_server(models_root=models_root)
    result = await _call(
        mcp,
        "predict",
        {
            "model_name": "server_test_model",
            "features": {"age": 25.0, "plan": "basic"},
            "top_n": 1,
        },
    )
    assert len(result["contributions"]) == 1


def _fixture_csv(n=200, seed=0) -> str:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "age": rng.randint(18, 80, size=n).astype(float),
            "plan": rng.choice(["basic", "premium"], size=n),
        }
    )
    df["churn"] = np.where(df["age"] < 40, "Yes", "No")
    return df.to_csv(index=False)


@pytest.mark.asyncio
async def test_train_model_trains_and_saves_a_new_model(models_root):
    mcp = server.build_server(models_root=models_root)
    result = await _call(
        mcp,
        "train_model",
        {
            "csv_content": _fixture_csv(),
            "target_column": "churn",
            "model_name": "brand_new_model",
        },
    )
    assert result["model_name"] == "brand_new_model"
    assert sorted(result["target_classes"]) == ["No", "Yes"]
    assert "age" in result["features_kept"]
    assert "plan" in result["features_kept"]

    metadata = engine.load_metadata(models_root / "brand_new_model")
    assert metadata["model_name"] == "brand_new_model"


@pytest.mark.asyncio
async def test_train_model_refuses_to_clobber_existing_model_without_overwrite(models_root):
    mcp = server.build_server(models_root=models_root)
    with pytest.raises(ToolError, match="overwrite=True"):
        await mcp.call_tool(
            "train_model",
            {
                "csv_content": _fixture_csv(),
                "target_column": "churn",
                "model_name": "server_test_model",
            },
        )


@pytest.mark.asyncio
async def test_train_model_overwrite_retrains_existing_model(models_root):
    mcp = server.build_server(models_root=models_root)
    result = await _call(
        mcp,
        "train_model",
        {
            "csv_content": _fixture_csv(seed=1),
            "target_column": "churn",
            "model_name": "server_test_model",
            "overwrite": True,
        },
    )
    assert result["model_name"] == "server_test_model"


@pytest.mark.asyncio
async def test_train_model_from_csv_path_avoids_inlining_data(models_root, tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(_fixture_csv())

    mcp = server.build_server(models_root=models_root, allowed_dirs=[tmp_path])
    result = await _call(
        mcp,
        "train_model",
        {
            "csv_path": str(csv_path),
            "target_column": "churn",
            "model_name": "from_path_model",
        },
    )
    assert result["model_name"] == "from_path_model"
    assert engine.load_metadata(models_root / "from_path_model")["model_name"] == "from_path_model"


@pytest.mark.asyncio
async def test_train_model_requires_exactly_one_of_csv_path_or_csv_content(models_root):
    mcp = server.build_server(models_root=models_root)
    with pytest.raises(ToolError, match="exactly one"):
        await mcp.call_tool(
            "train_model",
            {"target_column": "churn", "model_name": "neither_given"},
        )

    with pytest.raises(ToolError, match="exactly one"):
        await mcp.call_tool(
            "train_model",
            {
                "csv_path": "/does/not/matter.csv",
                "csv_content": _fixture_csv(),
                "target_column": "churn",
                "model_name": "both_given",
            },
        )


@pytest.mark.asyncio
async def test_train_model_invalid_dataset_raises(models_root):
    mcp = server.build_server(models_root=models_root)
    tiny_csv = "age,churn\n25,Yes\n30,No\n"
    with pytest.raises(ToolError, match="at least"):
        await mcp.call_tool(
            "train_model",
            {
                "csv_content": tiny_csv,
                "target_column": "churn",
                "model_name": "too_small",
            },
        )


@pytest.mark.asyncio
async def test_prepare_csv_for_training_prompt_mentions_target_column(models_root):
    mcp = server.build_server(models_root=models_root)
    prompts = await mcp.list_prompts()
    assert "prepare_csv_for_training" in {p.name for p in prompts}

    result = await mcp.get_prompt("prepare_csv_for_training", {"target_column": "churn"})
    text = result.messages[0].content.text
    assert "churn" in text
    assert "train_model" in text


@pytest.mark.asyncio
async def test_list_hyperparameters_matches_registry(models_root):
    from glin.hyperparameters import HYPERPARAMETERS

    mcp = server.build_server(models_root=models_root)
    structured = await _call(mcp, "list_hyperparameters", {})
    rows = {r["name"]: r for r in structured["result"]}
    assert set(rows) == set(HYPERPARAMETERS)
    assert rows["max_bins"]["default"] == 256
    assert rows["interactions"]["default"] == 10
    assert all(r["guidance"] for r in rows.values())


@pytest.mark.asyncio
async def test_train_model_applies_hyperparameters_and_reports_holdout_metrics(models_root):
    mcp = server.build_server(models_root=models_root)
    result = await _call(
        mcp,
        "train_model",
        {
            "csv_content": _fixture_csv(),
            "target_column": "churn",
            "model_name": "tuned_model",
            "hyperparameters": {"interactions": 0, "max_bins": 16, "outer_bags": 2},
        },
    )
    assert result["hyperparameters"]["interactions"] == 0
    assert result["hyperparameters"]["max_bins"] == 16
    assert result["hyperparameters"]["learning_rate"] == 0.015  # untouched default
    assert result["terms"]["interactions"] == 0

    evaluation = result["evaluation"]
    assert evaluation["holdout_rows"] == 40  # default holdout_fraction=0.2 of 200
    assert result["training_rows"] == 160
    assert 0 < evaluation["log_loss"] < evaluation["baseline_log_loss"]

    # what the agent was told is also what's persisted for list_models/inspect_model
    meta = engine.load_metadata(models_root / "tuned_model")
    assert meta["hyperparameters"]["max_bins"] == 16
    assert meta["evaluation"]["log_loss"] == evaluation["log_loss"]


@pytest.mark.asyncio
async def test_train_model_without_holdout_trains_on_everything(models_root):
    mcp = server.build_server(models_root=models_root)
    result = await _call(
        mcp,
        "train_model",
        {
            "csv_content": _fixture_csv(),
            "target_column": "churn",
            "model_name": "full_data_model",
            "holdout_fraction": 0,
            "hyperparameters": {"outer_bags": 2},
        },
    )
    assert result["evaluation"] is None
    assert result["training_rows"] == 200


@pytest.mark.asyncio
async def test_train_model_rejects_bad_hyperparameters_naming_every_problem(models_root):
    mcp = server.build_server(models_root=models_root)
    with pytest.raises(ToolError) as exc:
        await mcp.call_tool(
            "train_model",
            {
                "csv_content": _fixture_csv(),
                "target_column": "churn",
                "model_name": "bad_hp",
                "hyperparameters": {"learning_rate": 5, "no_such_knob": 1, "max_bins": 2.5},
            },
        )
    msg = str(exc.value)
    assert "learning_rate" in msg and "no_such_knob" in msg and "max_bins" in msg
    assert not (models_root / "bad_hp").exists()


@pytest.mark.asyncio
async def test_tune_model_hyperparameters_prompt_lists_live_knobs(models_root):
    mcp = server.build_server(models_root=models_root)
    assert "tune_model_hyperparameters" in {p.name for p in await mcp.list_prompts()}

    result = await mcp.get_prompt("tune_model_hyperparameters", {"model_name": "churn_v2"})
    text = result.messages[0].content.text
    assert "churn_v2" in text
    assert "min_samples_leaf" in text and "holdout_fraction=0" in text


@pytest.mark.asyncio
async def test_build_classifier_prompt_is_self_contained_and_generic(models_root):
    mcp = server.build_server(models_root=models_root)
    prompts = {p.name: p for p in await mcp.list_prompts()}
    assert {"build_classifier", "prepare_csv_for_training", "tune_model_hyperparameters"} <= set(prompts)
    # every argument optional, so the user can invoke it with just a file path
    assert not any(a.required for a in prompts["build_classifier"].arguments or [])

    result = await mcp.get_prompt(
        "build_classifier",
        {"csv_path": "/data/x.csv", "target_column": "churn", "model_name": "churn_v1"},
    )
    text = result.messages[0].content.text
    for needle in (
        "/data/x.csv", "churn_v1",
        "PHASE 1", "PHASE 2", "PHASE 3", "PHASE 4",
        "target leakage", "list_hyperparameters", "holdout_fraction=0",
        "random_state", "STOP",  # embedded prep checklist, tuning loop, early exit
        "Possible target leakage",  # leakage-warning review guidance
    ):
        assert needle in text

    # works with no arguments at all and tells the agent to ask for the data
    blank = (await mcp.get_prompt("build_classifier", {})).messages[0].content.text
    assert "ask the user where the data is" in blank


@pytest.mark.asyncio
async def test_explain_model_tool_returns_global_importance(models_root):
    mcp = server.build_server(models_root=models_root)
    out = await _call(mcp, "explain_model", {"model_name": "server_test_model"})
    assert {f["feature"] for f in out["features"]} == {"age", "plan"}
    assert out["features"][0]["feature"] == "age"  # churn is a function of age
    assert out["terms"][0]["share"] > 0.5


@pytest.mark.asyncio
async def test_stdio_real_mcp_client_round_trip(models_root):
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-c",
            "import sys; from pathlib import Path; from glin import server; "
            "server.run_stdio(Path(sys.argv[1]))",
            str(models_root),
        ],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            info = getattr(init, "serverInfo", None) or init.server_info
            assert (info.name, info.version) == ("glin", glin.__version__)

            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == {
                "list_models",
                "inspect_model",
                "predict",
                "train_model",
                "list_hyperparameters",
                "explain_model",
                "delete_model",
            }

            result = await session.call_tool(
                "predict",
                {
                    "model_name": "server_test_model",
                    "features": {"age": 25.0, "plan": "basic"},
                },
            )
            structured = getattr(result, "structuredContent", None) or result.structured_content
            assert structured["predicted_class"] in ("Yes", "No")


def test_serve_mode_stdio_keeps_stdout_pure_json_rpc(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-m", "glin.cli", "serve", "--mode", "stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        text=True,
    )
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    }
    try:
        stdout, stderr = proc.communicate(json.dumps(initialize) + "\n", timeout=300)
    finally:
        proc.kill()
    messages = [json.loads(line) for line in stdout.splitlines()]
    assert [m["id"] for m in messages] == [1]
    assert messages[0]["result"]["serverInfo"]["version"] == glin.__version__
    assert stderr.count("--mode is deprecated") == 1


def test_run_stdio_quiets_interpret_logging(monkeypatch, tmp_path):
    logger = logging.getLogger("interpret")
    previous = logger.level
    ran = []
    monkeypatch.setattr(server.MCPServer, "run", lambda self, transport: ran.append(transport))
    try:
        server.run_stdio(models_root=tmp_path)
        assert ran == ["stdio"]
        assert logger.getEffectiveLevel() == logging.WARNING
    finally:
        logger.setLevel(previous)


def test_all_tools_have_annotations(tmp_path):
    tools = asyncio.run(server.build_server(models_root=tmp_path).list_tools())
    assert len(tools) == 7
    for tool in tools:
        assert tool.annotations is not None, tool.name
        assert tool.annotations.title, tool.name
        assert _hint(tool.annotations, "readOnlyHint") is not None, tool.name
        assert _hint(tool.annotations, "destructiveHint") is not None, tool.name
    by_name = {t.name: t.annotations for t in tools}
    assert _hint(by_name["train_model"], "readOnlyHint") is False
    assert _hint(by_name["train_model"], "destructiveHint") is True
    assert by_name["delete_model"].title == "Delete Model"
    assert _hint(by_name["delete_model"], "readOnlyHint") is False
    assert _hint(by_name["delete_model"], "destructiveHint") is True
    for name, ann in by_name.items():
        if name not in ("train_model", "delete_model"):
            assert _hint(ann, "readOnlyHint") is True and _hint(ann, "destructiveHint") is False


async def _train_with_path(models_root, allowed, csv_path):
    mcp = server.build_server(models_root=models_root, allowed_dirs=allowed)
    return await _call(
        mcp,
        "train_model",
        {"csv_path": str(csv_path), "target_column": "churn", "model_name": "guarded"},
    )


@pytest.mark.asyncio
async def test_csv_path_rejects_non_csv_extension(models_root, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    with pytest.raises(ToolError, match=r"\.csv") as exc:
        await _train_with_path(models_root, [tmp_path], secret)
    assert "TOPSECRET" not in str(exc.value)


@pytest.mark.asyncio
async def test_csv_path_rejects_directory_and_missing(models_root, tmp_path):
    (tmp_path / "dir.csv").mkdir()
    with pytest.raises(ToolError, match="regular .csv file"):
        await _train_with_path(models_root, [tmp_path], tmp_path / "dir.csv")
    with pytest.raises(ToolError, match="not found or outside"):
        await _train_with_path(models_root, [tmp_path], tmp_path / "nope.csv")


@pytest.mark.asyncio
async def test_csv_path_rejects_outside_allowed_dirs_and_traversal(models_root, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text(_fixture_csv())
    with pytest.raises(ToolError, match="not found or outside"):
        await _train_with_path(models_root, [allowed], outside)
    with pytest.raises(ToolError, match="not found or outside"):
        await _train_with_path(models_root, [allowed], allowed / ".." / "outside.csv")


@pytest.mark.asyncio
async def test_csv_path_symlink_escaping_allowed_dir_is_rejected(models_root, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text(_fixture_csv())
    link = allowed / "link.csv"
    link.symlink_to(outside)
    with pytest.raises(ToolError, match="not found or outside"):
        await _train_with_path(models_root, [allowed], link)


@pytest.mark.asyncio
async def test_csv_path_symlink_inside_allowed_dir_is_accepted(models_root, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    real = allowed / "real.csv"
    real.write_text(_fixture_csv())
    link = allowed / "link.csv"
    link.symlink_to(real)
    result = await _train_with_path(models_root, [allowed], link)
    assert result["model_name"] == "guarded"


def test_default_allowed_dirs_skip_root_working_dir(monkeypatch, tmp_path):
    home = tmp_path / "home" / "me"
    home.mkdir(parents=True)
    monkeypatch.setattr(server.Path, "home", lambda: home)
    monkeypatch.setattr(server.Path, "cwd", lambda: server.Path("/"))
    assert server.default_allowed_dirs() == [home.resolve()]
    monkeypatch.setattr(server.Path, "cwd", lambda: home.parent)
    assert server.default_allowed_dirs() == [home.resolve()]
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(server.Path, "cwd", lambda: project)
    assert server.default_allowed_dirs() == [home.resolve(), project.resolve()]


def test_default_allowed_dirs_skip_root_home(monkeypatch):
    monkeypatch.setattr(server.Path, "home", lambda: server.Path("/"))
    monkeypatch.setattr(server.Path, "cwd", lambda: server.Path("/"))
    assert server.default_allowed_dirs() == []


def test_root_allowed_dir_warns_on_stderr(models_root, capsys):
    server.build_server(models_root=models_root, allowed_dirs=[server.Path("/")])
    assert "filesystem root" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_empty_allowed_dirs_disables_csv_path(models_root, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "default_allowed_dirs", lambda: [tmp_path])
    data = tmp_path / "data.csv"
    data.write_text(_fixture_csv())
    with pytest.raises(ToolError, match="csv_path is disabled.*--allow-dir"):
        await _train_with_path(models_root, [], data)
    mcp = server.build_server(models_root=models_root, allowed_dirs=[])
    result = await _call(
        mcp,
        "train_model",
        {"csv_content": _fixture_csv(), "target_column": "churn", "model_name": "inline"},
    )
    assert result["model_name"] == "inline"


@pytest.mark.asyncio
async def test_csv_path_over_size_limit_is_rejected_before_reading(models_root, tmp_path):
    big = tmp_path / "big.csv"
    with open(big, "wb") as fh:
        fh.truncate(3 * 1024**2)
    mcp = server.build_server(models_root=models_root, allowed_dirs=[tmp_path], max_file_mb=2)
    with pytest.raises(ToolError, match=r"3 MB, above .* limit of 2 MB .*--max-file-mb"):
        await _call(
            mcp,
            "train_model",
            {"csv_path": str(big), "target_column": "churn", "model_name": "big"},
        )


def test_default_size_limit_rejects_sparse_file(tmp_path):
    big = tmp_path / "big.csv"
    with open(big, "wb") as fh:
        fh.truncate((server.DEFAULT_MAX_FILE_MB + 1) * 1024**2)
    with pytest.raises(ValueError, match="--max-file-mb"):
        server.resolve_csv_path(str(big), [tmp_path])
    assert server.resolve_csv_path(str(big), [tmp_path], max_file_mb=None) == big.resolve()


@pytest.mark.asyncio
async def test_date_column_trains_inspects_and_predicts_over_mcp(models_root):
    rng = np.random.RandomState(0)
    days = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.randint(0, 700, 300), unit="D")
    df = pd.DataFrame(
        {
            "signup_date": days.strftime("%Y-%m-%d"),
            "churn": np.where(days.dayofweek >= 5, "Yes", "No"),
        }
    )
    mcp = server.build_server(models_root=models_root)
    trained = await _call(
        mcp,
        "train_model",
        {"csv_content": df.to_csv(index=False), "target_column": "churn", "model_name": "dated"},
    )
    derived = trained["date_columns"]["signup_date"]
    assert "signup_date__day_of_week" in derived
    assert trained["features_kept"] == derived
    assert trained["features_dropped"] == []

    meta = await _call(mcp, "inspect_model", {"model_name": "dated"})
    assert meta["features"]["date"]["signup_date"]["features"] == derived

    result = await _call(
        mcp, "predict", {"model_name": "dated", "features": {"signup_date": "2021-03-06"}}
    )
    assert result["predicted_class"] == "Yes"
    assert result["additivity_verified"] is True
    assert result["contributions"][0]["feature"] == "signup_date__day_of_week"
    assert result["contributions"][0]["value"] == "sat"


@pytest.mark.asyncio
async def test_predict_score_class_keeps_the_target_type_over_mcp(tmp_path):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"age": rng.randint(18, 80, size=200).astype(float)})
    df["churn"] = np.where(df["age"] < 40, 1, 0)
    engine.save_bundle(engine.train_model(df, "churn", model_name="int_target"), tmp_path)
    mcp = server.build_server(models_root=tmp_path)
    result = await _call(mcp, "predict", {"model_name": "int_target", "features": {"age": 25.0}})
    assert result["score_class"] == 1 and result["predicted_class"] == 1
    assert set(result["model_probability"]) == {"0", "1"}
