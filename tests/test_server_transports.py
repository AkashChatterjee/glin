import asyncio
import socket

import numpy as np
import pandas as pd
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from glin import engine, server


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
    }


@pytest.mark.asyncio
async def test_list_models_reads_metadata_without_unpickling(models_root):
    mcp = server.build_server(models_root=models_root)
    _, structured = await mcp.call_tool("list_models", {})
    models = structured["result"]
    assert len(models) == 1
    assert models[0]["model_name"] == "server_test_model"
    assert models[0]["target_classes"] == ["No", "Yes"]


@pytest.mark.asyncio
async def test_inspect_model_returns_feature_schema(models_root):
    mcp = server.build_server(models_root=models_root)
    _, meta = await mcp.call_tool("inspect_model", {"model_name": "server_test_model"})
    assert meta["features"]["numeric"] == ["age"]
    assert meta["features"]["categorical"] == ["plan"]


@pytest.mark.asyncio
async def test_inspect_model_unknown_name_raises(models_root):
    from mcp.server.fastmcp.exceptions import ToolError

    mcp = server.build_server(models_root=models_root)
    with pytest.raises(ToolError, match="does_not_exist"):
        await mcp.call_tool("inspect_model", {"model_name": "does_not_exist"})


@pytest.mark.asyncio
async def test_predict_returns_prediction_and_verified_additivity(models_root):
    mcp = server.build_server(models_root=models_root)
    _, result = await mcp.call_tool(
        "predict",
        {"model_name": "server_test_model", "features": {"age": 25.0, "plan": "basic"}},
    )
    assert result["predicted_class"] in ("Yes", "No")
    assert "probabilities" in result
    assert result["additivity_verified"] is True
    assert "base_rate" in result
    assert isinstance(result["contributions"], list)


@pytest.mark.asyncio
async def test_predict_top_n_truncates_contributions(models_root):
    mcp = server.build_server(models_root=models_root)
    _, result = await mcp.call_tool(
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
    _, result = await mcp.call_tool(
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
    from mcp.server.fastmcp.exceptions import ToolError

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
    _, result = await mcp.call_tool(
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
async def test_train_model_invalid_dataset_raises(models_root):
    from mcp.server.fastmcp.exceptions import ToolError

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
async def test_streamable_http_real_mcp_client_round_trip(models_root):
    """End-to-end proof that a genuine MCP client (not a bespoke protocol)
    can connect over the network and call glin's tools — the actual
    requirement behind the EC2 one-shot-deploy story."""
    port = _unused_tcp_port()
    mcp = server.build_server(host="127.0.0.1", port=port, models_root=models_root)
    app = mcp.streamable_http_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    uv_server = uvicorn.Server(config)
    serve_task = asyncio.create_task(uv_server.serve())

    try:
        for _ in range(100):
            if uv_server.started:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("uvicorn server did not start in time")

        url = f"http://127.0.0.1:{port}/mcp"
        async with streamable_http_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                assert {t.name for t in tools.tools} == {
                    "list_models",
                    "inspect_model",
                    "predict",
                    "train_model",
                }

                result = await session.call_tool(
                    "predict",
                    {
                        "model_name": "server_test_model",
                        "features": {"age": 25.0, "plan": "basic"},
                    },
                )
                assert result.structuredContent["predicted_class"] in ("Yes", "No")
    finally:
        uv_server.should_exit = True
        await serve_task
