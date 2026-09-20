"""Dual-transport MCP server.

Both transports share one set of tool definitions — ``stdio`` for local
clients (Claude Desktop, Cursor) and ``streamable-http`` (MCP's own standard
remote transport, no custom wire protocol) for agents calling a deployed
instance, e.g. on an EC2 box. ``stateless_http=True`` avoids sticky sessions
so the http mode sits behind a plain load balancer without special handling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from glin import engine
from glin.engine import DEFAULT_MODELS_ROOT


def build_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    models_root: Path = DEFAULT_MODELS_ROOT,
) -> FastMCP:
    mcp = FastMCP("glin", host=host, port=port, stateless_http=True)

    def _model_dir(model_name: str) -> Path:
        model_dir = models_root / model_name
        if not (model_dir / engine.METADATA_FILENAME).exists():
            raise ValueError(f"model '{model_name}' not found in {models_root}")
        return model_dir

    @mcp.tool()
    def list_models() -> list[dict[str, Any]]:
        """List all trained glin models available on this machine."""
        if not models_root.exists():
            return []
        return [
            engine.load_metadata(model_dir)
            for model_dir in sorted(models_root.iterdir())
            if (model_dir / engine.METADATA_FILENAME).exists()
        ]

    @mcp.tool()
    def inspect_model(model_name: str) -> dict[str, Any]:
        """Return feature names, types, and target classes for a trained model."""
        return engine.load_metadata(_model_dir(model_name))

    @mcp.tool()
    def predict(
        model_name: str, features: dict[str, Any], top_n: int = 10
    ) -> dict[str, Any]:
        """Predicted class and probabilities, plus the full glassbox audit:
        baseline rate, every contributing feature's exact score (sorted by
        magnitude), and an additivity check verifying the scores sum to the
        model's own predicted probability."""
        bundle = engine.load_bundle(_model_dir(model_name))
        return engine.explain_record(bundle, features, top_n=top_n)

    return mcp


def run_stdio(models_root: Path = DEFAULT_MODELS_ROOT) -> None:
    build_server(models_root=models_root).run(transport="stdio")


def run_http(
    host: str = "0.0.0.0",
    port: int = 8000,
    models_root: Path = DEFAULT_MODELS_ROOT,
) -> None:
    build_server(host=host, port=port, models_root=models_root).run(transport="streamable-http")
