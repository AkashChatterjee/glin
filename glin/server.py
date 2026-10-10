"""Dual-transport MCP server.

Both transports share one set of tool definitions — ``stdio`` for local
clients (Claude Desktop, Cursor) and ``streamable-http`` (MCP's own standard
remote transport, no custom wire protocol) for agents calling a deployed
instance, e.g. on an EC2 box. ``stateless_http=True`` avoids sticky sessions
so the http mode sits behind a plain load balancer without special handling.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from glin import engine, prompts
from glin.engine import DEFAULT_MODELS_ROOT
from glin.hyperparameters import HYPERPARAMETERS


def default_allowed_dirs() -> list[Path]:
    return [Path.home(), Path.cwd()]


def resolve_csv_path(csv_path: str, allowed_dirs: list[Path]) -> Path:
    """Validate a client-supplied CSV path. Errors never echo file contents,
    and don't reveal whether a path outside the allowed directories exists."""
    path = Path(csv_path).expanduser()
    if path.suffix.lower() != ".csv":
        raise ValueError("csv_path must point to a .csv file")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        resolved = None
    roots = [d.resolve() for d in allowed_dirs]
    if resolved is None or not any(resolved.is_relative_to(r) for r in roots):
        raise ValueError(
            "csv_path not found or outside the directories this server may read "
            "(see `glin serve --allow-dir`)"
        )
    if resolved.suffix.lower() != ".csv" or not resolved.is_file():
        raise ValueError("csv_path must be a regular .csv file")
    return resolved


def _read_only(title: str) -> ToolAnnotations:
    return ToolAnnotations(
        title=title, readOnlyHint=True, destructiveHint=False, openWorldHint=False
    )


_TRAIN_ANNOTATIONS = ToolAnnotations(
    title="Train Model",
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=False,
)


def build_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    models_root: Path = DEFAULT_MODELS_ROOT,
    allowed_dirs: Optional[list[Path]] = None,
) -> FastMCP:
    allowed = list(allowed_dirs) if allowed_dirs else default_allowed_dirs()
    mcp = FastMCP("glin", host=host, port=port, stateless_http=True)

    def _model_dir(model_name: str) -> Path:
        model_dir = models_root / model_name
        if not (model_dir / engine.METADATA_FILENAME).exists():
            raise ValueError(f"model '{model_name}' not found in {models_root}")
        return model_dir

    @mcp.tool(annotations=_read_only("List Models"))
    def list_models() -> list[dict[str, Any]]:
        """List all trained glin models available on this machine."""
        if not models_root.exists():
            return []
        return [
            engine.load_metadata(model_dir)
            for model_dir in sorted(models_root.iterdir())
            if (model_dir / engine.METADATA_FILENAME).exists()
        ]

    @mcp.tool(annotations=_read_only("Inspect Model"))
    def inspect_model(model_name: str) -> dict[str, Any]:
        """Return feature names, types, and target classes for a trained model."""
        return engine.load_metadata(_model_dir(model_name))

    @mcp.tool(annotations=_read_only("Predict"))
    def predict(
        model_name: str, features: dict[str, Any], top_n: int = 10
    ) -> dict[str, Any]:
        """Predicted class and probabilities, plus the full glassbox audit:
        baseline rate, every contributing feature's exact score (sorted by
        magnitude), and an additivity check verifying the scores sum to the
        model's own predicted probability. `warnings` names inputs the model never
        saw (unseen categories, numerics outside the training range); treat the
        prediction as less reliable when present. `missing_features` lists
        absent inputs, which are valid but worth mentioning."""
        bundle = engine.load_bundle(_model_dir(model_name))
        return engine.explain_record(bundle, features, top_n=top_n)

    @mcp.tool(annotations=_read_only("Explain Model"))
    def explain_model(
        model_name: str, top_n: int = 15, include_shapes: bool = True
    ) -> dict[str, Any]:
        """Global view of what a trained model learned, exact rather than
        sampled from example predictions: every feature ranked by its mean
        absolute contribution to the log-odds (`features`, with main-effect
        vs interaction split), the top terms including pairwise interactions
        (`terms`, each with its share of the total), and, for binary targets,
        a `shape` per main effect showing which categories or value ranges
        push the prediction up or down. Use this to answer "what drives this
        model" instead of running `predict` on a handful of rows."""
        bundle = engine.load_bundle(_model_dir(model_name))
        return engine.global_importance(bundle, top_n=top_n, include_shapes=include_shapes)

    @mcp.tool(annotations=_read_only("List Hyperparameters"))
    def list_hyperparameters() -> list[dict[str, Any]]:
        """Every EBM training hyperparameter `train_model` accepts in its
        `hyperparameters` argument: name, type, default, allowed range, and
        what raising/lowering it does. Fetch the `tune_model_hyperparameters`
        prompt for a search strategy built on these."""
        return [hp.describe() for hp in HYPERPARAMETERS.values()]

    @mcp.tool(annotations=_TRAIN_ANNOTATIONS)
    def train_model(
        target_column: str,
        model_name: str,
        csv_path: Optional[str] = None,
        csv_content: Optional[str] = None,
        overwrite: bool = False,
        hyperparameters: Optional[dict[str, Any]] = None,
        holdout_fraction: float = 0.2,
    ) -> dict[str, Any]:
        """Train an EBM classifier from a CSV and save it as model_name. This
        is also how to retrain an existing model: pass overwrite=True to
        replace it with a freshly trained model (e.g. on updated data) --
        otherwise a pre-existing model_name is rejected so it isn't silently
        clobbered.

        Pass exactly one of:
        - csv_path: a .csv file inside a directory this server allows (it does the
          reading itself). Always prefer this when the MCP client runs on
          the same machine as the server, e.g. the local stdio transport
          (Claude Desktop, Cursor) -- it works for CSVs of any size without
          the calling agent ever having to read or reproduce the file's
          contents itself.
        - csv_content: the complete raw CSV text (header row + data rows).
          Only use this when there's no path the server can reach, e.g. a
          remote streamable-http deployment with no shared filesystem, or
          data assembled in memory. Do not use this for large files --
          having an agent inline tens of thousands of rows as a tool-call
          argument is slow, burns enormous context, and risks truncation.
          Prefer writing the data to a file the server can see (csv_path)
          whenever that's possible instead.

        Either way, this tool does no cleanup of its own beyond what
        EBMTabularPreprocessor already does automatically (coercing dirty
        numeric formatting, dropping ID-like/constant/high-cardinality
        columns, imputing missing values). If the CSV came from an
        unfamiliar or messy source (wrong delimiter, banner rows before the
        header, embedded JSON cells, date columns, a continuous-looking
        target, etc.), work through the checklist in the `prepare_csv_for_training`
        MCP prompt first (or the `build_classifier` prompt for the full
        prepare -> tune -> report workflow) -- those are judgment calls this
        tool does not make for you.

        Tuning: `hyperparameters` is an optional object overriding any subset
        of the knobs listed by `list_hyperparameters` (e.g.
        {"interactions": 0, "min_samples_leaf": 30}); anything omitted keeps
        its default, and unknown names or out-of-range values are rejected
        with an error naming every problem. `holdout_fraction` (default 0.2)
        sets that share of rows aside, trains on the rest, and reports
        out-of-sample `evaluation` metrics for the saved model -- compare
        `evaluation.log_loss` (lower is better) across trials to tune. Because
        the saved model then saw only the remaining rows, once you've settled
        on hyperparameters retrain with holdout_fraction=0 and overwrite=True
        to fit on all the data (no evaluation is reported in that case). See
        the `tune_model_hyperparameters` prompt for the full loop.
        """
        if (csv_path is None) == (csv_content is None):
            raise ValueError("pass exactly one of csv_path or csv_content")

        model_dir = models_root / model_name
        if model_dir.exists() and not overwrite:
            raise ValueError(
                f"model '{model_name}' already exists at {model_dir} -- pass "
                "overwrite=True to retrain it with this data"
            )

        df = (
            pd.read_csv(resolve_csv_path(csv_path, allowed))
            if csv_path is not None
            else pd.read_csv(io.StringIO(csv_content))
        )
        bundle = engine.train_model(
            df,
            target_column,
            model_name=model_name,
            hyperparameters=hyperparameters,
            holdout_fraction=holdout_fraction,
        )
        engine.save_bundle(bundle, models_root)

        preprocessor = bundle["preprocessor"]
        term_features = bundle["model"].term_features_
        return {
            "model_name": model_name,
            "target_column": target_column,
            "rows_read": len(df),
            "target_classes": bundle["target_classes"],
            "features_kept": preprocessor.output_columns_,
            "features_dropped": preprocessor.dropped_columns_,
            "warnings": bundle["warnings"],
            "hyperparameters": bundle["hyperparameters"],
            "holdout_fraction": bundle["holdout_fraction"],
            "training_rows": bundle["training_rows"],
            "training_seconds": round(bundle["training_seconds"], 2),
            "terms": {
                "main_effects": sum(1 for t in term_features if len(t) == 1),
                "interactions": sum(1 for t in term_features if len(t) > 1),
            },
            "evaluation": bundle["evaluation"],
        }

    @mcp.prompt()
    def build_classifier(
        csv_path: str = "", target_column: str = "", model_name: str = ""
    ) -> str:
        """One-shot workflow for any dataset: prepare the CSV, drop leaky/ID/text
        columns, tune hyperparameters without overfitting, refit, and report."""
        return prompts.end_to_end_prompt(csv_path, target_column, model_name)

    @mcp.prompt()
    def prepare_csv_for_training(target_column: str = "") -> str:
        """Checklist for turning a possibly-messy CSV into input for train_model."""
        return prompts.prep_prompt(target_column)

    @mcp.prompt()
    def tune_model_hyperparameters(model_name: str = "", target_column: str = "") -> str:
        """Search strategy for tuning an EBM's hyperparameters with train_model."""
        return prompts.tuning_prompt(model_name, target_column)

    return mcp


def run_stdio(
    models_root: Path = DEFAULT_MODELS_ROOT,
    allowed_dirs: Optional[list[Path]] = None,
) -> None:
    build_server(models_root=models_root, allowed_dirs=allowed_dirs).run(
        transport="stdio"
    )


def run_http(
    host: str = "0.0.0.0",
    port: int = 8000,
    models_root: Path = DEFAULT_MODELS_ROOT,
    allowed_dirs: Optional[list[Path]] = None,
) -> None:
    build_server(
        host=host, port=port, models_root=models_root, allowed_dirs=allowed_dirs
    ).run(transport="streamable-http")
