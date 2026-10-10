"""Local stdio MCP server exposing glin's tools and prompts."""
from __future__ import annotations

import functools
import io
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from mcp.types import ToolAnnotations

from glin import engine, prompts
from glin.engine import DEFAULT_MODELS_ROOT
from glin.hyperparameters import HYPERPARAMETERS

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer
    from mcp.server.fastmcp.exceptions import ToolError


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
    models_root: Path = DEFAULT_MODELS_ROOT,
    allowed_dirs: Optional[list[Path]] = None,
) -> MCPServer:
    allowed = list(allowed_dirs) if allowed_dirs else default_allowed_dirs()
    mcp = MCPServer("glin")

    def tool(annotations: ToolAnnotations):
        # mcp 2.x hides non-ToolError messages from clients; ours are meant for the agent.
        def register(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except ValueError as exc:
                    raise ToolError(str(exc)) from exc

            return mcp.tool(annotations=annotations)(wrapper)

        return register

    def _model_dir(model_name: str) -> Path:
        model_dir = models_root / model_name
        if not (model_dir / engine.METADATA_FILENAME).exists():
            raise ValueError(f"model '{model_name}' not found in {models_root}")
        return model_dir

    @tool(annotations=_read_only("List Models"))
    def list_models() -> list[dict[str, Any]]:
        """List all trained glin models available on this machine."""
        if not models_root.exists():
            return []
        return [
            engine.load_metadata(model_dir)
            for model_dir in sorted(models_root.iterdir())
            if (model_dir / engine.METADATA_FILENAME).exists()
        ]

    @tool(annotations=_read_only("Inspect Model"))
    def inspect_model(model_name: str) -> dict[str, Any]:
        """Return feature names, types, and target classes for a trained model."""
        return engine.load_metadata(_model_dir(model_name))

    @tool(annotations=_read_only("Predict"))
    def predict(
        model_name: str, features: dict[str, Any], top_n: int = 10
    ) -> dict[str, Any]:
        """Predicted class and probabilities, plus the full glassbox audit:
        baseline rate, every contributing feature's exact score (sorted by
        magnitude), and an additivity check verifying the scores sum to the
        model's own predicted probability. `warnings` names inputs the model never
        saw (unseen categories, numerics or dates outside the training range,
        unparseable dates); treat the prediction as less reliable when present.
        `missing_features` lists absent inputs, which are valid but worth
        mentioning. Pass a date column as its raw value (e.g. "2024-03-06");
        glin derives the `<col>__<part>` terms shown in `contributions` itself."""
        bundle = engine.load_bundle(_model_dir(model_name))
        return engine.explain_record(bundle, features, top_n=top_n)

    @tool(annotations=_read_only("Explain Model"))
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

    @tool(annotations=_read_only("List Hyperparameters"))
    def list_hyperparameters() -> list[dict[str, Any]]:
        """Every EBM training hyperparameter `train_model` accepts in its
        `hyperparameters` argument: name, type, default, allowed range, and
        what raising/lowering it does. Fetch the `tune_model_hyperparameters`
        prompt for a search strategy built on these."""
        return [hp.describe() for hp in HYPERPARAMETERS.values()]

    @tool(annotations=_TRAIN_ANNOTATIONS)
    def train_model(
        target_column: str,
        model_name: str,
        csv_path: Optional[str] = None,
        csv_content: Optional[str] = None,
        overwrite: bool = False,
        hyperparameters: Optional[dict[str, Any]] = None,
        holdout_fraction: float = 0.2,
        suppression_codes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Train an EBM classifier from a CSV and save it as model_name. This
        is also how to retrain an existing model: pass overwrite=True to
        replace it with a freshly trained model (e.g. on updated data) --
        otherwise a pre-existing model_name is rejected so it isn't silently
        clobbered.

        Pass exactly one of:
        - csv_path: a .csv file inside a directory this server allows (it
          does the reading itself). Always prefer this -- it works for CSVs
          of any size without the calling agent ever having to read or
          reproduce the file's contents itself.
        - csv_content: the complete raw CSV text (header row + data rows).
          Only use this when there's no file on disk to point at, e.g.
          data pasted into the chat or assembled in memory. Do not use this for large files --
          having an agent inline tens of thousands of rows as a tool-call
          argument is slow, burns enormous context, and risks truncation.
          Prefer writing the data to a file the server can see (csv_path)
          whenever that's possible instead.

        Either way, this tool does no cleanup of its own beyond what
        EBMTabularPreprocessor already does automatically (coercing dirty
        numeric formatting such as currency symbols, thousands separators and
        trailing % -- percentages stay in percentage points -- and treating
        suppression codes as missing, expanding date/timestamp string columns
        into year/month/day-of-week/hour/days-since-start features named
        `<col>__<part>`, dropping ID-like/constant/high-cardinality columns,
        imputing missing values). If the CSV came from an
        unfamiliar or messy source (wrong delimiter, banner rows before the
        header, embedded JSON cells, dates stored as bare numbers, a
        continuous-looking target, etc.), work through the checklist in the `prepare_csv_for_training`
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

        `suppression_codes` lists the tokens treated as missing in otherwise
        numeric columns (case-insensitive, whitespace ignored); omit it for
        the default ["SUPP", "NE", "NP", "z"]. Passing a list replaces the
        default. If the result's `warnings` say a numeric column has repeated
        non-numeric values that look like missing-value codes, confirm with
        the user and pass them here.

        Leakage: if a single feature on its own (near-)perfectly separates the
        target (a threshold of a numeric column, or a category -> class
        mapping), `warnings` carries a "Possible target leakage" entry naming
        the column and the evidence. Training still succeeds; confirm with the
        user that the column would be known at prediction time, otherwise drop
        it from the CSV and retrain.
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
            suppression_codes=suppression_codes,
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
            "date_columns": preprocessor.date_feature_names_,
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

