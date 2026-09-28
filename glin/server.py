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
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP

from glin import engine
from glin.engine import DEFAULT_MODELS_ROOT
from glin.preprocessor import EBMTabularPreprocessor
from glin.validation import MAX_REASONABLE_TARGET_CLASSES, MIN_ROWS


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

    @mcp.tool()
    def train_model(
        csv_content: str,
        target_column: str,
        model_name: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Train an EBM classifier from raw CSV text and save it as model_name.
        This is also how to retrain an existing model: pass overwrite=True to
        replace it with a freshly trained model (e.g. on updated data) --
        otherwise a pre-existing model_name is rejected so it isn't silently
        clobbered.

        csv_content must be a complete, well-formed CSV (header row + data
        rows) -- this tool does no cleanup of its own beyond what
        EBMTabularPreprocessor already does automatically (coercing dirty
        numeric formatting, dropping ID-like/constant/high-cardinality
        columns, imputing missing values). If the CSV came from an
        unfamiliar or messy source (wrong delimiter, banner rows before the
        header, embedded JSON cells, date columns, a continuous-looking
        target, etc.), fetch the `prepare_csv_for_training` prompt first --
        those are judgment calls this tool does not make for you.
        """
        model_dir = models_root / model_name
        if model_dir.exists() and not overwrite:
            raise ValueError(
                f"model '{model_name}' already exists at {model_dir} -- pass "
                "overwrite=True to retrain it with this data"
            )

        df = pd.read_csv(io.StringIO(csv_content))
        bundle = engine.train_model(df, target_column, model_name=model_name)
        engine.save_bundle(bundle, models_root)

        preprocessor = bundle["preprocessor"]
        return {
            "model_name": model_name,
            "target_column": target_column,
            "rows_read": len(df),
            "target_classes": bundle["target_classes"],
            "features_kept": preprocessor.output_columns_,
            "features_dropped": preprocessor.dropped_columns_,
            "warnings": bundle["warnings"],
        }

    @mcp.prompt()
    def prepare_csv_for_training(target_column: str = "") -> str:
        """Checklist for turning a possibly-messy CSV into input for train_model."""
        defaults = EBMTabularPreprocessor()
        target_line = (
            f"The target column you plan to train on is '{target_column}'."
            if target_column
            else "You have not yet said which column is the target -- figure that out first."
        )
        return f"""You're about to train a glin model from a CSV file you were handed. \
Before calling the `train_model` tool, work through this checklist -- glin's \
preprocessor handles a lot automatically, but not everything, and it will \
happily train a bad model on data it wasn't designed to catch.

{target_line}

1. Parse the raw file correctly first.
   - Open the file as text and look at the first ~20 lines before assuming
     it's a clean, comma-delimited, one-header-row CSV. Exports from
     spreadsheet tools often prepend banner/title rows, blank rows, or a
     second header row -- skip those before the real header.
   - Confirm the delimiter is actually a comma (not ';', tab, or '|'). If
     not, convert it before building csv_content -- train_model always
     parses with pandas' default comma-delimited reader.
   - Watch for ragged rows (inconsistent column counts) and stray quoting
     -- these usually mean upstream data corruption, not something glin's
     preprocessor should paper over silently.

2. Pick the target column deliberately.
   - It must exist verbatim in the header (exact spelling/case/whitespace).
   - It needs at least 2 distinct non-null values, and the dataset needs
     at least {MIN_ROWS} rows total, or train_model will reject it.
   - If it's a numeric column with more than {MAX_REASONABLE_TARGET_CLASSES}
     distinct values, it's probably a continuous quantity -- glin trains
     classifiers, not regressors. Bin it into meaningful classes yourself
     (e.g. quantiles or domain thresholds) before training, or pick a
     different target.

3. Know what glin's preprocessor already handles for you (don't pre-clean
   these -- it's wasted work and can conflict with its own logic):
   - Dirty-but-numeric columns (currency symbols, commas, stray
     whitespace) are coerced to numeric automatically as long as
     >={defaults.numeric_coerce_threshold:.0%} of non-null values parse.
   - Missing values are imputed (numeric -> passed through as a native
     missing bin; categorical -> an explicit "{defaults.missing_token}"
     category).
   - ID-like columns (>{defaults.id_column_threshold:.0%} unique values,
     only checked once a dataset has >{defaults.min_rows_for_id_check}
     rows) are dropped automatically, as are constant columns.
   - High-cardinality categoricals (>{defaults.max_categorical_cardinality}
     distinct values) are dropped automatically.

4. Fix what glin's preprocessor does NOT handle -- these need your
   judgment before the CSV goes into train_model:
   - Date/datetime-looking columns are never parsed into features; they
     fall through to being treated as opaque categorical text (and are
     often then dropped for high cardinality, losing all signal). If a
     date matters, engineer it yourself first -- e.g. split into
     day-of-week, month, or a numeric "days since X" column.
   - List/dict-valued cells (e.g. a JSON blob or embedded array in a
     cell) are treated as opaque text, not expanded. If there's signal in
     there, flatten it into real columns yourself (one-hot flags, counts,
     extracted sub-fields) before training.
   - Duplicate or near-duplicate columns, leakage columns (anything that
     encodes the target, e.g. a "churned_flag" alongside a "churned"
     target), and obviously irrelevant free-text columns are not
     detected -- drop them yourself.

5. Sanity-check before and after training.
   - Call `list_models` first to make sure you're not about to silently
     clobber an unrelated model with the same name; only pass
     `overwrite=True` on `train_model` when you actually intend to retrain
     an existing model on new data.
   - After training, read the `warnings` and `features_dropped` fields in
     the result -- they tell you exactly what glin decided to ignore, so
     you can catch a mistake (like an important column being dropped as
     "ID-like") before trusting the model.

Once the CSV is clean, call `train_model` with the full CSV text (header +
rows) as `csv_content`, the exact target column name, and a `model_name` to
save it under."""

    return mcp


def run_stdio(models_root: Path = DEFAULT_MODELS_ROOT) -> None:
    build_server(models_root=models_root).run(transport="stdio")


def run_http(
    host: str = "0.0.0.0",
    port: int = 8000,
    models_root: Path = DEFAULT_MODELS_ROOT,
) -> None:
    build_server(host=host, port=port, models_root=models_root).run(transport="streamable-http")
