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

from glin import engine
from glin.engine import DEFAULT_MODELS_ROOT
from glin.hyperparameters import HYPERPARAMETERS, default_hyperparameters
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
    def list_hyperparameters() -> list[dict[str, Any]]:
        """Every EBM training hyperparameter `train_model` accepts in its
        `hyperparameters` argument: name, type, default, allowed range, and
        what raising/lowering it does. Fetch the `tune_model_hyperparameters`
        prompt for a search strategy built on these."""
        return [hp.describe() for hp in HYPERPARAMETERS.values()]

    @mcp.tool()
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
        - csv_path: a path this server process can read (it does the
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
        target, etc.), fetch the `prepare_csv_for_training` prompt first --
        those are judgment calls this tool does not make for you.

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
            pd.read_csv(csv_path)
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

1. Decide how you'll hand the CSV to train_model.
   - If you can reach the file on disk from wherever this MCP server runs
     (true for the common local stdio setup, e.g. Claude Desktop or
     Cursor talking to a glin server on the same machine), pass
     csv_path and let the server read the file itself.
   - Only pass csv_content (the raw CSV text inline) when there's truly no
     shared filesystem, e.g. a remote streamable-http deployment. Never
     inline a large file this way -- reproducing tens of thousands of rows
     as a tool-call argument is slow, burns enormous context, and risks
     truncation. If you had to clean the data in memory, write the
     cleaned result to a file and pass csv_path instead of inlining it.

2. Parse the raw file correctly first.
   - Open the file as text and look at the first ~20 lines before assuming
     it's a clean, comma-delimited, one-header-row CSV. Exports from
     spreadsheet tools often prepend banner/title rows, blank rows, or a
     second header row -- skip those before the real header.
   - Confirm the delimiter is actually a comma (not ';', tab, or '|'). If
     not, convert it before handing the CSV off -- train_model always
     parses with pandas' default comma-delimited reader.
   - Watch for ragged rows (inconsistent column counts) and stray quoting
     -- these usually mean upstream data corruption, not something glin's
     preprocessor should paper over silently.

3. Pick the target column deliberately.
   - It must exist verbatim in the header (exact spelling/case/whitespace).
   - It needs at least 2 distinct non-null values, and the dataset needs
     at least {MIN_ROWS} rows total, or train_model will reject it.
   - If it's a numeric column with more than {MAX_REASONABLE_TARGET_CLASSES}
     distinct values, it's probably a continuous quantity -- glin trains
     classifiers, not regressors. Bin it into meaningful classes yourself
     (e.g. quantiles or domain thresholds) before training, or pick a
     different target.

4. Know what glin's preprocessor already handles for you (don't pre-clean
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

5. Fix what glin's preprocessor does NOT handle -- these need your
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

6. Sanity-check before and after training.
   - Call `list_models` first to make sure you're not about to silently
     clobber an unrelated model with the same name; only pass
     `overwrite=True` on `train_model` when you actually intend to retrain
     an existing model on new data.
   - After training, read the `warnings` and `features_dropped` fields in
     the result -- they tell you exactly what glin decided to ignore, so
     you can catch a mistake (like an important column being dropped as
     "ID-like") before trusting the model.

Once the CSV is clean, call `train_model` with csv_path (preferred) or
csv_content, the exact target column name, and a `model_name` to save it
under."""

    @mcp.prompt()
    def tune_model_hyperparameters(model_name: str = "", target_column: str = "") -> str:
        """Search strategy for tuning an EBM's hyperparameters with train_model."""
        defaults = default_hyperparameters()
        knobs = "\n".join(
            f"   - {hp.name} (default {defaults[hp.name]}, range "
            f"{hp.minimum:g}..{hp.maximum:g}): {hp.guidance}"
            for hp in HYPERPARAMETERS.values()
        )
        name = model_name or "<final_model_name>"
        target_line = f" on target '{target_column}'" if target_column else ""
        return f"""You're going to tune a glin model{target_line} yourself by running \
`train_model` repeatedly with different `hyperparameters` and comparing the \
out-of-sample metrics it returns. Work through this loop.

How the model works (so you tune with intent, not at random): an EBM is an \
additive model -- a separate learned curve per feature, plus a few pairwise \
interaction tables -- built by cyclic boosting with tiny single-feature \
trees at a small learning rate, then averaged over `outer_bags` resamples. \
Every knob trades flexibility against smoothness: fine bins, many \
interactions, weak regularization and long training fit more detail (and \
more noise); the opposite settings are smoother and generalize better on \
small or noisy data. Tuning is finding the flexibility the data can support.

1. Fetch `list_hyperparameters` once if you need the machine-readable \
version of the table below.

2. Establish a baseline: call `train_model` with NO `hyperparameters` and a \
scratch name (e.g. "{name}_trial") with overwrite=True. Record from the \
result's `evaluation`:
   - log_loss: your primary score; lower is better. Compare it against \
baseline_log_loss (always guessing the class frequencies) -- if it isn't \
clearly lower, the features carry little signal and no tuning will fix that.
   - roc_auc / balanced_accuracy: secondary; prefer balanced_accuracy over \
accuracy when classes are imbalanced.
   - train_log_loss vs log_loss: a large gap (train much lower) means \
overfitting -> move toward smoother settings. A small gap with both high \
means underfitting -> move toward more flexible settings.
   - training_seconds: your cost per trial; budget your trial count by it.

3. Search coarse to fine, changing ONE or TWO knobs per trial so you can \
attribute the change. Suggested order, highest leverage first:
   a. interactions: try 0 against the default. If 0 is as good, keep it -- a \
purely additive model is simpler and fully readable per feature.
   b. min_samples_leaf (and reg_lambda): overfitting controls. Raise them if \
there's a train/holdout gap.
   c. max_bins: lower for small data, higher for large.
   d. learning_rate / early_stopping_rounds / outer_bags: refinements. Use \
a larger learning_rate and fewer outer_bags to make exploratory trials \
fast, then restore finer values for the final fit.
   Stop when a few consecutive trials improve log_loss by less than about \
1% or less than the run-to-run noise (see 4), or when you hit your trial budget.

4. Know the noise floor: the holdout split is fixed across all your trials, \
so differences are comparable -- but with a small holdout, small differences \
are luck. Re-run your best config with two other `random_state` values; if \
the spread is as large as the gain you were chasing, prefer the simpler \
configuration. Don't run dozens of trials hunting a tiny gain: every trial \
tunes you a little more to this one holdout set.

5. Finish: call `train_model` with the winning `hyperparameters`, the REAL \
model name ({name}), holdout_fraction=0 (train on every row) and \
overwrite=True only if replacing an existing model. This final model has no \
`evaluation` of its own -- report the holdout numbers from the winning \
trial as its expected performance. Tell the user the chosen hyperparameters \
and the trial table (config -> log_loss, balanced_accuracy, seconds).

Hyperparameters you can set:
{knobs}

Out-of-range or unknown values are rejected with a message naming every \
problem, so you can correct and retry."""

    return mcp


def run_stdio(models_root: Path = DEFAULT_MODELS_ROOT) -> None:
    build_server(models_root=models_root).run(transport="stdio")


def run_http(
    host: str = "0.0.0.0",
    port: int = 8000,
    models_root: Path = DEFAULT_MODELS_ROOT,
) -> None:
    build_server(host=host, port=port, models_root=models_root).run(transport="streamable-http")
