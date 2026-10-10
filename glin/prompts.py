"""Text of glin's MCP prompts.

MCP prompts are normally *selected by the user* in their client, not fetched
by the agent mid-conversation, so each prompt here is self-contained: the
end-to-end ``build_classifier`` prompt embeds the data-prep checklist and the
tuning loop rather than telling the agent to go fetch the other two. The
building blocks below are shared so the three prompts can't drift apart, and
every number in them (row minimums, ID thresholds, hyperparameter defaults and
ranges) is read live from the code that enforces it.
"""
from __future__ import annotations

from glin.hyperparameters import HYPERPARAMETERS, default_hyperparameters
from glin.preprocessor import EBMTabularPreprocessor
from glin.validation import MAX_REASONABLE_TARGET_CLASSES, MIN_ROWS


def prep_checklist(target_column: str = "") -> str:
    defaults = EBMTabularPreprocessor()
    target_line = (
        f"The target column you plan to train on is '{target_column}'."
        if target_column
        else "You have not yet said which column is the target -- figure that out first."
    )
    return f"""{target_line}

1. Decide how you'll hand the data to train_model.
   - If the file is on disk, pass csv_path and let the server read the
     file itself (the server runs locally, on the same machine). csv_path
     accepts .csv, .xlsx and .xls files (Excel needs the optional
     `glin-ml[excel]` install; if train_model says so, tell the user).
   - Only pass csv_content (the raw CSV text inline) when there's no file
     to point at, e.g. data pasted into the chat. Never
     inline a large file this way -- reproducing tens of thousands of rows
     as a tool-call argument is slow, burns enormous context, and risks
     truncation. If you had to clean the data in memory, write the
     cleaned result to a file and pass csv_path instead of inlining it.

2. Parse the raw file correctly first.
   - Open the file as text and look at the first ~20 lines before assuming
     it's a clean, comma-delimited, one-header-row CSV. Exports from
     spreadsheet tools often prepend banner/title rows, blank rows, or a
     second header row -- skip those before the real header.
   - Excel files (.xlsx/.xls) are read by train_model directly; don't
     convert them to CSV. Pick the sheet with `sheet` (name or 1-based
     position; required when the workbook has several sheets -- the error
     lists them). The header row is auto-detected, skipping title/banner
     rows and trailing footnotes. If detection reports an ambiguous
     header, or the sheet has a multi-row / merged header, look at the
     sheet's first rows and pass `header_rows` using the row numbers as
     Excel shows them (4, "3-4" or [3, 4]); the rows are joined into
     column names like "2023 - sales". Check the returned `sheet`,
     `header_rows` and `features_kept` to confirm the columns came out
     right, and reference columns (e.g. the target) by those flattened
     names.
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
     classifiers, not regressors (regression is planned separately). Bin
     it into meaningful classes yourself (e.g. quantiles or domain
     thresholds) before training, or pick a different target.
   - If it's ordinal (ordered levels such as reading levels or risk
     bands, or a small-integer score), a multiclass model ignores the
     order: predicting a neighbouring level is as wrong as a far-off one.
     Consider collapsing it to fewer bands or a binary threshold (e.g.
     "high risk" vs not), and tell the user the order is not modelled.
   - A numeric target stored as text (e.g. "85%") would make glin treat
     each distinct value as its own class -- convert and bin it first.
     When binning, make sure the bins cover the whole range -- gaps
     silently drop rows.

4. Know what glin's preprocessor already handles for you (don't pre-clean
   these -- it's wasted work and can conflict with its own logic):
   - Dirty-but-numeric columns (currency symbols, commas, stray
     whitespace, a trailing "%") are coerced to numeric automatically as
     long as >={defaults.numeric_coerce_threshold:.0%} of non-null values
     parse. Percentages stay in percentage points ("44.98%" -> 44.98).
   - Suppression codes ({", ".join(defaults.suppression_codes)}) in
     otherwise numeric columns are treated as missing. If train_model's
     `warnings` say a numeric column has other repeated code-like values,
     confirm with the user what they mean, then pass them via the
     `suppression_codes` argument (a list replaces the default set).
   - Date/timestamp string columns (2020-01-31, 01/31/2020, 31 Jan 2020,
     optionally with a time) are expanded automatically into named
     features: <col>__year, __month, __day_of_week, __hour (only if the
     time of day varies) and __days_since_start; parts that never vary
     are dropped. Don't pre-split them, and don't drop date columns as
     "raw dates". Prefer ISO dates: with dd/mm vs mm/dd and no value
     above 12 to settle it, glin assumes month-first and warns. Values
     that don't parse as dates become missing (also warned).
     Judge dates for leakage like any other column (post-event dates
     leak the outcome).
   - Missing values are imputed (numeric -> passed through as a native
     missing bin; categorical -> an explicit "{defaults.missing_token}"
     category).
   - ID-like columns are dropped automatically, as are constant columns.
     A numeric column counts as an ID only when it is >{defaults.id_column_threshold:.0%}
     unique, holds only whole numbers and has no repeated value at all
     (a row index or shuffled 1..N), and only once a dataset has
     >{defaults.min_rows_for_id_check} rows; a column with repeated
     values, or any float column, is kept as a measurement. Text columns
     that are >{defaults.id_column_threshold:.0%} unique are dropped.
     Every dropped column is named, with the reason, in the `warnings`
     returned by train_model (and listed in `features_dropped`). If a
     column that matters was dropped, round or bin it so values repeat.
   - High-cardinality categoricals (>{defaults.max_categorical_cardinality}
     distinct values) are dropped automatically.

5. Fix what glin's preprocessor does NOT handle -- these need your
   judgment before the CSV goes into train_model:
   - Dates stored as a bare number (a year, YYYYMMDD, epoch seconds) are
     numbers to glin, not dates: an epoch or YYYYMMDD column is binned as
     a meaningless continuous value. If it matters, convert it to a
     real date string (YYYY-MM-DD) first, or engineer year/month/etc.
   - List/dict-valued cells (e.g. a JSON blob or embedded array in a
     cell) are treated as opaque text, not expanded. If there's signal in
     there, flatten it into real columns yourself (one-hot flags, counts,
     extracted sub-fields) before training.
   - Duplicate or near-duplicate columns, leakage columns (anything that
     encodes the target, e.g. a "churned_flag" alongside a "churned"
     target), and obviously irrelevant free-text columns are not
     reliably detected -- drop them yourself. train_model only warns about
     the extreme case of a single column that all but perfectly separates
     the target (see step 6).

6. Sanity-check before and after training.
   - Call `list_models` first to make sure you're not about to silently
     clobber an unrelated model with the same name; only pass
     `overwrite=True` on `train_model` when you actually intend to retrain
     an existing model on new data.
   - After training, read the `warnings` and `features_dropped` fields in
     the result -- they tell you exactly what glin decided to ignore, so
     you can catch a mistake (like an important column being dropped as
     "ID-like") before trusting the model.
   - Review every "Possible target leakage" warning in `warnings`: glin
     found a single column (named, with the threshold or category mapping
     as evidence; date columns are named by their raw column) that almost
     perfectly separates the target, so the target may be a threshold or
     simple function of it. It is a warning, not an error -- training
     proceeded. Confirm with the user whether the column would really be
     known when making a prediction; if not, drop it from the CSV and
     retrain (a near-perfect score from it is not a good model). Keep it
     only if the user confirms the relationship is legitimate."""


def feature_selection_guide() -> str:
    return """Decide which columns are features. This judgment is yours -- glin will \
happily train on whatever it is given, including columns that make the \
model look brilliant and be worthless.

For every column ask: "Would this value be known at the moment I want to \
make the prediction?" If not, it is target leakage -- drop it. Typical \
leakage, in any domain:
   - The outcome under another name or in finer detail (a detailed/coded \
version of the label, a status or flag set because of the outcome).
   - Text written after the fact that describes the result: judgments, \
rulings, summaries, resolution/closing notes, "reason" or "outcome \
description" fields. Raw "facts"/"description" text is usually a trap too: \
glin has no NLP and treats it as an opaque label, so it is both useless and \
often leaky.
   - Anything computed from or recorded after the event: final amounts, \
post-event timestamps, downstream approvals.

Also drop:
   - Identifiers and names (record IDs, titles, person names, reference \
numbers). glin only removes the near-unique ones automatically; a name with \
a few repeats can slip through and be memorized.
   - Free-text and very long columns. Date columns are fine to keep (glin \
extracts year/month/day-of-week/hour/days-since-start itself) unless the date \
is recorded after the outcome.
   - Duplicate columns that say the same thing in several ways.
   - Columns you can't explain and that are nearly unique per row.

Keep the legitimate predictors: categoricals (type, region, category), \
numerics, booleans, date columns, and sensible engineered fields. If \
you're genuinely unsure whether a column leaks, train once with it and once \
without -- a large jump from including it is a red flag, not good news.

Red-flag check after every trial: a holdout ROC-AUC above ~0.95, or a \
log_loss collapsing far below baseline_log_loss, on real-world data almost \
always means leakage. Go back and re-examine the columns before celebrating. \
Also read the "Possible target leakage" entries in `train_model`'s \
`warnings`: they name single columns that all but perfectly separate the \
target. Ask the user whether each would be known at prediction time; if not, \
drop it and retrain before tuning.

Write the cleaned feature set (+ target) to a NEW csv file and pass it as \
csv_path; never overwrite the original."""


def tuning_loop(model_name: str = "", target_column: str = "") -> str:
    defaults = default_hyperparameters()
    knobs = "\n".join(
        f"   - {hp.name} (default {defaults[hp.name]}, range "
        f"{hp.minimum:g}..{hp.maximum:g}): {hp.guidance}"
        for hp in HYPERPARAMETERS.values()
    )
    name = model_name or "<final_model_name>"
    return f"""How the model works (so you tune with intent, not at random): an EBM is an \
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
result's `evaluation` (and check its `warnings` for "Possible target \
leakage" first -- drop flagged columns the user can't justify and re-run):
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


def tuning_prompt(model_name: str = "", target_column: str = "") -> str:
    target_line = f" on target '{target_column}'" if target_column else ""
    return f"""You're going to tune a glin model{target_line} yourself by running \
`train_model` repeatedly with different `hyperparameters` and comparing the \
out-of-sample metrics it returns. Work through this loop.

{tuning_loop(model_name, target_column)}"""


def prep_prompt(target_column: str = "") -> str:
    return f"""You're about to train a glin model from a CSV or Excel file you were handed. \
Before calling the `train_model` tool, work through this checklist -- glin's \
preprocessor handles a lot automatically, but not everything, and it will \
happily train a bad model on data it wasn't designed to catch.

{prep_checklist(target_column)}

Once the data is clean, call `train_model` with csv_path (preferred) or \
csv_content, the exact target column name, and a `model_name` to save it \
under."""


def end_to_end_prompt(
    csv_path: str = "", target_column: str = "", model_name: str = ""
) -> str:
    csv_line = (
        f"The data is at: {csv_path}"
        if csv_path
        else "No CSV path was given -- ask the user where the data is before doing anything."
    )
    final_name = model_name or "<ask the user, or derive one from the file name>"
    return f"""Build, tune, and ship a glin classifier for the data below, end to end, \
without asking the user to babysit each step. Work through the phases in \
order and report back at the end.

{csv_line}
Target column: {target_column or "(not given -- decide with the user in Phase 1)"}
Final model name: {final_name}

Before starting, confirm this glin server is current: the `list_hyperparameters` \
tool must exist and `train_model` must accept `hyperparameters` and \
`holdout_fraction`. If not, stop and tell the user the glin MCP server needs a \
restart (Claude Desktop must be fully quit and reopened to load new code).

=== PHASE 1: inspect and prepare the data ===
Look at the file yourself first (head, shape, dtypes, distinct counts, missing \
values, target balance), then apply this checklist.

{prep_checklist(target_column)}

=== PHASE 2: choose the features ===
{feature_selection_guide()}

=== PHASE 3: tune WITHOUT overfitting ===
Guardrails that apply to every trial:
   - Use a scratch model name ("{model_name or "scratch"}_trial") with \
overwrite=True and the default holdout_fraction (0.2). The holdout split is \
fixed, so trials are directly comparable. With under ~300 rows the holdout is \
tiny and noisy: use holdout_fraction 0.25-0.3, run fewer trials, and lean on \
regularization (min_samples_leaf, reg_lambda, interactions=0) rather than \
fine-grained search.
   - Hard budget: at most ~10 trials in total. Each extra trial tunes you a \
little more to this one holdout, so its score becomes optimistic.
   - Optimize holdout log_loss, but prefer the config where train_log_loss is \
CLOSE to it over the one with the single lowest holdout number. A big gap is \
overfitting even if the holdout score looks fine.
   - Never pick a winner by a difference smaller than the seed noise (step 4 \
below). When two configs tie, choose the simpler one: fewer interactions, more \
regularization, fewer bins.
   - Early exit: if after the baseline and 2-3 trials holdout log_loss is not \
clearly below baseline_log_loss (under ~2% better) and roc_auc is near 0.5, \
STOP. The features don't predict the target in this data. Tell the user that \
plainly; do not keep tuning, do not ship it as a success, and don't blame the \
model.

{tuning_loop(model_name, target_column)}

=== PHASE 4: report ===
Give the user a short, honest summary:
   - What you dropped and why (especially leakage decisions), and anything you \
engineered.
   - The trial table: config -> holdout log_loss, train_log_loss, \
balanced_accuracy, roc_auc, seconds; with baseline_log_loss for reference.
   - The chosen hyperparameters and why they won over simpler alternatives.
   - Expected performance = the winning trial's HOLDOUT numbers (the final \
model trained on all rows has no evaluation of its own), with the caveat that \
it was selected on that holdout and so is slightly optimistic.
   - Which features mattered most: call `explain_model` on the final model for \
the exact ranking, interaction terms, and per-feature shapes (don't \
approximate it from a few `predict` calls), and any caveats (small data, \
class imbalance, columns glin dropped that you'd have wanted to keep)."""
