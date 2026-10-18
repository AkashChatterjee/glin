# glin

☕ If glin is useful to you and you want to help maintain it, you can [buy me a coffee](https://buymeacoffee.com/akashchatterjee).

<!-- mcp-name: io.github.AkashChatterjee/glin -->

glin is an [MCP](https://modelcontextprotocol.io/) plugin. It gives AI agents a fast, statistical "gut feeling" (System 1 thinking). You install it into an agent such as Claude Code, Claude Desktop or Cursor. Then you talk to the agent in plain language. The agent calls the glin tools.

glin trains an [Explainable Boosting Machine](https://interpret.ml/) (EBM) on a CSV or Excel file. Each prediction comes with an exact breakdown of the features that drove it. The breakdown comes from the additive structure of the model. It is not a SHAP or LIME approximation.

To use glin, you do not write code and you do not type glin commands. The agent trains, tunes, predicts and explains. The server runs on your machine and talks to the agent over stdio.

> glin runs locally only (stdio). It has no remote or HTTP mode. Version 0.1.3 is the last release that had one.

## What you can ask the agent

- "Train a classifier on `~/data/leads.csv` that predicts `converted`."
- "Which of these 3 deals should I prioritize?" The agent runs each deal through the model and gives the reasons.
- "What drives churn in this model?"
- "Tune the model, and tell me how good it really is."

See [Example: ask Claude directly](#example-ask-claude-directly) for a full conversation.

**Changelog:** [CHANGELOG.md](https://github.com/AkashChatterjee/glin/blob/main/CHANGELOG.md) lists the changes in each release.

**Case study:** [The segment of 100,000 sales leads](https://akashchatterjee.com/tech/2026/10/11/glin-ml-case-study-sales-leads-synthetic). Claude with glin put 71% of new leads in the correct segment. Claude alone got 33%. With 1,000 labeled leads in the prompt, Claude got 60% and used 6.7 times more tokens.

## Contents

- [What you can ask the agent](#what-you-can-ask-the-agent)
- [Install](#install)
  - [Claude Code](#claude-code)
  - [Other MCP clients](#other-mcp-clients)
  - [MCP Registry](#mcp-registry)
  - [Install for development](#install-for-development)
- [Train a model](#train-a-model)
  - [Model names](#model-names)
  - [Excel input](#excel-input-xlsx--xls)
  - [Delete a model](#delete-a-model)
- [Tools exposed over MCP](#tools-exposed-over-mcp)
- [Prompts exposed over MCP](#prompts-exposed-over-mcp)
- [Restricting `csv_path`](#restricting-csv_path)
- [Let an agent tune the model](#let-an-agent-tune-the-model)
  - [Grouped holdout](#grouped-holdout)
- [Example: ask Claude directly](#example-ask-claude-directly)
- [Case study](#case-study)
- [Data requirements](#data-requirements)
  - [Feature columns: supported today](#feature-columns-supported-today)
  - [Dates and timestamps](#dates-and-timestamps)
  - [Feature columns: not yet supported](#feature-columns-not-yet-supported)
  - [Target leakage check](#target-leakage-check)
  - [Target column requirements](#target-column-requirements)
- [Releasing (maintainers)](#releasing-maintainers)
  - [Release steps](#release-steps)
  - [One-time setup that makes this the only path](#one-time-setup-that-makes-this-the-only-path)
  - [Publishing to the MCP Registry](#publishing-to-the-mcp-registry)
  - [Checking a release](#checking-a-release)

## Install

glin needs [`uv`](https://docs.astral.sh/uv/) on your machine. The MCP client starts the server with `uvx`. You do not install glin yourself.

The server code is the PyPI package `glin-ml`. The name `glin` belongs to an unrelated project. `uvx` downloads `glin-ml` the first time the client starts the server, and caches it.

### Claude Code

The plugin adds the glin MCP server and a skill. The skill tells the agent when and how to build, tune, predict with and explain a model. The plugin runs `uvx --from 'glin-ml[excel]>=0.1.4' glin serve`. This command includes Excel support, and it does not start a version older than 0.1.4.

In Claude Code, run:

```text
/plugin marketplace add AkashChatterjee/glin
/plugin install glin@glin-marketplace
```

The files are in [`.claude-plugin/marketplace.json`](https://github.com/AkashChatterjee/glin/blob/main/.claude-plugin/marketplace.json) and [`plugins/glin/`](https://github.com/AkashChatterjee/glin/tree/main/plugins/glin).

### Other MCP clients

Add glin to the config of your MCP client, for example Claude Desktop or Cursor. In Claude Desktop, the file is `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "glin": {
      "command": "uvx",
      "args": ["--from", "glin-ml[excel]", "glin", "serve"]
    }
  }
}
```

Restart the client. The glin tools and prompts then appear. The `[excel]` extra adds support for `.xlsx` and `.xls` files. Remove it if you only use CSV files. See [Excel input](#excel-input-xlsx--xls).

The server gives the glin version in `serverInfo.version`. `glin serve` writes only MCP messages to stdout. Log messages go to stderr. During `serve`, glin sets the log level of the `interpret` library to WARNING, so each training does not fill the client log.

To add server flags such as `--allow-dir`, put them after `serve` in `args`. See [Restricting `csv_path`](#restricting-csv_path).

To run glin from a copy that is already installed (`uv tool install glin-ml`, `pipx install glin-ml` or `pip install glin-ml`), set `"command": "glin"` and `"args": ["serve"]`.

### MCP Registry

After each release, the maintainer publishes glin to the [official MCP Registry](https://registry.modelcontextprotocol.io) as `io.github.AkashChatterjee/glin`. This is a manual step (see [Publishing to the MCP Registry](#publishing-to-the-mcp-registry)). The metadata is in [`server.json`](https://github.com/AkashChatterjee/glin/blob/main/server.json). The registry points to the `glin-ml` package on PyPI. The package also installs a `glin-ml` command, which is the same as `glin`, so `uvx glin-ml serve` works.

The registry entry installs `glin-ml` without the `excel` extra, because the registry format has no field for extras. To read Excel files, set your MCP client to run `uvx --from 'glin-ml[excel]' glin serve` (command `uvx`, arguments `--from`, `glin-ml[excel]`, `glin`, `serve`).

### Install for development

Install an editable copy from a clone of this repository:

```bash
git clone https://github.com/AkashChatterjee/glin.git
cd glin
pip install -e ".[dev]"      # also installs the Excel libraries, and xlwt to build .xls test files
```

To try the checkout in an MCP client, set `"command"` to the `glin` program in your environment (for example `/path/to/glin/.venv/bin/glin`) and `"args"` to `["serve"]`.

glin works with the `mcp` 1.x and 2.x Python SDK lines (`mcp>=1.21.1,<3`). Versions before 1.21.1 do not work. mcp 1.21.0 and earlier fail on import with pydantic 2.14 and later. The earliest 1.x versions also do not have the API that glin uses. To test one line, add a constraint. For example: `pip install -e ".[dev]" "mcp>=2,<3"`.

The `Tests` GitHub Actions workflow ([.github/workflows/test.yml](.github/workflows/test.yml)) runs the test suite on each pull request and on each push to `main`. It runs on Python 3.10 and 3.12 with the latest mcp 1.x and the latest mcp 2.x. It also runs on Python 3.10 with the lowest supported mcp (`mcp==1.21.1`).

## Train a model

Ask the agent. Give it a data file and the column to predict:

> "Train a classifier on `~/data/churn.csv` that predicts `churn`. Call it `churn_v1`."

The agent calls `train_model`. For a careful build, ask the agent to use the `build_classifier` prompt. It cleans the data, drops leakage, tunes the model and reports the honest performance. See [Prompts exposed over MCP](#prompts-exposed-over-mcp).

glin saves each model in `~/.glin/models/<name>/`.

### Model names

When you make a model (`model_name` in `train_model`), the name must obey this rule:

- Use 1 to 64 characters.
- Use only letters, digits, `-`, `_` and `.`.
- Do not start the name with `.`. Thus `.`, `..` and hidden names are not valid.
- Do not use `/` or `\`. Thus a path or an absolute path is not valid.

When you use or delete a model (`inspect_model`, `predict`, `explain_model`, `delete_model`), the name must be one folder name in the models folder. It must not contain `/` or `\`, and it must not be empty, `.` or `..`. Thus you can still use models that an earlier glin version saved with other names, for example a name with a space.

glin rejects a name that does not obey the applicable rule. The error message gives the rule. glin also makes sure that the model folder is directly in the models folder, also when a symbolic link is in the path. Thus a model name cannot write, read or delete files outside the models folder.

### Excel input (`.xlsx` / `.xls`)

glin trains directly on spreadsheets that have title rows and merged headers, for example government tables. Excel support is an optional extra, so the base install stays small. `openpyxl` reads `.xlsx` files. `xlrd` reads old `.xls` files. The Claude Code plugin includes the extra. For other clients, use `glin-ml[excel]` in the server command (see [Other MCP clients](#other-mcp-clients)).

Ask the agent in plain language, for example:

> "Train a classifier on `~/data/report.xlsx`, sheet `Data 2023`, header rows 3-4, that predicts `outcome`."

The agent passes `sheet` and `header_rows` to `train_model`. These rules apply:

- **Sheet:** `sheet` takes a name or a 1-based position. A workbook with one sheet needs no `sheet`. A workbook with several sheets and no `sheet` gives an error that lists the sheets. glin ignores case and spaces at the start and end of a name only if one sheet matches. If two or more sheets match (for example `Data` and `data `), glin gives an error that lists them. Then use the exact name or the position.
- **Header detection:** Without `header_rows`, glin finds the first row that is mostly filled and looks like labels. It skips title rows above it. If the choice is not clear, glin stops with an error. The error names the rows and asks for `header_rows`. glin does not guess. Two cases are not clear. In the first, a sparse group-label row sits next to the header row. In the second, the first full row looks like data.
- **Footnotes:** glin drops footnote rows at the end of the sheet. A footnote row has text in half of the columns or fewer, and no numbers. Its first filled cell starts with `Source`, `Note`, `Footnote`, `*`, `†`, `‡` or `©`, or it is a sentence of six words or more. A sparse last row that does not look like a note stays as data. glin gives a warning with the sheet row numbers of the rows that it dropped.
- **`header_rows`:** Use 1-based row numbers as shown in Excel: `4`, `"3-4"`, `"3,4"` or `[3, 4]`. glin joins a multi-row header into one name for each column with ` - `, for example `2023 - sales`. A blank cell in an upper header row continues the label from the cell on its left (merged group headers). glin drops blanks and repeated parts. Duplicate names get the suffixes `_2` and `_3`. A header cell that holds a date becomes an ISO date, for example `2024-01-01`. Data starts below the last header row. The result shows the sheet, the header rows and the number of rows that glin read (`rows_read`).
- **Formulas:** glin uses the cached results of the cells. A spreadsheet app must calculate and save the formulas first.
- **Empty rows and columns:** glin drops fully empty rows and columns.
- **CSV files:** CSV behavior does not change. `sheet` and `header_rows` are errors for CSV files.

### Delete a model

Ask the agent, for example: "Delete the model `churn_v1`." The agent calls `delete_model`. The deletion is permanent. To replace a model with a new one, train it again with `overwrite=True`. You can also remove the folder in `~/.glin/models/` yourself.

The `glin` program only starts the MCP server (`glin serve`). It has no commands to train, list or delete models. Version 0.1.4 is the last release that had them.

## Tools exposed over MCP

Every tool has MCP annotations (title, `readOnlyHint`, `destructiveHint`). All tools are read-only except `train_model` and `delete_model`. These two tools are marked destructive: `train_model` with `overwrite=True` replaces a model, and `delete_model` removes one. A client can approve the read-only tools automatically and ask before it runs `train_model` or `delete_model`.

- `list_models()` lists all trained models.
- `inspect_model(model_name)` returns the feature schema and the target classes. It lists date columns under `features.date`. Each date column shows its derived terms and the `start_date` that `days_since_start` counts from.
- `predict(model_name, features, top_n=10)` returns the predicted class and the probabilities, plus a full audit:
  - `base_logit` is the intercept.
  - `contributions` lists the contribution of every term, sorted by size.
  - `additivity_verified` checks that `reconstructed_probability` equals `model_probability`. Both are keyed by class name.
  - For binary targets, `score_class` is the second entry of `target_classes`. glin sorts the classes by value, so this is not always the alphabetically second class. The scores and `base_logit` are log-odds for this class. A positive score pushes toward it. A negative score pushes toward the other class.
  - `predicted_class`, `score_class` and the entries of `target_classes` have the type of the target values (for example `1`, or `true` in JSON). The keys of `probabilities`, `model_probability` and `reconstructed_probability` are always text (for example `"1"` or `"True"`). To find the probability of `score_class` in Python, use `model_probability[str(score_class)]`.
  - For multiclass targets, `base_logit` is per class, and each term has `score_by_class`.
  - `warnings` lists inputs that the model never saw. Each warning has `feature`, `value` and `issue`. The issue is `unseen_category`, `out_of_range`, `unparseable_number`, `unparseable_date`, `ambiguous_date_format` or `date_order_mismatch`. `out_of_range` warnings also have `training_range`. Date format warnings also have `parsed_as`, the date that glin used. `unparseable_number` is for a numeric input that is not a finite number, for example `abc` or `inf`. glin then treats the input as missing. A blank input or a suppression code does not give this warning. Each `value` is safe for JSON: glin sends `inf` as the string `"inf"`.
  - `missing_features` lists inputs that are absent or blank. These inputs are valid. glin reports them separately.
  - To send a date, pass the raw value, for example `"signup_date": "2024-03-06"`. glin derives the same `signup_date__*` terms as at training time. Contributions name each term, for example `signup_date__day_of_week` = `sat`. A date that is absent, blank or not parseable counts as missing. A date that does not parse also adds a warning.
- `train_model(target_column, model_name, csv_path=None, csv_content=None, overwrite=False, hyperparameters=None, holdout_fraction=0.2, holdout_group_column=None, suppression_codes=None, sheet=None, header_rows=None)` trains a new model. It can also retrain an existing model on new data with `overwrite=True`. If the name exists and `overwrite` is not set, the call fails. Nothing is replaced by accident.
  - Pass exactly one of `csv_path` and `csv_content`.
  - `csv_path` is a `.csv`, `.xlsx` or `.xls` file that the server can read. See [Restricting `csv_path`](#restricting-csv_path). Use it for local stdio setups. It works for large files (up to 200 MB by default), and the agent never copies the file contents.
  - `csv_content` is the raw CSV text. Use it only when the client and the server share no file system, and only for small data. Tens of thousands of rows in a tool argument are slow and use a lot of context.
  - `holdout_group_column` makes the holdout a [grouped holdout](#grouped-holdout).
  - `suppression_codes` replaces the tokens that count as missing in numeric columns. The default is `["SUPP", "NE", "NP", "z", "*", ".."]`. See [Feature columns: supported today](#feature-columns-supported-today).
  - For Excel files (they need `glin-ml[excel]`), two more arguments apply. `sheet` is a name or a 1-based position. It is required for workbooks with several sheets. `header_rows` is `4`, `"3-4"` or `[3, 4]`. glin detects it if you omit it. See [Excel input](#excel-input-xlsx--xls). The result shows the `sheet` and `header_rows` that glin used.
  - The result has the kept and dropped features, the resolved `hyperparameters`, `training_seconds` and the term counts. `date_columns` maps each expanded date column to its derived terms. If `holdout_fraction > 0`, the result also has an `evaluation` block of out-of-sample metrics. See [Let an agent tune the model](#let-an-agent-tune-the-model).
  - `warnings` has the reason for each dropped column and any "Possible target leakage" entries. See [Target leakage check](#target-leakage-check).
- `explain_model(model_name, top_n=15, include_shapes=True)` shows what the model learned. glin reads it exactly from the additive structure. It does not sample example predictions. The result has:
  - every feature, ranked by mean absolute log-odds contribution, with its main-effect part and its interaction part;
  - the top terms, including pairwise interactions, with their share of the total;
  - for binary targets, a shape for each feature that shows which categories or value ranges push toward which class. For a continuous feature, the shape gives the value range (bin edges, in the units of the feature) where the curve is highest and lowest. The edges have 4 significant digits. Values below 10,000,000 do not use exponent notation (for example `21140`, not `2.114e+04`).
- `list_hyperparameters()` lists every EBM hyperparameter. Each entry has its type, default, allowed range and a note on what a higher or lower value does.
- `delete_model(model_name)` permanently deletes a trained model and its folder. It uses the same name rules as `inspect_model` (see [Model names](#model-names)). If the model does not exist, the call fails with a "not found" error. The result has `model_name`, `deleted` and `path`.

## Prompts exposed over MCP

- `build_classifier(csv_path="", target_column="", model_name="")` runs the whole workflow for any dataset in one prompt:
  1. Inspect and clean the CSV.
  2. Decide which columns are real predictors. Drop target leakage, such as text that describes the outcome, identifiers, free text and dates after the outcome.
  3. Tune the hyperparameters with these guardrails against overfitting:
     - a fixed holdout;
     - a budget of about 10 trials;
     - a check of the gap between train and holdout;
     - a check of seed noise;
     - a preference for the simpler configuration;
     - an early exit when the features carry no signal.
  4. Refit on all rows.
  5. Report the expected performance honestly.

  Select it in the prompt picker of your MCP client. Claude Desktop lists server prompts in the attach (`+`) menu. Give it a file path and a target. All other arguments are optional. The prompt is self-contained, because clients show prompts to the user and do not let the agent fetch them.
- `tune_model_hyperparameters(model_name="", target_column="")` runs the tuning loop in [Let an agent tune the model](#let-an-agent-tune-the-model). It includes the live hyperparameter table. glin reads the defaults and ranges from the code.
- `prepare_csv_for_training(target_column="")` is a checklist for an agent to follow before it calls `train_model`. It covers:
  - parsing problems: banner rows, a wrong delimiter, ragged rows, and the choice of Excel sheet and header rows;
  - how to choose a valid target column;
  - what the preprocessor already handles: dirty numeric formats, date and timestamp expansion, missing values, and identifier-like, constant and high-cardinality columns;
  - what the preprocessor does not handle: dates stored as bare numbers, embedded JSON or list cells, leakage columns and targets that look continuous.

  glin reads this guidance from the actual defaults of the preprocessor, so it cannot drift from the code.

## Restricting `csv_path`

`train_model(csv_path=...)` reads only a regular file that ends in `.csv`, `.xlsx` or `.xls`. The name `csv_path` stays for compatibility, and it also takes Excel files. The fully resolved path (symlinks and `..` followed) must be inside an allowed directory. glin rejects directories, devices, other extensions, missing files, path traversal and symlinks that point outside. The error never includes file contents.

By default, the allowed directories are:

- the home directory;
- the working directory of the server, but only when it is not a filesystem root (`/`) and not a parent of the home directory (for example `/Users`).

Some clients, for example Claude Desktop on macOS, start the server from `/`. Then only the home directory is allowed. Claude Code starts the server from the project directory, so that directory is also allowed. To set the directories yourself, use a repeatable flag:

```bash
glin serve --allow-dir /data/csvs --allow-dir /srv/uploads
```

If an allowed directory is `/`, glin writes a warning to stderr, because then `csv_path` can read any `.csv`, `.xlsx` or `.xls` file on the machine. In Python, `build_server(allowed_dirs=[])` allows no directories. Then `csv_path` is disabled and only `csv_content` works.

glin rejects a file that is larger than 200 MB. It checks the size before it reads the file. To change the limit, use `--max-file-mb`, for example `glin serve --max-file-mb 500`.

`csv_content` is not affected by these rules. In the config of your MCP client, add the flags to `args`, for example `["serve", "--allow-dir", "/data/csvs"]`.

These checks have two known limits. Both need write access inside an allowed directory:

- A hard link inside an allowed directory to a file outside it is read. A hard link is the file itself, so glin cannot see where it came from.
- glin checks the path and then reads it. If the path changes between the check and the read, glin reads the new target.

Thus, do not allow a directory that untrusted users or processes can write to.

The older flag `--mode stdio` is still accepted so that existing client configs keep working. glin ignores it and writes one deprecation notice to stderr. You can remove it from `args`. `--mode http` and `--mode sse` stop with exit code 2 and this message: "HTTP mode was removed in v0.1.4; v0.1.3 is the last release with it. glin runs over stdio only."

## Let an agent tune the model

An EBM is a sum of one learned curve for each feature, plus a few pairwise interaction tables. Cyclic boosting with tiny single-feature trees fits it. Most of its settings trade flexibility against smoothness. `train_model(hyperparameters={...})` exposes them: `max_bins`, `max_interaction_bins`, `interactions`, `learning_rate`, `max_rounds`, `early_stopping_rounds`, `validation_size`, `outer_bags`, `inner_bags`, `min_samples_leaf`, `max_leaves`, `reg_alpha`, `reg_lambda`, `smoothing_rounds`, `greedy_ratio` and `random_state`. Settings that you omit keep their defaults. glin rejects unknown names, wrong types and values out of range. The error message lists every problem. `glin/hyperparameters.py` defines each setting once. Add or widen a setting there.

An agent needs a score to optimize. `train_model` holds out `holdout_fraction` of the rows and trains on the rest. The holdout is stratified. Its split uses a seed that does not depend on `random_state`, so every trial is scored on the same rows. The result has an `evaluation` block:

- `log_loss`
- `baseline_log_loss` (a guess from the class frequencies)
- `train_log_loss` (compare it with `log_loss` to see overfitting or underfitting)
- `accuracy`
- `balanced_accuracy`
- `roc_auc`

glin saves the hyperparameters and the metrics in the metadata of the model.

The `tune_model_hyperparameters` prompt guides the agent through this loop:

1. Train a baseline.
2. Change one or two settings at a time.
3. Check the seed noise.
4. Refit on all rows with `holdout_fraction=0`.

Trials must use a scratch `model_name` with `overwrite=True`.

### Grouped holdout

Rows from the same state, region, customer, site or user look alike. A random holdout puts near-duplicates on both sides, so the metrics are too optimistic. In one real case, a model scored 82% accuracy and 0.89 AUC on a random holdout. It scored 63% and 0.77 on states that it never saw.

Pass `holdout_group_column="state"` (or any other column) to `train_model`. glin then holds out whole groups until about `holdout_fraction` of the rows are in the holdout. The other rows train the model.

- The split uses a seed that does not depend on `random_state`, so every trial uses the same rows. glin tries 25 seeded group orders. It keeps the order whose holdout class shares are closest to the overall class shares.
- The group column stays a normal feature. At prediction time, a held-out group is therefore an unseen category. glin fits the preprocessing on the training rows only.
- glin groups text values in the same way as the preprocessor: it removes spaces at the start and end and changes letters to lowercase. Thus `"S06"` and `"s06 "` are one group, and that group is on one side of the split only. Blank text, `nan` and `none` are missing values. If two or more spellings become one group, glin returns a warning.
- `evaluation` also reports `holdout_group_column`, `holdout_groups` and `train_groups`. These counts use the normalised groups. Missing values count as one group. glin also saves the column name in the model metadata.
- glin rejects these cases with a clear error:
  - the column does not exist, or it is the target;
  - fewer than 5 distinct non-null groups exist;
  - one group has more than half of the rows. If this group is the missing values, the error says so. Fill the missing values, or use a different column;
  - every group is too large for the holdout target. The error gives the size of the smallest group. Use a larger `holdout_fraction`;
  - the holdout has fewer than 2 classes, or the split leaves a class out of training.

  If `holdout_fraction=0`, glin ignores the column and returns a warning.
- The `build_classifier` and `tune_model_hyperparameters` prompts tell the agent to use this option when such a column exists. They also tell the agent to report this holdout as the honest estimate.

## Example: ask Claude directly

When a model is trained and the MCP server is attached, talk to Claude in plain language. You do not need to know the tool schema.

> "I've got 3 deals to prioritize before quarter close: Northwind Systems ($42k, Opportunity stage, came from a referral, VP contact, owned by Carla Nguyen, 8 touches logged), Summit Retail Group ($8k, still a Lead, paid social source, owned by Brian Kessler), and Anchor Nonprofit ($3k, still a Lead, owned by Frank Suarez, no activity logged yet). Run them through deal_predictor_v1 and tell me which to prioritize."

Claude calls `predict` once for each deal. It returns a ranked answer with reasons, not only a number:

| Deal | Win probability | Why |
|---|---|---|
| Northwind Systems | 69% | The Opportunity stage, the referral and the VP contact all favor it. The $42k deal size is the one drag. |
| Summit Retail Group | 35% | It is still a Lead, and it came from paid social. Both are large drags with nothing against them. |
| Anchor Nonprofit | 34% | No activity is logged and it is still a Lead. Nothing in the record favors it. |

Note: a field that you do not mention reaches the model as missing. It does not reach the model as "average". Name a rep and a lifecycle stage, and use the categories that the model learned, for example `Lead`, `Opportunity` and `Customer`. This gives a better answer than extra detail in fields that the model already has.

## Case study

[The segment of 100,000 sales leads](https://akashchatterjee.com/tech/2026/10/11/glin-ml-case-study-sales-leads-synthetic) uses glin on synthetic CRM data with known rules. Because the rules are known, the study can check each prediction, explanation and what-if against the truth.

- Claude with glin put 71% of new leads in the correct segment. The best possible score on these leads is 68%.
- Claude alone got 33%. With 1,000 labeled leads in the prompt, Claude got 60% and used 83,000 tokens for each lead. With glin, Claude used approximately 12,500 tokens.
- glin gave the same answer for the same lead in each new session.
- The effects that glin learned agree with the true effects (correlation 0.97 to 1.00).
- One 4-class model and two binary models gave the same accuracy. Two models give a separate explanation for each outcome.

An earlier study, [Child stunting across Indian districts (NFHS-5)](https://akashchatterjee.com/tech/2026/10/11/glin-ml-case-study-child-stunting-nfhs5), uses glin on real public-sector data. A glin model predicts whether a district has high child stunting. The model gives the exact reason for each prediction.

## Data requirements

`train_model` checks your data before it does any work. The data is a CSV, or a sheet of an Excel workbook after header detection. A hard problem stops training with a clear error. An example is a target column with only one class. A soft problem gives a warning, and training continues. An example is a numeric target with many distinct values.

The rules are in `glin/validation.py` as a flat list. To support a new data shape, add one rule and one preprocessing case. The rest of the pipeline does not change.

In the Python API, a DataFrame can have column names that are not text, for example `0` and `1`. glin changes them to text (`"0"`, `"1"`), and the results use these names. `predict` accepts either form as a key. Two names that are the same as text, such as `0` and `"0"`, give an error.

### Feature columns: supported today

| Data shape | What happens |
|---|---|
| Numeric (int, float), including `NaN` | glin passes the values through. EBM puts missing values in their own bin. glin keeps a column with almost all distinct values if a value repeats or the values have decimals, for example rupee amounts. glin also keeps unique whole numbers with large, uneven gaps, for example salaries. For the columns that glin drops as identifiers, see "Row identifiers" below. |
| Dirty numeric strings (blanks, for example `" "` for the total charges of a new customer) | glin converts the column to `float64` if at least 80% of the non-null values parse as numbers. Cells that do not parse become `NaN`. |
| Categorical strings, up to 250 unique values | glin lowercases and strips the text. EBM bins each category. Missing and unseen values map to a `__missing__` marker. |
| Currency and thousands-separated strings (`$1,234.56`, `£1,234`, `€1.234,56`, `USD 100`, `Rs. 100`, `100 EUR`) | glin removes currency symbols, spaces and common currency codes (`USD`, `EUR`, `GBP`, `INR`, `Rs.` and others) before or after the number. Then it converts the column as above. glin reads a decimal comma only when the column has clear European values (`1.234,56`, `12,5`, `1.234.567`). A plain `1,234` is a thousands group. The same rules apply at predict time. glin does not change native numbers (int, float). In a value with both separators, glin uses the last separator as the decimal mark. Thus `1,500.50` and `1.500,50` both become 1500.5. In a European column, `1500.5` is 1500.5, but `1.500` is 1500. A column that mixes both styles is parsed as US style, and glin returns a warning. In that column, a European value with only a comma, such as `12,5`, becomes 125. |
| Negative numbers (`-5`, `−5`, `(1,234)`, `1234-`) | glin reads the Unicode minus sign (`−`, common in UK tables), brackets and a trailing minus (common in accounting exports) as a negative sign. |
| Infinity (`inf`, `-inf`, `inf%`, a float `inf`) | glin treats infinity as missing (`NaN`). `train_model` returns a warning that gives the number of infinite values. At predict time, an infinite input gives `unparseable_number`. |
| Percent strings (`44.98%`, `12 %`, `(5%)`) | glin removes the trailing `%` before it converts the value. The value stays in percentage points: `44.98%` becomes `44.98`, not `0.4498`. This works together with the currency rules. `50%` and `50` in one column give the same value. |
| Suppression codes in numeric columns (`SUPP`, `NE`, `NP`, `z`, `*`, `..`) | glin treats these codes as missing (`NaN`). It ignores them when it decides if a column is numeric, so the column stays numeric. Matching ignores case and surrounding spaces. To change the set, use `EBMTabularPreprocessor(suppression_codes=[...])` or `train_model(suppression_codes=[...])`. A list replaces the default. Categorical columns do not change, so `NE` as a region stays a category. glin also treats a few short, repeated values that are not in the set (for example `x` or `c`) as missing. `train_model` then returns a warning that suggests the `suppression_codes` to confirm. If unknown codes are more than 20% of the values, the column becomes a category. If at least half of the values are numbers, `train_model` returns a warning that suggests the `suppression_codes` that make the column numeric. `<5` and `—` are not in the default set, because they can be real values: `<5` is a value below 5, and `—` is often zero in financial tables. |
| Boolean columns | glin treats them as a 0/1 continuous numeric feature. |
| Dates and timestamps | glin detects them and splits them into named terms. See [Dates and timestamps](#dates-and-timestamps). |

### Dates and timestamps

These strings work: `2024-03-06`, `2024-03-06 14:30:00`, `2024-03-06T14:30:00Z`, `03/06/2024`, `6 Mar 2024` and `Mar 6, 2024`. Real datetimes work in the Python API.

- **Detection:** At least 80% of the non-null values must be full calendar dates. You can change the threshold with `EBMTabularPreprocessor(date_parse_threshold=...)`. glin checks dates before the identifier and cardinality rules, so a date that is unique for each row stays.
- **Terms:** glin makes `<col>__year`, `<col>__month` (`jan` to `dec`), `<col>__day_of_week` (`mon` to `sun`), `<col>__hour` and `<col>__days_since_start`. `days_since_start` counts whole days since the earliest training date. glin stores that date with the model, and `inspect_model` shows it.
- **Reading explanations:** Month and day of week are categorical, so an explanation reads `signup_date__day_of_week = sat`. The other terms are ordered numbers.
- **Dropped parts:** glin drops a part that never changes in training. Single-year data has no `__year`. Date-only data has no `__hour`. A column with one distinct date is dropped.
- **No sin/cos terms:** An EBM learns a free-form curve for each feature, so it can already show the December-to-January wrap-around. Sin and cos terms are hard to read.
- **Time zones:** glin converts values that have a time zone to UTC.
- **Day and month order:** glin reads `dd/mm` or `mm/dd` from the values. A field above 12 decides. If nothing decides, glin assumes month first, and `train_model` returns a warning. Use ISO `YYYY-MM-DD` to avoid the warning.
- **Values that do not parse:** They become missing, with a warning.
- **Bare numbers are never dates.** Years, `20240306`, epoch seconds and IDs stay numeric. Month-only and quarter strings (`Jan`, `Q1 2020`) and version strings (`1.2.3`) stay categorical. If the date matters, convert these values to date strings first.
- **At predict time:** Send the raw date. A date outside the training range returns an `out_of_range` warning. A date that does not parse returns `unparseable_date`. A month or weekday that training never saw returns `unseen_category`. A numeric date such as `05/03/2024` on a column with no numeric dates in training (ISO or month names) returns `ambiguous_date_format` if the day and the month can change places. glin reads it month first. A numeric date that cannot be in the training order returns `date_order_mismatch`, for example `02/13/2024` on a `dd/mm` column. glin reads it in the other order. Both warnings show the date that glin used in `parsed_as`. An ISO date never gets these warnings.

### Feature columns: not yet supported

| Data shape | What happens | Why |
|---|---|---|
| Dates stored as bare numbers (year, `YYYYMMDD`, epoch seconds) | glin treats them as ordinary numbers, not dates. | glin cannot tell a plain integer column from an ID or a count. Convert the column to date strings if it holds dates. |
| Free text | glin drops the column if it has more than 250 unique values. Below that limit, the text becomes a set of categories that almost certainly have no use. | glin does not do natural language processing. A text column is not a set of classes. |
| Lists, dicts or nested JSON in a cell | glin does not parse them. It treats the cell as an opaque string. | No structured extraction exists. |
| Row identifiers (sequential IDs, UUIDs, hashes) | glin drops them on purpose. This is not a gap. glin checks only data with more than 50 rows. A numeric column is an identifier if all of its values are distinct whole numbers and one more condition is true:<br>- the values fill at least 20% of the range from the minimum to the maximum (also with a constant step, for example `10, 20, 30`), or<br>- the column name has an ID word: `id`, `idx`, `key`, `uuid`, `guid` or `pk` (for example `row_id` or `PassengerId`), or the name is `index`.<br>The same rule applies to a text column with mostly numeric values, for example IDs with one `TBD` value. glin also drops other text columns that are more than 95% unique. | IDs carry no predictive signal. |
| Any dropped column (constant, identifier-like, more than 250 categories, or a date with no varying part) | `features_dropped` lists the column. `train_model` also returns a `warnings` entry with the column name and the reason. For numeric identifiers, the entry suggests that you round or bin the column if it is a real measurement. | A drop is never silent. |

### Target leakage check

After preprocessing, `train_model` looks for single feature columns that separate the target almost perfectly on their own. This is the usual sign that someone derived the target from that column with a threshold, a band or a category mapping. The model then only reads the target back.

glin flags such a column in `warnings` as `Possible target leakage: ...`. The warning shows the evidence: a numeric threshold or band such as `> 62.5 -> 'yes'`, or a category mapping such as `{p, q} -> 'ok'`. It suggests that you drop the column and retrain if the column is not known at prediction time. The check is only a warning. Training continues. The Python result also has `leakage_suspects`. The `build_classifier`, `prepare_csv_for_training` and tuning prompts tell the agent to review these warnings with the user.

- **Method:** glin fits a single-feature stump for each feature. It scores the stump by balanced accuracy on the training rows. The holdout rows are not used. For numeric features, the stump is a decision tree with at most `n_classes + 1` leaves (one threshold, or a band with two cut points). For categorical features, the stump maps each category to its majority class. glin also uses this category map for a numeric feature with 20 or fewer distinct values, because such a column is often a category stored as integer codes (for example survey answer codes). Missing values are one more category. A target such as `q12_code in {2, 5, 7}` is not monotone, so a threshold cannot show it, but the category map can. glin flags a feature at balanced accuracy of 0.99 or more. For a binary target, this also means a one-vs-rest AUC of 0.99 or more. A strong but imperfect predictor does not trigger the check.
- **As the model sees it:** glin checks percent, currency and suppression-coded columns after numeric conversion. Missingness counts. A column that is blank exactly when the target is positive is flagged as `missing -> ...`. glin checks date columns through their derived terms and reports them under the raw column, for example `'signup' (via derived term 'signup__days_since_start')`.
- **Guards against chance and memorization:**
  - Each numeric leaf needs at least 5 rows (more on bigger data).
  - Each category needs at least 5 rows. This floor does not increase with the data size. Thus glin can also flag a column with 100 to 250 categories (the preprocessor keeps at most 250).
  - A categorical column must have at least 90% of its rows in categories with 5 or more rows. A near-unique column therefore cannot "separate" the target by memorization. In a test, 200 noise columns with 250 categories and 20,000 rows gave no flags.
  - glin leaves out classes with fewer than 10 rows.
  - glin skips the check below 30 rows.
  - The cost is a few milliseconds for each column. glin samples at most 20,000 rows. 500 columns take about 2 seconds.
- glin reports at most 10 columns, the best first.

### Target column requirements

- The target needs at least 2 distinct non-null values (binary or multiclass). A single-class target is a hard error.
- glin drops rows with a missing target value and returns a warning. The other rows are not affected.
- A numeric target with more than 20 distinct values triggers a warning that it looks like a regression target. glin trains classifiers, not regressors, so each value becomes a separate class. Band the target into a few classes (quantiles or domain thresholds), or pick another target.
- A numeric target with 3 to 20 distinct values (for example a score or an ordinal level) triggers a warning. The multiclass model ignores the order of the values. The warning gives an example with the real values of the target. Use fewer bands or a binary threshold. The `prepare_csv_for_training` and `build_classifier` prompts give the same guidance.
- These two checks also apply to a text target whose values are numbers, for example `"85%"` or `"1"` to `"5"`. glin parses the values with the same rules as a numeric feature column. The model still uses each text value as a class.
- A text target with more than 20 distinct values triggers a warning. Each value becomes its own class, and each class needs many rows.
- Full regression (the upstream `ExplainableBoostingRegressor`) is planned separately. It is not yet supported.

## Releasing (maintainers)

**Only the `Publish to PyPI` GitHub Actions workflow ([`.github/workflows/publish.yml`](https://github.com/AkashChatterjee/glin/blob/main/.github/workflows/publish.yml)) publishes releases to PyPI. Nobody uploads from a local machine. Do not use `twine upload`, `uv publish` or personal API tokens.** To release, push a tag of the form `vX.Y.Z`:

```bash
git tag v0.1.4
git push origin v0.1.4
```

The workflow runs the `Tests` workflow first (the same matrix as for pull requests). If a test fails, the workflow stops and does not publish. Then it builds the sdist and the wheel, and checks that both carry the version of the tag. Then it publishes them to PyPI as `glin-ml` at that exact version, and creates the GitHub Release from `CHANGELOG.md`.

- `hatch-vcs` derives the package version from the git tag. `pyproject.toml` has no fixed version, so you do not bump the package version before a release. The plugin and registry files have their own versions (see [Release steps](#release-steps)).
- An untagged local build gets a dev version such as `0.1.4.dev1+g3fa91a0`. A source tree with no git metadata reports `0.0.0`.
- `glin.__version__` reads the installed package metadata.
- The workflow ignores tags that do not match `vX.Y.Z`.
- PyPI rejects an upload of a version number that already exists. To fix a bad release, tag a new version.

### Release steps

The owner does these steps for each release:

1. In `CHANGELOG.md`, move the entries under `[Unreleased]` to a new heading `## [X.Y.Z] - YYYY-MM-DD`. Add the compare link at the bottom. Compare the entries with `gh issue list --milestone vX.Y.Z --state closed`.
2. Set the release version in `server.json` (`version` and `packages[0].version`) and in `plugins/glin/.claude-plugin/plugin.json` (`version`). If the plugin needs the new release, also change the minimum version in `plugins/glin/.mcp.json`. Merge these changes into `main`.
3. Tag the release and push the tag, for example `git tag v0.1.4` and `git push origin v0.1.4`.
4. In GitHub Actions, open the `Publish to PyPI` run and approve the `pypi` environment. The upload step starts only after this approval.
5. The workflow publishes to PyPI. Then it creates the GitHub Release with the `CHANGELOG.md` section of the version as the notes. If `CHANGELOG.md` has no section for the version, the workflow stops before it publishes.
6. Check that the new version is on PyPI and that the GitHub Release exists.
7. Publish to the MCP Registry with `mcp-publisher login github` and `mcp-publisher publish`. See [Publishing to the MCP Registry](#publishing-to-the-mcp-registry).

To compare the changelog with the merged pull requests, open a draft release on GitHub and click "Generate release notes". [`.github/release.yml`](https://github.com/AkashChatterjee/glin/blob/main/.github/release.yml) groups the pull requests by label.

### One-time setup that makes this the only path

The workflow publishes through PyPI [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC). This also makes local uploads impossible.

1. **PyPI → the `glin-ml` project → Publishing:** add a trusted publisher for owner `AkashChatterjee`, repository `glin`, workflow `publish.yml` and environment `pypi`.
2. **PyPI → Account settings → API tokens:** delete every API token that can upload `glin-ml`. Do not create new tokens. With no tokens and 2FA on, the only credential that can publish is the short-lived OIDC identity of the workflow. Also review the collaborators of the project. Remove anyone who does not need upload rights.
3. **GitHub → Settings → Rules → Rulesets:** add a tag ruleset for `v*`. Restrict who can create tags, and block deleting or moving them. A new matching tag starts a release.
4. **GitHub → Settings → Environments → `pypi`:** add required reviewers. Every release then needs a manual approval before the upload step runs.

### Publishing to the MCP Registry

The registry holds metadata only. It checks that the PyPI package has the line `mcp-name: io.github.AkashChatterjee/glin` in its README. The README has this line as a hidden comment. Publish to PyPI first, then to the registry:

1. Release the new version on PyPI (see above).
2. Check that `version` and `packages[0].version` in `server.json` are the released version.
3. Run these commands from the repository root:

```bash
mcp-publisher login github
mcp-publisher publish
```

4. Check the result: `curl "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.AkashChatterjee/glin"`.

The registry rejects a version that already exists. Update `server.json` for each release.

### Checking a release

To check a build without publishing it, run `uv build` locally and inspect `dist/`. Never upload it. After a release, confirm that the public package works with a clean install:

```bash
uvx --refresh --from glin-ml glin --version
uvx --from glin-ml glin serve --help
```

The first command must show the released version. Then start the server from an MCP client, ask the agent to train a small model, and delete it with `delete_model`.
