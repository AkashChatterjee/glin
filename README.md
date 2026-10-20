# glin

☕ If glin is useful to you and you want to help maintain it, you can [buy me a coffee](https://buymeacoffee.com/akashchatterjee).

<!-- mcp-name: io.github.AkashChatterjee/glin -->

glin lets your AI agent train a classifier on your data and then use it. You install glin into an agent such as Claude Code, Claude Desktop or Cursor. Then you talk to the agent in plain language. The agent trains the model, makes predictions and explains them. You do not write code.

glin trains an [Explainable Boosting Machine](https://interpret.ml/) (EBM) on a CSV or Excel file. Each prediction comes with an exact breakdown of the features that caused it. The breakdown comes from the additive structure of the model. It is not a SHAP or LIME approximation.

glin is an [MCP](https://modelcontextprotocol.io/) server. It runs on your machine and talks to the agent over stdio. Your data and your models stay on your machine.

**Changelog:** [CHANGELOG.md](https://github.com/AkashChatterjee/glin/blob/main/CHANGELOG.md) lists the changes in each release.

**Case study:** [The segment of 100,000 sales leads](https://akashchatterjee.com/tech/2026/10/11/glin-ml-case-study-sales-leads-synthetic). Claude with glin put 71% of new leads in the correct segment. Claude alone got 33%. With 1,000 labeled leads in the prompt, Claude got 60% and used 6.7 times more tokens.

## How glin works

1. You give the agent a data file and the column to predict.
2. The agent calls glin. glin checks the data, trains a model and saves it in `~/.glin/models/<name>/`.
3. glin tests the model on rows that it did not train on. The agent tells you how good the model is.
4. You ask about new cases. The agent sends each case to the model and gets a prediction with the exact reasons.
5. You ask what drives the outcome. The agent reads it from the model.

A saved model stays available. You can use it again in a new chat.

## What you can ask the agent

- "Train a classifier on `~/data/leads.csv` that predicts `converted`. Call it `leads_v1`."
- "Which of these 3 deals should I prioritize?" The agent sends each deal to the model and gives the reasons.
- "What drives churn in this model?"
- "Tune the model, and tell me how good it really is."
- "Which models do I have?"
- "Delete the model `leads_v1`."

## Contents

- [How glin works](#how-glin-works)
- [What you can ask the agent](#what-you-can-ask-the-agent)
- [Install](#install)
- [Use glin from your agent](#use-glin-from-your-agent)
  - [Train a model](#train-a-model)
  - [Get predictions](#get-predictions)
  - [Explain a model](#explain-a-model)
  - [Tune a model](#tune-a-model)
  - [Manage models](#manage-models)
- [Your data](#your-data)
  - [Files that glin can read](#files-that-glin-can-read)
  - [Excel files](#excel-files)
  - [Data checks](#data-checks)
  - [How glin reads columns](#how-glin-reads-columns)
  - [Dates and timestamps](#dates-and-timestamps)
  - [Columns that glin does not use](#columns-that-glin-does-not-use)
  - [Target leakage check](#target-leakage-check)
- [Reference](#reference)
  - [MCP tools](#mcp-tools)
  - [MCP prompts](#mcp-prompts)
  - [Server flags](#server-flags)
- [Case studies](#case-studies)
- [Development](#development)
- [Releasing (maintainers)](#releasing-maintainers)

## Install

Your MCP client starts glin with `uvx`. `uvx` downloads the PyPI package `glin-ml` and keeps it in a cache. You do not install glin with `pip`. (The PyPI name `glin` belongs to an unrelated project.)

Do step 1 for all clients. Then do step 2 for your client.

### Step 1: Install uv and download glin

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/). Use one of these commands:

   ```bash
   # macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # macOS with Homebrew
   brew install uv

   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Open a new terminal.
3. Run this command:

   ```bash
   uvx --from 'glin-ml[excel]>=0.1.5' glin --version
   ```

   The command shows `glin-ml 0.1.5` or a later version. The first run downloads glin and its libraries into the cache. This can take approximately one minute.

**Do not skip step 1.3.** Some clients stop a server that does not start in 30 seconds. Claude Code is one of them. When the cache is full, glin starts in a few seconds.

### Step 2: Add glin to your client

Do the steps for one of these clients:

- [Claude Code, with the plugin (recommended)](#claude-code-with-the-plugin-recommended)
- [Claude Code, without the plugin](#claude-code-without-the-plugin)
- [Claude Desktop](#claude-desktop)
- [Cursor and other MCP clients](#cursor-and-other-mcp-clients)
- [A client with a registry search](#a-client-with-a-registry-search)

#### Claude Code, with the plugin (recommended)

The plugin adds the glin MCP server and a skill. The skill tells the agent when and how to build, tune, predict with and explain a model. The plugin includes Excel support.

1. Start Claude Code.
2. Add the glin marketplace:

   ```text
   /plugin marketplace add AkashChatterjee/glin
   ```

3. Install the plugin:

   ```text
   /plugin install glin@glin-marketplace
   ```

4. If Claude Code tells you to restart it, restart it.
5. Run `/mcp`. Make sure that the list shows the glin server as connected.

The plugin starts `uvx --from 'glin-ml[excel]>=0.1.5' glin serve`. The plugin files are in [`.claude-plugin/marketplace.json`](https://github.com/AkashChatterjee/glin/blob/main/.claude-plugin/marketplace.json) and [`plugins/glin/`](https://github.com/AkashChatterjee/glin/tree/main/plugins/glin).

#### Claude Code, without the plugin

Use this option to add only the MCP server, without the skill.

1. In a terminal, run this command:

   ```bash
   claude mcp add --scope user glin -- uvx --from 'glin-ml[excel]>=0.1.5' glin serve
   ```

   `--scope user` makes glin available in all your projects. To add glin to the current project only, remove `--scope user`.

2. Start Claude Code.
3. Run `/mcp`. Make sure that the list shows the glin server as connected.

#### Claude Desktop

1. Open Claude Desktop.
2. Open **Settings > Developer** and click **Edit Config**. This opens the file `claude_desktop_config.json`:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
3. Add the `glin` entry to `mcpServers`. If the file has other servers, keep them:

   ```json
   {
     "mcpServers": {
       "glin": {
         "command": "uvx",
         "args": ["--from", "glin-ml[excel]>=0.1.5", "glin", "serve"]
       }
     }
   }
   ```

4. Save the file.
5. Quit Claude Desktop fully, then start it again. On macOS, closing the window is not sufficient.
6. Make sure that the glin tools show in the tools menu of a new chat.

If Claude Desktop cannot find `uvx` (for example `spawn uvx ENOENT`), use the full path. Run `which uvx` (macOS and Linux) or `where uvx` (Windows), and put the result in `"command"`.

#### Cursor and other MCP clients

1. Open the MCP config file of your client:
   - Cursor: `~/.cursor/mcp.json` for all projects, or `.cursor/mcp.json` in one project.
   - Other clients: see the documentation of the client.
2. Add the same `glin` entry as for [Claude Desktop](#claude-desktop):

   ```json
   {
     "mcpServers": {
       "glin": {
         "command": "uvx",
         "args": ["--from", "glin-ml[excel]>=0.1.5", "glin", "serve"]
       }
     }
   }
   ```

3. Save the file and restart the client.
4. Make sure that the client shows glin as an enabled MCP server.

If the client cannot find `uvx`, use the full path to `uvx` in `"command"`, as for Claude Desktop.

#### A client with a registry search

glin is in the [official MCP Registry](https://registry.modelcontextprotocol.io) as `io.github.AkashChatterjee/glin`. The registry is a catalog of metadata. It is not an installer. MCP marketplaces and some clients read it and show glin in their server lists. For example, VS Code installs servers from the [GitHub MCP Registry](https://github.com/mcp).

1. In your client, search the server list for `glin`.
2. Install the entry `io.github.AkashChatterjee/glin`.
3. Make sure that the client shows glin as an enabled MCP server.

The registry entry starts `uvx glin-ml serve`, without the `excel` extra. The registry format has no field for extras. To read Excel files, use the steps in [Cursor and other MCP clients](#cursor-and-other-mcp-clients) instead. If your client has no registry search, use those steps too.

### Install options

- **CSV only:** If you do not use Excel files, you can remove `[excel]` from the arguments.
- **Server flags:** Put flags such as `--allow-dir` after `serve` in the arguments. See [Server flags](#server-flags).
- **Update glin:** Run `uvx --refresh --from 'glin-ml[excel]>=0.1.5' glin --version`. Then restart your client.
- **An installed copy:** To use a copy from `uv tool install 'glin-ml[excel]'` or `pipx install 'glin-ml[excel]'`, set `"command"` to `glin` and `"args"` to `["serve"]`. You must update this copy yourself, for example with `uv tool upgrade glin-ml`.

## Use glin from your agent

When glin is installed, talk to the agent in plain language. The agent selects the glin tools. You do not need to know the tool names. This section shows what you can ask and what you get. For the exact tool arguments, see [MCP tools](#mcp-tools).

### Train a model

Give the agent a data file and the column to predict:

> "Train a classifier on `~/data/churn.csv` that predicts `churn`. Call it `churn_v1`."

The agent calls `train_model`. glin checks the data, prepares the columns and trains the model. Then glin returns:

- the features that it kept and the features that it dropped, each with a reason;
- warnings, for example a column that possibly leaks the target;
- test results on a holdout: 20% of the rows that the model did not train on.

The test results are `log_loss`, `baseline_log_loss` (a guess from the class shares only), `train_log_loss`, `accuracy`, `balanced_accuracy` and `roc_auc`. A useful model has a `log_loss` that is clearly lower than `baseline_log_loss`.

For a careful build, ask the agent to use the `build_classifier` prompt. The prompt tells the agent to clean the data, remove leakage, tune the model and report the honest performance. See [MCP prompts](#mcp-prompts).

If a model with the same name exists, training fails. To replace a model, for example with new data, ask the agent to retrain it with `overwrite=True`.

### Get predictions

Ask about one case or many cases. Give the values that you know:

> "I've got 3 deals to prioritize before quarter close: Northwind Systems ($42k, Opportunity stage, came from a referral, VP contact, owned by Carla Nguyen, 8 touches logged), Summit Retail Group ($8k, still a Lead, paid social source, owned by Brian Kessler), and Anchor Nonprofit ($3k, still a Lead, owned by Frank Suarez, no activity logged yet). Run them through deal_predictor_v1 and tell me which to prioritize."

The agent calls `predict` once for each deal. It gives a ranked answer with the reasons:

| Deal | Win probability | Why |
|---|---|---|
| Northwind Systems | 69% | The Opportunity stage, the referral and the VP contact all favor it. The $42k deal size is the one drag. |
| Summit Retail Group | 35% | It is still a Lead, and it came from paid social. Both are large drags with nothing against them. |
| Anchor Nonprofit | 34% | No activity is logged and it is still a Lead. Nothing in the record favors it. |

Each prediction has an audit. The audit gives the score of each feature, and the scores add up exactly to the probability. glin checks this sum in each prediction (`additivity_verified`).

Some important points:

- A value that you do not give is missing for the model. It is not an average value.
- Use the categories that are in the training data, for example `Lead`, `Opportunity` and `Customer`. Other spellings are new categories for the model.
- glin warns about values that the model did not see in training, for example a new category or a number out of the training range. Such a prediction is less reliable.

### Explain a model

Ask what drives the outcome:

> "What drives churn in `churn_v1`?"

The agent calls `explain_model`. glin reads the answer directly from the model. It does not sample predictions. The result gives:

- each feature, ranked by its effect, with the part that comes from interactions;
- the top terms, including pairs of features, with their share of the total;
- for a binary target, the values or categories of each feature that push toward each class.

### Tune a model

Ask the agent to tune the model:

> "Tune `churn_v1`, and tell me how good it really is."

The `tune_model_hyperparameters` prompt gives the agent a safe procedure:

1. Train a baseline.
2. Change one or two settings in each trial. All trials use the same holdout rows.
3. Check the noise with other random seeds.
4. Stop after approximately 10 trials, or when the gain is small. Prefer the simpler setting.
5. Train the final model on all rows.

The agent can get all settings, with defaults and ranges, from `list_hyperparameters`. glin rejects unknown settings and values out of range.

#### Grouped holdout

Rows from the same state, region, customer, site or user look alike. A random holdout then gives metrics that are too optimistic. In one real case, a model got 82% accuracy and 0.89 AUC on a random holdout. It got 63% and 0.77 on states that it did not see.

If your data has such a column, ask the agent to hold out by that column:

> "Train on `~/data/districts.csv`, predict `high_stunting`, and hold out by `state`."

The agent passes `holdout_group_column="state"`. glin then puts whole groups in the holdout. The test results then show the performance on groups that the model did not see. The prompts tell the agent to use this option when such a column exists.

- glin puts approximately 20% of the rows (`holdout_fraction`) in the holdout. It tries 25 group orders and keeps the one whose class shares are closest to the full data.
- Text values are grouped without case and outer spaces. Thus `"S06"` and `"s06 "` are one group. Blank text, `nan` and `none` are missing values. Missing values are one group.
- The column stays a feature. A group in the holdout is an unseen category at prediction time.
- glin stops with an error if the column does not exist or is the target, if it has fewer than 5 groups, if one group has more than half of the rows, if all groups are too large for the holdout, or if the split loses a class.

### Manage models

- **List models:** ask "Which models do I have?" The agent calls `list_models`.
- **Look at a model:** ask "What does `churn_v1` need as input?" The agent calls `inspect_model`. The result gives the features by type (numeric, categorical and date), the dropped columns, the target classes, the settings and the test results.
- **Delete a model:** ask "Delete the model `churn_v1`." The agent calls `delete_model`. The deletion is permanent. You can also remove the folder in `~/.glin/models/` yourself.

**Model names:** A new model name has 1 to 64 characters. Use only letters, digits, `-`, `_` and `.`. Do not start the name with `.`. To use or delete a model, the name must be one folder name in `~/.glin/models/`. glin makes sure that a model name cannot read, write or delete files outside this folder.

## Your data

### Files that glin can read

The agent gives glin a file path (`csv_path`). glin reads the file itself, so the agent does not copy the data into the chat. `csv_path` takes `.csv`, `.xlsx` and `.xls` files.

glin reads a file only if it is in an allowed folder. By default, the allowed folders are:

- your home folder;
- the folder where the client starts the server, but only if it is not `/` and not a parent of your home folder.

Claude Desktop on macOS starts the server from `/`. Then only your home folder is allowed. Claude Code starts the server from the project folder, so that folder is also allowed. For data in other places, for example `/Volumes/data`, use `--allow-dir`. See [Server flags](#server-flags).

More rules:

- The file must be 200 MB or smaller. To change the limit, use `--max-file-mb`.
- glin follows symbolic links and `..` in the path. The real path must be in an allowed folder.
- glin does not read folders, devices or files with other extensions. Error messages do not show the file contents.
- For a small table in the chat, the agent can send the CSV text (`csv_content`). Use this only for small data, because the data goes through the chat.

The folder rules have two known limits. A hard link in an allowed folder to a file outside it is read. If the path changes between the check and the read, glin reads the new file. Thus, do not allow a folder that other users or processes can write to.

### Excel files

glin trains directly on Excel sheets, including sheets with title rows, merged headers and footnotes. Excel support needs the `excel` extra. The install steps above include it.

Tell the agent the sheet and the header rows if you know them:

> "Train a classifier on `~/data/report.xlsx`, sheet `Data 2023`, header rows 3-4, that predicts `outcome`."

The agent passes `sheet` and `header_rows` to `train_model`. These rules apply:

- **Sheet:** a name or a position (1 is the first sheet). A workbook with one sheet needs no `sheet`. For a workbook with several sheets, glin gives an error that lists the sheets. glin ignores case and outer spaces in the name if only one sheet matches.
- **Header rows:** row numbers as in Excel, for example `4`, `"3-4"`, `"3,4"` or `[3, 4]`. If you do not give them, glin finds the header row. It skips title rows above it. If the header is not clear, glin stops and asks for `header_rows`. glin does not guess.
- **Multi-row headers:** glin joins the rows into one name for each column, for example `2023 - sales`. A blank cell under a merged label gets the label from the left. Duplicate names get `_2`, `_3` and so on. A date in a header becomes an ISO date, for example `2024-01-01`.
- **Footnotes:** glin removes footnote rows at the end of the sheet and gives a warning with their row numbers. A footnote row has no numbers and starts with `Source`, `Note`, `Footnote`, `*`, `†`, `‡` or `©`, or it is a sentence of 6 words or more.
- **Formulas:** glin uses the saved results of the formulas. Calculate and save the file in a spreadsheet app first.
- **Empty rows and columns:** glin removes them.

The result shows the sheet, the header rows and the number of rows that glin read. `sheet` and `header_rows` are errors for a CSV file.

### Data checks

Before training, glin checks the data. A hard problem stops training with a clear error. A soft problem gives a warning, and training continues.

- The data must have at least 20 rows.
- The target must have at least 2 classes. glin removes rows with no target value and gives a warning.
- glin trains classifiers, not regression models. A numeric target with more than 20 values gives a warning: each value becomes a separate class. Put the target into a few bands, or use a different target.
- A numeric target with 3 to 20 values, for example a score from 1 to 5, gives a warning. The model ignores the order of the values. Use fewer bands or a yes/no threshold.
- The two checks above also apply to a text target that holds numbers, for example `"1"` to `"5"`.
- A text target with more than 20 values gives a warning. Each class needs many rows.

### How glin reads columns

glin prepares each column before training. It uses the same rules at prediction time.

| Data | What glin does |
|---|---|
| Numbers, with missing values | glin uses them as numbers. Missing values get their own bin. |
| Numbers stored as text, with blanks | glin converts the column to numbers if at least 80% of the values are numbers. Other values become missing. |
| Currency and thousands separators (`$1,234.56`, `€1.234,56`, `USD 100`, `Rs. 100`) | glin removes the currency symbols and codes. It reads a decimal comma only if the column has clear European values, for example `1.234,56` or `12,5`. A plain `1,234` is one thousand two hundred and thirty-four. If a value has both separators, the last one is the decimal mark. If a column mixes both styles, glin reads US style and gives a warning. |
| Negative numbers (`-5`, `−5`, `(1,234)`, `1234-`) | glin reads the Unicode minus, brackets and a trailing minus as negative. |
| Percents (`44.98%`) | glin removes the `%`. The value stays in percentage points: `44.98%` becomes `44.98`. |
| Infinity (`inf`, `-inf`) | glin makes it missing and gives a warning. |
| Codes for missing values (`SUPP`, `NE`, `NP`, `z`, `*`, `..`) | In a numeric column, glin makes these codes missing. Case and outer spaces do not matter. To use other codes, give `suppression_codes`. If a numeric column has other short codes, for example `x`, glin makes them missing and gives a warning. If unknown codes are more than 20% of a column, the column becomes text, and the warning suggests the codes to add. glin does not use `<5` and `—` as codes, because they can be real values. |
| Text categories, up to 250 values | glin uses each value as a category, without case and outer spaces. A new value at prediction time is a new category and gives a warning. |
| True/false | glin uses them as 0 and 1. |
| Dates and timestamps | glin splits them into date parts. See [Dates and timestamps](#dates-and-timestamps). |

### Dates and timestamps

glin finds date columns automatically. At least 80% of the values must be full dates, for example `2024-03-06`, `2024-03-06 14:30:00`, `03/06/2024`, `6 Mar 2024` or `Mar 6, 2024`.

- **Date parts:** glin makes these features from each date column: `<col>__year`, `<col>__month` (`jan` to `dec`), `<col>__day_of_week` (`mon` to `sun`), `<col>__hour` and `<col>__days_since_start`. `days_since_start` counts days from the earliest date in training.
- **Unused parts:** glin removes a part that does not change, for example `__hour` for dates with no time.
- **Day and month order:** glin finds `dd/mm` or `mm/dd` from the values. If the values do not show the order, glin uses month first and gives a warning. Use ISO dates (`YYYY-MM-DD`) to prevent this.
- **Time zones:** glin converts times with a time zone to UTC.
- **Values that are not dates:** they become missing, with a warning.
- **Predictions:** give the raw date, for example `"2024-03-06"`. glin warns if the date is out of the training range, if it is not a date, or if its day and month order is not clear. The warning shows the date that glin used (`parsed_as`).

### Columns that glin does not use

glin removes these columns before training. The result lists each removed column with the reason.

| Data | What glin does |
|---|---|
| Columns with one value | glin removes them. |
| Identifiers (row numbers, sequential IDs, UUIDs, hashes) | In data with more than 50 rows, glin removes a column of distinct whole numbers if the numbers fill a range (for example `1` to `N`) or if the name has an ID word (`id`, `idx`, `key`, `uuid`, `guid`, `pk`, or the name `index`). It also removes text columns where more than 95% of the values are different. Distinct numbers with large, uneven gaps, for example salaries, stay as numbers. |
| Text with more than 250 values, for example free text | glin removes it. glin does not process natural language. |
| Lists or JSON in a cell | glin does not read them. It uses the cell as one text value. |
| Dates stored as numbers (`2024`, `20240306`, epoch seconds) | glin uses them as numbers, not dates. Change them to date text if they are dates. |

### Target leakage check

A column leaks the target if it gives the answer directly. This often happens when someone made the target from that column, for example with a threshold. A model with such a column looks very good in tests but is not useful.

After preparation, glin checks each feature alone. If one feature predicts the target with 99% balanced accuracy or more, glin gives a `Possible target leakage` warning. The warning shows the evidence, for example `> 62.5 -> 'yes'` or `{p, q} -> 'ok'`. Training continues. The agent asks you about each warning. If the value is not known at the time of the prediction, remove the column and train again.

- glin finds thresholds, bands and category maps. It also finds numeric codes (20 values or fewer) that map to classes, and missing values that map to a class.
- glin checks date columns through their date parts.
- To prevent false alarms, each rule needs at least 5 rows. glin ignores classes with fewer than 10 rows and skips data with fewer than 30 rows.
- glin uses at most 20,000 rows for the check and reports at most 10 columns.

## Reference

### MCP tools

All tools have MCP annotations. `train_model` and `delete_model` are marked destructive, because they can replace or remove a model. The other tools are read-only. Thus a client can run the read-only tools automatically and ask you before the others.

- `train_model(target_column, model_name, csv_path=None, csv_content=None, overwrite=False, hyperparameters=None, holdout_fraction=0.2, holdout_group_column=None, suppression_codes=None, sheet=None, header_rows=None)` trains a model and saves it.
  - Give `csv_path` or `csv_content`, not both.
  - `holdout_fraction=0` trains on all rows and gives no test results.
  - `hyperparameters` takes EBM settings, for example `{"interactions": 0}`. See `list_hyperparameters`.
  - The result has `rows_read`, `target_classes`, `features_kept`, `features_dropped`, `date_columns`, `warnings`, `hyperparameters`, `training_rows`, `training_seconds`, `terms` and `evaluation`. For an Excel file, it also has `sheet` and `header_rows`.
  - `evaluation` is `null` if `holdout_fraction=0`. A grouped holdout adds `holdout_groups` and `train_groups`.
- `predict(model_name, features, top_n=10)` gives the predicted class, the probabilities and the audit:
  - `base_logit` is the intercept. `contributions` gives the score of each term, the largest first.
  - For a binary target, the scores are log-odds of `score_class`, the second class in `target_classes`. A positive score pushes toward `score_class`. For a multiclass target, each term has `score_by_class`.
  - `additivity_verified` is true if the scores add up to the model probability.
  - The class values keep the type of the target, for example `1`. The keys of the probability dictionaries are always text, for example `"1"`.
  - `warnings` lists inputs that the model did not see: `unseen_category`, `out_of_range`, `unparseable_number`, `unparseable_date`, `ambiguous_date_format` and `date_order_mismatch`.
  - `missing_features` lists the inputs that are absent or blank.
- `explain_model(model_name, top_n=15, include_shapes=True)` gives the feature ranking, the top terms and, for a binary target, the shape of each feature.
- `inspect_model(model_name)` gives the features by type (`numeric`, `categorical` and `date`), the dropped columns, the target classes, the settings and the test results. For each date column, it gives the date parts and the start date of `days_since_start`.
- `list_models()` lists all saved models.
- `list_hyperparameters()` lists each EBM setting with its type, default, range and effect.
- `delete_model(model_name)` deletes a model permanently. The result has `model_name`, `deleted` and `path`.

### MCP prompts

Prompts are ready-made instructions for the agent. In most clients, you select them in a prompt menu. Claude Desktop shows them in the attach (`+`) menu.

- `build_classifier(csv_path, target_column, model_name)` does the full procedure: check and clean the data, remove leakage, tune with a fixed holdout and a budget of approximately 10 trials, train on all rows, and report the expected performance.
- `tune_model_hyperparameters(model_name, target_column)` gives the tuning procedure in [Tune a model](#tune-a-model), with the current settings table.
- `prepare_csv_for_training(target_column)` is a checklist for the data before training: file problems, the choice of target, and what glin does and does not do to the columns.

All arguments are optional. glin makes the prompts from its current defaults, so they agree with the code.

### Server flags

Put these flags after `serve` in the arguments of your client config, for example `["--from", "glin-ml[excel]>=0.1.5", "glin", "serve", "--allow-dir", "/Volumes/data"]`.

| Flag | Effect |
|---|---|
| `--allow-dir DIR` | Allow `csv_path` to read files in `DIR`. You can use the flag more than once. It replaces the default folders. If `DIR` is `/`, glin gives a warning on stderr. |
| `--max-file-mb MB` | Set the largest file that `csv_path` reads. The default is 200. |

`glin --version` shows the version. `glin serve` writes only MCP messages to stdout and writes log messages to stderr.

## Case studies

[The segment of 100,000 sales leads](https://akashchatterjee.com/tech/2026/10/11/glin-ml-case-study-sales-leads-synthetic) uses glin on synthetic CRM data with known rules. Because the rules are known, the study can check each prediction and explanation against the truth.

- Claude with glin put 71% of new leads in the correct segment. The best possible score on these leads is 68%.
- Claude alone got 33%. With 1,000 labeled leads in the prompt, Claude got 60% and used 83,000 tokens for each lead. With glin, Claude used approximately 12,500 tokens.
- glin gave the same answer for the same lead in each new session.
- The effects that glin learned agree with the true effects (correlation 0.97 to 1.00).

[Child stunting across Indian districts (NFHS-5)](https://akashchatterjee.com/tech/2026/10/11/glin-ml-case-study-child-stunting-nfhs5) uses glin on real public data. A glin model predicts whether a district has high child stunting, and gives the exact reason for each prediction.

## Development

Install an editable copy from a clone of this repository:

```bash
git clone https://github.com/AkashChatterjee/glin.git
cd glin
pip install -e ".[dev]"      # also installs the Excel libraries, and xlwt to build .xls test files
pytest
```

To try the checkout in an MCP client, set `"command"` to the `glin` program in your environment (for example `/path/to/glin/.venv/bin/glin`) and `"args"` to `["serve"]`.

- glin works with the `mcp` Python SDK 1.x and 2.x (`mcp>=1.21.1,<3`). To test one line, add a constraint, for example `pip install -e ".[dev]" "mcp>=2,<3"`.
- The `Tests` workflow ([.github/workflows/test.yml](.github/workflows/test.yml)) runs on each pull request and each push to `main`. It uses Python 3.10 and 3.12 with the latest mcp 1.x and 2.x, and Python 3.10 with `mcp==1.21.1`.
- The data rules are in `glin/validation.py` and `glin/preprocessor.py`. The EBM settings are in `glin/hyperparameters.py`. Add or change a setting there.
- In Python, `EBMTabularPreprocessor` takes `suppression_codes` and `date_parse_threshold`. A DataFrame with column names that are not text, for example `0`, gets text names, for example `"0"`.

## Releasing (maintainers)

**Only the `Publish to PyPI` GitHub Actions workflow ([`.github/workflows/publish.yml`](https://github.com/AkashChatterjee/glin/blob/main/.github/workflows/publish.yml)) publishes releases to PyPI. Nobody uploads from a local machine. Do not use `twine upload`, `uv publish` or personal API tokens.** To release, push a tag of the form `vX.Y.Z`:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
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
2. Set the release version in `server.json` (`version` and `packages[0].version`) and in `plugins/glin/.claude-plugin/plugin.json` (`version`). If the plugin needs the new release, also change the minimum version in `plugins/glin/.mcp.json` and in the commands and configs in [Install](#install). Merge these changes into `main`.
3. Tag the release and push the tag: `git tag vX.Y.Z` and `git push origin vX.Y.Z`.
4. In GitHub Actions, open the `Publish to PyPI` run and approve the `pypi` environment. The upload step starts only after this approval.
5. The workflow stops at the start if `CHANGELOG.md` has no section for the version, or if `server.json` or `plugin.json` has a different version than the tag. If all is correct, it publishes to PyPI. Then it creates the GitHub Release with the `CHANGELOG.md` section of the version as the notes. Also, the `mcp-registry` job publishes to the MCP Registry (see [Publishing to the MCP Registry](#publishing-to-the-mcp-registry)).
6. Check that the new version is on PyPI and that the GitHub Release exists.
7. Check the listing in the MCP Registry. The new version must show `isLatest: true`:

```bash
curl "https://registry.modelcontextprotocol.io/v0.1/servers?search=io.github.AkashChatterjee/glin"
```

If the `mcp-registry` job fails, use the manual fallback in [Publishing to the MCP Registry](#publishing-to-the-mcp-registry).

To compare the changelog with the merged pull requests, open a draft release on GitHub and click "Generate release notes". [`.github/release.yml`](https://github.com/AkashChatterjee/glin/blob/main/.github/release.yml) groups the pull requests by label.

### One-time setup that makes this the only path

The workflow publishes through PyPI [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC). This also makes local uploads impossible.

1. **PyPI → the `glin-ml` project → Publishing:** add a trusted publisher for owner `AkashChatterjee`, repository `glin`, workflow `publish.yml` and environment `pypi`.
2. **PyPI → Account settings → API tokens:** delete every API token that can upload `glin-ml`. Do not create new tokens. With no tokens and 2FA on, the only credential that can publish is the short-lived OIDC identity of the workflow. Also review the collaborators of the project. Remove anyone who does not need upload rights.
3. **GitHub → Settings → Rules → Rulesets:** add a tag ruleset for `v*`. Restrict who can create tags, and block deleting or moving them. A new matching tag starts a release.
4. **GitHub → Settings → Environments → `pypi`:** add required reviewers. Every release then needs a manual approval before the upload step runs.

### Publishing to the MCP Registry

The registry holds metadata only. It checks that the PyPI package has the line `mcp-name: io.github.AkashChatterjee/glin` in its README. The README has this line as a hidden comment.

The `mcp-registry` job in `publish.yml` publishes to the registry. It runs after the `publish` job, so the version is on PyPI first. The job does these steps:

1. It checks that `version` and `packages[0].version` in `server.json`, and `version` in `plugins/glin/.claude-plugin/plugin.json`, equal the tag without the `v`. The `build` job does the same check, so a mismatch stops the run before the upload to PyPI. The check is in `.github/scripts/check-versions.sh`.
2. It downloads a pinned release of `mcp-publisher` (Linux, amd64) and checks its SHA-256. The version is in `MCP_PUBLISHER_VERSION` in `publish.yml`. When you change it, also change `MCP_PUBLISHER_SHA256` to the value in the `registry_<version>_checksums.txt` file of that release.
3. It logs in with `mcp-publisher login github-oidc`. The job has the `id-token: write` permission. No secret or token is stored in GitHub.
4. It runs `mcp-publisher publish`. The PyPI CDN can lag for a minute or more after an upload, and the registry then cannot see the new version. If the command fails, the job waits 60 seconds and tries again, up to 5 tries in total.
5. It queries the registry and fails if the version of the tag is not listed with `isLatest: true`.

The registry rejects a version that already exists. If the job fails after the registry accepted the version, do not run it again. Check the listing instead.

#### Manual fallback

If the job cannot publish, set the versions in `server.json` and run these commands from the repository root, after the version is on PyPI:

```bash
mcp-publisher login github
mcp-publisher publish
```

Then check the listing with the `curl` command in the release steps.

### Checking a release

To check a build without publishing it, run `uv build` locally and inspect `dist/`. Never upload it. After a release, confirm that the public package works with a clean install:

```bash
uvx --refresh --from 'glin-ml==X.Y.Z' glin --version
uvx --from 'glin-ml==X.Y.Z' glin serve --help
```

Use the released version in place of `X.Y.Z`. The first command must show this version. A pinned version also prevents `uvx` from using an older copy from `uv tool install`. Then start the server from an MCP client, ask the agent to train a small model, and delete it with `delete_model`.
