# glin

CLI tool and Python library that equips AI agents with an instant, statistical "gut feeling" (System 1 thinking).

`glin` trains an [Explainable Boosting Machine](https://interpret.ml/) on a CSV, then exposes it to LLM agents over [MCP](https://modelcontextprotocol.io/) — locally over stdio, or remotely over MCP's standard `streamable-http` transport. Every prediction comes with an exact, zero-approximation breakdown of which features drove it, straight from the model's own additive structure (no SHAP/LIME approximation).

## Install

The PyPI package is named `glin-ml` (the name `glin` was already taken by an unrelated project) — but the CLI command and Python import are both just `glin`.

```bash
uv tool install glin-ml     # recommended: installs the `glin` command in an isolated env
```

```bash
pipx install glin-ml        # equivalent, if you use pipx instead of uv
```

```bash
pip install glin-ml         # plain pip also works, into whatever environment is active
```

Either way, you get the `glin` command:

```bash
glin --help
```

**For development** (editable install from a clone of this repo):

```bash
git clone https://github.com/AkashChatterjee/glin.git
cd glin
pip install -e ".[dev]"
```

## Train a model

```bash
glin train path/to/data.csv --target churn --name churn_v1
```

Models are saved under `~/.glin/models/<name>/`.

```bash
glin list
```

```bash
glin delete churn_v1        # prompts for confirmation
glin delete churn_v1 --yes  # skips the prompt
```

## Use it locally (Claude Desktop, Cursor, ...)

Add to your MCP client's config (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "glin": {
      "command": "glin",
      "args": ["serve", "--mode", "stdio"]
    }
  }
}
```

## Deploy it remotely (e.g. one EC2 box, any MCP-aware agent)

```bash
docker build -t glin .
docker run -p 8000:8000 -v ~/.glin:/root/.glin glin
```

Then point any MCP client at the standard streamable-http endpoint:

```bash
claude mcp add --transport http glin http://<host>:8000/mcp
```

## Tools exposed over MCP

Every tool carries MCP annotations (title, `readOnlyHint`, `destructiveHint`): all are read-only except `train_model`, which is flagged destructive since `overwrite=True` replaces an existing model — so clients can auto-approve the former and confirm the latter.

- `list_models()` — all trained models available.
- `inspect_model(model_name)` — feature schema and target classes.
- `predict(model_name, features, top_n=10)` — predicted class and probabilities, plus the full glassbox audit: base rate, every term's contribution (sorted by magnitude), and an explicit additivity check against the model's own predicted probability. Also returns `warnings` for inputs the model never saw (each with `feature`, `value`, `issue` of `unseen_category` or `out_of_range`, and the `training_range` for numerics) and `missing_features` for absent/blank inputs, which are valid but reported separately.
- `train_model(target_column, model_name, csv_path=None, csv_content=None, overwrite=False, hyperparameters=None, holdout_fraction=0.2)` — train a new model, or retrain an existing one (`overwrite=True`) on refreshed data; a name collision without `overwrite=True` is rejected so nothing gets silently clobbered. Pass exactly one of `csv_path` (a path the server process can read — always prefer this for local/stdio setups, since it works for CSVs of any size without the calling agent ever reproducing the file's contents) or `csv_content` (the raw CSV text inline — only for cases with no shared filesystem, e.g. a remote http deployment, and only for small data; inlining tens of thousands of rows as a tool argument is slow and context-expensive). Returns the resolved `hyperparameters`, `training_seconds`, term counts, and — when `holdout_fraction > 0` — an `evaluation` block of out-of-sample metrics (see below).
- `explain_model(model_name, top_n=15, include_shapes=True)` — what the model learned, globally and exactly (read off the additive structure, not sampled from example predictions): every feature ranked by mean absolute log-odds contribution with its main-effect vs interaction split, the top terms including pairwise interactions with their share of the total, and for binary targets a per-feature shape (which categories or value ranges push toward which class).
- `list_hyperparameters()` — every tunable EBM hyperparameter with its type, default, allowed range, and a note on what raising or lowering it does.

## Letting an agent tune the model

An EBM is a sum of one learned curve per feature (plus a few pairwise interaction tables), fit by cyclic boosting with tiny single-feature trees. Its knobs mostly trade flexibility against smoothness, so they're exposed through `train_model(hyperparameters={...})`: `max_bins`, `max_interaction_bins`, `interactions`, `learning_rate`, `max_rounds`, `early_stopping_rounds`, `validation_size`, `outer_bags`, `inner_bags`, `min_samples_leaf`, `max_leaves`, `reg_alpha`, `reg_lambda`, `smoothing_rounds`, `greedy_ratio`, `random_state`. Omitted ones keep their defaults; unknown names, wrong types, and out-of-range values are rejected with a message listing every problem. They are defined once in `glin/hyperparameters.py`, which is also where to add or widen one.

To give the agent something to optimize, `train_model` holds out `holdout_fraction` of the rows (stratified, split seeded independently of the model's `random_state` so every trial is scored on the same rows), trains on the rest, and returns `evaluation`: `log_loss`, `baseline_log_loss` (class-frequency guess), `train_log_loss` (compare for over/underfitting), `accuracy`, `balanced_accuracy`, `roc_auc`. The same hyperparameters and metrics are saved in the model's metadata. The `tune_model_hyperparameters` prompt walks the agent through the loop: baseline → vary one or two knobs at a time → check seed noise → refit on all rows with `holdout_fraction=0`. Trials should use a scratch `model_name` with `overwrite=True`; the CLI's `glin train` still trains on every row with default hyperparameters.

## Prompts exposed over MCP

- `build_classifier(csv_path="", target_column="", model_name="")` — the whole workflow in one prompt, for any dataset: inspect and clean the CSV, decide which columns are real predictors (dropping target leakage such as outcome-describing text, identifiers, free text and raw dates), tune hyperparameters with overfitting guardrails (fixed holdout, ~10-trial budget, train-vs-holdout gap check, seed-noise check, prefer-the-simpler-config rule, and an early exit when the features carry no signal), refit on all rows, and report honest expected performance. Select it from your MCP client's prompt picker (Claude Desktop lists server prompts in the attach/`+` menu) and give it a file path and target; everything else is optional. It is self-contained, since clients show prompts to the user rather than letting the agent fetch them.
- `tune_model_hyperparameters(model_name="", target_column="")` — the tuning loop described above, with the live hyperparameter table (defaults and ranges read from the code) embedded.
- `prepare_csv_for_training(target_column="")` — a checklist an agent should work through before calling `train_model`: parsing gotchas (banner rows, wrong delimiter, ragged rows), how to pick a valid target column, what `glin`'s preprocessor already handles automatically (dirty numeric formatting, missing values, ID-like/constant/high-cardinality columns), and what it doesn't (dates, embedded JSON/list cells, leakage columns, continuous-looking targets) — all pulled live from the preprocessor's actual defaults, so the guidance can't drift out of sync with the code.

## Example: ask Claude directly

Once a model's trained and the MCP server is attached (see above), just talk to Claude in plain language — no need to know the tool schema:

> "I've got 3 deals to prioritize before quarter close: Northwind Systems ($42k, Opportunity stage, came from a referral, VP contact, owned by Carla Nguyen, 8 touches logged), Summit Retail Group ($8k, still a Lead, paid social source, owned by Brian Kessler), and Anchor Nonprofit ($3k, still a Lead, owned by Frank Suarez, no activity logged yet). Run them through deal_predictor_v1 and tell me which to prioritize."

Claude calls `predict` once per deal and comes back with a ranked, *explained* answer, not just a number:

| Deal | Win Prob | Why |
|---|---|---|
| Northwind Systems | 69% | Opportunity stage + referral + VP contact all favor it; $42k deal size is the one drag |
| Summit Retail Group | 35% | Still Lead stage + paid social source — both large, unopposed drags |
| Anchor Nonprofit | 34% | Zero logged activity, still a Lead — nothing in the record favors it |

Worth knowing before you ask: any field you don't mention comes through as missing to the model, not as "average" — so naming a rep and a lifecycle stage (using the model's own trained categories, e.g. `Lead` / `Opportunity` / `Customer`) matters far more for a well-differentiated answer than adding extra color to fields the model already has.

## Data requirements

`glin train` validates your CSV before doing any work — hard problems (e.g. a target column with only one class) stop training with a clear error; soft issues (e.g. a date-like column) print a warning and training proceeds anyway. These rules live in `glin/validation.py` as a flat, appendable list, so support for a currently-unsupported shape below can be added by adding one rule and one preprocessing case, without touching the rest of the pipeline.

**Feature columns — supported today:**

| Data shape | What happens |
|---|---|
| Numeric (int/float), including `NaN` | Passed through as-is; EBM natively bins missing values into their own split. |
| Dirty numeric strings (blanks, e.g. `" "` for a new customer's total charges) | Coerced to `float64` if ≥80% of non-null values parse as numbers; unparseable cells become `NaN`. |
| Categorical strings, any cardinality up to 250 unique values | Standardized (lowercased, stripped) and one-hot-style binned by EBM; missing/unseen values map to a `__missing__` sentinel. |
| Currency / thousands-separated strings (`$1,234.56`, `£1,234`, `€1.234,56`, `(1,234)`) | Currency symbols and whitespace are stripped, then the column is coerced as above. A decimal comma is used only when the column has unambiguous European values (`1.234,56`, `12,5`, `1.234.567`); a bare `1,234` is read as a thousands group. Columns mixing both conventions are parsed as US and a warning is returned. |
| Boolean columns | Treated as a 0/1 continuous numeric feature. |

**Feature columns — not yet supported** (each is flagged by `glin train`'s validator when detected):

| Data shape | What actually happens | Why |
|---|---|---|
| Dates / timestamps | No date features are extracted. The column is either dropped (if high-cardinality) or kept as a meaningless categorical label — no time-based signal survives either way. | Cut from V1 scope; a real dataset need should drive adding cyclical date-feature extraction. |
| Free text / natural language | Dropped once it exceeds 250 unique values; below that threshold it becomes a set of (almost certainly useless) categorical labels. | glin doesn't do NLP; a text column isn't a set of classes. |
| Lists / dicts / nested JSON in a cell | Not parsed structurally; treated as an opaque string. | No structured extraction implemented. |
| Row identifiers (sequential IDs, UUIDs, hashes) | Dropped intentionally — not a gap, this is by design. | IDs carry no predictive signal. |

**Target column requirements:**

- At least 2 distinct non-null values (binary or multiclass) — a single-class target is a hard error.
- Rows with a missing target value are dropped automatically (with a warning); the rest are unaffected.
- A numeric target with many distinct values (>20) triggers a warning that it looks like a regression target — `glin` trains classifiers, not regressors, so each value would be its own unrelated class. Band it into a few classes (quantiles or domain thresholds) or pick another target.
- A numeric target with 3–20 distinct values (e.g. a score or ordinal level) triggers a warning that the multiclass model ignores the values' order; consider fewer bands or a binary threshold. The `prepare_csv_for_training` and `build_classifier` prompts carry the same guidance.
- Full regression (upstream `ExplainableBoostingRegressor`) is planned separately and not yet supported.

## Releasing (maintainers)

**Releases are published to PyPI only by the `Publish to PyPI` GitHub Actions workflow ([.github/workflows/publish.yml](.github/workflows/publish.yml)). Nobody uploads from a local machine — no `twine upload`, no `uv publish`, no personal API tokens.** To release, push a tag of the form `vX.Y.Z`:

```bash
git tag v0.1.3
git push origin v0.1.3
```

The workflow runs the tests on Python 3.10 and 3.12, builds the sdist and wheel, checks that the artifacts carry the tag's version, and publishes them to PyPI as `glin-ml` at that exact version. The package version is derived from the git tag by `hatch-vcs` (`pyproject.toml` has no hardcoded version), so there is nothing to bump or commit before a release. Untagged local builds get a dev version such as `0.1.4.dev1+g3fa91a0`, and a source tree with no git metadata (like the Docker build) reports `0.0.0`. `glin.__version__` reads the installed package metadata. Tags that don't match `vX.Y.Z` are ignored. PyPI rejects re-uploading a version number, so a bad release can only be fixed by tagging a new version.

### One-time setup that makes this the only path

The workflow publishes through PyPI [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), which is also what keeps local uploads impossible:

1. **PyPI → the `glin-ml` project → Publishing:** add a trusted publisher for owner `AkashChatterjee`, repository `glin`, workflow `publish.yml`, environment `pypi`.
2. **PyPI → Account settings → API tokens:** delete every existing API token that can upload `glin-ml`, and don't create new ones. With no tokens and 2FA on, the only credential that can publish is the workflow's short-lived OIDC identity. Also review the project's Collaborators and remove anyone who doesn't need upload rights.
3. **GitHub → Settings → Rules → Rulesets:** add a tag ruleset for `v*` that restricts who can create tags (and blocks deleting or moving them), since creating a matching tag is what triggers a release.
4. **GitHub → Settings → Environments → `pypi`:** add required reviewers, so every release needs a manual approval before the upload step runs.

### Checking a release

To sanity-check a build without publishing anything, `uv build` locally and inspect `dist/` — but never upload it. After a release, confirm the public package works with a clean install:

```bash
uv tool install glin-ml
glin train some.csv --target y --name smoke_check
```
