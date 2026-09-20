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

- `list_models()` — all trained models available.
- `inspect_model(model_name)` — feature schema and target classes.
- `predict(model_name, features, top_n=10)` — predicted class and probabilities, plus the full glassbox audit: base rate, every term's contribution (sorted by magnitude), and an explicit additivity check against the model's own predicted probability.

## Data requirements

`glin train` validates your CSV before doing any work — hard problems (e.g. a target column with only one class) stop training with a clear error; soft issues (e.g. a date-like column) print a warning and training proceeds anyway. These rules live in `glin/validation.py` as a flat, appendable list, so support for a currently-unsupported shape below can be added by adding one rule and one preprocessing case, without touching the rest of the pipeline.

**Feature columns — supported today:**

| Data shape | What happens |
|---|---|
| Numeric (int/float), including `NaN` | Passed through as-is; EBM natively bins missing values into their own split. |
| Dirty numeric strings (blanks, e.g. `" "` for a new customer's total charges) | Coerced to `float64` if ≥80% of non-null values parse as numbers; unparseable cells become `NaN`. |
| Categorical strings, any cardinality up to 250 unique values | Standardized (lowercased, stripped) and one-hot-style binned by EBM; missing/unseen values map to a `__missing__` sentinel. |
| Boolean columns | Treated as a 0/1 continuous numeric feature. |

**Feature columns — not yet supported** (each is flagged by `glin train`'s validator when detected):

| Data shape | What actually happens | Why |
|---|---|---|
| Dates / timestamps | No date features are extracted. The column is either dropped (if high-cardinality) or kept as a meaningless categorical label — no time-based signal survives either way. | Cut from V1 scope; a real dataset need should drive adding cyclical date-feature extraction. |
| Free text / natural language | Dropped once it exceeds 250 unique values; below that threshold it becomes a set of (almost certainly useless) categorical labels. | glin doesn't do NLP; a text column isn't a set of classes. |
| Currency symbols / thousands separators (`$`, `€`, `£`, `,`) | Not stripped. `"$1,234.56"` fails the ≥80% numeric-parse threshold and falls back to categorical (i.e. garbage). Clean these before training. | Cut from V1 scope under an "assume a healthy dataset" simplification. |
| Lists / dicts / nested JSON in a cell | Not parsed structurally; treated as an opaque string. | No structured extraction implemented. |
| Row identifiers (sequential IDs, UUIDs, hashes) | Dropped intentionally — not a gap, this is by design. | IDs carry no predictive signal. |

**Target column requirements:**

- At least 2 distinct non-null values (binary or multiclass) — a single-class target is a hard error.
- Rows with a missing target value are dropped automatically (with a warning); the rest are unaffected.
- A numeric target with many distinct values (>20) triggers a warning that it looks like a regression target — `glin` trains classifiers, not regressors.

## Releasing (maintainers)

PyPI rejects re-uploading a version number, so bump `version` in `pyproject.toml` first and double check the metadata (description, classifiers, URLs) before publishing — there's no fixing a released version after the fact, only shipping a new one.

```bash
rm -rf dist
uv build
```

Dry run against TestPyPI first:

```bash
twine upload --repository testpypi dist/*
# username: __token__, password: a TestPyPI API token
```

Verify it actually installs from there before touching the real index:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ glin-ml
```

Then publish for real:

```bash
twine upload dist/*
# username: __token__, password: a PyPI API token (separate account/token from TestPyPI)
```

Confirm the public release works with a clean install:

```bash
uv tool install glin-ml
glin train some.csv --target y --name smoke_check
```
