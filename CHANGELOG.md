# Changelog

This file records the changes in each release of glin (`glin-ml` on PyPI). The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The version numbers follow [Semantic Versioning](https://semver.org/).

The numbers in parentheses are GitHub issues.

## [Unreleased]

## [0.1.4] - 2026-10-11

### Removed

- The HTTP and SSE transports. glin now runs locally over stdio only. Version 0.1.3 is the last release with HTTP mode (#8).
- The `base_rate` field in the `predict` audit. Use `score_class` and `base_logit` (#31).

### Changed

- `csv_path` reads only `.csv`, `.xlsx` and `.xls` files inside the allowed folders. `glin serve --allow-dir` sets these folders (#7).
- New default allowed folders: the home folder, plus the working folder only when it is not `/` and not a parent folder of home. Claude Desktop starts servers from `/`, so it gets the home folder only. Data outside home (for example `/Volumes/...`) needs `--allow-dir` (#47).
- `build_server(allowed_dirs=[])` now turns `csv_path` off. Before, it used the default folders (#47).
- `csv_path` files have a size limit of 200 MB by default. `glin serve --max-file-mb` changes it (#47).
- `score_class` now has the same type as `predicted_class` (for example `1`, not `"1"`). The keys of the probability dictionaries stay strings, so look up `model_probability[str(score_class)]` (#54).
- A new model name must have 1 to 64 letters, digits, `-`, `_` or `.`, and must not start with `.`. Models saved with other names stay usable (#48).
- glin supports the `mcp` 1.x and 2.x SDK lines. The lowest supported version is now `mcp>=1.21.1`. Older versions fail at import (#5, #49).
- `glin serve --mode stdio` is deprecated. glin accepts it and writes a notice to stderr. `--mode http` and `--mode sse` give an error that says why (#41, #50).
- The default suppression codes now include `*` and `..` (#17, #43).
- A column of unique whole numbers is dropped as an identifier only when its values fill a range or its name looks like an ID (for example `row_id`). Other unique whole numbers, such as salaries, stay as measurements. glin warns about each dropped column (#32, #44).
- The Claude Code plugin starts `glin-ml[excel]>=0.1.4`, so Excel input works in the plugin (#55).

### Added

- Excel input (`.xlsx`, `.xls`) with messy headers: title rows, multi-row headers and footnotes. `train_model` takes `sheet` and `header_rows`. Install the extra with `pip install 'glin-ml[excel]'` (#12, #53).
- Numbers with currency symbols, currency codes (`USD 100`, `Rs. 100`), thousands separators, decimal commas (`1.234,56`), the Unicode minus and a trailing minus are read as numbers (#16, #42, #43).
- Percents are read as percentage points. Suppression codes are read as missing values. `train_model` takes `suppression_codes` (#17).
- Date and timestamp columns become date features (#15).
- A derived-target leakage check. `train_model` returns `leakage_suspects` and `leakage_warnings` (#18, #51).
- Grouped holdout: `train_model` takes `holdout_group_column` and holds out whole groups (#33, #52).
- `predict` warnings for inputs that the model did not see: `unseen_category`, `out_of_range` (#14), `unparseable_number` (#43), `ambiguous_date_format` and `date_order_mismatch` (#45).
- Warnings for numeric and ordinal targets, with workarounds. Numbers stored as text get the same warnings (#13, #54).
- The `predict` audit names the class that the scores describe: `score_class` and `base_logit` (#31).
- MCP tool annotations: titles and read-only and destructive hints (#6).
- `glin --version`. The MCP `serverInfo.version` is now the glin version (#50).
- A listing for the MCP Registry (`server.json`) and a Claude Code plugin with a skill (#9, #55).
- A case study: child stunting across Indian districts (NFHS-5) (#11).
- A test workflow for pull requests and pushes to `main` (#49).
- This changelog. Each release on GitHub uses its section as the release notes (#56).

### Fixed

- `explain_model` reported score error bars as the shape ranges. The ranges are now in feature units (#38).
- Decimal-comma columns had wrong training ranges, and `predict` misread some values by a factor of 10 or more (#42).
- Infinite values stopped training with a raw scikit-learn error. glin now treats them as missing, with a warning (#43).
- A `model_name` such as `../x` could write and load models outside the models folder (#48).
- An ID column with one text value stayed as a feature (#44).
- `predict` on wide data wrote thousands of pandas warnings to stderr (#46).
- The leakage check missed categoricals with many levels and categories stored as integer codes (#51).
- Grouped holdout gave a wrong error when every group was too large, split spelling variants of one group, and was slow for many groups (#52).
- The Excel loader kept some footnote rows, dropped some real final rows, accepted `true` as a sheet, and used MCP names in CLI errors (#53).
- The interpret library wrote its progress messages to stderr during `serve` (#50).
- The sdist included the `.claude/` and `.github/` folders (#55).
- `fit` failed on a frame with integer column names and a date column. glin now changes column names to text. Column names that are the same as text, such as `0` and `"0"`, give an error (#57).

## Earlier versions

Versions 0.1.0 to 0.1.3 have no entries in this file. See the [tags on GitHub](https://github.com/AkashChatterjee/glin/tags).

[Unreleased]: https://github.com/AkashChatterjee/glin/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/AkashChatterjee/glin/compare/v0.1.3...v0.1.4
