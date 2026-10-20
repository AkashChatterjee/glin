---
name: glin
description: Use glin to build, tune, predict with and explain a classifier on tabular data (CSV or Excel). Use it when the user wants a fast, statistical "gut feeling" on a row or a batch of rows, wants to know what drives an outcome, or needs a prediction with an exact reason. The glin MCP tools do the work.
---

# Use glin

glin trains an Explainable Boosting Machine (EBM) on a CSV or Excel file. Each prediction has an exact breakdown by feature. The breakdown is not a SHAP or LIME estimate. glin runs on the local machine only.

glin handles classification. It cannot do regression. A target must have at least 2 classes.

## Tools

- `list_models()` lists the trained models.
- `inspect_model(model_name)` returns the features, the types and the target classes. Call it before `predict`.
- `predict(model_name, features, top_n=10)` returns the class, the probabilities and the full audit.
- `explain_model(model_name, top_n=15, include_shapes=True)` shows what the model learned.
- `list_hyperparameters()` lists the tuning settings with defaults and ranges.
- `train_model(target_column, model_name, csv_path=..., ...)` trains or retrains a model.
- `delete_model(model_name)` deletes a model permanently. Ask the user to confirm before you call it.

## Build a model

1. Get the data as a file. Pass it as `csv_path` (`.csv`, `.xlsx`, `.xls`). Use `csv_content` only for small data pasted in the chat.
2. Decide which columns are features. For each column, ask: "Is this value known at the time of the prediction?" If not, it is leakage. Drop it.
3. Drop identifiers, names, free text and duplicate columns. Keep dates. glin splits dates into year, month, day of week, hour and days since start.
4. Write the cleaned data (with the target) to a NEW file. Never overwrite the original.
5. Call `train_model` with the exact `target_column` and a new `model_name`. A numeric target with many values is a regression target. Band it into a few classes first.
6. Read the result:
   - `warnings` lists dropped columns, suppression codes and "Possible target leakage". Ask the user about each leakage warning. Drop the column and retrain if it is not known at prediction time.
   - `evaluation.log_loss` must be clearly lower than `evaluation.baseline_log_loss`. If it is not, the features carry little signal.
   - A holdout `roc_auc` above 0.95 on real data usually means leakage. Check the columns again.

Excel files need `glin-ml[excel]`. With uvx, the server command is `uvx --from 'glin-ml[excel]>=0.1.5' glin serve`. Pass `sheet` for a workbook with several sheets. Pass `header_rows` (for example `"3-4"`) if the header detection is not clear.

Numeric columns can have codes for missing values. The default codes are `SUPP`, `NE`, `NP`, `z`, `*` and `..`. If a warning names other codes, confirm with the user. Then pass them in `suppression_codes`. A warning can also say that codes keep a column from being numeric. Do the same for that warning.

## Tune a model

Tune only if the user asks for the best model, or if the first result looks weak.

1. Train a baseline with no `hyperparameters`. Use a scratch name such as `<name>_trial` and `overwrite=True`.
2. If a column groups the rows (state, region, customer, site, user), pass it as `holdout_group_column` in every trial. glin then holds out whole groups. The metrics show performance on groups the model never saw. Keep this setting the same in all trials.
3. Compare `evaluation.log_loss` (lower is better). Compare `train_log_loss` with `log_loss`. A large gap means overfitting. Move to smoother settings.
4. Change one or two settings in each trial. Start with `interactions` (try 0), then `min_samples_leaf` and `reg_lambda`, then `max_bins`. Call `list_hyperparameters` for the ranges.
5. Use a budget of about 10 trials. Stop when the gain is below 1% or below the noise. Check the noise with two other `random_state` values. Prefer the simpler setting.
6. Refit with the best `hyperparameters`, the real `model_name` and `holdout_fraction=0`. Add `overwrite=True` only to replace a model. This model has no `evaluation`. Report the holdout numbers of the best trial as the expected performance.

## Predict

1. Call `inspect_model` to get the exact feature names and categories.
2. Call `predict` with the `features` object. Use the categories the model learned. Send dates as raw values, for example `"2024-03-06"`.
3. Report the class and the probability. Then report the top contributions, with their direction.

Read the audit like this:
- For a binary target, scores and `base_logit` are log-odds of `score_class`. A positive score pushes toward `score_class`. A negative score pushes toward the other class.
- `score_class` and `predicted_class` have the type of the target values (for example `1`). The probability keys are always text (for example `"1"`).
- For a multiclass target, each term has `score_by_class`.
- `additivity_verified` must be true. It proves that the scores add up to the model probability.
- `warnings` lists inputs that the model never saw (`unseen_category`, `out_of_range`, `unparseable_number`, `unparseable_date`). Tell the user that the prediction is less reliable. `ambiguous_date_format` and `date_order_mismatch` tell you that glin read a date in an order that the user possibly did not mean. Show the user the `parsed_as` date. If it is not correct, send the date again as ISO `YYYY-MM-DD`.
- `missing_features` lists absent inputs. A missing field is not an average value. Say which fields were missing.

For many rows, call `predict` once for each row. Then rank the rows and give the reason for each.

## Explain a model

Call `explain_model` to answer "what drives this outcome?". It ranks every feature by its mean absolute contribution. It also lists the top terms, with pairwise interactions, and (for binary targets) the shape of each main effect. Use this instead of running `predict` on a few rows.

## Rules

- Never claim more than the holdout shows. State how the metrics were measured.
- `train_model` is destructive when `overwrite=True`. Ask before you replace a model the user did not create in this session.
- `csv_path` must be inside a directory that the server allows: by default the home directory, and the working directory when it is not `/`. The file must also be smaller than the size limit (200 MB by default). Otherwise the server rejects the path. If the error says `csv_path` is disabled, use `csv_content`.
