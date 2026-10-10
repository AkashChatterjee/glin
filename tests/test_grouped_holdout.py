import numpy as np
import pandas as pd
import pytest

from glin import engine, server
from glin.grouped_split import split_by_group
from tests.mcp_compat import call_structured

FAST = {"interactions": 0, "outer_bags": 2, "max_rounds": 100}


def _grouped_frame(n_groups=12, per_group=40, offset=2.5, seed=0):
    """Binary target driven by a weak real signal plus a strong per-group offset."""
    rng = np.random.RandomState(seed)
    offsets = rng.normal(scale=offset, size=n_groups)
    rows = []
    for g in range(n_groups):
        size = per_group + 5 * g
        x = rng.normal(size=size)
        logit = 0.8 * x + offsets[g]
        y = rng.uniform(size=size) < 1 / (1 + np.exp(-logit))
        rows.append(
            pd.DataFrame(
                {"state": f"S{g:02d}", "x": x, "z": rng.normal(size=size), "y": np.where(y, "yes", "no")}
            )
        )
    return pd.concat(rows, ignore_index=True)


def _encode(df):
    classes = sorted(df["y"].unique())
    return np.array([classes.index(v) for v in df["y"]]), len(classes)


def test_no_group_in_both_train_and_holdout():
    df = _grouped_frame()
    y, k = _encode(df)
    split = split_by_group(df["state"], y, k, 0.2, seed=0, column="state")
    train_groups = set(df["state"].iloc[split.train_idx])
    holdout_groups = set(df["state"].iloc[split.holdout_idx])
    assert holdout_groups and train_groups
    assert not train_groups & holdout_groups
    assert len(split.train_idx) + len(split.holdout_idx) == len(df)
    assert split.holdout_groups == len(holdout_groups)
    assert split.train_groups == len(train_groups)


def test_holdout_size_and_class_balance_are_close_to_targets():
    df = _grouped_frame()
    y, k = _encode(df)
    split = split_by_group(df["state"], y, k, 0.2, seed=0, column="state")
    assert abs(len(split.holdout_idx) / len(df) - 0.2) < 0.08
    overall = np.bincount(y, minlength=k) / len(y)
    held = np.bincount(y[split.holdout_idx], minlength=k) / len(split.holdout_idx)
    assert np.abs(held - overall).max() < 0.15


def test_missing_group_values_form_their_own_group():
    df = _grouped_frame()
    df.loc[df["state"] == "S03", "state"] = None
    y, k = _encode(df)
    split = split_by_group(df["state"], y, k, 0.3, seed=0, column="state")
    in_holdout = df["state"].isna().iloc[split.holdout_idx]
    in_train = df["state"].isna().iloc[split.train_idx]
    assert not (in_holdout.any() and in_train.any())


def test_engine_reports_group_fields_and_split_ignores_random_state():
    df = _grouped_frame()
    runs = [
        engine.train_model(
            df,
            "y",
            model_name="m",
            holdout_fraction=0.2,
            holdout_group_column="state",
            hyperparameters={**FAST, "random_state": seed},
        )
        for seed in (1, 2)
    ]
    ev = [r["evaluation"] for r in runs]
    for e in ev:
        assert e["holdout_group_column"] == "state"
        assert e["holdout_groups"] + e["train_groups"] == df["state"].nunique()
        assert e["holdout_rows"] + runs[0]["training_rows"] == len(df)
        assert e["holdout_groups"] >= 1
    assert ev[0]["holdout_rows"] == ev[1]["holdout_rows"]
    assert ev[0]["holdout_groups"] == ev[1]["holdout_groups"]
    assert runs[0]["training_rows"] == runs[1]["training_rows"]
    assert runs[0]["holdout_group_column"] == "state"
    assert "state" in runs[0]["preprocessor"].output_columns_  # still a feature


def test_group_fields_are_saved_in_metadata(tmp_path):
    bundle = engine.train_model(
        _grouped_frame(),
        "y",
        model_name="grouped",
        holdout_fraction=0.2,
        holdout_group_column="state",
        hyperparameters=FAST,
    )
    engine.save_bundle(bundle, tmp_path)
    meta = engine.load_metadata(tmp_path / "grouped")
    assert meta["holdout_group_column"] == "state"
    assert meta["evaluation"]["holdout_groups"] == bundle["evaluation"]["holdout_groups"]
    assert meta["evaluation"]["train_groups"] == bundle["evaluation"]["train_groups"]


def test_default_holdout_is_unchanged_without_group_column():
    df = _grouped_frame()
    bundle = engine.train_model(
        df, "y", model_name="plain", holdout_fraction=0.2, hyperparameters=FAST
    )
    ev = bundle["evaluation"]
    assert "holdout_group_column" not in ev
    assert "holdout_groups" not in ev
    assert ev["holdout_rows"] == int(round(len(df) * 0.2))
    assert bundle["holdout_group_column"] is None


def test_group_column_ignored_with_warning_when_no_holdout():
    bundle = engine.train_model(
        _grouped_frame(),
        "y",
        model_name="nohold",
        holdout_fraction=0,
        holdout_group_column="state",
        hyperparameters=FAST,
    )
    assert bundle["evaluation"] is None
    assert bundle["holdout_group_column"] is None
    assert any("holdout_group_column" in w and "ignored" in w for w in bundle["warnings"])


def _train(df, **kw):
    return engine.train_model(
        df, "y", model_name="m", holdout_fraction=0.2, hyperparameters=FAST, **kw
    )


def test_missing_group_column_is_rejected():
    with pytest.raises(ValueError, match="'nope' is not a column"):
        _train(_grouped_frame(), holdout_group_column="nope")


def test_target_as_group_column_is_rejected():
    with pytest.raises(ValueError, match="cannot be the target"):
        _train(_grouped_frame(), holdout_group_column="y")


def test_fewer_than_five_groups_is_rejected():
    df = _grouped_frame(n_groups=4)
    with pytest.raises(ValueError, match="at least 5 groups"):
        _train(df, holdout_group_column="state")


def test_dominant_group_is_rejected():
    df = _grouped_frame(n_groups=6, per_group=20)
    big = _grouped_frame(n_groups=1, per_group=400, seed=5).assign(state="BIG")
    with pytest.raises(ValueError, match="more than half"):
        _train(pd.concat([df, big], ignore_index=True), holdout_group_column="state")


def test_holdout_without_two_classes_is_rejected():
    df = _grouped_frame()
    # one class lives in a single group: a holdout can't have both classes and leave one in training
    df["y"] = np.where(df["state"] == "S00", "yes", "no")
    with pytest.raises(ValueError, match="at least 2 classes"):
        _train(df, holdout_group_column="state")


def test_grouped_holdout_is_not_more_optimistic_than_random():
    # Group offsets dominate; a random split sees every group in training.
    df = _grouped_frame(n_groups=15, per_group=60, offset=3.0, seed=3)
    assert _train(df)["evaluation"]["log_loss"] > 0
    grouped_ev = _train(df, holdout_group_column="state")["evaluation"]
    assert grouped_ev["holdout_groups"] >= 1
    assert grouped_ev["holdout_groups"] + grouped_ev["train_groups"] == 15


@pytest.mark.asyncio
async def test_mcp_train_model_accepts_holdout_group_column(tmp_path):
    mcp = server.build_server(models_root=tmp_path)
    result = await call_structured(
        mcp,
        "train_model",
        {
            "csv_content": _grouped_frame().to_csv(index=False),
            "target_column": "y",
            "model_name": "grouped_mcp",
            "holdout_group_column": "state",
            "hyperparameters": FAST,
        },
    )
    assert result["holdout_group_column"] == "state"
    ev = result["evaluation"]
    assert ev["holdout_group_column"] == "state"
    assert ev["holdout_groups"] + ev["train_groups"] == 12


@pytest.mark.asyncio
async def test_mcp_train_model_surfaces_group_errors(tmp_path):
    mcp = server.build_server(models_root=tmp_path)
    with pytest.raises(server.ToolError, match="is not a column"):
        await mcp.call_tool(
            "train_model",
            {
                "csv_content": _grouped_frame().to_csv(index=False),
                "target_column": "y",
                "model_name": "bad",
                "holdout_group_column": "nope",
            },
        )
