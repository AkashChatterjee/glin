import numpy as np
import pandas as pd
import pytest

from glin import engine, server
from glin.leakage import find_leaky_features, leakage_warnings
from glin.preprocessor import EBMTabularPreprocessor
from tests.mcp_compat import call_structured


def _suspects(df, target):
    """Run the check the way train_model does: on the preprocessed features."""
    y_raw = df[target]
    classes = sorted(y_raw.unique())
    y = np.array([classes.index(v) for v in y_raw])
    pre = EBMTabularPreprocessor()
    X = pre.fit_transform(df.drop(columns=[target]))
    return find_leaky_features(
        X,
        y,
        classes,
        categorical=[c for c, t in pre.feature_types_.items() if t == "nominal"],
        source_columns={n: c for c, ns in pre.date_feature_names_.items() for n in ns},
    )


def _columns(suspects):
    return [s["column"] for s in suspects]


def _noise_frame(n=400, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(size=n).round(1),
            "b": rng.randint(0, 100, size=n).astype(float),
            "region": rng.choice(list("ABCDE"), size=n),
        }
    )


def test_threshold_of_numeric_feature_is_flagged_with_the_threshold():
    df = _noise_frame()
    df["score"] = np.random.RandomState(1).uniform(0, 100, size=len(df)).round(1)
    df["bad"] = np.where(df["score"] > 62.5, "yes", "no")
    found = _suspects(df, "bad")
    assert _columns(found) == ["score"]
    assert found[0]["score"] >= 0.99
    assert "> 62" in found[0]["rule"] and "'yes'" in found[0]["rule"]


def test_function_of_categorical_feature_is_flagged_with_the_mapping():
    df = _noise_frame()
    df["grade"] = np.random.RandomState(2).choice(["p", "q", "r", "s"], size=len(df))
    df["pass"] = np.where(df["grade"].isin(["p", "q"]), "ok", "fail")
    found = _suspects(df, "pass")
    assert _columns(found) == ["grade"]
    assert "'ok'" in found[0]["rule"] and "'fail'" in found[0]["rule"]


def test_band_function_of_numeric_feature_is_flagged():
    df = _noise_frame()
    df["t"] = np.random.RandomState(3).uniform(0, 100, size=len(df)).round(1)
    df["mid"] = np.where((df["t"] > 30) & (df["t"] < 60), "in", "out")
    assert _columns(_suspects(df, "mid")) == ["t"]


def test_percent_string_column_leaking_is_flagged():
    df = _noise_frame()
    pct = np.random.RandomState(4).uniform(0, 100, size=len(df)).round(2)
    df["coverage"] = [f"{p}%" for p in pct]
    df["low_coverage"] = np.where(pct < 50, "yes", "no")
    found = _suspects(df, "low_coverage")
    assert _columns(found) == ["coverage"]


def test_date_derived_leak_is_reported_under_the_raw_column():
    rng = np.random.RandomState(5)
    days = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.randint(0, 1500, size=400), unit="D")
    df = _noise_frame()
    df["signup"] = days.strftime("%Y-%m-%d")
    df["early"] = np.where(days < pd.Timestamp("2022-01-01"), "yes", "no")
    found = _suspects(df, "early")
    assert _columns(found) == ["signup"]
    assert found[0]["feature"].startswith("signup__")
    assert "signup" in leakage_warnings(found, "early")[0]


def test_multiclass_derived_target_is_flagged():
    df = _noise_frame()
    x = np.random.RandomState(6).uniform(0, 90, size=len(df)).round(1)
    df["x"] = x
    df["level"] = np.select([x < 30, x < 60], ["low", "mid"], "high")
    assert _columns(_suspects(df, "level")) == ["x"]


def test_missing_in_leaking_column_with_missingness_tied_to_class_is_flagged():
    df = _noise_frame()
    t = np.random.RandomState(7).uniform(0, 100, size=len(df)).round(1)
    df["t"] = np.where(t < 20, np.nan, t)
    df["low"] = np.where(t < 20, "yes", "no")
    found = _suspects(df, "low")
    assert _columns(found) == ["t"]
    assert "missing -> 'yes'" in found[0]["rule"]


def test_strong_but_imperfect_predictor_is_not_flagged():
    df = _noise_frame(n=1000)
    x = np.random.RandomState(8).uniform(0, 100, size=len(df)).round(1)
    flip = np.random.RandomState(9).rand(len(df)) < 0.05
    df["x"] = x
    df["y"] = np.where((x > 50) ^ flip, "yes", "no")
    assert _suspects(df, "y") == []


def test_overlapping_gaussians_are_not_flagged():
    rng = np.random.RandomState(10)
    y = rng.choice(["a", "b"], size=1000)
    df = _noise_frame(n=1000)
    df["x"] = rng.normal(loc=(y == "b") * 2.5, scale=1.0).round(1)  # AUC ~0.96
    df["y"] = y
    assert _suspects(df, "y") == []


@pytest.mark.parametrize("n", [30, 60, 400, 2000])
def test_pure_noise_with_many_columns_is_not_flagged(n):
    rng = np.random.RandomState(11)
    df = pd.DataFrame(rng.normal(size=(n, 300)).round(1), columns=[f"f{i}" for i in range(300)])
    for i in range(30):
        df[f"c{i}"] = rng.choice(list("ABC"), size=n)
    df["y"] = rng.choice(["a", "b"], size=n, p=[0.8, 0.2])
    assert _suspects(df, "y") == []


def test_high_cardinality_categorical_does_not_memorise_the_target():
    rng = np.random.RandomState(12)
    n = 600
    df = _noise_frame(n)
    df["code"] = [f"k{i}" for i in rng.randint(0, 200, size=n)]  # ~3 rows each
    df["y"] = rng.choice(["a", "b"], size=n)
    assert _suspects(df, "y") == []


def test_categorical_with_enough_support_per_value_can_still_be_flagged():
    rng = np.random.RandomState(13)
    n = 600
    df = _noise_frame(n)
    df["code"] = [f"k{i}" for i in rng.randint(0, 30, size=n)]  # ~20 rows each
    df["y"] = np.where(df["code"].isin([f"k{i}" for i in range(10)]), "a", "b")
    assert _columns(_suspects(df, "y")) == ["code"]


@pytest.mark.parametrize("levels", [101, 150, 240])
def test_categorical_with_more_than_100_levels_recoded_into_target_is_flagged(levels):
    rng = np.random.RandomState(20)
    n = 4000
    district = rng.randint(0, levels, n)
    df = pd.DataFrame({"noise": rng.normal(size=n), "district": [f"D{d:03d}" for d in district]})
    df["y"] = np.where(district % 2 == 0, "urban", "rural")
    assert _columns(_suspects(df, "y")) == ["district"]


def test_train_model_flags_high_cardinality_recode_on_training_rows():
    rng = np.random.RandomState(21)
    district = rng.randint(0, 150, 4000)
    df = pd.DataFrame({"noise": rng.normal(size=4000), "district": [f"D{d:03d}" for d in district]})
    df["y"] = np.where(district % 2 == 0, "urban", "rural")
    bundle = engine.train_model(
        df, "y", model_name="m", holdout_fraction=0.2, hyperparameters={"interactions": 0, "outer_bags": 1}
    )
    assert [s["column"] for s in bundle["leakage_suspects"]] == ["district"]


def test_noise_categoricals_with_many_small_levels_are_not_flagged():
    rng = np.random.RandomState(22)
    n = 20_000
    X = pd.DataFrame({f"c{i}": rng.randint(0, 250, n).astype(str) for i in range(200)})
    y = rng.randint(0, 2, n)
    assert find_leaky_features(X, y, [0, 1], categorical=list(X.columns)) == []


@pytest.mark.parametrize("n", [60, 400, 1500])
def test_noise_categoricals_near_the_support_floor_are_not_flagged(n):
    rng = np.random.RandomState(23)
    X = pd.DataFrame({f"c{i}": rng.randint(0, max(2, n // 6), n).astype(str) for i in range(100)})
    y = (rng.rand(n) < 0.2).astype(int)
    assert find_leaky_features(X, y, [0, 1], categorical=list(X.columns)) == []


def test_integer_coded_category_recoded_into_target_is_flagged():
    rng = np.random.RandomState(24)
    n = 3000
    code = rng.randint(1, 11, n)
    df = pd.DataFrame({"noise": rng.normal(size=n), "q12_code": code})
    df["y"] = np.where(np.isin(code, [2, 5, 7]), "yes", "no")
    found = _suspects(df, "y")
    assert _columns(found) == ["q12_code"]
    assert "{2, 5, 7} -> 'yes'" in found[0]["rule"]


def test_integer_code_with_missing_as_own_category_is_flagged():
    rng = np.random.RandomState(25)
    n = 1000
    code = rng.randint(1, 6, n).astype(float)
    code[rng.rand(n) < 0.1] = np.nan
    X = pd.DataFrame({"q": code})
    y = np.isin(code, [1, 3]) | np.isnan(code)
    found = find_leaky_features(X, y.astype(int), ["no", "yes"], categorical=[])
    assert _columns(found) == ["q"]
    assert "missing" in found[0]["rule"]


def test_noise_integer_codes_are_not_flagged():
    rng = np.random.RandomState(26)
    for n in (30, 100, 2000):
        X = pd.DataFrame({f"q{i}": rng.randint(0, 20, n).astype(float) for i in range(100)})
        y = (rng.rand(n) < 0.3).astype(int)
        assert find_leaky_features(X, y, [0, 1], categorical=[]) == []


def test_numeric_with_more_than_20_values_gets_no_category_map():
    rng = np.random.RandomState(27)
    n = 2000
    code = rng.randint(0, 30, n)
    X = pd.DataFrame({"q": code.astype(float)})
    y = np.isin(code, [2, 5, 7, 11, 19, 23]).astype(int)
    assert find_leaky_features(X, y, [0, 1], categorical=[]) == []


def test_tiny_dataset_is_skipped():
    df = _noise_frame(n=24)
    df["y"] = np.where(df["a"] > 0, "yes", "no")
    assert _suspects(df, "y") == []


def test_tiny_classes_are_ignored_not_memorised():
    rng = np.random.RandomState(14)
    n = 400
    df = _noise_frame(n)
    y = np.array(["a"] * n, dtype=object)
    y[rng.choice(n, 4, replace=False)] = "rare"  # below MIN_CLASS_ROWS
    y[rng.choice(n, 150, replace=False)] = "b"
    df["y"] = y
    assert _suspects(df, "y") == []


def test_rare_but_checkable_class_is_flagged_when_perfectly_separated():
    rng = np.random.RandomState(15)
    n = 1000
    df = _noise_frame(n)
    df["x"] = rng.uniform(0, 100, size=n).round(1)
    df["y"] = np.where(df["x"] > 98.5, "yes", "no")  # ~15 positives
    assert _columns(_suspects(df, "y")) == ["x"]


def test_missing_values_in_features_do_not_raise_or_trigger():
    rng = np.random.RandomState(16)
    df = _noise_frame()
    df.loc[rng.rand(len(df)) < 0.4, "a"] = np.nan
    df.loc[rng.rand(len(df)) < 0.4, "region"] = np.nan
    df["y"] = rng.choice(["a", "b"], size=len(df))
    assert _suspects(df, "y") == []


def test_warnings_are_capped_and_name_the_column_and_evidence():
    found = [
        {"feature": f"c{i}", "column": f"c{i}", "score": 1.0, "rule": "<= 1 -> 'a'; > 1 -> 'b'"}
        for i in range(13)
    ]
    messages = leakage_warnings(found, "y")
    assert len(messages) == 11
    assert "'c0'" in messages[0] and "1.000" in messages[0] and "drop" in messages[0]
    assert "3 more" in messages[-1]


def _leaky_df(n=300, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "noise": rng.normal(size=n),
            "region": rng.choice(["n", "s", "e", "w"], size=n),
            "tox": rng.uniform(0, 10, size=n).round(2),
        }
    )
    df["unsafe"] = np.where(df["tox"] > 6, "yes", "no")
    return df


def test_train_model_warns_but_still_trains():
    bundle = engine.train_model(_leaky_df(), "unsafe", model_name="leaky")
    assert bundle["model"] is not None
    assert [s["column"] for s in bundle["leakage_suspects"]] == ["tox"]
    assert any("Possible target leakage" in w and "'tox'" in w for w in bundle["warnings"])
    assert bundle["leakage_warnings"] == [w for w in bundle["warnings"] if "leakage" in w]


def test_train_model_leakage_check_uses_training_rows_only():
    bundle = engine.train_model(_leaky_df(), "unsafe", model_name="leaky", holdout_fraction=0.2)
    assert bundle["training_rows"] == 240
    assert [s["column"] for s in bundle["leakage_suspects"]] == ["tox"]


def test_train_model_clean_data_has_no_leakage_warning():
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"a": rng.normal(size=300).round(1), "b": rng.choice(list("xyz"), size=300)})
    df["y"] = np.where(df["a"] + rng.normal(scale=1.0, size=300) > 0, "yes", "no")
    bundle = engine.train_model(df, "y", model_name="clean")
    assert bundle["leakage_suspects"] == []
    assert not any("leakage" in w for w in bundle["warnings"])


@pytest.mark.asyncio
async def test_mcp_train_model_returns_leakage_warning_and_trains(tmp_path):
    mcp = server.build_server(models_root=tmp_path)
    result = await call_structured(
        mcp,
        "train_model",
        {
            "csv_content": _leaky_df().to_csv(index=False),
            "target_column": "unsafe",
            "model_name": "leaky_mcp",
        },
    )
    assert result["model_name"] == "leaky_mcp"
    assert (tmp_path / "leaky_mcp" / "bundle.joblib").exists()
    leak = [w for w in result["warnings"] if "Possible target leakage" in w]
    assert len(leak) == 1 and "'tox'" in leak[0] and "drop it and retrain" in leak[0]
