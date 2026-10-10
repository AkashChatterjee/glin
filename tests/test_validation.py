import numpy as np
import pandas as pd

from glin.validation import MIN_ROWS, validate_dataset


def _base_df(n=50) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    return pd.DataFrame(
        {
            "age": rng.randint(18, 80, size=n),
            "plan": rng.choice(["basic", "premium"], size=n),
            "churn": rng.choice(["Yes", "No"], size=n),
        }
    )


def test_valid_dataset_has_no_errors():
    result = validate_dataset(_base_df(), "churn")
    assert result.is_valid
    assert result.errors == []


def test_missing_target_column_is_an_error():
    result = validate_dataset(_base_df(), "does_not_exist")
    assert not result.is_valid
    assert any("does_not_exist" in i.message for i in result.errors)


def test_too_few_rows_is_an_error():
    df = _base_df(n=MIN_ROWS - 1)
    result = validate_dataset(df, "churn")
    assert not result.is_valid
    assert any("rows" in i.message for i in result.errors)


def test_single_class_target_is_an_error():
    df = _base_df()
    df["churn"] = "Yes"
    result = validate_dataset(df, "churn")
    assert not result.is_valid
    assert any("distinct value" in i.message for i in result.errors)


def test_missing_target_values_is_a_warning_not_an_error():
    df = _base_df()
    df.loc[0:2, "churn"] = np.nan
    result = validate_dataset(df, "churn")
    assert result.is_valid
    assert any("missing target value" in i.message for i in result.warnings)


def test_continuous_looking_target_is_a_warning():
    df = _base_df()
    df["price"] = np.random.RandomState(1).normal(100, 10, size=len(df)).round(2)
    result = validate_dataset(df, "price")
    assert result.is_valid  # still trainable, just flagged
    assert any("regressor" in i.message for i in result.warnings)
    assert any("band the target" in i.message for i in result.warnings)


def test_ordinal_looking_numeric_target_warns_order_is_ignored():
    df = _base_df()
    df["level"] = np.random.RandomState(2).randint(1, 6, size=len(df))
    result = validate_dataset(df, "level")
    assert result.is_valid
    assert any("ignores their order" in i.message for i in result.warnings)


def test_binary_numeric_target_has_no_target_shape_warning():
    df = _base_df()
    df["flag"] = np.random.RandomState(3).randint(0, 2, size=len(df))
    result = validate_dataset(df, "flag")
    assert not any("flag" in i.message for i in result.warnings)


def test_date_column_is_supported_and_not_flagged():
    # As strings, matching how pandas actually reads a CSV date column.
    df = _base_df()
    df["signup_date"] = pd.date_range("2020-01-01", periods=len(df)).astype(str)
    result = validate_dataset(df, "churn")
    assert result.is_valid
    assert not any("signup_date" in i.message for i in result.issues)


def test_dirty_numeric_string_column_is_not_mistaken_for_a_date():
    df = _base_df(n=48)
    df["total_charges"] = ["480.47", "521.94", " ", "396.98"] * 12
    result = validate_dataset(df, "churn")
    assert not any("total_charges" in i.message for i in result.warnings)


def test_list_valued_column_is_a_warning():
    df = _base_df()
    df["tags"] = [["a", "b"]] * len(df)
    result = validate_dataset(df, "churn")
    assert result.is_valid
    assert any("tags" in i.message for i in result.warnings)


def test_train_model_raises_with_validation_error_message():
    import pytest

    from glin import engine

    df = _base_df(n=MIN_ROWS - 1)
    with pytest.raises(ValueError, match="rows"):
        engine.train_model(df, "churn", model_name="too_small")


def test_train_model_drops_rows_with_missing_target():
    from glin import engine

    df = _base_df(n=100)
    df.loc[0:4, "churn"] = np.nan
    bundle = engine.train_model(df, "churn", model_name="missing_target_test")
    assert any("missing target value" in w for w in bundle["warnings"])


def _dfe_style_df(n=100, extra_code=None):
    rng = np.random.RandomState(1)
    pct = [f"{v:.2f}%" for v in rng.uniform(5, 95, size=n)]
    pct[::10] = ["SUPP"] * len(pct[::10])
    if extra_code:
        pct[5::10] = [extra_code] * len(pct[5::10])
    return pd.DataFrame({"pct": pct, "churn": rng.choice(["Yes", "No"], size=n)})


def test_train_model_handles_percent_and_suppression_columns_without_warning():
    from glin import engine

    bundle = engine.train_model(
        _dfe_style_df(), "churn", model_name="pct_test", hyperparameters={"interactions": 0}
    )
    assert bundle["preprocessor"].numeric_columns_ == ["pct"]
    assert bundle["warnings"] == []


def test_train_model_surfaces_unrecognized_code_warning_and_custom_codes_silence_it():
    from glin import engine

    df = _dfe_style_df(extra_code="xx")
    bundle = engine.train_model(
        df, "churn", model_name="code_test", hyperparameters={"interactions": 0}
    )
    assert any("'xx'" in w and "suppression_codes" in w for w in bundle["warnings"])

    bundle = engine.train_model(
        df,
        "churn",
        model_name="code_test",
        hyperparameters={"interactions": 0},
        suppression_codes=["SUPP", "xx"],
    )
    assert bundle["warnings"] == []
