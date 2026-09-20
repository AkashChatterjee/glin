import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from glin.preprocessor import EBMTabularPreprocessor


def _fit(df: pd.DataFrame, **kwargs) -> EBMTabularPreprocessor:
    return EBMTabularPreprocessor(**kwargs).fit(df)


def test_drops_native_numeric_id_column():
    n = 200
    df = pd.DataFrame(
        {
            "passenger_id": range(1, n + 1),
            "age": np.random.RandomState(0).randint(1, 90, size=n).astype(float),
        }
    )
    pre = _fit(df)
    assert "passenger_id" in pre.dropped_columns_
    assert "age" in pre.numeric_columns_


def test_full_precision_continuous_column_is_dropped_as_id_like():
    """Known, intentional limitation: a genuinely continuous numeric column
    with essentially no repeated values (e.g. raw high-precision floats, as
    opposed to real-world CSV data which almost always carries some rounding)
    is indistinguishable from a sequential identifier by a pure uniqueness
    check, and is dropped per the PRD's literal id-column rule. Real CSV
    numeric columns (currency, ages, measurements) virtually always have
    enough rounding to produce natural repeats and stay well under the
    threshold — see test_dirty_numeric_object_column_survives_high_cardinality
    for the case this is meant to handle."""
    n = 200
    df = pd.DataFrame({"measurement": np.random.RandomState(0).normal(size=n)})
    pre = _fit(df)
    assert "measurement" in pre.dropped_columns_


def test_drops_string_id_column():
    n = 200
    df = pd.DataFrame(
        {
            "customer_id": [f"cust-{i}" for i in range(n)],
            "plan": np.random.RandomState(0).choice(["a", "b"], size=n),
        }
    )
    pre = _fit(df)
    assert "customer_id" in pre.dropped_columns_
    assert "plan" in pre.categorical_columns_


def test_drops_high_cardinality_free_text():
    # 260 distinct free-text-ish values, each repeated twice -> unique_ratio
    # is only 0.5 (well under the 0.95 id-ratio rule), isolating the separate
    # cardinality>250 rule specifically.
    values = [f"note-{i}" for i in range(260)] * 2
    df = pd.DataFrame({"remarks": values})
    pre = _fit(df)
    assert "remarks" in pre.dropped_columns_


def test_drops_constant_and_all_null_columns():
    n = 100
    df = pd.DataFrame(
        {
            "constant": ["same"] * n,
            "all_null": [np.nan] * n,
            "age": list(range(20, 70)) * 2,  # repeated values, not a sequential id
        }
    )
    pre = _fit(df)
    assert "constant" in pre.dropped_columns_
    assert "all_null" in pre.dropped_columns_
    assert "age" in pre.numeric_columns_


def test_dirty_numeric_object_column_survives_high_cardinality():
    """Regression test: a numeric-but-object-dtype column (blank rows force
    object dtype, e.g. Telco's TotalCharges) must NOT be dropped as an ID or
    as free text just because nearly every value is unique — EBM bins
    continuous values regardless of cardinality."""
    n = 300
    rng = np.random.RandomState(0)
    values = (rng.normal(500, 100, size=n)).round(2).astype(str)
    values[:5] = " "  # blank rows, e.g. new customers with no charges yet
    df = pd.DataFrame({"total_charges": values, "age": rng.randint(18, 80, size=n)})

    pre = _fit(df)
    assert "total_charges" in pre.numeric_columns_
    assert "total_charges" not in pre.dropped_columns_

    transformed = pre.transform(df)
    assert transformed["total_charges"].dtype == np.float64
    assert transformed["total_charges"].isna().sum() == 5


def test_numeric_coercion_threshold_rejects_mostly_non_numeric_column():
    n = 300
    rng = np.random.RandomState(0)
    # Only ~20% parseable as numeric, and low cardinality (5 repeated
    # non-numeric labels) -> should land as categorical, not numeric or dropped.
    values = [
        str(rng.uniform(0, 1)) if i % 5 == 0 else f"label-{i % 5}" for i in range(n)
    ]
    df = pd.DataFrame({"mixed": values})
    pre = _fit(df)
    assert "mixed" in pre.categorical_columns_


def test_categorical_standardization_and_missing_token():
    df = pd.DataFrame({"plan": [" Basic", "premium", None, "PREMIUM", np.nan]})
    pre = _fit(df, id_column_threshold=0.99, min_rows_for_id_check=0)
    transformed = pre.transform(df)
    assert set(transformed["plan"]) == {"basic", "premium", "__missing__"}
    assert "__missing__" in pre.categorical_vocab_["plan"]


def test_transform_dict_matches_transform_dataframe():
    df = pd.DataFrame(
        {
            "age": [22.0, 38.0, 26.0, 35.0, 40.0] * 20,
            "plan": ["basic", "premium", "basic", "enterprise", "premium"] * 20,
        }
    )
    pre = _fit(df)

    record = {"age": 30.0, "plan": "premium"}
    from_dict = pre.transform(record)
    from_frame = pre.transform(pd.DataFrame([record]))

    pd.testing.assert_frame_equal(
        from_dict.reset_index(drop=True), from_frame.reset_index(drop=True)
    )


def test_unseen_category_maps_to_missing_token():
    df = pd.DataFrame({"plan": ["basic", "premium"] * 50})
    pre = _fit(df)
    out = pre.transform({"plan": "totally_new_plan"})
    assert out["plan"].iloc[0] == "__missing__"


def test_missing_column_at_inference_fills_defaults():
    df = pd.DataFrame(
        {
            "age": [22.0, 38.0, 26.0] * 20,
            "plan": ["basic", "premium", "enterprise"] * 20,
        }
    )
    pre = _fit(df)
    out = pre.transform({"age": 40.0})  # 'plan' entirely omitted
    assert out["plan"].iloc[0] == "__missing__"
    assert out["age"].iloc[0] == 40.0

    out2 = pre.transform({"plan": "basic"})  # 'age' entirely omitted
    assert pd.isna(out2["age"].iloc[0])


def test_extra_unknown_keys_are_dropped_silently():
    df = pd.DataFrame({"age": [22.0, 38.0, 26.0] * 20})
    pre = _fit(df)
    out = pre.transform({"age": 30.0, "some_field_the_model_never_saw": "x"})
    assert list(out.columns) == ["age"]


def test_sklearn_clone_and_get_params_round_trip():
    pre = EBMTabularPreprocessor(id_column_threshold=0.9)
    cloned = clone(pre)
    assert cloned.get_params() == pre.get_params()

    df = pd.DataFrame({"age": range(100), "plan": ["a", "b"] * 50})
    pre.fit(df)
    # fitted state (trailing underscore attrs) must not leak into get_params
    assert "dropped_columns_" not in pre.get_params()


def test_missing_target_column_style_usage_preserves_native_nan():
    """Numeric columns keep native NaN rather than being imputed — EBM bins
    missing values into their own split natively."""
    df = pd.DataFrame({"age": [22.0, np.nan, 26.0] * 20})
    pre = _fit(df)
    out = pre.transform(df)
    assert out["age"].isna().sum() == df["age"].isna().sum()
