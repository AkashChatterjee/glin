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


def _col(values, n=60):
    return pd.DataFrame({"x": (values * n)[:n], "y": range(n)})


@pytest.mark.parametrize(
    "values,expected",
    [
        (["$1,234.56", "$10.00", "$2,000"], [1234.56, 10.0, 2000.0]),
        (["£1,234", "£99", "£1,000,000"], [1234.0, 99.0, 1000000.0]),
        (["€1.234,56", "€10,50", "€2.000,00"], [1234.56, 10.5, 2000.0]),
        (["(1,234.50)", "$5", "$6"], [-1234.5, 5.0, 6.0]),
    ],
)
def test_currency_and_separator_formats_become_numeric(values, expected):
    pre = _fit(_col(values))
    assert "x" in pre.numeric_columns_
    out = pre.transform(_col(values))["x"].tolist()[: len(values)]
    assert out == pytest.approx(expected)
    assert pre.warnings_ == []


def test_european_decimal_comma_detected_per_column():
    pre = _fit(_col(["12,5", "3,25", "100"]))
    assert pre.decimal_comma_columns_ == ["x"]
    assert pre.transform({"x": "12,5"})["x"].iloc[0] == pytest.approx(12.5)


def test_ambiguous_single_group_is_treated_as_thousands():
    pre = _fit(_col(["1,234", "2,500", "3,999"]))
    assert pre.decimal_comma_columns_ == []
    assert pre.transform({"x": "1,234"})["x"].iloc[0] == 1234.0


def test_mixed_conventions_warn_and_fall_back_to_us():
    pre = _fit(_col(["1,234.56", "1.234,56", "5", "6", "7"]))
    assert pre.decimal_comma_columns_ == []
    assert any("mixes" in w for w in pre.warnings_)


@pytest.mark.parametrize(
    "values,expected",
    [
        (["44.98%", "12%", "100 %"], [44.98, 12.0, 100.0]),
        (["1,234.5%", "(5%)", "7%"], [1234.5, -5.0, 7.0]),
        (["12,5%", "3,25%", "100%"], [12.5, 3.25, 100.0]),
        (["$1,200", "50%", "7"], [1200.0, 50.0, 7.0]),
    ],
)
def test_percent_strings_become_numeric_in_percentage_points(values, expected):
    pre = _fit(_col(values))
    assert "x" in pre.numeric_columns_
    out = pre.transform(_col(values))["x"].tolist()[: len(values)]
    assert out == pytest.approx(expected)
    assert pre.warnings_ == []


def test_percent_only_stripped_when_trailing():
    pre = _fit(_col(["5", "6", "7%"]))
    assert np.isnan(pre.transform({"x": "%5"})["x"].iloc[0])


@pytest.mark.parametrize("code", ["SUPP", "NE", "NP", "z"])
def test_each_default_suppression_code_is_missing_not_categorical(code):
    values = ["10", "20.5", "30", code]
    pre = _fit(_col(values))
    assert "x" in pre.numeric_columns_
    out = pre.transform(_col(values))["x"].tolist()[:4]
    assert out[:3] == pytest.approx([10, 20.5, 30])
    assert np.isnan(out[3])
    assert pre.warnings_ == []


def test_codes_match_case_insensitively_ignoring_whitespace():
    values = ["10", "20", " supp ", "Ne", "Z", "np"]
    pre = _fit(_col(values))
    out = pre.transform(_col(values))["x"].tolist()[:6]
    assert out[:2] == pytest.approx([10, 20])
    assert all(np.isnan(v) for v in out[2:])


def test_mostly_suppressed_column_stays_numeric_and_is_not_dropped():
    n = 200
    values = [str(i % 40) if i % 4 == 0 else "SUPP" for i in range(n)]
    df = pd.DataFrame({"x": values, "y": np.arange(n) % 7})
    pre = _fit(df)
    assert "x" in pre.numeric_columns_
    assert pre.dropped_columns_ == []


def test_mixed_percent_and_suppression_codes():
    values = ["44.98%", "SUPP", "12.5%", "NE", "z", "NP", "60%"]
    pre = _fit(_col(values))
    assert "x" in pre.numeric_columns_
    out = pre.transform(_col(values))["x"].tolist()[:7]
    assert [out[0], out[2], out[6]] == pytest.approx([44.98, 12.5, 60.0])
    assert all(np.isnan(out[i]) for i in (1, 3, 4, 5))
    assert pre.warnings_ == []


def test_percent_and_codes_predict_time_parity():
    pre = _fit(_col(["44.98%", "SUPP", "12.5%", "NE", "60%"]))
    batch = pre.transform(pd.DataFrame({"x": ["44.98%", "SUPP", "z"]}))["x"]
    assert batch.iloc[0] == pytest.approx(44.98)
    assert batch.iloc[1:].isna().all()
    assert pre.transform({"x": "44.98%"})["x"].iloc[0] == pytest.approx(44.98)
    assert np.isnan(pre.transform({"x": "SUPP"})["x"].iloc[0])

    check = pre.check_inputs({"x": "NP"})
    assert check["missing_features"] == ["x"] and check["warnings"] == []
    check = pre.check_inputs({"x": "500%"})
    assert [w["issue"] for w in check["warnings"]] == ["out_of_range"]
    assert pre.check_inputs({"x": "30%"}) == {"warnings": [], "missing_features": []}


def test_custom_suppression_codes_replace_defaults():
    values = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "[c]", "SUPP"]
    pre = _fit(_col(values), suppression_codes=["[c]"])
    out = pre.transform(_col(values))["x"].tolist()[:11]
    assert out[:3] == pytest.approx([10, 20, 30])
    assert np.isnan(out[9]) and np.isnan(out[10])
    assert "('SUPP' x" in pre.warnings_[0]
    assert clone(pre).suppression_codes == ["[c]"]


def test_empty_suppression_codes_disable_recognition():
    pre = _fit(_col(["a", "b", "c", "NE"]), suppression_codes=())
    assert "x" in pre.categorical_columns_


def test_categorical_column_keeps_codes_as_values():
    pre = _fit(_col(["NE", "NW", "SE", "SW"]))
    assert "x" in pre.categorical_columns_
    assert pre.warnings_ == []
    assert pre.transform({"x": "ne"})["x"].iloc[0] == "ne"


def test_unrecognized_repeated_tokens_in_numeric_column_warn():
    values = [str(i) for i in range(10, 100, 5)] + ["x", "x", "c"]
    pre = _fit(_col(values, n=len(values) * 5))
    assert "x" in pre.numeric_columns_
    [warning] = pre.warnings_
    assert "'x'" in warning and "'c'" in warning
    assert "suppression_codes=['SUPP', 'NE', 'NP', 'z', 'x', 'c']" in warning


def test_unrecognized_code_warning_is_quiet_when_codes_are_known():
    values = ["10", "20", "30", "40", "50", "SUPP", "SUPP", "NE", "NE"]
    assert _fit(_col(values, n=90)).warnings_ == []
    values = ["10", "20", "30", "40", "50", "x", "x", "x"]
    assert _fit(_col(values, n=80), suppression_codes=["x"]).warnings_ == []


@pytest.mark.parametrize(
    "leftovers",
    [
        ["oops"],
        ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"],
        ["not applicable", "not applicable"],
    ],
)
def test_unrecognized_code_warning_ignores_non_code_like_leftovers(leftovers):
    numbers = [str(i) for i in range(100, 160)]
    df = pd.DataFrame({"x": numbers + leftovers, "y": range(60 + len(leftovers))})
    pre = _fit(df)
    assert "x" in pre.numeric_columns_
    assert pre.warnings_ == []


def test_warning_does_not_fire_on_normal_categoricals():
    pre = _fit(_col(["red", "green", "blue", "N/A?", "red"]))
    assert "x" in pre.categorical_columns_
    assert pre.warnings_ == []
