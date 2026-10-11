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


def test_full_precision_continuous_column_is_kept():
    n = 200
    df = pd.DataFrame({"measurement": np.random.RandomState(0).normal(size=n)})
    pre = _fit(df)
    assert "measurement" in pre.numeric_columns_
    assert pre.warnings_ == []


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
    return pd.DataFrame({"x": (values * n)[:n], "y": [i % 7 for i in range(n)]})


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


def test_mixed_conventions_parse_values_with_both_separators():
    pre = _fit(_col(["1,234.56", "1.234,56", "12,5", "6", "7"]))
    warning = next(w for w in pre.warnings_ if "mixes" in w)
    assert "becomes 125" in warning and "missing" not in warning
    out = pre.transform(pd.DataFrame({"x": ["1,234.56", "1.234,56", "12,5"]}))["x"].tolist()
    assert out == pytest.approx([1234.56, 1234.56, 125.0])


def test_decimal_comma_range_and_check_inputs_use_feature_units():
    pre = _fit(pd.DataFrame({"c": ["12,5", "3,25", "100,1", "7,0", "8,9"]}))
    assert pre.numeric_ranges_["c"] == pytest.approx((3.25, 100.1))
    assert pre.check_inputs({"c": "5,0"})["warnings"] == []
    warnings = pre.check_inputs({"c": "500,0"})["warnings"]
    assert [w["issue"] for w in warnings] == ["out_of_range"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1.500,50", 1500.5),
        ("980,50", 980.5),
        ("1.500", 1500.0),
        ("1500.5", 1500.5),
        ("1,500.50", 1500.5),
        ("1,500,000", 1500000.0),
        ("0.125", 0.125),
        (1500.5, 1500.5),
        (12, 12.0),
    ],
)
def test_decimal_comma_column_predict_inputs(value, expected):
    pre = _fit(
        pd.DataFrame({"c": ["1.234,56", "2.500,00", "3.100,10", "980,50", "4.000,00"]})
    )
    assert pre.decimal_comma_columns_ == ["c"]
    assert pre.transform({"c": value})["c"].iloc[0] == pytest.approx(expected)


def test_decimal_comma_object_batch_keeps_native_numbers():
    pre = _fit(
        pd.DataFrame({"c": ["1.234,56", "2.500,00", "3.100,10", "980,50", "4.000,00"]})
    )
    batch = pd.DataFrame({"c": pd.Series([1500.5, "1.000,0", 12], dtype=object)})
    assert pre.transform(batch)["c"].tolist() == pytest.approx([1500.5, 1000.0, 12.0])


def test_mixed_object_column_at_training():
    values = pd.Series([1234.5, "1.234,56", 7.25] * 4, dtype=object)
    pre = _fit(pd.DataFrame({"c": values}))
    assert pre.decimal_comma_columns_ == ["c"]
    out = pre.transform(pd.DataFrame({"c": values}))["c"].tolist()[:3]
    assert out == pytest.approx([1234.5, 1234.56, 7.25])
    assert pre.numeric_ranges_["c"] == pytest.approx((7.25, 1234.56))


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


@pytest.mark.parametrize("code", ["SUPP", "NE", "NP", "z", "*", ".."])
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

    check = pre.check_inputs({"x": "NP", "y": 1})
    assert check["missing_features"] == ["x"] and check["warnings"] == []
    check = pre.check_inputs({"x": "500%", "y": 1})
    assert [w["issue"] for w in check["warnings"]] == ["out_of_range"]
    assert pre.check_inputs({"x": "30%", "y": 1}) == {"warnings": [], "missing_features": []}


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
    assert "suppression_codes=['SUPP', 'NE', 'NP', 'z', '*', '..', 'x', 'c']" in warning


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
    numbers = [f"{i}.5" for i in range(100, 160)]
    df = pd.DataFrame({"x": numbers + leftovers, "y": [i % 7 for i in range(60 + len(leftovers))]})
    pre = _fit(df)
    assert "x" in pre.numeric_columns_
    assert pre.warnings_ == []


def _x(values) -> pd.DataFrame:
    return pd.DataFrame({"x": values})


def test_unknown_codes_below_numeric_threshold_warn():
    values = [str(i) for i in range(1, 21)] + ["*", "*", "<5", "<5", "NA", "NA", "S", "S", "—", "—"]
    pre = _fit(_x(values))
    assert pre.categorical_columns_ == ["x"]
    [warning] = pre.warnings_
    assert "below the 80% numeric threshold" in warning
    assert "'<5' x2" in warning and "'—' x2" in warning and "'*' x" not in warning
    assert "suppression_codes=['SUPP', 'NE', 'NP', 'z', '*', '..', '<5', 'NA', 'S', '—']" in warning


def test_unknown_codes_warning_with_custom_codes():
    values = [str(i) for i in range(1, 9)] + ["x", "x", "SUPP", "SUPP", "SUPP"]
    pre = _fit(_x(values), suppression_codes=["x"])
    assert pre.categorical_columns_ == ["x"]
    [warning] = pre.warnings_
    assert "suppression_codes=['x', 'SUPP']" in warning
    retrained = _fit(_x(values), suppression_codes=["x", "SUPP"])
    assert retrained.numeric_columns_ == ["x"] and retrained.warnings_ == []


@pytest.mark.parametrize(
    "values",
    [
        ["a", "b", "c", "d", "1", "2"] * 5,
        ["1", "2", "3"] + ["low", "high"] * 3 + ["mid"],
        ["1", "2", "3", "4", "5", "6", "7", "8", "not applicable", "not applicable", "-", "-"],
    ],
    ids=["mostly_text", "single_token", "long_token"],
)
def test_unknown_codes_warning_is_quiet_for_real_categoricals(values):
    pre = _fit(_x(values))
    assert pre.categorical_columns_ == ["x"]
    assert pre.warnings_ == []


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("−5", -5.0),
        ("−3.2%", -3.2),
        ("USD 100", 100.0),
        ("usd100", 100.0),
        ("Rs. 1,250", 1250.0),
        ("Rs.100", 100.0),
        ("INR 100", 100.0),
        ("100 EUR", 100.0),
        ("-USD 100", -100.0),
        ("(USD 50)", -50.0),
        ("1234-", -1234.0),
        ("1,234.5-", -1234.5),
    ],
)
def test_minus_signs_and_currency_codes_parse(raw, expected):
    values = ["10", "20", "30", "40", raw]
    pre = _fit(_x(values))
    assert pre.numeric_columns_ == ["x"]
    assert pre.transform({"x": raw})["x"].iloc[0] == pytest.approx(expected)


def test_unicode_minus_in_decimal_comma_column():
    values = ["−12,5", "3,25", "100,1", "7,0", "−8,9"]
    pre = _fit(_x(values))
    assert pre.decimal_comma_columns_ == ["x"]
    assert pre.transform(_x(values))["x"].tolist() == pytest.approx(
        [-12.5, 3.25, 100.1, 7.0, -8.9]
    )


@pytest.mark.parametrize("values", [["USD", "EUR", "INR"] * 4, ["Lot 100", "Lot 200", "Lot 300"] * 4])
def test_currency_code_alone_or_other_words_stay_categorical(values):
    assert _fit(_x(values)).categorical_columns_ == ["x"]


def test_infinite_values_are_missing_with_warning():
    values = ["inf", "inf%", "-inf", "1", "2", "3", "4", "5"]
    pre = _fit(_x(values))
    assert pre.numeric_ranges_["x"] == (1.0, 5.0)
    [warning] = pre.warnings_
    assert "3 infinite value(s)" in warning
    out = pre.transform(_x(values))["x"]
    assert out.iloc[:3].isna().all()
    assert np.isnan(pre.transform({"x": float("inf")})["x"].iloc[0])


def test_train_model_with_infinite_values_succeeds():
    from glin import engine

    rng = np.random.RandomState(0)
    x = [f"{v:.2f}%" for v in rng.uniform(0, 100, 120)]
    x[0], x[1] = "inf%", "-inf"
    df = pd.DataFrame({"x": x, "y": rng.choice(["a", "b"], 120)})
    bundle = engine.train_model(df, "y", model_name="inf", hyperparameters={"interactions": 0})
    assert any("2 infinite value(s)" in w for w in bundle["warnings"])
    result = engine.explain_record(bundle, {"x": float("inf")})
    assert [w["issue"] for w in result["warnings"]] == ["unparseable_number"]


def test_native_infinite_values_are_missing():
    pre = _fit(pd.DataFrame({"x": [1.0, 2.0, np.inf, 4.0, -np.inf]}))
    assert pre.numeric_ranges_["x"] == (1.0, 4.0)
    assert pre.transform(pd.DataFrame({"x": [np.inf]}))["x"].isna().all()


@pytest.mark.parametrize("value", ["abc", "inf", float("inf"), float("-inf")])
def test_unparseable_number_at_predict_warns(value):
    import json

    pre = _fit(_x(["10", "20", "30", "40"]))
    check = pre.check_inputs({"x": value})
    assert check["missing_features"] == ["x"]
    assert [w["issue"] for w in check["warnings"]] == ["unparseable_number"]
    json.dumps(check, allow_nan=False)


@pytest.mark.parametrize("value", ["", "  ", None, float("nan"), "SUPP", " z "])
def test_blank_or_code_at_predict_is_missing_without_warning(value):
    pre = _fit(_x(["10", "20", "30", "40"]))
    assert pre.check_inputs({"x": value}) == {"warnings": [], "missing_features": ["x"]}


def test_warning_values_are_json_native():
    import json

    pre = _fit(pd.DataFrame({"x": [1, 2, 3, 4], "c": ["a", "b", "a", "b"]}))
    check = pre.check_inputs({"x": np.int64(9), "c": np.str_("z")})
    assert [w["value"] for w in check["warnings"]] == [9, "z"]
    assert type(check["warnings"][0]["value"]) is int
    json.dumps(check, allow_nan=False)


def test_warning_does_not_fire_on_normal_categoricals():
    pre = _fit(_col(["red", "green", "blue", "N/A?", "red"]))
    assert "x" in pre.categorical_columns_
    assert pre.warnings_ == []


N_ROWS = 563


def _warning_for(pre: EBMTabularPreprocessor, col: str) -> str:
    matches = [w for w in pre.warnings_ if f"'{col}'" in w]
    assert len(matches) == 1, pre.warnings_
    return matches[0]


def test_integer_measurement_with_repeats_is_kept():
    rng = np.random.RandomState(0)
    values = rng.randint(0, 20000, size=N_ROWS)
    assert 0.9 < len(set(values)) / N_ROWS < 1.0
    pre = _fit(pd.DataFrame({"expenditure": values}))
    assert "expenditure" in pre.numeric_columns_
    assert pre.dropped_columns_ == []
    assert pre.warnings_ == []


def test_integer_measurement_with_missing_values_is_kept():
    rng = np.random.RandomState(1)
    values = rng.randint(0, 20000, size=N_ROWS).astype(float)
    values[:10] = np.nan
    pre = _fit(pd.DataFrame({"expenditure": values}))
    assert "expenditure" in pre.numeric_columns_


@pytest.mark.parametrize(
    "values",
    [
        np.arange(1, N_ROWS + 1),
        np.random.RandomState(0).permutation(np.arange(1, N_ROWS + 1)),
        np.arange(0, N_ROWS),
        np.random.RandomState(0).randint(0, 10**9, size=N_ROWS),
    ],
    ids=["sequential", "shuffled", "zero_based", "large_unique"],
)
def test_distinct_integer_column_is_dropped_with_warning(values):
    pre = _fit(pd.DataFrame({"row_id": values, "x": np.arange(N_ROWS) % 7}))
    assert pre.dropped_columns_ == ["row_id"]
    message = _warning_for(pre, "row_id")
    assert "identifier" in message
    assert "round or bin" in message


def test_integer_id_with_missing_values_is_dropped():
    values = np.arange(1, N_ROWS + 1).astype(float)
    values[:5] = np.nan
    pre = _fit(pd.DataFrame({"row_id": values}))
    assert pre.dropped_columns_ == ["row_id"]


@pytest.mark.parametrize(
    "values",
    [
        np.random.default_rng(0).choice(np.arange(20_000, 200_000), N_ROWS, replace=False),
        np.random.default_rng(1).choice(
            np.arange(1_600_000_000, 1_700_000_000), N_ROWS, replace=False
        ),
        np.random.default_rng(2).choice(np.arange(0, 10**7, 100), N_ROWS, replace=False),
    ],
    ids=["salary", "epoch_seconds", "rounded_amount"],
)
def test_unique_integer_measurement_is_kept(values):
    pre = _fit(pd.DataFrame({"salary": values}))
    assert pre.numeric_columns_ == ["salary"]
    assert pre.warnings_ == []


@pytest.mark.parametrize(
    "values",
    [
        np.arange(1, N_ROWS + 1),
        np.random.RandomState(0).permutation(np.arange(1000, 1000 + N_ROWS)),
        np.arange(0, 10 * N_ROWS, 10),
        np.random.RandomState(0).choice(np.arange(1, 2 * N_ROWS), N_ROWS, replace=False),
    ],
    ids=["sequential", "shuffled", "constant_step", "half_filled"],
)
def test_range_filling_integers_are_dropped_without_id_name(values):
    pre = _fit(pd.DataFrame({"number": values}))
    assert pre.dropped_columns_ == ["number"]
    assert "round or bin" in _warning_for(pre, "number")


@pytest.mark.parametrize(
    "name", ["row_id", "PassengerId", "customerID", "ID", "uuid", "order_key", "index"]
)
def test_sparse_unique_integers_with_id_name_are_dropped(name):
    values = np.random.default_rng(0).choice(np.arange(10**9), N_ROWS, replace=False)
    pre = _fit(pd.DataFrame({name: values}))
    assert pre.dropped_columns_ == [name]


@pytest.mark.parametrize("name", ["valid", "paid", "body_mass_index", "kidney"])
def test_id_substring_in_name_is_not_an_id_name(name):
    values = np.random.default_rng(0).choice(np.arange(10**9), N_ROWS, replace=False)
    assert _fit(pd.DataFrame({name: values})).numeric_columns_ == [name]


def test_object_id_with_one_text_value_is_dropped():
    ids = [str(i) for i in range(1, N_ROWS)] + ["TBD"]
    df = pd.DataFrame({"row_id": ids, "x": np.arange(N_ROWS) % 7})
    pre = _fit(df)
    assert pre.dropped_columns_ == ["row_id"]
    assert pre.numeric_columns_ == ["x"]
    assert "identifier" in _warning_for(pre, "row_id")


def test_object_column_of_numeric_strings_uses_integer_id_rule():
    shuffled = np.random.RandomState(0).permutation(N_ROWS).astype(str)
    salaries = np.random.default_rng(0).choice(
        np.arange(20_000, 200_000), N_ROWS, replace=False
    )
    df = pd.DataFrame(
        {"code": shuffled, "salary": [f"${v:,}" for v in salaries[:-1]] + [""]}
    )
    pre = _fit(df)
    assert pre.dropped_columns_ == ["code"]
    assert pre.numeric_columns_ == ["salary"]


def test_uuid_column_is_dropped_with_warning():
    import uuid

    ids = [str(uuid.UUID(int=i * 7919 + 12345)) for i in range(N_ROWS)]
    pre = _fit(pd.DataFrame({"uid": ids}))
    assert pre.dropped_columns_ == ["uid"]
    assert "identifier" in _warning_for(pre, "uid")


def test_constant_column_drop_is_warned():
    pre = _fit(pd.DataFrame({"same": ["a"] * 100, "empty": [np.nan] * 100}))
    assert set(pre.dropped_columns_) == {"same", "empty"}
    assert "constant" in _warning_for(pre, "same")
    assert "constant" in _warning_for(pre, "empty")


def test_high_cardinality_drop_is_warned():
    values = [f"note-{i}" for i in range(260)] * 2
    pre = _fit(pd.DataFrame({"remarks": values}))
    message = _warning_for(pre, "remarks")
    assert "260" in message and "250" in message


def test_dropped_date_column_is_warned():
    pre = _fit(pd.DataFrame({"when": ["2024-03-06"] * 10 + ["2024-03-06 "] * 10}))
    assert pre.dropped_columns_ == ["when"]
    assert "date" in _warning_for(pre, "when")


def test_kept_columns_produce_no_drop_warnings():
    df = pd.DataFrame({"age": list(range(20, 70)) * 2, "plan": ["a", "b"] * 50})
    assert _fit(df).warnings_ == []


def test_train_model_forwards_drop_warnings_once():
    from glin import engine

    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "row_id": np.arange(200),
            "same": ["a"] * 200,
            "spend": rng.randint(0, 5000, size=200),
            "x": rng.normal(size=200).round(1),
        }
    )
    df["y"] = np.where(df["x"] + rng.normal(size=200) > 0, "yes", "no")
    bundle = engine.train_model(df, "y", model_name="drops")
    for col in ("row_id", "same"):
        assert sum(f"'{col}'" in w for w in bundle["warnings"]) == 1
    assert not any("'spend'" in w for w in bundle["warnings"])


def _int_named_frame(n=200):
    rng = np.random.RandomState(0)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({0: dates.strftime("%Y-%m-%d"), 1: rng.normal(size=n), 3: rng.choice(["a", "b", "c"], n)})


def test_integer_column_names_with_a_date_column_fit_and_transform():
    df = _int_named_frame()
    pre = _fit(df)

    out = pre.transform(df)

    assert {"0__month", "0__days_since_start", "1", "3"} <= set(out.columns)
    assert len(out) == len(df)


def test_check_inputs_accepts_integer_or_string_keys():
    pre = _fit(_int_named_frame())

    by_int = pre.check_inputs({0: "2030-01-01", 1: 99.0, 3: "zzz"})
    by_str = pre.check_inputs({"0": "2030-01-01", "1": 99.0, "3": "zzz"})

    assert by_int == by_str
    assert {(w["feature"], w["issue"]) for w in by_int["warnings"]} == {
        ("0", "out_of_range"), ("1", "out_of_range"), ("3", "unseen_category")
    }


def test_column_names_that_collide_as_text_are_rejected():
    df = pd.DataFrame({0: [1.0, 2.0, 3.0], "0": [4.0, 5.0, 6.0]})

    with pytest.raises(ValueError, match="unique as text"):
        _fit(df)
