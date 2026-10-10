import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from glin import engine
from glin.preprocessor import CONTINUOUS, NOMINAL, EBMTabularPreprocessor


def _fit(values, name="d", **kwargs) -> EBMTabularPreprocessor:
    return EBMTabularPreprocessor(**kwargs).fit(pd.DataFrame({name: values}))


def _daily(n=120, start="2021-01-01", fmt="%Y-%m-%d") -> list[str]:
    return list(pd.date_range(start, periods=n, freq="D").strftime(fmt))


def test_iso_dates_become_named_features_with_types():
    pre = _fit(_daily(n=800))
    assert pre.date_columns_ == ["d"]
    assert pre.output_columns_ == [
        "d__year",
        "d__month",
        "d__day_of_week",
        "d__days_since_start",
    ]
    assert pre.feature_types_["d__year"] == CONTINUOUS
    assert pre.feature_types_["d__days_since_start"] == CONTINUOUS
    assert pre.feature_types_["d__month"] == NOMINAL
    assert pre.feature_types_["d__day_of_week"] == NOMINAL
    assert pre.dropped_columns_ == []
    assert pre.get_feature_types() == [pre.feature_types_[c] for c in pre.output_columns_]


def test_unique_per_row_date_is_not_dropped_as_an_id():
    df = pd.DataFrame({"d": _daily(n=300), "x": np.arange(300) % 7})
    pre = EBMTabularPreprocessor().fit(df)
    assert "d" not in pre.dropped_columns_
    assert "d__days_since_start" in pre.output_columns_


@pytest.mark.parametrize(
    "values",
    [
        ["2020-01-31", "31/01/2020", "Jan 31, 2020", "31 Jan 2020", "2020-01-31 10:15:00"] * 4,
        ["01/31/2020", "02/01/2020", "03/04/2020", "12/12/2021"] * 5,
        ["2020-01-31T10:15:00Z", "2020-02-01T23:59:59+05:30"] * 10,
    ],
)
def test_common_date_formats_are_detected(values):
    assert _fit(values).date_columns_ == ["d"]


def test_known_date_derives_expected_values():
    pre = _fit(_daily(n=800))
    row = pre.transform({"d": "2021-03-06"}).iloc[0]
    assert row["d__year"] == 2021
    assert row["d__month"] == "mar"
    assert row["d__day_of_week"] == "sat"
    assert row["d__days_since_start"] == 64  # 2021-01-01 is day 0


def test_plain_integer_columns_are_never_dates():
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "year": rng.randint(1990, 2020, 200),
            "epoch": rng.randint(1_600_000_000, 1_700_000_000, 200),
            "yyyymmdd_int": rng.choice([20200101, 20200215, 20210301, 20220701], 200),
            "yyyymmdd_str": rng.choice(["20200101", "20200215", "20210301", "20220701"], 200),
            "year_str": rng.choice(["2019", "2020", "2021"], 200),
        }
    )
    pre = EBMTabularPreprocessor().fit(df)
    assert pre.date_columns_ == []
    assert {"year", "yyyymmdd_int", "yyyymmdd_str", "year_str"} <= set(pre.numeric_columns_)
    assert "year" in pre.numeric_columns_


@pytest.mark.parametrize(
    "values",
    [
        ["Jan", "Feb", "Mar", "May"] * 10,
        ["1.2.3", "1.10.0", "2.0.1", "10.4.2"] * 10,
        ["Q1 2020", "Q2 2020", "Q3 2020"] * 10,
        ["Dec 2020", "Jan 2021", "Feb 2021"] * 10,
        ["3-4-5", "9-8-7", "1-1-1"] * 10,
    ],
)
def test_date_lookalikes_stay_categorical(values):
    pre = _fit(values)
    assert pre.date_columns_ == []
    assert pre.categorical_columns_ == ["d"]


def test_mostly_non_date_strings_are_not_dates():
    pre = _fit(["2020-01-01", "2020-01-02"] + ["pending"] * 18)
    assert pre.date_columns_ == []


def test_constant_parts_are_dropped():
    # One year, date-only: no year and no hour.
    pre = _fit(_daily(n=100, start="2021-01-01"))
    assert "d__year" not in pre.output_columns_
    assert "d__hour" not in pre.output_columns_

    # Time of day varies: hour kept.
    stamps = list(pd.date_range("2021-01-01", periods=100, freq="5h").strftime("%Y-%m-%d %H:%M:%S"))
    assert "d__hour" in _fit(stamps).output_columns_

    # One calendar day, only the time varies.
    same_day = list(pd.date_range("2021-03-06", periods=24, freq="h").strftime("%Y-%m-%d %H:%M"))
    assert _fit(same_day).output_columns_ == ["d__hour"]


def test_constant_date_column_is_dropped():
    pre = _fit(["2021-03-06"] * 30)
    assert pre.dropped_columns_ == ["d"]
    assert pre.output_columns_ == []


def test_day_first_inferred_from_values_above_12():
    pre = _fit(["13/01/2020", "02/03/2020", "25/12/2020", "05/06/2020"] * 5)
    assert pre.date_dayfirst_["d"] is True
    assert pre.warnings_ == []
    out = pre.transform({"d": "02/03/2020"}).iloc[0]
    assert out["d__month"] == "mar"


def test_month_first_inferred_from_second_field_above_12():
    pre = _fit(["01/13/2020", "02/03/2020", "12/25/2020"] * 7)
    assert pre.date_dayfirst_["d"] is False
    assert pre.warnings_ == []
    assert pre.transform({"d": "02/03/2020"}).iloc[0]["d__month"] == "feb"


def test_ambiguous_day_month_order_warns_and_assumes_month_first():
    pre = _fit(["01/02/2020", "03/04/2020", "05/06/2020", "07/08/2020"] * 5)
    assert pre.date_dayfirst_["d"] is False
    assert len(pre.warnings_) == 1 and "ambiguous" in pre.warnings_[0]
    assert pre.transform({"d": "01/02/2020"}).iloc[0]["d__month"] == "jan"


def test_symmetric_slash_dates_are_not_flagged_ambiguous():
    pre = _fit(["01/01/2020", "02/02/2020", "03/03/2021", "12/12/2021"] * 5)
    assert pre.date_columns_ == ["d"]
    assert pre.warnings_ == []


def test_inconsistent_day_month_order_warns():
    pre = _fit(["13/01/2020", "01/13/2020", "02/03/2020", "04/05/2020"] * 5)
    assert any("mixes" in w for w in pre.warnings_)


def test_iso_dates_unaffected_by_day_first_columns():
    pre = _fit(["13/01/2020", "2020-03-04", "25/12/2020", "05/06/2020"] * 5)
    assert pre.date_dayfirst_["d"] is True
    assert pre.transform({"d": "2020-03-04"}).iloc[0]["d__month"] == "mar"


def test_unparseable_stragglers_become_missing_with_warning():
    pre = _fit(_daily(n=50, start="2020-01-01") + ["not a date", "n/a"])
    assert pre.date_columns_ == ["d"]
    assert any("not parseable" in w for w in pre.warnings_)
    out = pre.transform({"d": "not a date"}).iloc[0]
    assert out["d__month"] == pre.missing_token
    assert np.isnan(out["d__days_since_start"])


def test_native_datetime_and_tz_aware_inputs():
    stamps = pd.date_range("2021-01-01", periods=60, freq="D")
    pre = EBMTabularPreprocessor().fit(pd.DataFrame({"d": stamps}))
    assert pre.date_columns_ == ["d"]
    out = pre.transform({"d": pd.Timestamp("2021-02-03 04:05", tz="UTC")}).iloc[0]
    assert out["d__month"] == "feb" and out["d__days_since_start"] == 33
    aware = pre.transform({"d": pd.Timestamp("2021-02-03 00:30", tz="Asia/Kolkata")}).iloc[0]
    assert aware["d__days_since_start"] == 32  # converted to UTC: 2021-02-02 19:00


def test_batch_and_single_record_agree():
    train = pd.DataFrame({"d": _daily(n=800), "x": np.arange(800) % 5})
    pre = EBMTabularPreprocessor().fit(train)
    batch_in = pd.DataFrame({"d": ["2021-03-06", "2022-07-15", None, "garbage"], "x": [1, 2, 3, 4]})
    batch = pre.transform(batch_in)
    for i, record in enumerate(batch_in.to_dict("records")):
        single = pre.transform(record)
        pd.testing.assert_frame_equal(single.reset_index(drop=True), batch.iloc[[i]].reset_index(drop=True))


def test_missing_date_column_at_predict_time_is_missing_not_error():
    pre = EBMTabularPreprocessor().fit(pd.DataFrame({"d": _daily(n=800), "x": np.arange(800) % 5}))
    out = pre.transform({"x": 1}).iloc[0]
    assert out["d__month"] == pre.missing_token
    assert np.isnan(out["d__year"])
    assert pre.check_inputs({"x": 1})["missing_features"] == ["d"]


def test_check_inputs_for_dates():
    pre = _fit(_daily(n=800, start="2021-01-01"))
    ok = pre.check_inputs({"d": "2021-06-01"})
    assert ok == {"warnings": [], "missing_features": []}

    blank = pre.check_inputs({"d": None})
    assert blank == {"warnings": [], "missing_features": ["d"]}
    assert pre.check_inputs({"d": "  "}) == {"warnings": [], "missing_features": ["d"]}

    bad = pre.check_inputs({"d": "next tuesday"})
    assert bad["missing_features"] == ["d"]
    assert [(w["feature"], w["issue"]) for w in bad["warnings"]] == [("d", "unparseable_date")]

    numeric = pre.check_inputs({"d": 1700000000})
    assert numeric["missing_features"] == ["d"]
    assert numeric["warnings"][0]["issue"] == "unparseable_date"

    late = pre.check_inputs({"d": "2030-01-01"})
    assert late["missing_features"] == []
    assert [(w["feature"], w["issue"]) for w in late["warnings"]] == [("d", "out_of_range")]
    assert late["warnings"][0]["training_range"] == ["2021-01-01", "2023-03-11"]
    assert [w["issue"] for w in pre.check_inputs({"d": "2020-12-31"})["warnings"]] == ["out_of_range"]


def test_unseen_month_is_flagged():
    dates = [d for d in _daily(n=800, start="2021-01-01") if d[5:7] != "03"]
    pre = _fit(dates)
    result = pre.check_inputs({"d": "2022-03-10"})
    assert [(w["feature"], w["issue"]) for w in result["warnings"]] == [
        ("d__month", "unseen_category")
    ]


def test_clone_round_trip_keeps_date_threshold():
    pre = EBMTabularPreprocessor(date_parse_threshold=0.5)
    assert clone(pre).get_params()["date_parse_threshold"] == 0.5


def _weekday_df(n=500, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.randint(0, 700, n), unit="D")
    label = np.where(dates.dayofweek >= 5, "weekend_signup", "weekday_signup")
    flip = rng.rand(n) < 0.04
    label = np.where(flip, np.where(label == "weekend_signup", "weekday_signup", "weekend_signup"), label)
    return pd.DataFrame(
        {
            "signup_date": dates.strftime("%Y-%m-%d"),
            "noise": np.round(rng.normal(size=n), 1),
            "label": label,
        }
    )


def test_model_finds_signal_that_lives_only_in_the_weekday():
    df = _weekday_df()
    bundle = engine.train_model(
        df, "label", model_name="weekday", hyperparameters={"interactions": 0}, holdout_fraction=0.25
    )
    ev = bundle["evaluation"]
    assert ev["roc_auc"] > 0.9
    assert ev["log_loss"] < 0.8 * ev["baseline_log_loss"]

    imp = engine.global_importance(bundle)
    assert imp["features"][0]["feature"] == "signup_date__day_of_week"
    shape = next(t for t in imp["terms"] if t["term"] == "signup_date__day_of_week")["shape"]
    assert {c["category"] for c in shape["highest"][:2]} == {"sat", "sun"}


def test_month_only_signal():
    rng = np.random.RandomState(1)
    dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.randint(0, 730, 600), unit="D")
    hot = dates.month.isin([6, 7, 8])
    label = np.where(hot ^ (rng.rand(600) < 0.05), "summer", "other")
    df = pd.DataFrame({"when": dates.strftime("%d %b %Y"), "label": label})
    bundle = engine.train_model(
        df, "label", model_name="month", hyperparameters={"interactions": 0}, holdout_fraction=0.25
    )
    assert "when__month" in bundle["preprocessor"].output_columns_
    assert engine.global_importance(bundle)["features"][0]["feature"] == "when__month"
    ev = bundle["evaluation"]
    assert ev["roc_auc"] > 0.8
    assert ev["log_loss"] < 0.9 * ev["baseline_log_loss"]


def test_additivity_and_named_terms_with_date_features():
    df = _weekday_df()
    bundle = engine.train_model(df, "label", model_name="dates_add", hyperparameters={"interactions": 3})
    raw = df.drop(columns=["label"])
    proba = bundle["model"].predict_proba(bundle["preprocessor"].transform(raw))

    for i in range(15):
        explanation = engine.explain_record(bundle, raw.iloc[i].to_dict(), top_n=50)
        assert explanation["additivity_verified"]
        reconstructed = explanation["reconstructed_probability"][str(bundle["target_classes"][1])]
        assert abs(reconstructed - proba[i][1]) < 1e-3
        features = {c["feature"] for c in explanation["contributions"]}
        assert any(f.startswith("signup_date__") for f in features)
        assert "signup_date" not in features
        assert explanation["warnings"] == [] and explanation["missing_features"] == []


def test_train_to_predict_parity_with_raw_date_strings():
    df = _weekday_df()
    bundle = engine.train_model(df, "label", model_name="parity", hyperparameters={"interactions": 0})
    raw = df.drop(columns=["label"])
    batch = bundle["model"].predict_proba(bundle["preprocessor"].transform(raw))
    for i in (0, 7, 42):
        out = engine.predict_record(bundle, raw.iloc[i].to_dict())
        classes = bundle["target_classes"]
        assert out["probabilities"][str(classes[1])] == pytest.approx(batch[i][1], abs=1e-9)


def test_contribution_values_for_derived_terms():
    df = _weekday_df()
    bundle = engine.train_model(df, "label", model_name="vals", hyperparameters={"interactions": 0})
    explanation = engine.explain_record(bundle, {"signup_date": "2021-03-06", "noise": 0.0}, top_n=50)
    by_name = {c["feature"]: c["value"] for c in explanation["contributions"]}
    assert by_name["signup_date__day_of_week"] == "sat"
    assert by_name["signup_date__month"] == "mar"
    assert by_name["signup_date__days_since_start"] == (
        pd.Timestamp("2021-03-06") - pd.Timestamp(bundle["preprocessor"].date_references_["signup_date"])
    ).days
    assert by_name["noise"] == 0.0


def test_predict_with_bad_or_missing_dates_still_explains_exactly():
    df = _weekday_df()
    bundle = engine.train_model(df, "label", model_name="bad", hyperparameters={"interactions": 0})

    out = engine.explain_record(bundle, {"signup_date": "whenever", "noise": 0.1})
    assert out["additivity_verified"]
    assert out["missing_features"] == ["signup_date"]
    assert out["warnings"][0]["issue"] == "unparseable_date"

    out = engine.explain_record(bundle, {"noise": 0.1})
    assert out["additivity_verified"]
    assert out["missing_features"] == ["signup_date"] and out["warnings"] == []

    out = engine.explain_record(bundle, {"signup_date": "2035-05-05", "noise": 0.1})
    assert out["additivity_verified"]
    assert [w["issue"] for w in out["warnings"]] == ["out_of_range"]


def test_date_only_dataset_trains_and_bundle_metadata_lists_date_features(tmp_path):
    df = _weekday_df().drop(columns=["noise"])
    bundle = engine.train_model(df, "label", model_name="meta", hyperparameters={"interactions": 0})
    engine.save_bundle(bundle, tmp_path)
    meta = json.loads((tmp_path / "meta" / engine.METADATA_FILENAME).read_text())
    date = meta["features"]["date"]["signup_date"]
    assert date["features"] == bundle["preprocessor"].date_feature_names_["signup_date"]
    assert date["start_date"] == bundle["preprocessor"].date_references_["signup_date"].date().isoformat()
    assert meta["features"]["numeric"] == [] and meta["features"]["categorical"] == []
    assert "signup_date" not in meta["dropped_columns"]

    reloaded = engine.load_bundle(tmp_path / "meta")
    out = engine.explain_record(reloaded, {"signup_date": "2021-03-06"})
    assert out["additivity_verified"]


def test_pandas_dtypes_do_not_break_wide_frames():
    df = pd.DataFrame(
        {
            "created": _daily(n=60),
            "flag": [True, False] * 30,
            "name": ["a", "b", "c"] * 20,
        }
    )
    pre = EBMTabularPreprocessor().fit(df)
    out = pre.transform(df)
    assert list(out.columns) == pre.output_columns_
    assert not out.isna().any().any()
