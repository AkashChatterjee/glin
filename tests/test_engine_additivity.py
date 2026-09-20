import numpy as np
import pandas as pd
import pytest

from glin import engine


def _synthetic_binary_df(n=400, seed=0) -> pd.DataFrame:
    # Rounded to a realistic CSV precision: a raw full-precision float column
    # has a unique-value ratio of ~1.0 by construction (float collisions are
    # essentially impossible), which would trip the id-column drop rule even
    # though it's a perfectly ordinary continuous feature. Real-world CSV
    # columns (currency, ages, measurements) always carry some rounding that
    # naturally keeps their uniqueness ratio well under the drop threshold.
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "num1": np.round(rng.normal(size=n), 1),
            "num2": np.round(rng.normal(size=n), 1),
            "cat1": rng.choice(["a", "b", "c"], size=n),
        }
    )
    score = df["num1"] + (df["cat1"] == "a").astype(float) * 2 + rng.normal(scale=0.1, size=n)
    df["label"] = np.where(score > 0.5, "positive", "negative")
    return df


def _synthetic_multiclass_df(n=500, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "num1": np.round(rng.normal(size=n), 1),
            "cat1": rng.choice(["a", "b", "c"], size=n),
        }
    )
    score = df["num1"] + (df["cat1"] == "a").astype(float) * 2
    df["label"] = pd.cut(score, bins=3, labels=["low", "mid", "high"]).astype(str)
    return df


def test_binary_additivity_matches_predict_proba():
    df = _synthetic_binary_df()
    bundle = engine.train_model(df, "label", model_name="binary_test", interactions=3)

    X = bundle["preprocessor"].transform(df.drop(columns=["label"]))
    proba = bundle["model"].predict_proba(X)

    for i in range(20):
        record = df.drop(columns=["label"]).iloc[i].to_dict()
        explanation = engine.explain_record(bundle, record)
        assert explanation["additivity_verified"]
        assert abs(explanation["reconstructed_probability"] - proba[i][1]) < 1e-3


def test_binary_direction_labels_match_score_sign():
    df = _synthetic_binary_df()
    bundle = engine.train_model(df, "label", model_name="binary_direction_test", interactions=3)
    target_classes = bundle["target_classes"]

    record = df.drop(columns=["label"]).iloc[0].to_dict()
    explanation = engine.explain_record(bundle, record)
    for c in explanation["contributions"]:
        if c["score"] > 0:
            assert c["direction"] == f"favors {target_classes[1]}"
        else:
            assert c["direction"] == f"favors {target_classes[0]}"


def test_multiclass_additivity_matches_predict_proba_via_softmax():
    df = _synthetic_multiclass_df()
    bundle = engine.train_model(df, "label", model_name="multiclass_test", interactions=0)

    X = bundle["preprocessor"].transform(df.drop(columns=["label"]))
    proba = bundle["model"].predict_proba(X)

    for i in range(20):
        record = df.drop(columns=["label"]).iloc[i].to_dict()
        explanation = engine.explain_record(bundle, record)
        assert explanation["additivity_verified"]
        reconstructed = explanation["reconstructed_probability"]
        for k, class_label in enumerate(bundle["target_classes"]):
            assert abs(reconstructed[str(class_label)] - proba[i][k]) < 1e-3


def test_predict_top_n_truncates_contributions():
    df = _synthetic_binary_df()
    bundle = engine.train_model(df, "label", model_name="topn_test", interactions=3)
    record = df.drop(columns=["label"]).iloc[0].to_dict()
    explanation = engine.explain_record(bundle, record, top_n=2)
    assert len(explanation["contributions"]) == 2


def test_bundle_round_trip_preserves_predictions(tmp_path):
    df = _synthetic_binary_df()
    bundle = engine.train_model(df, "label", model_name="roundtrip_test", interactions=3)
    record = df.drop(columns=["label"]).iloc[0].to_dict()

    before = engine.predict_record(bundle, record)

    model_dir = engine.save_bundle(bundle, tmp_path)
    reloaded = engine.load_bundle(model_dir)
    after = engine.predict_record(reloaded, record)

    assert before == after


def test_metadata_json_is_plain_json_serializable(tmp_path):
    import json

    df = _synthetic_binary_df()
    bundle = engine.train_model(df, "label", model_name="metadata_test", interactions=3)
    model_dir = engine.save_bundle(bundle, tmp_path)

    metadata = engine.load_metadata(model_dir)
    # round-trips through json.dumps without a custom encoder -> everything
    # in it is already a native python type, not a numpy scalar.
    json.dumps(metadata)
    assert metadata["target_classes"] == ["negative", "positive"]


def test_train_model_raises_on_missing_target_column():
    df = _synthetic_binary_df()
    with pytest.raises(ValueError):
        engine.train_model(df, "does_not_exist", model_name="bad_target")
