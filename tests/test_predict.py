"""Run with:  python -m tests.test_predict   (from the project root)."""
import tempfile

import numpy as np
import pandas as pd

import predict
from src.classification import run_classification_stage
from src.clustering import run_clustering_stage
from src.data_pipeline import CLF_TARGET, FEATURE_COLS
from src.model_info import write_model_info
from src.regression import run_regression_stage

GOOD = {"plot_area_ha": 1.2, "rainfall_mm": 81, "soil_ph": 5.7,
        "seed_kg": 210, "distance_km": 14, "arrival_hour": 9}


def _build_models(d):
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 100, size=(120, 6))
    y = X @ np.arange(1, 7) + rng.normal(0, 1, 120)
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df[CLF_TARGET] = (X[:, 0] + rng.normal(0, 20, 120) > 50).astype(int)
    run_regression_stage(X, y, d, d)
    run_classification_stage(df, d, d)
    run_clustering_stage(X, np.arange(120).astype(str), d, d)
    write_model_info(d, "AI-GTEST", 42, "abc", FEATURE_COLS)


def _rejects(rec_text):
    try:
        predict.validate_record(predict.parse_record(rec_text))
    except predict.RecordError:
        return True
    return False


def test_valid_record_returns_all_fields():
    with tempfile.TemporaryDirectory() as d:
        _build_models(d)
        values = predict.validate_record(GOOD)
        out = predict.predict_one(values, predict.load_models(d))
    assert set(out) == {"regression_prediction", "classification_prediction",
                        "classification_probability", "cluster_label",
                        "group_code", "model_version"}
    assert out["group_code"] == "AI-GTEST"
    assert 0.0 <= out["classification_probability"] <= 1.0
    assert out["classification_prediction"] in (0, 1)
    assert np.isfinite(out["regression_prediction"])


def test_bad_records_are_rejected():
    import json
    missing = dict(GOOD); del missing["soil_ph"]
    text = dict(GOOD, soil_ph="acidic")
    extra = dict(GOOD, record_id="R1")
    boolean = dict(GOOD, soil_ph=True)
    assert _rejects(json.dumps(missing))
    assert _rejects(json.dumps(text))
    assert _rejects(json.dumps(extra))
    assert _rejects(json.dumps(boolean))
    assert _rejects("{not json")
    assert _rejects("[1, 2, 3]")
    assert _rejects('{"plot_area_ha": NaN}')


def test_missing_models_give_clear_error():
    with tempfile.TemporaryDirectory() as d:
        try:
            predict.load_models(d)
        except predict.ModelError as exc:
            assert "run_all.py" in str(exc)
            return
    raise AssertionError("expected ModelError")


def test_main_exit_codes():
    with tempfile.TemporaryDirectory() as d:
        _build_models(d)
        import json
        assert predict.main(["--record", json.dumps(GOOD), "--models", d]) == 0
        assert predict.main(["--record", "{}", "--models", d]) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
