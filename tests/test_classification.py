"""Run with:  python -m tests.test_classification   (from the project root)."""
import numpy as np
import pandas as pd

from src.classification import (ClassificationError, classification_metrics,
                                run_classification_stage,
                                split_stratified_if_possible)
from src.data_pipeline import CLF_TARGET, FEATURE_COLS


def _df(n=100, pos=0.3, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.normal(size=(n, 6)), columns=FEATURE_COLS)
    d[CLF_TARGET] = (rng.random(n) < pos).astype(int)
    return d


def test_stratified_keeps_class_ratio():
    d = _df()
    X_tr, X_te, y_tr, y_te, flag = split_stratified_if_possible(
        d[FEATURE_COLS], d[CLF_TARGET], 0.2, 1)
    assert flag is True
    assert abs(y_tr.mean() - y_te.mean()) < 0.1


def test_falls_back_when_class_too_small():
    d = _df(n=30)
    d[CLF_TARGET] = 0
    d.loc[0, CLF_TARGET] = 1  # a single positive: cannot stratify
    *_, flag = split_stratified_if_possible(d[FEATURE_COLS], d[CLF_TARGET], 0.2, 1)
    assert flag is False


def test_metrics_from_known_confusion():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    m = classification_metrics(y_true, y_pred)
    c = m["confusion_matrix"]
    assert (c["TP"], c["FN"], c["FP"], c["TN"]) == (2, 1, 1, 4)
    assert abs(m["accuracy"] - 6 / 8) < 1e-12
    assert abs(m["precision"] - 2 / 3) < 1e-12
    assert abs(m["recall"] - 2 / 3) < 1e-12
    assert abs(m["f1"] - 2 / 3) < 1e-12


def test_single_class_is_rejected():
    d = _df()
    d[CLF_TARGET] = 0
    try:
        run_classification_stage(d, "/tmp/_o", "/tmp/_m")
    except ClassificationError:
        return
    raise AssertionError("single class should raise ClassificationError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
