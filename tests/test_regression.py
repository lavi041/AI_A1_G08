"""Run with:  python -m tests.test_regression   (from the project root)."""
import numpy as np

from src.regression import (add_bias, fit_standardizer, gradient_descent,
                            predict_raw, regression_metrics,
                            train_test_split_np)


def test_gd_matches_closed_form():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = 3.0 + X @ np.array([2.0, -1.0, 0.5]) + rng.normal(0, 0.1, 200)
    Xb = add_bias(X)
    w_gd, hist = gradient_descent(Xb, y, lr=0.1, epochs=3000)
    w_ref = np.linalg.lstsq(Xb, y, rcond=None)[0]  # reference check only
    assert np.allclose(w_gd, w_ref, atol=1e-3), (w_gd, w_ref)
    assert hist[-1] < hist[0] and np.all(np.diff(hist) <= 1e-12)


def test_split_is_disjoint_and_complete():
    tr, te = train_test_split_np(50, 0.2, 7)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == 50
    tr2, te2 = train_test_split_np(50, 0.2, 7)
    assert (tr == tr2).all() and (te == te2).all()  # reproducible


def test_metrics_perfect_and_mean_baseline():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert regression_metrics(y, y)["R2"] == 1.0
    assert abs(regression_metrics(y, np.full(4, y.mean()))["R2"]) < 1e-12


def test_constant_column_does_not_break_scaler():
    A = np.column_stack([np.ones(10), np.arange(10.0)])
    mean, std = fit_standardizer(A)
    assert np.all(np.isfinite((A - mean) / std))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
