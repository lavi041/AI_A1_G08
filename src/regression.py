"""Linear regression + batch gradient descent, NumPy only (Member 2).

Model      : y_hat = Xb @ w          (Xb = [1 | X_scaled], w[0] is the bias)
Loss       : J(w) = 1/(2n) * sum((y_hat - y)^2)
Gradient   : dJ/dw = (1/n) * Xb.T @ (Xb @ w - y)
Update     : w <- w - lr * dJ/dw     (all rows used every step = batch GD)
Scaling    : X and y are standardized with TRAIN statistics only.
No library estimator is used.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_pipeline import FEATURE_COLS

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_LR = 0.1
DEFAULT_EPOCHS = 2000


def train_test_split_np(n_rows, test_size, seed):
    """Random index split; returns (train_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_rows)
    n_test = max(1, int(round(n_rows * test_size)))
    return idx[n_test:], idx[:n_test]


def fit_standardizer(A):
    mean = A.mean(axis=0)
    std = A.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # constant column -> avoid divide by zero
    return mean, std


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def gradient_descent(Xb, y, lr, epochs):
    n = Xb.shape[0]
    w = np.zeros(Xb.shape[1])
    history = np.empty(epochs)
    for i in range(epochs):
        residual = Xb @ w - y
        history[i] = (residual @ residual) / (2 * n)
        if not np.isfinite(history[i]):
            raise FloatingPointError(
                f"Loss became non-finite at epoch {i}; lower the learning rate.")
        w -= lr * (Xb.T @ residual) / n
    return w, history


def regression_metrics(y_true, y_pred):
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def predict_raw(X, model):
    """X in original units -> predictions in original units."""
    Xs = (X - np.asarray(model["x_mean"])) / np.asarray(model["x_std"])
    y_s = add_bias(Xs) @ np.asarray(model["weights"])
    return y_s * model["y_std"] + model["y_mean"]


def run_regression_stage(X, y, output_dir, models_dir, seed=DEFAULT_SEED,
                         test_size=DEFAULT_TEST_SIZE, lr=DEFAULT_LR,
                         epochs=DEFAULT_EPOCHS):
    output_dir, models_dir = Path(output_dir), Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    tr, te = train_test_split_np(len(y), test_size, seed)
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

    x_mean, x_std = fit_standardizer(X_tr)      # fitted on TRAIN only
    y_mean, y_std = fit_standardizer(y_tr.reshape(-1, 1))
    y_mean, y_std = float(y_mean[0]), float(y_std[0])

    Xb_tr = add_bias((X_tr - x_mean) / x_std)
    w, history = gradient_descent(Xb_tr, (y_tr - y_mean) / y_std, lr, epochs)

    model = {"feature_names": FEATURE_COLS, "weights": w.tolist(),
             "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
             "y_mean": y_mean, "y_std": y_std}
    pred_tr, pred_te = predict_raw(X_tr, model), predict_raw(X_te, model)
    if not (np.all(np.isfinite(pred_tr)) and np.all(np.isfinite(pred_te))):
        raise FloatingPointError("Non-finite predictions produced.")

    metrics = {
        "seed": seed, "test_size": test_size, "learning_rate": lr,
        "epochs": epochs, "n_train": int(len(tr)), "n_test": int(len(te)),
        "loss_definition": "1/(2n)*sum((pred-y)^2) on standardized target",
        "initial_loss": float(history[0]), "final_loss": float(history[-1]),
        "weights_standardized": dict(zip(["bias"] + FEATURE_COLS, w.tolist())),
        "train_metrics": regression_metrics(y_tr, pred_tr),
        "test_metrics": regression_metrics(y_te, pred_te),
        "scalers_fitted_on": "training split only",
    }
    with open(output_dir / "regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(models_dir / "regression_model.json", "w") as f:
        json.dump(model, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, epochs + 1), history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (standardized units)")
    ax.set_title("Batch gradient descent: loss history")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "regression_loss.png", dpi=150)
    plt.close(fig)

    t = metrics["test_metrics"]
    print(f"Regression | test MAE={t['MAE']:.2f} RMSE={t['RMSE']:.2f} "
          f"R2={t['R2']:.3f} | final loss={metrics['final_loss']:.5f}")
    return model, metrics
