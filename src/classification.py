"""Interpretable classifier for dispatch_attention (Member 3).

Model : logistic regression inside Pipeline(StandardScaler -> LogisticRegression).
        Pipeline.fit fits the scaler on the TRAINING split only.
Split : stratified when possible, otherwise a plain random split (recorded).
No tuning is done on the test set; the decision threshold is a fixed setting.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import CLF_TARGET, FEATURE_COLS

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_THRESHOLD = 0.5

COST_INTERPRETATION = (
    "A false negative (a consignment that needs dispatch attention but is "
    "predicted 0) is treated as the more costly error: it is not flagged, so "
    "the issue may only be discovered after dispatch. A false positive only "
    "causes an unnecessary manual check by staff. Recall for class 1 is "
    "therefore watched closely. (Group judgement; edit to match the group's "
    "own reasoning.)"
)


class ClassificationError(Exception):
    """Raised when classification cannot be trained on the given data."""


def split_stratified_if_possible(X, y, test_size, seed):
    """Returns X_tr, X_te, y_tr, y_te, stratified_flag."""
    try:
        parts = train_test_split(X, y, test_size=test_size,
                                 random_state=seed, stratify=y)
        return (*parts, True)
    except ValueError:
        parts = train_test_split(X, y, test_size=test_size, random_state=seed)
        return (*parts, False)


def classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "confusion_matrix": {"labels": [0, 1], "matrix": cm.tolist(),
                             "TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_classification_stage(df, output_dir, models_dir, seed=DEFAULT_SEED,
                             test_size=DEFAULT_TEST_SIZE,
                             threshold=DEFAULT_THRESHOLD):
    output_dir, models_dir = Path(output_dir), Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 < threshold < 1.0:
        raise ClassificationError("threshold must be between 0 and 1.")

    X = df[FEATURE_COLS]
    y = df[CLF_TARGET].astype(int)
    if y.nunique() < 2:
        raise ClassificationError(
            "dispatch_attention contains only one class; cannot train a classifier.")

    X_tr, X_te, y_tr, y_te, stratified = split_stratified_if_possible(
        X, y, test_size, seed)
    if y_tr.nunique() < 2:
        raise ClassificationError("Training split contains only one class.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    m = classification_metrics(y_te.to_numpy(), pred)

    coefs = model.named_steps["clf"].coef_[0]
    metrics = {
        "seed": seed, "test_size": test_size, "threshold": threshold,
        "stratified_split": stratified, "model": "LogisticRegression",
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "class_counts_train": {str(k): int(v) for k, v in y_tr.value_counts().items()},
        "class_counts_test": {str(k): int(v) for k, v in y_te.value_counts().items()},
        "scaler_fitted_on": "training split only",
        **m,
        "coefficients_standardized": dict(zip(FEATURE_COLS, map(float, coefs))),
        "intercept": float(model.named_steps["clf"].intercept_[0]),
        "cost_interpretation": COST_INTERPRETATION,
    }
    with open(output_dir / "classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump({"pipeline": model, "threshold": threshold,
                 "feature_names": FEATURE_COLS},
                models_dir / "classification_model.joblib")

    cm = np.array(m["confusion_matrix"]["matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["0 (no attention)", "1 (attention)"],
                yticklabels=["0 (no attention)", "1 (attention)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (test set, threshold={threshold})")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"Classification | acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
          f"rec={m['recall']:.3f} f1={m['f1']:.3f} | stratified={stratified}")
    return model, metrics
