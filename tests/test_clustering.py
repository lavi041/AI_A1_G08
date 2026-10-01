"""Run with:  python -m tests.test_clustering   (from the project root)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering import (ClusteringError, run_clustering_stage, select_k)
from src.data_pipeline import FEATURE_COLS


def _blobs(per=30, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0] * 6, [10] * 6, [20, 0, 20, 0, 20, 0]], dtype=float)
    X = np.vstack([c + rng.normal(0, 0.5, size=(per, 6)) for c in centres])
    ids = np.array([f"R{i}" for i in range(len(X))])
    return X, ids


def test_selects_true_k_and_labels_every_record():
    X, ids = _blobs()
    with tempfile.TemporaryDirectory() as d:
        _, m = run_clustering_stage(X, ids, d, d, seed=1)
        out = pd.read_csv(Path(d) / "clusters.csv")
    assert m["selected_k"] == 3
    assert len(out) == len(X) and out["cluster_label"].notna().all()
    assert set(m["silhouette_scores"]) == {"2", "3", "4", "5"}


def test_output_has_no_target_columns():
    X, ids = _blobs()
    with tempfile.TemporaryDirectory() as d:
        run_clustering_stage(X, ids, d, d, seed=1)
        cols = list(pd.read_csv(Path(d) / "clusters.csv").columns)
    assert cols == ["record_id", "cluster_label"] + FEATURE_COLS


def test_same_seed_same_labels():
    X, ids = _blobs()
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        run_clustering_stage(X, ids, d1, d1, seed=5)
        run_clustering_stage(X, ids, d2, d2, seed=5)
        a = pd.read_csv(Path(d1) / "clusters.csv")["cluster_label"]
        b = pd.read_csv(Path(d2) / "clusters.csv")["cluster_label"]
    assert (a == b).all()


def test_tie_goes_to_smaller_k():
    assert select_k({2: 0.5, 3: 0.5, 4: 0.4}) == 2


def test_too_few_rows_is_rejected():
    X = np.zeros((2, 6))
    with tempfile.TemporaryDirectory() as d:
        try:
            run_clustering_stage(X, np.array(["a", "b"]), d, d)
        except ClusteringError:
            return
    raise AssertionError("2 rows should raise ClusteringError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
