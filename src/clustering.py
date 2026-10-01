"""KMeans clustering on the INPUT features only (Member 4).

Inputs  : the six feature columns. actual_yield_kg, dispatch_attention and
          record_id are never used for fitting.
Scaling : StandardScaler fitted on all rows (clustering is unsupervised, so
          there is no train/test split here).
Choice  : k is tried from k_min to k_max; the k with the highest silhouette
          score is selected (ties go to the smaller k).
Caution : clusters are statistical groups, not verified real-world categories.
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import FEATURE_COLS

DEFAULT_SEED = 42
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 5
WEAK_SILHOUETTE = 0.25   # below this, structure is described as weak
N_INIT = 10

CAUTION = ("Clusters are statistical groupings of the input features only. "
           "They are not verified real-world categories and should be reviewed "
           "by cooperative staff before any operational use.")


class ClusteringError(Exception):
    """Raised when clustering cannot be performed on the given data."""


def evaluate_k_values(Xs, k_values, seed):
    """Returns (silhouette_by_k, inertia_by_k) for every usable k."""
    scores, inertias = {}, {}
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(Xs)
        if len(np.unique(km.labels_)) < 2:
            continue
        scores[k] = float(silhouette_score(Xs, km.labels_))
        inertias[k] = float(km.inertia_)
    return scores, inertias


def select_k(scores):
    """Highest silhouette; ties resolved towards the smaller k."""
    return max(scores, key=lambda k: (scores[k], -k))


def build_justification(scores, best_k):
    ordered = sorted(scores, key=lambda k: -scores[k])
    text = (f"k={best_k} was selected because it has the highest silhouette "
            f"score ({scores[best_k]:.3f}) among the tested values "
            f"{sorted(scores)}.")
    if len(ordered) > 1:
        runner = ordered[1]
        text += f" The next best was k={runner} ({scores[runner]:.3f})."
    if scores[best_k] < WEAK_SILHOUETTE:
        text += (f" The best silhouette is below {WEAK_SILHOUETTE}, so the "
                 "cluster structure is weak and the groups should be read "
                 "with extra caution.")
    return text


def run_clustering_stage(X, ids, output_dir, models_dir, seed=DEFAULT_SEED,
                         k_min=DEFAULT_K_MIN, k_max=DEFAULT_K_MAX):
    output_dir, models_dir = Path(output_dir), Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if k_min < 2 or k_max < k_min:
        raise ClusteringError("Need 2 <= k_min <= k_max.")

    k_values = [k for k in range(k_min, k_max + 1) if k <= n - 1]
    if not k_values:
        raise ClusteringError(f"Too few rows ({n}) to evaluate k={k_min}..{k_max}.")

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    scores, inertias = evaluate_k_values(Xs, k_values, seed)
    if not scores:
        raise ClusteringError("No k produced at least two distinct clusters.")
    best_k = select_k(scores)

    km = KMeans(n_clusters=best_k, n_init=N_INIT, random_state=seed).fit(Xs)
    labels = km.labels_

    out = pd.DataFrame(X, columns=FEATURE_COLS)
    out.insert(0, "cluster_label", labels)
    out.insert(0, "record_id", np.asarray(ids))
    out.to_csv(output_dir / "clusters.csv", index=False)

    profile = (pd.DataFrame(X, columns=FEATURE_COLS)
               .groupby(labels).mean().round(4))
    metrics = {
        "seed": seed, "k_range_tested": [k_min, k_max],
        "k_values_evaluated": sorted(scores),
        "silhouette_scores": {str(k): scores[k] for k in sorted(scores)},
        "inertia": {str(k): inertias[k] for k in sorted(inertias)},
        "selected_k": int(best_k),
        "selected_silhouette": scores[best_k],
        "justification": build_justification(scores, best_k),
        "n_samples": int(n),
        "cluster_sizes": {str(int(c)): int(s) for c, s in
                          zip(*np.unique(labels, return_counts=True))},
        "cluster_feature_means": {str(int(c)): row.to_dict()
                                  for c, row in profile.iterrows()},
        "features_used": FEATURE_COLS,
        "excluded_from_clustering": ["record_id", "actual_yield_kg",
                                     "dispatch_attention"],
        "scaler_fitted_on": "all rows (unsupervised, input features only)",
        "interpretation_caution": CAUTION,
    }
    with open(output_dir / "clustering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump({"scaler": scaler, "kmeans": km, "selected_k": int(best_k),
                 "feature_names": FEATURE_COLS},
                models_dir / "clustering_model.joblib")

    _plot(Xs, labels, scores, best_k, output_dir / "cluster_plot.png")

    print(f"Clustering | silhouette by k: "
          + ", ".join(f"{k}={scores[k]:.3f}" for k in sorted(scores))
          + f" | selected k={best_k}")
    return km, metrics


def _plot(Xs, labels, scores, best_k, path):
    pca = PCA(n_components=2).fit(Xs)
    Z = pca.transform(Xs)
    var = pca.explained_variance_ratio_
    palette = plt.get_cmap("tab10").colors  # distinct colours
    markers = ["o", "s", "^", "D", "P", "X"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ks = sorted(scores)
    ax1.plot(ks, [scores[k] for k in ks], marker="o")
    ax1.axvline(best_k, linestyle="--", color="gray", label=f"selected k={best_k}")
    ax1.set_xticks(ks)
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Silhouette score")
    ax1.set_title("Silhouette score by k")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for c in np.unique(labels):
        m = labels == c
        ax2.scatter(Z[m, 0], Z[m, 1], s=28, alpha=0.8,
                    color=palette[int(c) % len(palette)],
                    marker=markers[int(c) % len(markers)],
                    label=f"Cluster {int(c)} (n={int(m.sum())})")
    ax2.set_xlabel(f"PC1 ({var[0]:.0%} of variance)")
    ax2.set_ylabel(f"PC2 ({var[1]:.0%} of variance)")
    ax2.set_title("Clusters (2-D PCA view of standardized features)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
