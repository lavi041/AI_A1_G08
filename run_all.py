"""Pipeline entry point. Stages are added one by one."""
import argparse
import sys
from pathlib import Path

from src.data_pipeline import (run_data_stage, DataValidationError,
                               sha256_of_file, FEATURE_COLS)
from src.model_info import write_model_info
from src.classification import (run_classification_stage, ClassificationError,
                                DEFAULT_THRESHOLD)
from src.clustering import (run_clustering_stage, ClusteringError,
                            DEFAULT_K_MIN, DEFAULT_K_MAX)
from src.regression import (run_regression_stage, DEFAULT_SEED, DEFAULT_TEST_SIZE,
                            DEFAULT_LR, DEFAULT_EPOCHS)


def main():
    p = argparse.ArgumentParser(description="Musanze HarvestLink pipeline")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--k-min", type=int, default=DEFAULT_K_MIN)
    p.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    p.add_argument("--models", default=str(Path(__file__).resolve().parent / "models"))
    args = p.parse_args()
    try:
        df, X, y_reg, y_clf, ids = run_data_stage(args.data, args.output, args.group)
    except DataValidationError as exc:
        print(f"DATA ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        run_regression_stage(X, y_reg, args.output, args.models, seed=args.seed,
                             test_size=args.test_size, lr=args.lr, epochs=args.epochs)
    except FloatingPointError as exc:
        print(f"REGRESSION ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        run_classification_stage(df, args.output, args.models, seed=args.seed,
                                 test_size=args.test_size, threshold=args.threshold)
    except ClassificationError as exc:
        print(f"CLASSIFICATION ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        run_clustering_stage(X, ids, args.output, args.models, seed=args.seed,
                             k_min=args.k_min, k_max=args.k_max)
    except ClusteringError as exc:
        print(f"CLUSTERING ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    write_model_info(args.models, args.group, args.seed,
                     sha256_of_file(args.data), FEATURE_COLS)
    print(f"Done. Models saved in {args.models}; artifacts saved in {args.output}")


if __name__ == "__main__":
    main()
