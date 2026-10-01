"""Predict command: one JSON record in, one JSON result out (Member 5).

Usage:
  python predict.py --record '{"plot_area_ha":1.2,"rainfall_mm":81,"soil_ph":5.7,
                               "seed_kg":210,"distance_km":14,"arrival_hour":9}'
Run run_all.py first: it creates the saved models in models/.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_pipeline import FEATURE_COLS
from src.model_info import read_model_info
from src.regression import predict_raw

DEFAULT_MODELS = Path(__file__).resolve().parent / "models"


class RecordError(Exception):
    """The input record is missing, malformed or has invalid fields."""


class ModelError(Exception):
    """Saved models are missing or unreadable."""


def _reject_constant(name):
    raise RecordError(f"Invalid number '{name}' in JSON; use finite numbers only.")


def parse_record(text):
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise RecordError(f"Record is not valid JSON: {exc}") from exc


def validate_record(rec):
    if not isinstance(rec, dict):
        raise RecordError("Record must be a JSON object with the six feature fields.")
    problems = []
    missing = [c for c in FEATURE_COLS if c not in rec]
    extra = [k for k in rec if k not in FEATURE_COLS]
    if missing:
        problems.append(f"Missing field(s): {missing}")
    if extra:
        problems.append(f"Unexpected field(s): {extra}")
    for c in FEATURE_COLS:
        if c not in rec:
            continue
        v = rec[c]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            problems.append(f"Field '{c}' must be a number, got {v!r}")
        elif not math.isfinite(v):
            problems.append(f"Field '{c}' must be finite, got {v!r}")
    if problems:
        raise RecordError("; ".join(problems) + f". Required fields: {FEATURE_COLS}")
    return [float(rec[c]) for c in FEATURE_COLS]


def load_models(models_dir):
    models_dir = Path(models_dir)
    needed = ["regression_model.json", "classification_model.joblib",
              "clustering_model.joblib", "model_info.json"]
    absent = [n for n in needed if not (models_dir / n).is_file()]
    if absent:
        raise ModelError(f"Missing model file(s) in {models_dir}: {absent}. "
                         "Run run_all.py first.")
    try:
        with open(models_dir / "regression_model.json") as f:
            reg = json.load(f)
        return {"reg": reg,
                "clf": joblib.load(models_dir / "classification_model.joblib"),
                "clu": joblib.load(models_dir / "clustering_model.joblib"),
                "info": read_model_info(models_dir)}
    except Exception as exc:
        raise ModelError(f"Could not load saved models: {exc}") from exc


def predict_one(values, models):
    x = np.array([values], dtype=float)
    reg_pred = float(predict_raw(x, models["reg"])[0])
    clf = models["clf"]
    proba = float(clf["pipeline"].predict_proba(
        pd.DataFrame(x, columns=FEATURE_COLS))[0, 1])
    clu = models["clu"]
    label = int(clu["kmeans"].predict(clu["scaler"].transform(x))[0])
    if not all(math.isfinite(v) for v in (reg_pred, proba)):
        raise RecordError("Input produced a non-finite prediction.")
    return {
        "regression_prediction": reg_pred,
        "classification_prediction": int(proba >= clf["threshold"]),
        "classification_probability": proba,
        "cluster_label": label,
        "group_code": models["info"]["group_code"],
        "model_version": models["info"]["model_version"],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Musanze HarvestLink prediction")
    p.add_argument("--record", required=True, help="one JSON record")
    p.add_argument("--models", default=str(DEFAULT_MODELS))
    args = p.parse_args(argv)
    try:
        values = validate_record(parse_record(args.record))
        result = predict_one(values, load_models(args.models))
    except (RecordError, ModelError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
