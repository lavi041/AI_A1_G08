"""Model metadata written by run_all.py and read by predict.py (Member 5)."""
import json
from pathlib import Path

MODEL_VERSION = "1.0.0"
INFO_FILE = "model_info.json"


def write_model_info(models_dir, group_code, seed, dataset_sha256, feature_names):
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    info = {"model_version": MODEL_VERSION, "group_code": group_code,
            "seed": seed, "dataset_sha256": dataset_sha256,
            "feature_names": list(feature_names)}
    with open(models_dir / INFO_FILE, "w") as f:
        json.dump(info, f, indent=2)
    return info


def read_model_info(models_dir):
    path = Path(models_dir) / INFO_FILE
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)
