"""Data loading, validation, cleaning and vectorization (Member 1)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "record_id"
FEATURE_COLS = ["plot_area_ha", "rainfall_mm", "soil_ph",
                "seed_kg", "distance_km", "arrival_hour"]
REG_TARGET = "actual_yield_kg"
CLF_TARGET = "dispatch_attention"
REQUIRED_COLS = [ID_COL] + FEATURE_COLS + [REG_TARGET, CLF_TARGET]


class DataValidationError(Exception):
    """Raised when the CSV does not match the published schema."""


def sha256_of_file(path):
    if not Path(path).is_file():
        raise DataValidationError(f"Data file not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path):
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"Data file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise DataValidationError(f"Cannot read CSV: {exc}") from exc


def validate_schema(df):
    """Check column names and convert numeric columns. Raises on problems."""
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLS]
    if extra_cols:
        raise DataValidationError(f"Unexpected columns: {extra_cols}")

    df = df.copy()
    df[ID_COL] = df[ID_COL].astype("string")
    for col in FEATURE_COLS + [REG_TARGET, CLF_TARGET]:
        converted = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & converted.isna()  # present but not numeric
        if bad.any():
            raise DataValidationError(
                f"Column '{col}' has {int(bad.sum())} non-numeric value(s).")
        df[col] = converted
    return df


def build_report(df_raw, df_clean, file_hash, group_code):
    missing = {c: int(df_raw[c].isna().sum()) for c in df_raw.columns}
    stats = df_clean[FEATURE_COLS + [REG_TARGET, CLF_TARGET]].describe().T
    stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    return {
        "group_code": group_code,
        "sha256": file_hash,
        "row_count_raw": int(len(df_raw)),
        "row_count_clean": int(len(df_clean)),
        "feature_count": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
        "missing_values": missing,
        "duplicate_rows": int(df_raw.duplicated().sum()),
        "duplicate_record_ids": int(df_raw[ID_COL].duplicated().sum()),
        "cleaning_rule": ("Exact duplicate rows removed (first kept); rows with "
                          "any missing value in a required column removed."),
        "descriptive_statistics": json.loads(stats.to_json(orient="index")),
    }


def run_data_stage(data_path, output_dir, group_code):
    """Load -> validate -> clean -> vectorize. Writes data_report.json.

    Returns (df_clean, X, y_reg, y_clf, ids).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_hash = sha256_of_file(data_path)
    print(f"Group code     : {group_code}")
    print(f"Dataset SHA-256: {file_hash}")

    df_raw = validate_schema(load_csv(data_path))

    df_clean = df_raw.drop_duplicates().dropna(subset=REQUIRED_COLS)
    df_clean = df_clean.reset_index(drop=True)
    if len(df_clean) < 10:
        raise DataValidationError(
            f"Only {len(df_clean)} usable rows after cleaning; need at least 10.")
    if not set(df_clean[CLF_TARGET].unique()) <= {0, 1}:
        raise DataValidationError("dispatch_attention must contain only 0 or 1.")

    ids = df_clean[ID_COL].to_numpy()                   # identifier, never a feature
    X = df_clean[FEATURE_COLS].to_numpy(dtype=float)    # NumPy feature matrix
    y_reg = df_clean[REG_TARGET].to_numpy(dtype=float)
    y_clf = df_clean[CLF_TARGET].to_numpy(dtype=int)

    report = build_report(df_raw, df_clean, file_hash, group_code)
    with open(output_dir / "data_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Rows (raw/clean): {report['row_count_raw']}/{report['row_count_clean']}"
          f" | features: {X.shape[1]} | matrix shape: {X.shape}")
    return df_clean, X, y_reg, y_clf, ids
