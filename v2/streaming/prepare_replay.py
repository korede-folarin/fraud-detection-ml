"""
Prepare the streaming replay input and the batch reference (D14)
Run from the repository root (needs data/creditcard.csv from DVC):
    python v2/streaming/prepare_replay.py

Plain English: picks out the raw transactions the stream will replay — the 99
validate payments needed to warm up the 100-payment window, then all 42,722
out-of-time test payments — and writes the v1 batch answers for the same
payments, so the stream's answers can be compared one by one.

Technical: verifies the raw file against the DVC md5 (dataset_version), then
recomputes v1's batch features with v1's pandas logic and confirms they equal
the frozen v1 test features exactly (row numbers = position in the time-sorted
raw data, which is already in time order). Reference scores come from the
frozen v1 pickle on the frozen v1 features — the v1 batch result.

What the code does: writes v2/streaming/data/replay.parquet (seq, Time, Amount,
V1..V28 — no label) and reference.parquet (seq, label, v1 features, v1 score,
v1 decision). Both are gitignored: they are derived from raw data.
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "creditcard.csv"
BUNDLE = ROOT / "v1_frozen"
OUT = ROOT / "v2" / "streaming" / "data"
WARMUP = 99


def main():
    manifest = json.loads((BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
    expected_md5 = manifest["dataset_version"].split("md5:")[1]
    md5 = hashlib.md5(RAW.read_bytes()).hexdigest()
    if md5 != expected_md5:
        sys.exit(f"{RAW} md5 {md5} != dataset_version {expected_md5}")

    raw = pd.read_csv(RAW)
    assert raw["Time"].is_monotonic_increasing, "raw data is expected in time order"

    # v1 batch features (DataTransformation.create_features), whole dataset
    batch = pd.DataFrame({
        "Amount_log": np.log1p(raw["Amount"]),
        "Hour": ((raw["Time"] // 3600) % 24).astype(int),
        "rolling_std_amount": raw["Amount"].rolling(100, min_periods=1).std().fillna(0),
    })

    test = pd.read_parquet(BUNDLE / "data" / "test.parquet")
    start, end = int(test.index.min()), int(test.index.max())
    for col in batch.columns:
        if not np.array_equal(batch.loc[test.index, col].to_numpy(), test[col].to_numpy()):
            sys.exit(f"recomputed {col} differs from frozen v1 features — refusing to build a reference")

    replay = raw.loc[start - WARMUP:end].drop(columns="Class").copy()
    replay.insert(0, "seq", replay.index.astype("int64"))

    model = joblib.load(BUNDLE / "models" / "xgboost.pkl")
    threshold = json.loads((ROOT / "artifacts" / "model_evaluation" / "evaluation_results.json")
                           .read_text(encoding="utf-8"))["champion"]["business_threshold"]
    features = test.drop(columns="Class")
    score = model.predict_proba(features)[:, 1]

    reference = pd.DataFrame({
        "seq": test.index.astype("int64"),
        "Class": test["Class"].astype("int8"),
        "Amount_log": test["Amount_log"],
        "Hour": test["Hour"].astype(int),
        "rolling_std_amount": test["rolling_std_amount"],
        "v1_score": score.astype("float64"),
        "v1_decision": np.where(score >= threshold, "DECLINE", "APPROVE"),
    })

    OUT.mkdir(parents=True, exist_ok=True)
    replay.to_parquet(OUT / "replay.parquet", index=False)
    reference.to_parquet(OUT / "reference.parquet", index=False)
    print(f"dataset_version {manifest['dataset_version']} verified")
    print(f"replay: {len(replay):,} events (seq {start - WARMUP}..{end}: {WARMUP} warm-up + {len(test):,} test)")
    print(f"reference: {len(reference):,} rows, {int((reference.v1_decision == 'DECLINE').sum())} v1 declines, "
          f"threshold {threshold}")


if __name__ == "__main__":
    main()
