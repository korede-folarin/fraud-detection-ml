"""
Freeze v1 — one-off script, kept for provenance
Run once from the repository root: python v2/regression/build_v1_bundle.py

Plain English: copies exactly what v1 was evaluated on (the three trained
models, the scaler, and the validate/test sets) into v1_frozen/, and writes a
manifest with a fingerprint (SHA-256 hash) of every file.

Technical: the v1 pickles and feature sets are gitignored and the raw data sits
behind an authenticated DVC remote, so CI could not otherwise recompute v1.
This bundle lets the regression check (check_v1.py) rebuild every number in
evaluation_results.json with no secrets.

What the code does: copies models, writes validate/test features + labels as
parquet keeping the original row number, verifies the saved scaled arrays are
reproduced by scaler.transform, and records hashes, library versions and the
MLflow run IDs that correspond to v1.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_DATA = ROOT / "artifacts" / "data_transformation"
SRC_MODELS = ROOT / "artifacts" / "model_trainer" / "models"
OUT = ROOT / "v1_frozen"

# the six MLflow runs (DagsHub, experiment 0, logged 2026-05-30) whose metrics
# match artifacts/model_evaluation/evaluation_results.json; later runs in the
# same experiment come from the weekly Airflow DAG with different library
# versions and are NOT v1 (see docs/decisions.md, D10)
V1_MLFLOW_RUNS = {
    "Logistic Regression_validate": "d2a07d44e93d49659e27c17f7fa29e99",
    "Logistic Regression_test": "c7fd10a21cd7430db5b2ac510db741de",
    "Random Forest_validate": "836b673e7c174a16af47f7400814c7fa",
    "Random Forest_test": "f15dd076cc3c4d138ef4e19c2f029bd1",
    "XGBoost_validate": "3a7dd45ce810448ab16353124bc73240",
    "XGBoost_test": "f121c779f84c40a6bdc2a361fe7ca739",
}

LIBRARIES = ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "joblib", "pyarrow"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(parents=True, exist_ok=True)

    for name in ["xgboost.pkl", "random_forest.pkl", "logistic_regression.pkl"]:
        shutil.copy2(SRC_MODELS / name, OUT / "models" / name)
    shutil.copy2(SRC_DATA / "scaler.pkl", OUT / "models" / "scaler.pkl")

    scaler = joblib.load(SRC_DATA / "scaler.pkl")
    for split in ["validate", "test"]:
        X = joblib.load(SRC_DATA / f"X_{split}.pkl")
        y = joblib.load(SRC_DATA / f"y_{split}.pkl")
        saved_scaled = joblib.load(SRC_DATA / f"X_{split}_scaled.pkl")

        # the bundle stores raw features only; prove the scaled copy is reproducible
        if not np.array_equal(scaler.transform(X), np.asarray(saved_scaled)):
            sys.exit(f"scaler.transform does not reproduce X_{split}_scaled — refusing to freeze")

        df = X.copy()
        df["Class"] = np.asarray(y).astype("int8")
        df.index.name = "row_id"   # row number in the time-sorted raw data
        df.to_parquet(OUT / "data" / f"{split}.parquet", compression="zstd")

    commit = subprocess.run(["git", "rev-list", "-n", "1", "v1.0-evaluated"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    dvc_md5 = (ROOT / "data" / "creditcard.csv.dvc").read_text().split("md5:")[1].split()[0]

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    manifest = {
        "version": "v1",
        "git_tag": "v1.0-evaluated",
        "git_commit": commit,
        "dataset_version": f"ulb-creditcard@md5:{dvc_md5}",
        "evaluation_results": {
            "path": "artifacts/model_evaluation/evaluation_results.json",
            "sha256": sha256(ROOT / "artifacts" / "model_evaluation" / "evaluation_results.json"),
        },
        "mlflow": {
            "tracking_uri": "https://dagshub.com/korede-folarin/fraud-detection-ml.mlflow",
            "experiment_id": "0",
            "runs": V1_MLFLOW_RUNS,
        },
        "environment": {"python": "3.14.3", **{lib: version(lib) for lib in LIBRARIES}},
        "files": {str(p.relative_to(OUT)).replace("\\", "/"): sha256(p) for p in files},
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    for name in manifest["files"]:
        print(f"{(OUT / name).stat().st_size / 1e6:7.2f} MB  {name}")


if __name__ == "__main__":
    main()
