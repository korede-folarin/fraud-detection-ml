"""
Register the frozen v1 champion in the local BentoML model store
Run from the repository root: python v2/ci/register_v1_model.py

Plain English: the API container needs the model inside it. v1's
bento_service/save_model.py reads the model from a gitignored folder, so a
fresh CI machine can't run it. This does the same job from the committed,
hash-checked copy in v1_frozen/.

Technical: same call as v1's save_model.py — bentoml.xgboost.save_model with
the feature order and the deployment threshold from evaluation_results.json —
but it verifies the pickle's SHA-256 against v1_frozen/MANIFEST.json first,
and adds labels saying which baseline the model came from.

What the code does: refuses to register if the frozen model was altered;
otherwise prints the new model tag.
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import bentoml

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "v1_frozen"


def main():
    manifest = json.loads((BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
    model_path = BUNDLE / "models" / "xgboost.pkl"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest["files"]["models/xgboost.pkl"]:
        sys.exit("v1_frozen/models/xgboost.pkl does not match MANIFEST.json — refusing to register")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    features = json.loads((ROOT / "artifacts" / "data_transformation" / "final_features.json")
                          .read_text(encoding="utf-8"))["features"]
    results = json.loads((ROOT / "artifacts" / "model_evaluation" / "evaluation_results.json")
                         .read_text(encoding="utf-8"))

    saved = bentoml.xgboost.save_model(
        "fraud_xgboost",
        model,
        custom_objects={"feature_names": features},
        labels={
            "baseline": manifest["git_tag"],
            "dataset_version": manifest["dataset_version"].replace("@", "-").replace(":", "-"),
        },
        metadata={
            "ks": results["test"]["XGBoost"]["KS"],
            "auprc": results["test"]["XGBoost"]["AUPRC"],
            "business_threshold": results["champion"]["business_threshold"],
            "model_type": results["champion"]["model"],
        },
    )
    print(f"Registered {saved.tag} from v1_frozen (threshold {results['champion']['business_threshold']})")


if __name__ == "__main__":
    main()
