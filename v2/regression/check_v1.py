"""
v1 regression check — fails CI if any v1 result changes
Run from the repository root: python v2/regression/check_v1.py

Plain English: v1 is frozen. This script proves it is still v1. It checks that
no frozen file has been edited, then re-scores the frozen validate and test
sets with the frozen v1 models and confirms every number in
evaluation_results.json comes out identical.

Technical: two layers. (1) Integrity — SHA-256 of every file in v1_frozen/
and of evaluation_results.json must match v1_frozen/MANIFEST.json.
(2) Reproduction — recompute all metrics with v1's own ModelEvaluation
methods (imported, not copied) and compare to the committed JSON exactly.
Exact comparison is possible because v1 rounds every metric (4 dp, thresholds
floored to 6 dp) and the library versions are pinned to the v1 environment.

What the code does: exits 0 when v1 is intact, 1 with a list of every
difference otherwise. Needs no secrets and no network.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from frauddetection.components.model_evaluation import ModelEvaluation, floor_6dp  # noqa: E402

BUNDLE = ROOT / "v1_frozen"
RESULTS = ROOT / "artifacts" / "model_evaluation" / "evaluation_results.json"
MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}
# v1 business inputs (params.yaml at v1.0-evaluated)
COST_FN, COST_FP = 150, 10


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_integrity(manifest):
    problems = []
    for name, digest in manifest["files"].items():
        path = BUNDLE / name
        if not path.exists():
            problems.append(f"missing frozen file: v1_frozen/{name}")
        elif sha256(path) != digest:
            problems.append(f"frozen file changed: v1_frozen/{name}")
    if sha256(RESULTS) != manifest["evaluation_results"]["sha256"]:
        problems.append("artifacts/model_evaluation/evaluation_results.json changed")
    return problems


def recompute():
    """Rebuilds evaluation_results.json the way v1's ModelEvaluation.run() does"""
    ev = ModelEvaluation(SimpleNamespace(cost_false_negative=COST_FN, cost_false_positive=COST_FP))
    scaler = joblib.load(BUNDLE / "models" / "scaler.pkl")

    data, labels, scores, results = {}, {}, {"validate": {}, "test": {}}, {"validate": {}, "test": {}}
    for split in ("validate", "test"):
        df = pd.read_parquet(BUNDLE / "data" / f"{split}.parquet")
        labels[split] = df.pop("Class").astype("int64")
        data[split] = (df, scaler.transform(df))

    for name, file in MODELS.items():
        model = joblib.load(BUNDLE / "models" / file)
        for split in ("validate", "test"):
            X, X_scaled = data[split]
            scores[split][name] = ev.predict(name, model, X, X_scaled)
            results[split][name] = ev.evaluate_model(scores[split][name], labels[split])

    business = floor_6dp(ev.calculate_business_cost(labels["validate"], scores["validate"]["XGBoost"])[0])
    f1 = floor_6dp(ev.find_best_f1_threshold(labels["validate"], scores["validate"]["XGBoost"])[1])

    results["at_business_threshold"] = {
        split: {m: ev.metrics_at_threshold(labels[split], scores[split][m], business) for m in MODELS}
        for split in ("validate", "test")
    }
    results["xgboost_at_f1_threshold"] = {
        split: ev.metrics_at_threshold(labels[split], scores[split]["XGBoost"], f1)
        for split in ("validate", "test")
    }
    results["champion"] = {
        "test_metrics": results["test"]["XGBoost"],
        # v1's run() floors these a second time when writing the champion block,
        # which turns the float32 threshold into an exact 6-decimal float64
        "business_threshold": floor_6dp(business),
        "f1_threshold": floor_6dp(f1),
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
    }
    # same serialisation as v1 (json), so float/int types compare like for like
    return json.loads(json.dumps(results))


def diff(expected, actual, path=""):
    out = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for k in sorted(set(expected) | set(actual)):
            if k not in actual:
                out.append(f"{path}/{k}: missing from recomputation")
            elif k not in expected:
                out.append(f"{path}/{k}: not in committed results")
            else:
                out += diff(expected[k], actual[k], f"{path}/{k}")
    elif expected != actual:
        out.append(f"{path}: committed {expected!r} vs recomputed {actual!r}")
    return out


def main():
    manifest = json.loads((BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
    problems = check_integrity(manifest)

    committed = json.loads(RESULTS.read_text(encoding="utf-8"))
    champion = committed["champion"]
    # rationale and model name are fixed text in v1 code, not computed numbers
    numbers_only = {k: v for k, v in champion.items() if k not in ("model", "rationale")}
    committed_numbers = {**committed, "champion": numbers_only}
    problems += diff(committed_numbers, recompute())

    if problems:
        print(f"v1 REGRESSION CHECK FAILED — {len(problems)} difference(s):")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    n = sum(1 for _ in _leaves(committed_numbers))
    print(f"v1 regression check passed: {len(manifest['files'])} frozen files intact, "
          f"{n} committed values reproduced exactly (tag {manifest['git_tag']}).")


def _leaves(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from _leaves(v)
        else:
            yield v


if __name__ == "__main__":
    main()
