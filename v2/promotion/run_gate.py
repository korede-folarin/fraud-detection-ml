"""
Run the promotion gate on the v1 challengers and log the outcome either way
Run from the repository root: python v2/promotion/run_gate.py

Plain English: v1 kept Random Forest and Logistic Regression as challengers to
the XGBoost champion, but nothing ever formally tested whether they should take
over. This runs that test with the D11 rules on the frozen v1 models and data,
and records the answer — including "challenger did not win".

Technical: each model is scored on the frozen validate/test sets; its threshold
is its own validate cost-optimal threshold (v1's ModelEvaluation methods, so the
numbers match v1). Results go to a new MLflow experiment, fraud-v2-promotion,
tagged with dataset_version, and to a JSON record in v2/promotion/records/.

What the code does: prints each gate decision; logs to DagsHub MLflow only if
DAGSHUB_TOKEN is set in the environment or .env (never hard-coded).
"""
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from frauddetection.components.model_evaluation import ModelEvaluation, floor_6dp  # noqa: E402
from v2.promotion.gate import evaluate_gate  # noqa: E402

BUNDLE = ROOT / "v1_frozen"
RECORDS = ROOT / "v2" / "promotion" / "records"
EXPERIMENT = "fraud-v2-promotion"
CHAMPION = "XGBoost"
CHALLENGERS = ["Random Forest", "Logistic Regression"]
FILES = {
    "XGBoost": "xgboost.pkl",
    "Random Forest": "random_forest.pkl",
    "Logistic Regression": "logistic_regression.pkl",
}


def model_metrics(ev, name, data, labels):
    """Metrics for one model at its own validate cost-optimal threshold"""
    model = joblib.load(BUNDLE / "models" / FILES[name])
    scores = {s: ev.predict(name, model, *data[s]) for s in data}
    threshold = floor_6dp(ev.calculate_business_cost(labels["validate"], scores["validate"])[0])
    out = {"threshold": threshold}
    for split in data:
        m = ev.metrics_at_threshold(labels[split], scores[split], threshold)
        m["KS"] = round(float(ev.calculate_ks(labels[split], scores[split])[0]), 4)
        out[split] = m
    return out


def log_mlflow(manifest, champion_name, challenger_name, metrics, result):
    import mlflow

    mlflow.set_tracking_uri(manifest["mlflow"]["tracking_uri"])
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"gate: {challenger_name} vs {champion_name}"):
        mlflow.set_tags({
            "dataset_version": manifest["dataset_version"],
            "baseline": manifest["git_tag"],
            "gate": "D11",
            "champion": champion_name,
            "challenger": challenger_name,
            "outcome": result.outcome,
            "stage_reached": result.stage_reached,
        })
        for role, name in (("champion", champion_name), ("challenger", challenger_name)):
            m = metrics[name]
            mlflow.log_metric(f"{role}_threshold", m["threshold"])
            for split in ("validate", "test"):
                for k in ("Precision", "Recall", "Cost", "KS"):
                    mlflow.log_metric(f"{role}_{split}_{k}", m[split][k])
        mlflow.log_metric("promoted", int(result.promote))
        mlflow.log_text(result.summary(), "gate_summary.txt")


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    manifest = json.loads((BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
    ev = ModelEvaluation(SimpleNamespace(cost_false_negative=150, cost_false_positive=10))
    scaler = joblib.load(BUNDLE / "models" / "scaler.pkl")
    data, labels = {}, {}
    for split in ("validate", "test"):
        df = pd.read_parquet(BUNDLE / "data" / f"{split}.parquet")
        labels[split] = df.pop("Class").astype("int64")
        data[split] = (df, scaler.transform(df))

    metrics = {name: model_metrics(ev, name, data, labels) for name in [CHAMPION] + CHALLENGERS}

    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        os.environ.setdefault("MLFLOW_TRACKING_USERNAME", "korede-folarin")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    record = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "gate": "D11",
        "dataset_version": manifest["dataset_version"],
        "baseline": manifest["git_tag"],
        "champion": CHAMPION,
        "metrics": metrics,
        "decisions": [],
        "mlflow_logged": bool(token),
    }
    for challenger in CHALLENGERS:
        result = evaluate_gate(metrics[CHAMPION], metrics[challenger])
        print(f"\n{challenger} vs {CHAMPION}\n{result.summary()}")
        record["decisions"].append({
            "challenger": challenger, "promote": result.promote, "outcome": result.outcome,
            "stage_reached": result.stage_reached,
            "checks": [{"rule": r, "passed": ok, "detail": d} for r, ok, d in result.checks],
        })
        if token:
            log_mlflow(manifest, CHAMPION, challenger, metrics, result)

    RECORDS.mkdir(parents=True, exist_ok=True)
    path = RECORDS / f"{record['run_at'][:10]}-v1-challengers.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nRecord written: {path.relative_to(ROOT)}"
          + ("" if token else "  (MLflow skipped: DAGSHUB_TOKEN not set)"))


if __name__ == "__main__":
    main()
