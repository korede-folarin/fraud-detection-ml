"""
Builds the data files for the static GitHub Pages demo (site/)
Run from the repository root: python -m src.site.build

Every number the site shows is read here from files the pipeline already
writes — nothing is typed into the HTML. Scores and SHAP values are computed
once, now, with the same code the API uses; the site only replays them.

Note: this folder deliberately has no __init__.py, so setuptools never
installs it as a top-level package called `site` (which would shadow the
standard library module of the same name).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from frauddetection.inference import decide, describe_feature, shap_contributions  # noqa: E402

EVAL_DIR   = ROOT / "artifacts" / "model_evaluation"
ASSETS_DIR = ROOT / "dashboard" / "assets"
OUT_DIR    = ROOT / "site" / "data"
DAG_FILE   = ROOT / "dags" / "fraud_detection_dag.py"
MODELS     = ["XGBoost", "Random Forest", "Logistic Regression"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def r(x, digits=4):
    """Rounding for display only — applied when writing, never before choosing"""
    return round(float(x), digits)


def ks_retrain_trigger():
    """The retraining trigger lives in the Airflow drift_check task — read it from there"""
    match = re.search(r"xgb_ks\s*<\s*([0-9.]+)", DAG_FILE.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"KS retraining trigger not found in {DAG_FILE}")
    return float(match.group(1))


def pick_samples(demo, threshold):
    """
    One real out-of-time test transaction per sample type, chosen by a fixed
    rule so a rebuild always shows the same rows:
      fraud        the caught fraud with the median score among caught frauds
      legitimate   the legitimate row with the median score among approved legit
      borderline   the nearest approval: the highest-scoring row still approved
      false alarm  the flagged legitimate row with the median score among alarms
    """
    def median_row(pool):
        pool = pool.sort_values("score")
        return pool.index[len(pool) // 2]

    fraud  = demo[(demo["Class"] == 1) & (demo["score"] >= threshold)]
    legit  = demo[(demo["Class"] == 0) & (demo["score"] < threshold)]
    alarms = demo[(demo["Class"] == 0) & (demo["score"] >= threshold)]

    return {
        "fraud":       median_row(fraud),
        "legitimate":  median_row(legit),
        "borderline":  demo[demo["score"] < threshold]["score"].idxmax(),
        "false_alarm": median_row(alarms),
    }


def build_samples(meta, demo, model):
    features  = meta["features"]
    threshold = meta["business_threshold"]
    samples   = {}

    for kind, row_id in pick_samples(demo, threshold).items():
        row = demo.loc[[row_id], features].astype(float)
        fraud_prob = float(model.predict_proba(row)[0, 1])
        contributions, base = shap_contributions(model, row)

        # the same decision + notice function the API and dashboard call
        result = decide(fraud_prob, contributions, threshold)

        ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        samples[kind] = {
            "test_row": int(row_id),
            "actual": "fraud" if demo.loc[row_id, "Class"] == 1 else "legitimate",
            "amount_eur": r(np.expm1(row["Amount_log"].iloc[0]), 2),
            "hour": int(row["Hour"].iloc[0]),
            "fraud_probability": fraud_prob,
            "decision": result["decision"],
            "risk_level": result["risk_level"],
            "notice": result["adverse_action_notice"],
            "base_value": r(base),
            "shap": [
                {"feature": f, "label": describe_feature(f), "value": r(v)}
                for f, v in ordered[:8]
            ],
            "other_features": {
                "count": len(ordered) - 8,
                "value": r(sum(v for _, v in ordered[8:])),
            },
        }
    return samples


def build_overview(results, dash, meta):
    biz  = results["at_business_threshold"]
    f1   = results["xgboost_at_f1_threshold"]
    test = results["test"]["XGBoost"]
    xgb_test = biz["test"]["XGBoost"]

    return {
        "threshold": meta["business_threshold"],
        "f1_threshold": results["champion"]["f1_threshold"],
        "cost_fn": meta["cost_fn"],
        "cost_fp": meta["cost_fp"],
        "glance": {
            "caught": xgb_test["TP"],
            "frauds": xgb_test["TP"] + xgb_test["FN"],
            "recall": xgb_test["Recall"],
            "precision": xgb_test["Precision"],
            "alerts_per_10k": r(xgb_test["Flag_rate"] * 1e4, 1),
            "ks": test["KS"],
            "auprc": test["AUPRC"],
        },
        "splits": dash["splits"],
        "split_hours": dash["hourly"]["split_hours"],
        "ranking": {
            split: {m: {k: results[split][m][k] for k in ("KS", "AUPRC", "F1")} for m in MODELS}
            for split in ("validate", "test")
        },
        "at_threshold": {
            m: {"precision": biz["test"][m]["Precision"], "flag_rate": biz["test"][m]["Flag_rate"]}
            for m in MODELS
        },
        "mean_score_validate": {m: results["validate"][m]["Mean_score"] for m in MODELS},
        "fraud_rate_validate": results["validate"]["XGBoost"]["Fraud_rate"],
        "frozen": {
            split: {
                "cost_optimal": {k: biz[split]["XGBoost"][k] for k in ("TP", "FP", "FN", "Recall", "Precision", "Cost")},
                "f1_optimal":   {k: f1[split][k] for k in ("TP", "FP", "FN", "Recall", "Precision", "Cost")},
            }
            for split in ("validate", "test")
        },
        "rationale": results["champion"]["rationale"],
        "drift": {
            "ks_validate": results["validate"]["XGBoost"]["KS"],
            "ks_test": test["KS"],
            "ks_trigger": ks_retrain_trigger(),
            "psi": dash["psi"],
            "hourly": {
                "hour": dash["hourly"]["hour"],
                "fraud_rate": [r(x, 6) for x in dash["hourly"]["fraud_rate"]],
                "fraud": dash["hourly"]["fraud"],
                "count": dash["hourly"]["count"],
            },
        },
    }


def build_threshold(dash, meta, results):
    curves = dash["cost_curves"]
    grid = curves["validate"]["threshold"]
    return {
        "threshold": grid,
        "marks": {
            "cost_optimal": grid.index(meta["business_threshold"]),
            "f1_optimal": grid.index(results["champion"]["f1_threshold"]),
        },
        "splits": {
            split: {
                "frauds": dash["splits"][split]["fraud"],
                "rows": dash["splits"][split]["rows"],
                "tp": curves[split]["tp"],
                "fp": curves[split]["fp"],
                "fn": curves[split]["fn"],
                "cost": curves[split]["cost"],
            }
            for split in ("validate", "test")
        },
    }


def main():
    results = load_json(EVAL_DIR / "evaluation_results.json")
    dash    = load_json(EVAL_DIR / "dashboard_data.json")
    meta    = load_json(ASSETS_DIR / "model_meta.json")
    demo    = pd.read_csv(ASSETS_DIR / "demo_transactions.csv", index_col="test_row")

    model = xgb.XGBClassifier()
    model.load_model(ASSETS_DIR / "xgboost_model.json")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "overview.json":  build_overview(results, dash, meta),
        "samples.json":   build_samples(meta, demo, model),
        "threshold.json": build_threshold(dash, meta, results),
    }
    for name, data in outputs.items():
        with open(OUT_DIR / name, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"wrote site/data/{name}")


if __name__ == "__main__":
    main()
