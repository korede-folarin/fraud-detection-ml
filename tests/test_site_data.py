"""
The static demo (site/) must replay exactly what the API would return
Run: pytest tests/
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from frauddetection.inference import score_transaction  # noqa: E402

SITE_DATA = ROOT / "site" / "data"
ASSETS = ROOT / "dashboard" / "assets"

pytestmark = pytest.mark.skipif(not (SITE_DATA / "samples.json").exists(), reason="site data not built")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def samples():
    return load(SITE_DATA / "samples.json")


def test_site_samples_match_api_decision_and_notice(samples):
    import xgboost as xgb

    meta = load(ASSETS / "model_meta.json")
    model = xgb.XGBClassifier()
    model.load_model(ASSETS / "xgboost_model.json")
    demo = pd.read_csv(ASSETS / "demo_transactions.csv", index_col="test_row")

    for kind, s in samples.items():
        row = demo.loc[[s["test_row"]], meta["features"]].astype(float)
        api = score_transaction(model, row, meta["business_threshold"])

        assert s["decision"] == api["decision"], kind
        assert s["notice"] == api["adverse_action_notice"], kind
        assert s["fraud_probability"] == pytest.approx(api["fraud_probability"], abs=5e-5), kind


def test_only_positive_contributions_named_as_reasons(samples):
    for kind, s in samples.items():
        if s["decision"] != "DECLINE":
            assert s["notice"] is None
            continue
        for d in s["shap"]:
            if d["value"] <= 0:
                assert d["label"] not in s["notice"], (kind, d["feature"])


def test_one_sample_per_type_with_both_outcomes(samples):
    assert set(samples) == {"fraud", "legitimate", "borderline", "false_alarm"}
    assert {s["decision"] for s in samples.values()} == {"APPROVE", "DECLINE"}


def test_headline_figures_come_from_evaluation_results():
    overview = load(SITE_DATA / "overview.json")
    results = load(ROOT / "artifacts" / "model_evaluation" / "evaluation_results.json")
    xgb_test = results["at_business_threshold"]["test"]["XGBoost"]

    assert overview["threshold"] == results["champion"]["business_threshold"]
    assert overview["glance"]["caught"] == xgb_test["TP"]
    assert overview["glance"]["precision"] == xgb_test["Precision"]
    assert overview["glance"]["ks"] == results["test"]["XGBoost"]["KS"]
