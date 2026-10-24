"""
Decision and explanation rules for the fraud API / demo
Run: pytest tests/
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from frauddetection.inference import (  # noqa: E402
    build_adverse_action_notice,
    decide,
    describe_feature,
    risk_level,
    score_transaction,
    shap_contributions,
)

THRESHOLD = 0.0716
ASSETS = ROOT / "dashboard" / "assets"


def test_negative_contributions_never_cited_as_reasons():
    contributions = {"V14": 1.2, "V4": -3.0, "Amount_log": 0.4, "V12": -0.1}
    notice = build_adverse_action_notice(contributions)

    assert "V14" in notice
    assert "transaction amount" in notice
    assert "V4" not in notice
    assert "V12" not in notice


def test_at_most_three_reasons_in_contribution_order():
    contributions = {"V1": 0.1, "V2": 0.9, "V3": 0.5, "V5": 0.7, "V6": 0.3}
    notice = build_adverse_action_notice(contributions)

    assert notice.index("V2") < notice.index("V5") < notice.index("V3")
    assert "V1" not in notice and "V6" not in notice


def test_generic_notice_when_no_positive_driver():
    notice = build_adverse_action_notice({"V1": -0.2, "V2": -0.5})
    assert "overall risk assessment" in notice
    assert "manual review" in notice


def test_pca_components_not_given_invented_meaning():
    assert describe_feature("V14") == "behavioural risk indicator (V14)"
    assert describe_feature("Hour") == "the time of day the transaction occurred"


def test_threshold_boundary_is_inclusive():
    assert decide(THRESHOLD, {"V14": 1.0}, THRESHOLD)["decision"] == "DECLINE"
    assert decide(THRESHOLD - 1e-6, {"V14": 1.0}, THRESHOLD)["decision"] == "APPROVE"


def test_approved_transactions_get_no_notice():
    result = decide(0.01, {"V14": 2.0}, THRESHOLD)
    assert result["decision"] == "APPROVE"
    assert result["adverse_action_notice"] is None
    assert result["risk_level"] == "CLEAR"


@pytest.mark.parametrize("prob, band", [(0.9, "HIGH"), (0.5, "MEDIUM"), (0.1, "LOW"), (0.01, "CLEAR")])
def test_risk_bands(prob, band):
    assert risk_level(prob, THRESHOLD) == band


@pytest.mark.skipif(not (ASSETS / "xgboost_model.json").exists(), reason="demo assets not built")
def test_shap_contributions_reconstruct_model_score():
    import xgboost as xgb

    meta = json.loads((ASSETS / "model_meta.json").read_text())
    model = xgb.XGBClassifier()
    model.load_model(ASSETS / "xgboost_model.json")

    demo = pd.read_csv(ASSETS / "demo_transactions.csv", index_col="test_row")
    row = demo[demo["Class"] == 1].iloc[[0]][meta["features"]]

    contributions, base = shap_contributions(model, row)
    prob = float(model.predict_proba(row)[0, 1])
    reconstructed = 1 / (1 + math.exp(-(base + sum(contributions.values()))))

    # SHAP values are additive in log-odds; compare in probability space
    # because float32 log-odds lose precision for scores near 1
    assert np.isclose(reconstructed, prob, atol=1e-5)

    result = score_transaction(model, row, meta["business_threshold"])
    assert result["fraud_probability"] == round(prob, 4)
