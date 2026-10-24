"""
Shared scoring and explanation logic
Used by both the BentoML API (bento_service/service.py) and the Streamlit
demo (dashboard/app.py) so the decision a recruiter sees in the browser is
exactly the decision the API would return
"""
import numpy as np
import pandas as pd


# ── feature name → plain English mapping ──────────────────────
# only engineered features have a known meaning; V1-V28 are anonymised
# PCA components, so they are never given an invented interpretation
FEATURE_DESCRIPTIONS = {
    "Amount_log": "the transaction amount",
    "Hour": "the time of day the transaction occurred",
    "rolling_std_amount": "recent volatility in spending pattern",
}

MAX_REASONS = 3


def describe_feature(name: str) -> str:
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    return f"behavioural risk indicator ({name})"


def risk_level(fraud_prob: float, threshold: float) -> str:
    if fraud_prob >= 0.75:
        return "HIGH"
    if fraud_prob >= 0.25:
        return "MEDIUM"
    if fraud_prob >= threshold:
        return "LOW"
    return "CLEAR"


def shap_contributions(model, X: pd.DataFrame):
    """
    Exact TreeSHAP values for one row via XGBoost's native pred_contribs
    Identical to shap.TreeExplainer output (log-odds scale) without the
    shap dependency. Returns ({feature: contribution}, base_value)
    """
    import xgboost as xgb

    booster = model.get_booster() if hasattr(model, "get_booster") else model
    raw = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    return dict(zip(X.columns, raw[0, :-1].astype(float))), float(raw[0, -1])


def build_adverse_action_notice(contributions: dict) -> str:
    """
    Plain-English decline reason
    Only positive SHAP contributions are cited: a feature that pushed the
    score away from fraud must never be given as a reason for blocking
    """
    reasons = [
        name for name, value in sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        if value > 0
    ][:MAX_REASONS]

    if not reasons:
        return (
            "This transaction was declined based on an overall "
            "risk assessment. You may request a manual review."
        )

    reasons_text = ", ".join(describe_feature(name) for name in reasons)
    return (
        f"This transaction was declined. The principal factors "
        f"contributing to this decision were: {reasons_text}. "
        f"This is an automated decision; you may request a "
        f"manual review."
    )


def decide(fraud_prob: float, contributions: dict, threshold: float) -> dict:
    """
    Turns a score and its SHAP contributions into the API response
    Pure function — no model needed, so the decision rules are unit tested
    """
    decision = "DECLINE" if fraud_prob >= threshold else "APPROVE"
    drivers = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

    return {
        "fraud_probability": round(float(fraud_prob), 4),
        "decision": decision,
        "threshold_used": float(threshold),
        "risk_level": risk_level(fraud_prob, threshold),
        "top_features": {name: round(float(v), 4) for name, v in drivers[:5]},
        "adverse_action_notice": (
            build_adverse_action_notice(contributions) if decision == "DECLINE" else None
        ),
        "message": (
            f"Transaction {'flagged as likely fraud' if decision == 'DECLINE' else 'approved'}. "
            f"Fraud probability: {fraud_prob:.1%}. "
            f"Business threshold: {threshold:.4f}."
        ),
    }


def score_transaction(model, X: pd.DataFrame, threshold: float) -> dict:
    """
    Scores a single-row DataFrame (columns in model feature order)
    """
    fraud_prob = float(model.predict_proba(X)[0, 1])
    contributions, _ = shap_contributions(model, X)
    return decide(fraud_prob, contributions, threshold)
