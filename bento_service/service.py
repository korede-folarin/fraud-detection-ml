import bentoml
import pandas as pd
from pydantic import BaseModel
from typing import List


# ── feature name → plain English mapping ──────────────────────
FEATURE_DESCRIPTIONS = {
    "Amount_log": "the transaction amount",
    "Hour": "the time of day the transaction occurred",
    "rolling_std_amount": "recent volatility in spending pattern",
}


def describe_feature(name: str) -> str:
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    return f"behavioural risk indicator ({name})"


class TransactionInput(BaseModel):
    features: List[float]


class FraudPrediction(BaseModel):
    fraud_probability: float
    decision: str
    threshold_used: float
    risk_level: str
    top_features: dict
    adverse_action_notice: str | None
    message: str


@bentoml.service(
    name="fraud_detection",
    resources={"cpu": "1"},
    traffic={"timeout": 10},
)
class FraudDetectionService:
    bento_model = bentoml.models.get("fraud_xgboost:latest")

    def __init__(self):
        self.model = self.bento_model.load_model()
        self.feature_names = self.bento_model.custom_objects["feature_names"]
        self.business_threshold = self.bento_model.info.metadata["business_threshold"]
        self.explainer = self.bento_model.custom_objects["shap_explainer"]

    @bentoml.api
    def predict(self, transaction: TransactionInput) -> FraudPrediction:
        if len(transaction.features) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, "
                f"got {len(transaction.features)}"
            )

        X = pd.DataFrame([transaction.features], columns=self.feature_names)

        prob = self.model.predict_proba(X)
        fraud_prob = float(prob[0][1])

        decision = "DECLINE" if fraud_prob >= self.business_threshold else "APPROVE"

        if fraud_prob >= 0.75:
            risk_level = "HIGH"
        elif fraud_prob >= 0.25:
            risk_level = "MEDIUM"
        elif fraud_prob >= self.business_threshold:
            risk_level = "LOW"
        else:
            risk_level = "CLEAR"

        shap_values = self.explainer.shap_values(X)
        contributions = dict(zip(self.feature_names, shap_values[0]))

        fraud_drivers = sorted(
            contributions.items(), key=lambda x: x[1], reverse=True
        )

        top_features = {
            name: round(float(value), 4) for name, value in fraud_drivers[:5]
        }

        adverse_action_notice = None
        if decision == "DECLINE":
            top_reasons = [
                name for name, value in fraud_drivers if value > 0
            ][:3]

            if top_reasons:
                reasons_text = ", ".join(
                    describe_feature(name) for name in top_reasons
                )
                adverse_action_notice = (
                    f"This transaction was declined. The principal factors "
                    f"contributing to this decision were: {reasons_text}. "
                    f"This is an automated decision; you may request a "
                    f"manual review."
                )
            else:
                adverse_action_notice = (
                    "This transaction was declined based on an overall "
                    "risk assessment. You may request a manual review."
                )

        message = (
            f"Transaction {'flagged as likely fraud' if decision == 'DECLINE' else 'approved'}. "
            f"Fraud probability: {fraud_prob:.1%}. "
            f"Business threshold: {self.business_threshold:.4f}."
        )

        return FraudPrediction(
            fraud_probability=round(fraud_prob, 4),
            decision=decision,
            threshold_used=self.business_threshold,
            risk_level=risk_level,
            top_features=top_features,
            adverse_action_notice=adverse_action_notice,
            message=message,
        )