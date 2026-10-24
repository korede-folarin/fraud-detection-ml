import sys
from pathlib import Path

import bentoml
import pandas as pd
from pydantic import BaseModel
from typing import List

# shared scoring + explanation logic (same code the Streamlit demo runs)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from frauddetection.inference import score_transaction  # noqa: E402


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

    @bentoml.api
    def predict(self, transaction: TransactionInput) -> FraudPrediction:
        if len(transaction.features) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, "
                f"got {len(transaction.features)}"
            )

        X = pd.DataFrame([transaction.features], columns=self.feature_names)
        return FraudPrediction(**score_transaction(self.model, X, self.business_threshold))
