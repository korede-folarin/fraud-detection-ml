"""
Fraud scoring API, v2 — v1 decisions plus service monitoring (D13)

Plain English: the same model and the same customer notices as v1, but the
service now reports how it is doing: how many requests, how fast, how many
fail, and what scores it is handing out — so a sudden shift in scores is
visible on a dashboard, not discovered weeks later.

Technical: a BentoML service that reuses frauddetection.inference (identical
decisions to v1, the dashboard and the demo site) and adds Prometheus metrics
with prometheus_client. BentoML already exports request count and latency per
endpoint at /metrics; this adds:
  fraud_api_predictions_total{decision,risk_level}  counter
  fraud_api_fraud_probability                       histogram of scores
  fraud_api_inference_seconds                       model + SHAP time only
  fraud_api_rejected_requests_total{reason}         input-validation failures
  fraud_api_model_info{...}                         baseline facts as labels
Malformed input now returns HTTP 400 (v1 returned 500, see D12).

What the code does: loads the champion registered from v1_frozen, scores each
transaction, records the metrics, returns the v1 response shape.
"""
import sys
import time
from pathlib import Path

import bentoml
import pandas as pd
from bentoml.exceptions import InvalidArgument
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from frauddetection.inference import score_transaction  # noqa: E402

# score buckets are dense below the 0.0716 deployment threshold, where almost all traffic sits
SCORE_BUCKETS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.07159, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)

PREDICTIONS = Counter("fraud_api_predictions_total", "Scored transactions", ["decision", "risk_level"])
SCORES = Histogram("fraud_api_fraud_probability", "Fraud probability returned", buckets=SCORE_BUCKETS)
INFERENCE = Histogram("fraud_api_inference_seconds", "Model + SHAP time per request",
                      buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25))
REJECTED = Counter("fraud_api_rejected_requests_total", "Requests rejected before scoring", ["reason"])
MODEL_INFO = Gauge("fraud_api_model_info", "Champion model in service (value is always 1)",
                   ["model_tag", "baseline", "threshold", "v1_validate_flag_rate"])


class TransactionInput(BaseModel):
    features: list[float]


class FraudPrediction(BaseModel):
    fraud_probability: float
    decision: str
    threshold_used: float
    risk_level: str
    top_features: dict
    adverse_action_notice: str | None
    message: str


@bentoml.service(
    name="fraud_detection_v2",
    resources={"cpu": "1"},
    traffic={"timeout": 10},
)
class FraudDetectionV2:
    bento_model = bentoml.models.BentoModel("fraud_xgboost:latest")

    def __init__(self):
        self.model = self.bento_model.load_model()
        self.feature_names = self.bento_model.custom_objects["feature_names"]
        self.threshold = self.bento_model.info.metadata["business_threshold"]
        MODEL_INFO.labels(
            model_tag=str(self.bento_model.tag),
            baseline=self.bento_model.info.labels.get("baseline", "unknown"),
            threshold=str(self.threshold),
            # v1 validate flag rate at this threshold (evaluation_results.json): the
            # reference a live decline rate is compared against on the dashboard
            v1_validate_flag_rate="0.002271",
        ).set(1)

    @bentoml.api
    def predict(self, transaction: TransactionInput) -> FraudPrediction:
        if len(transaction.features) != len(self.feature_names):
            REJECTED.labels(reason="wrong_feature_count").inc()
            raise InvalidArgument(
                f"Expected {len(self.feature_names)} features, got {len(transaction.features)}"
            )

        X = pd.DataFrame([transaction.features], columns=self.feature_names)
        start = time.perf_counter()
        result = score_transaction(self.model, X, self.threshold)
        INFERENCE.observe(time.perf_counter() - start)

        SCORES.observe(result["fraud_probability"])
        PREDICTIONS.labels(decision=result["decision"], risk_level=result["risk_level"]).inc()
        return FraudPrediction(**result)
