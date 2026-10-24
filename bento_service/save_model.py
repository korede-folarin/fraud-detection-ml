"""
Registers the champion model in the BentoML model store
Run from the repository root: python bento_service/save_model.py
"""
import bentoml
import pickle
import json

# load the champion model
with open("artifacts/model_trainer/models/xgboost.pkl", "rb") as f:
    model = pickle.load(f)

# load feature order
with open("artifacts/data_transformation/final_features.json") as f:
    features = json.load(f)["features"]

# load evaluation results — single source of truth
with open("artifacts/model_evaluation/evaluation_results.json") as f:
    results = json.load(f)

champion = results["champion"]
xgb_test = results["test"]["XGBoost"]

# SHAP values come from XGBoost's native pred_contribs at request time
# (exact TreeSHAP), so no explainer object needs to be stored
bento_model = bentoml.xgboost.save_model(
    "fraud_xgboost",
    model,
    custom_objects={
        "feature_names": features,
    },
    metadata={
        "ks": xgb_test["KS"],
        "auprc": xgb_test["AUPRC"],
        "business_threshold": champion["business_threshold"],
        "model_type": champion["model"],
    },
)

print(f"Model saved: {bento_model}")
