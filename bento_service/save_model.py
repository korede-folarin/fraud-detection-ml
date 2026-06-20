import bentoml
import pickle
import json
import shap

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

# build a SHAP TreeExplainer once at save-time
# TreeExplainer is fast — exact for tree models, no sampling needed
explainer = shap.TreeExplainer(model)

# save to BentoML model store
bento_model = bentoml.xgboost.save_model(
    "fraud_xgboost",
    model,
    custom_objects={
        "feature_names": features,
        "shap_explainer": explainer,
    },
    metadata={
        "ks": xgb_test["KS"],
        "auprc": xgb_test["AUPRC"],
        "business_threshold": champion["business_threshold"],
        "model_type": champion["model"],
    },
)

print(f"Model saved: {bento_model}")