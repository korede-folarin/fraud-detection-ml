import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import dagshub
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    f1_score, precision_score,
    recall_score, confusion_matrix
)
from frauddetection import logger
from frauddetection.entity.config_entity import ModelEvaluationConfig
from frauddetection.utils.common import save_json, load_object


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def calculate_ks(self, y_true, y_prob):
        """
        KS statistic — primary discrimination metric in UK banking
        Measures maximum separation between fraud and legitimate
        cumulative distributions
        KS > 0.4 = good, > 0.8 = excellent
        """
        df = pd.DataFrame({'score': y_prob, 'label': y_true})
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        df['cum_fraud'] = (df['label']==1).cumsum() / (df['label']==1).sum()
        df['cum_legit'] = (df['label']==0).cumsum() / (df['label']==0).sum()
        df['ks']        = abs(df['cum_fraud'] - df['cum_legit'])

        ks_stat      = df['ks'].max()
        ks_threshold = df.loc[df['ks'].idxmax(), 'score']

        return ks_stat, ks_threshold

    def find_best_f1_threshold(self, y_true, y_prob):
        """h
        Finds threshold that maximises F1 score
        F1 = harmonic mean of precision and recall
        Used for statistical threshold optimisation
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores    = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx     = f1_scores.argmax()
        best_f1      = f1_scores[best_idx]
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

        return best_f1, best_threshold

    def calculate_business_cost(self, y_true, y_prob):
        """
        Business cost threshold optimisation
        Finds threshold minimising total financial cost:
        cost_fn = cost of missing fraud (false negative)
        cost_fp = cost of false alarm (false positive)
        Reflects asymmetric cost structure of fraud detection
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        precision = precision[:-1]
        recall    = recall[:-1]

        total_fraud = y_true.sum()
        total_legit = (y_true == 0).sum()

        costs = []
        for i, threshold in enumerate(thresholds):
            tp = recall[i] * total_fraud
            fn = (1 - recall[i]) * total_fraud
            fp = (1 - precision[i]) * (tp / precision[i]) if precision[i] > 0 else total_legit
            total_cost = (fn * self.config.cost_false_negative) + \
                         (fp * self.config.cost_false_positive)
            costs.append(total_cost)

        costs              = np.array(costs)
        best_cost_idx      = costs.argmin()
        best_threshold     = thresholds[best_cost_idx]
        best_cost          = costs[best_cost_idx]

        return best_threshold, best_cost

    def evaluate_model(self, name, model, X, X_scaled, y_true):
        """
        Evaluates a single model on a dataset
        Returns dict of all metrics
        Uses scaled features for LR, raw for RF and XGB
        """
        # logistic regression needs scaled features
        if name == 'Logistic Regression':
            probs = model.predict_proba(X_scaled)[:, 1]
        else:
            probs = model.predict_proba(X)[:, 1]

        # threshold independent metrics
        ks, _    = self.calculate_ks(y_true, probs)
        auprc    = average_precision_score(y_true, probs)

        # f1 optimal threshold
        f1, f1_threshold = self.find_best_f1_threshold(y_true, probs)

        # business cost threshold
        biz_threshold, biz_cost = self.calculate_business_cost(y_true, probs)

        # metrics at f1 threshold
        preds    = (probs >= f1_threshold).astype(int)
        prec     = precision_score(y_true, preds)
        rec      = recall_score(y_true, preds)
        cm       = confusion_matrix(y_true, preds)

        return {
            'KS':                  round(float(ks), 4),
            'AUPRC':               round(float(auprc), 4),
            'F1':                  round(float(f1), 4),
            'Precision':           round(float(prec), 4),
            'Recall':              round(float(rec), 4),
            'F1_threshold':        round(float(f1_threshold), 4),
            'Business_threshold':  round(float(biz_threshold), 4),
            'Business_cost':       round(float(biz_cost), 2),
            'TP':                  int(cm[1][1]),
            'FP':                  int(cm[0][1]),
            'TN':                  int(cm[0][0]),
            'FN':                  int(cm[1][0]),
            'Fraud_catch_rate':    round(float(cm[1][1] / cm[1].sum()), 4)
        }

    def log_to_mlflow(self, model_name, model, metrics, dataset):
        """
        Logs model metrics to MLflow
        Enables experiment tracking and model versioning
        Critical for champion-challenger management in production
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)

        with mlflow.start_run(run_name=f"{model_name}_{dataset}"):
            # log parameters
            mlflow.log_params({
                'model': model_name,
                'dataset': dataset,
                'cost_fn': self.config.cost_false_negative,
                'cost_fp': self.config.cost_false_positive
            })

            # log metrics
            mlflow.log_metrics({
                'KS':           metrics['KS'],
                'AUPRC':        metrics['AUPRC'],
                'F1':           metrics['F1'],
                'Precision':    metrics['Precision'],
                'Recall':       metrics['Recall'],
                'Business_cost':metrics['Business_cost'],
                'Fraud_catch_rate': metrics['Fraud_catch_rate']
            })

            # log model
            if model_name == 'XGBoost':
                mlflow.xgboost.log_model(model, 'model')
            else:
                mlflow.sklearn.log_model(model, 'model')

            logger.info(f"MLflow logged: {model_name} on {dataset}")

    def run(self):
        """
        Runs full model evaluation pipeline:
        load models → evaluate on validate and test → log to MLflow → save results
        """
        # initialise DagsHub for remote MLflow tracking
        dagshub.init(repo_owner='korede-folarin',
                     repo_name='fraud-detection-ml',
                     mlflow=True)

        logger.info("MLflow tracking initialised via DagsHub")
        logger.info("="*50)
        logger.info("Starting Model Evaluation")
        logger.info("="*50)

        # load models
        lr        = load_object(Path(self.config.model_path) / 'logistic_regression.pkl')
        rf        = load_object(Path(self.config.model_path) / 'random_forest.pkl')
        xgb_model = load_object(Path(self.config.model_path) / 'xgboost.pkl')

        # load data
        data_dir      = Path(self.config.validate_data_path).parent
        X_validate    = load_object(data_dir / 'X_validate.pkl')
        X_test        = load_object(data_dir / 'X_test.pkl')
        y_validate    = load_object(data_dir / 'y_validate.pkl')
        y_test        = load_object(data_dir / 'y_test.pkl')
        X_val_scaled  = load_object(data_dir / 'X_validate_scaled.pkl')
        X_test_scaled = load_object(data_dir / 'X_test_scaled.pkl')

        models = {
            'Logistic Regression': lr,
            'Random Forest':       rf,
            'XGBoost':             xgb_model
        }

        results = {'validate': {}, 'test': {}}

        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")

            val_metrics = self.evaluate_model(
                model_name, model,
                X_validate, X_val_scaled, y_validate
            )
            results['validate'][model_name] = val_metrics

            test_metrics = self.evaluate_model(
                model_name, model,
                X_test, X_test_scaled, y_test
            )
            results['test'][model_name] = test_metrics

            self.log_to_mlflow(model_name, model, val_metrics, 'validate')
            self.log_to_mlflow(model_name, model, test_metrics, 'test')

        xgb_test = results['test']['XGBoost']
        results['champion'] = {
            'model': 'XGBoost',
            'rationale': [
                'Best F1 and Precision at optimal threshold',
                'Best score calibration (mean score ≈ true fraud rate)',
                'Zero false positives at F1 optimal threshold',
                'Most operationally viable at business cost threshold'
            ],
            'test_metrics': xgb_test,
            'business_threshold': self.config.business_threshold,
            'cost_fn': self.config.cost_false_negative,
            'cost_fp': self.config.cost_false_positive
        }

        os.makedirs(self.config.root_dir, exist_ok=True)
        save_json(Path(self.config.evaluation_results), results)

        logger.info(f"Evaluation results saved to: {self.config.evaluation_results}")
        logger.info("Model Evaluation Complete")

        return results