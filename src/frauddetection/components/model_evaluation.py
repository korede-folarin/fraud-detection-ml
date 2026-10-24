import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score, confusion_matrix
)
from frauddetection import logger
from frauddetection.entity.config_entity import ModelEvaluationConfig
from frauddetection.utils.common import save_json, load_object


def floor_6dp(threshold):
    """
    Rounds a threshold down for storage — rounding up would silently stop
    flagging the transaction that sits exactly on the threshold
    """
    return float(np.floor(threshold * 1e6) / 1e6)


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
        """
        Finds threshold that maximises F1 score
        F1 = harmonic mean of precision and recall
        Used for statistical threshold optimisation
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        precision, recall = precision[:-1], recall[:-1]
        f1_scores    = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx     = f1_scores.argmax()

        return f1_scores[best_idx], thresholds[best_idx]

    def threshold_sweep(self, y_true, y_prob):
        """
        Exact confusion counts at every distinct score threshold
        Rule: flag when score >= threshold
        Returns DataFrame with threshold, tp, fp, fn, cost
        """
        y_true = np.asarray(y_true)
        order  = np.argsort(-y_prob, kind='mergesort')
        scores = y_prob[order]
        labels = y_true[order]

        tp = np.cumsum(labels == 1)
        fp = np.cumsum(labels == 0)

        # keep the last index of each distinct score — ties are flagged together
        last = np.r_[scores[1:] != scores[:-1], True]
        tp, fp, thresholds = tp[last], fp[last], scores[last]
        fn = (y_true == 1).sum() - tp

        cost = fn * self.config.cost_false_negative + fp * self.config.cost_false_positive
        return pd.DataFrame({'threshold': thresholds, 'tp': tp, 'fp': fp, 'fn': fn, 'cost': cost})

    def calculate_business_cost(self, y_true, y_prob):
        """
        Business cost threshold optimisation
        Finds threshold minimising total financial cost:
        cost_fn = cost of missing fraud (false negative)
        cost_fp = cost of false alarm (false positive)
        Reflects asymmetric cost structure of fraud detection
        """
        sweep = self.threshold_sweep(y_true, y_prob)
        best  = sweep.loc[sweep['cost'].idxmin()]
        return best['threshold'], best['cost']

    def metrics_at_threshold(self, y_true, y_prob, threshold):
        """
        Operating-point metrics for a fixed decision threshold
        Used to compare all models at the champion's deployment threshold
        """
        y_true = np.asarray(y_true)
        flagged = y_prob >= threshold
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        fn = int((~flagged & (y_true == 1)).sum())
        tn = int((~flagged & (y_true == 0)).sum())

        return {
            'threshold':  round(float(threshold), 6),
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Precision':  round(tp / (tp + fp), 4) if tp + fp else 0.0,
            'Recall':     round(tp / (tp + fn), 4) if tp + fn else 0.0,
            'Flag_rate':  round(float(flagged.mean()), 6),
            'Cost':       float(fn * self.config.cost_false_negative + fp * self.config.cost_false_positive)
        }

    def predict(self, name, model, X, X_scaled):
        """
        Logistic regression needs scaled features, RF and XGB use raw
        """
        if name == 'Logistic Regression':
            return model.predict_proba(X_scaled)[:, 1]
        return model.predict_proba(X)[:, 1]

    def evaluate_model(self, probs, y_true):
        """
        Evaluates a single model's scores on a dataset
        Returns dict of all metrics
        """
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
            'Mean_score':          round(float(probs.mean()), 6),
            'Fraud_rate':          round(float(np.mean(y_true)), 6),
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
        import mlflow
        import mlflow.xgboost
        import mlflow.sklearn

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
        load models → evaluate on validate and test → pick business threshold
        on validate → apply it unchanged to test → log to MLflow → save results
        """
        # remote MLflow tracking only when a DagsHub token is available
        log_mlflow = bool(os.getenv('DAGSHUB_TOKEN'))
        if log_mlflow:
            import mlflow
            os.environ['MLFLOW_TRACKING_URI'] = self.config.mlflow_uri
            os.environ['MLFLOW_TRACKING_USERNAME'] = 'korede-folarin'
            os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN')
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            logger.info("MLflow tracking initialised via DagsHub")
        else:
            logger.warning("DAGSHUB_TOKEN not set — skipping MLflow logging")

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
        scores  = {'validate': {}, 'test': {}}

        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")

            scores['validate'][model_name] = self.predict(model_name, model, X_validate, X_val_scaled)
            scores['test'][model_name]     = self.predict(model_name, model, X_test, X_test_scaled)

            val_metrics  = self.evaluate_model(scores['validate'][model_name], y_validate)
            test_metrics = self.evaluate_model(scores['test'][model_name], y_test)
            results['validate'][model_name] = val_metrics
            results['test'][model_name]     = test_metrics

            if log_mlflow:
                self.log_to_mlflow(model_name, model, val_metrics, 'validate')
                self.log_to_mlflow(model_name, model, test_metrics, 'test')

        # deployment threshold is chosen on validate only, then frozen for test
        business_threshold = floor_6dp(
            self.calculate_business_cost(y_validate, scores['validate']['XGBoost'])[0]
        )
        f1_threshold = floor_6dp(
            self.find_best_f1_threshold(y_validate, scores['validate']['XGBoost'])[1]
        )
        logger.info(f"Business threshold (validate cost-optimal): {business_threshold:.4f}")

        labels = {'validate': y_validate, 'test': y_test}
        results['at_business_threshold'] = {
            split: {
                m: self.metrics_at_threshold(labels[split], scores[split][m], business_threshold)
                for m in models
            }
            for split in ('validate', 'test')
        }
        results['xgboost_at_f1_threshold'] = {
            split: self.metrics_at_threshold(labels[split], scores[split]['XGBoost'], f1_threshold)
            for split in ('validate', 'test')
        }

        results['champion'] = {
            'model': 'XGBoost',
            'rationale': [
                'Best score calibration (mean score closest to true fraud rate)',
                'Only model whose alerts stay usable at the deployment threshold',
                'Threshold chosen on validate by minimising business cost, frozen for test'
            ],
            'test_metrics': results['test']['XGBoost'],
            'business_threshold': floor_6dp(business_threshold),
            'f1_threshold': floor_6dp(f1_threshold),
            'cost_fn': self.config.cost_false_negative,
            'cost_fp': self.config.cost_false_positive
        }

        os.makedirs(self.config.root_dir, exist_ok=True)
        save_json(Path(self.config.evaluation_results), results)
        logger.info(f"Evaluation results saved to: {self.config.evaluation_results}")

        from frauddetection.components.dashboard_export import DashboardExport
        DashboardExport(self).run(models, scores, labels, results, X_test)

        logger.info("Model Evaluation Complete")

        return results
