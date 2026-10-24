import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import precision_recall_curve
from frauddetection import logger
from frauddetection.utils.common import load_object


class DashboardExport:
    """
    Writes everything the Streamlit dashboard plots, computed from the real
    model scores — no chart in the dashboard is simulated:
      - dashboard_data.json     curves, cost sweeps, drift, importances
      - assets/xgboost_model.json + model_meta.json   champion for the live demo
      - assets/demo_transactions.csv   real out-of-time test transactions
    """

    POINTS = 200

    def __init__(self, evaluation):
        self.evaluation = evaluation
        self.config = evaluation.config

    def ks_curve(self, y_true, y_prob):
        """
        Cumulative share of fraud and legitimate transactions captured
        as the review population grows from highest score downwards
        """
        order  = np.argsort(-y_prob, kind='mergesort')
        labels = np.asarray(y_true)[order]
        cum_fraud = np.cumsum(labels == 1) / (labels == 1).sum()
        cum_legit = np.cumsum(labels == 0) / (labels == 0).sum()
        population = np.arange(1, len(labels) + 1) / len(labels)

        # dense sampling near the top of the ranking, where the curves move
        idx = np.unique(np.r_[
            np.geomspace(1, len(labels), self.POINTS).astype(int) - 1,
            np.argmax(np.abs(cum_fraud - cum_legit))
        ])
        return {
            'population': population[idx].round(6).tolist(),
            'fraud':      cum_fraud[idx].round(4).tolist(),
            'legit':      cum_legit[idx].round(4).tolist(),
        }

    def pr_curve(self, y_true, y_prob):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        # keep both ends of every recall step — the curve only changes shape there
        step = np.flatnonzero(np.diff(recall) != 0)
        idx  = np.unique(np.r_[0, step, step + 1, len(recall) - 1])
        return {
            'precision': precision[idx].round(4).tolist(),
            'recall':    recall[idx].round(4).tolist(),
        }

    def cost_curve(self, y_true, y_prob, marked=()):
        """
        Exact confusion counts on a log-spaced threshold grid
        The dashboard threshold slider reads from this grid
        """
        y_true     = np.asarray(y_true)
        # marked thresholds (deployment, F1) are added so the grid hits them exactly
        grid       = np.unique(np.r_[np.geomspace(1e-4, 0.9999, 300), 0.5, list(marked)].round(6))
        fraud_desc = np.sort(y_prob[y_true == 1])
        legit_desc = np.sort(y_prob[y_true == 0])

        # count of scores >= t for every grid threshold
        tp = len(fraud_desc) - np.searchsorted(fraud_desc, grid, side='left')
        fp = len(legit_desc) - np.searchsorted(legit_desc, grid, side='left')
        fn = len(fraud_desc) - tp
        cost = fn * self.config.cost_false_negative + fp * self.config.cost_false_positive

        return {
            'threshold': grid.tolist(),
            'tp': tp.tolist(), 'fp': fp.tolist(), 'fn': fn.tolist(),
            'cost': cost.astype(float).tolist(),
        }

    def hourly_fraud(self):
        """
        Fraud rate per elapsed hour across the 48-hour window,
        plus the hour boundaries of the 70/15/15 out-of-time split
        """
        raw = pd.read_csv(self.config.raw_data_path, usecols=['Time', 'Class'])
        raw = raw.sort_values('Time').reset_index(drop=True)
        raw['hour'] = (raw['Time'] // 3600).astype(int)

        hourly = raw.groupby('hour')['Class'].agg(['mean', 'sum', 'count'])

        n = len(raw)
        params = self.evaluation_params()
        cut_validate = raw.loc[int(n * params['train_ratio']), 'Time'] / 3600
        cut_test     = raw.loc[int(n * params['validate_ratio']), 'Time'] / 3600

        return {
            'hour':       hourly.index.tolist(),
            'fraud_rate': hourly['mean'].round(6).tolist(),
            'fraud':      hourly['sum'].astype(int).tolist(),
            'count':      hourly['count'].astype(int).tolist(),
            'split_hours': {'validate_start': round(float(cut_validate), 2),
                            'test_start':     round(float(cut_test), 2)},
        }

    def evaluation_params(self):
        from frauddetection.constant import PARAMS_FILE_PATH
        from frauddetection.utils.common import read_yaml
        return read_yaml(PARAMS_FILE_PATH).data_transformation

    def psi(self, expected, actual, bins=10):
        """
        Population Stability Index — train distribution vs test distribution
        < 0.10 stable · 0.10–0.25 moderate shift · > 0.25 significant shift
        """
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
        e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
        return float(np.sum((a - e) * np.log(a / e)))

    def importances(self, xgb_model, X_test):
        booster  = xgb_model.get_booster()
        contribs = booster.predict(xgb.DMatrix(X_test), pred_contribs=True)[:, :-1]
        shap_imp = pd.Series(np.abs(contribs).mean(axis=0), index=X_test.columns)

        gain = pd.Series(booster.get_score(importance_type='gain')).reindex(X_test.columns).fillna(0)
        iv   = pd.read_csv(self.config.iv_rankings_path, index_col=0)['IV'].reindex(X_test.columns)

        top = shap_imp.sort_values(ascending=False).index[:12]
        return {
            'features': top.tolist(),
            'shap':     shap_imp[top].round(5).tolist(),
            'gain':     (gain[top] / gain.max()).round(4).tolist(),
            'iv':       iv[top].round(4).tolist(),
        }

    def demo_assets(self, xgb_model, X_test, y_test, xgb_scores, business_threshold):
        """
        Champion model in XGBoost's portable JSON format plus a sample of
        real out-of-time test transactions for the interactive demo
        """
        out = Path(self.config.demo_assets_dir)
        os.makedirs(out, exist_ok=True)

        xgb_model.save_model(out / 'xgboost_model.json')

        df = X_test.copy()
        df['Class'] = np.asarray(y_test)
        df['score'] = xgb_scores.round(6)

        rng      = np.random.default_rng(42)
        fraud    = df[df['Class'] == 1]
        legit    = df[df['Class'] == 0]
        # every false alarm the champion raises, plus a random legit sample
        alarms   = legit[legit['score'] >= business_threshold]
        rest     = legit.drop(alarms.index)
        sample   = rest.iloc[rng.choice(len(rest), 450, replace=False)]
        demo     = pd.concat([fraud, alarms, sample]).sort_index()
        demo.index.name = 'test_row'
        demo.to_csv(out / 'demo_transactions.csv')

        meta = {
            'features': X_test.columns.tolist(),
            'business_threshold': round(float(business_threshold), 6),
            'cost_fn': self.config.cost_false_negative,
            'cost_fp': self.config.cost_false_positive,
            'feature_ranges': {
                c: [round(float(X_test[c].quantile(0.001)), 4), round(float(X_test[c].quantile(0.999)), 4)]
                for c in X_test.columns
            },
        }
        with open(out / 'model_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Demo assets written to: {out} ({len(demo)} transactions)")

    def run(self, models, scores, labels, results, X_test):
        logger.info("Exporting dashboard data...")

        X_train = load_object(self.config.train_data_path)
        y_train = load_object(Path(self.config.train_data_path).parent / 'y_train.pkl')
        business_threshold = results['champion']['business_threshold']

        data = {
            'splits': {
                'train':    {'rows': int(len(y_train)), 'fraud': int(y_train.sum()), 'fraud_rate': round(float(y_train.mean()), 6)},
                'validate': {'rows': int(len(labels['validate'])), 'fraud': int(labels['validate'].sum()), 'fraud_rate': round(float(labels['validate'].mean()), 6)},
                'test':     {'rows': int(len(labels['test'])), 'fraud': int(labels['test'].sum()), 'fraud_rate': round(float(labels['test'].mean()), 6)},
            },
            'ks_curves':   {s: {m: self.ks_curve(labels[s], scores[s][m]) for m in models} for s in scores},
            'pr_curves':   {s: {m: self.pr_curve(labels[s], scores[s][m]) for m in models} for s in scores},
            'cost_curves': {s: self.cost_curve(labels[s], scores[s]['XGBoost'],
                                               (business_threshold, results['champion']['f1_threshold']))
                            for s in scores},
            'hourly':      self.hourly_fraud(),
            'importance':  self.importances(models['XGBoost'], X_test),
        }

        top = data['importance']['features'][:8]
        data['psi'] = {f: round(self.psi(X_train[f].values, X_test[f].values), 4) for f in top}

        with open(self.config.dashboard_data, 'w') as f:
            json.dump(data, f)
        logger.info(f"Dashboard data saved to: {self.config.dashboard_data}")

        self.demo_assets(models['XGBoost'], X_test, labels['test'], scores['test']['XGBoost'], business_threshold)
