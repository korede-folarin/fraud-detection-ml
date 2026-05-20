import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from frauddetection import logger
from frauddetection.entity.config_entity import ModelTrainerConfig
from frauddetection.utils.common import save_object, load_object


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, params: dict):
        self.config = config
        self.params = params

    def get_scale_pos_weight(self, y_train: pd.Series) -> float:
        """
        Calculates scale_pos_weight for XGBoost
        ratio of legitimate to fraud transactions
        handles severe class imbalance without distorting distribution
        """
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")
        return scale_pos_weight

    def train_logistic_regression(self, 
                                   X_train_scaled: np.ndarray,
                                   y_train: pd.Series) -> LogisticRegression:
        """
        Trains Logistic Regression — interpretable FCA-aligned baseline
        class_weight='balanced' handles imbalance without SMOTE
        Scaled features required for LR convergence
        """
        logger.info("Training Logistic Regression...")

        lr = LogisticRegression(
            class_weight=self.params['logistic_regression']['class_weight'],
            max_iter=self.params['logistic_regression']['max_iter'],
            C=self.params['logistic_regression']['C'],
            solver=self.params['logistic_regression']['solver'],
            random_state=self.params['logistic_regression']['random_state']
        )
        lr.fit(X_train_scaled, y_train)
        logger.info("Logistic Regression trained ✅")
        return lr

    def train_random_forest(self,
                             X_train: pd.DataFrame,
                             y_train: pd.Series) -> RandomForestClassifier:
        """
        Trains Random Forest — ensemble benchmark
        class_weight='balanced' handles imbalance
        Raw features used — tree models scale invariant
        """
        logger.info("Training Random Forest...")

        rf = RandomForestClassifier(
            n_estimators=self.params['random_forest']['n_estimators'],
            max_depth=self.params['random_forest']['max_depth'],
            min_samples_split=self.params['random_forest']['min_samples_split'],
            min_samples_leaf=self.params['random_forest']['min_samples_leaf'],
            max_features=self.params['random_forest']['max_features'],
            class_weight=self.params['random_forest']['class_weight'],
            random_state=self.params['random_forest']['random_state'],
            n_jobs=self.params['random_forest']['n_jobs']
        )
        rf.fit(X_train, y_train)
        logger.info("Random Forest trained ✅")
        return rf

    def train_xgboost(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series) -> xgb.XGBClassifier:
        """
        Trains XGBoost — champion model for fraud detection
        scale_pos_weight calculated from actual class ratio
        Raw features used — XGBoost handles non-linearity internally
        """
        logger.info("Training XGBoost...")

        scale_pos_weight = self.get_scale_pos_weight(y_train)

        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=self.params['xgboost']['n_estimators'],
            max_depth=self.params['xgboost']['max_depth'],
            learning_rate=self.params['xgboost']['learning_rate'],
            subsample=self.params['xgboost']['subsample'],
            colsample_bytree=self.params['xgboost']['colsample_bytree'],
            min_child_weight=self.params['xgboost']['min_child_weight'],
            gamma=self.params['xgboost']['gamma'],
            random_state=self.params['xgboost']['random_state'],
            eval_metric=self.params['xgboost']['eval_metric'],
            n_jobs=self.params['xgboost']['n_jobs']
        )
        xgb_model.fit(X_train, y_train)
        logger.info("XGBoost trained ✅")
        return xgb_model

    def run(self):
        """
        Runs full model training pipeline:
        load data → train all models → save all models
        """
        logger.info("="*50)
        logger.info("Starting Model Training")
        logger.info("="*50)

        # load transformed data
        X_train        = load_object(Path(self.config.train_data_path).parent / 'X_train.pkl')
        X_train_scaled = load_object(Path(self.config.train_data_path).parent / 'X_train_scaled.pkl')
        y_train        = load_object(Path(self.config.train_data_path).parent / 'y_train.pkl')

        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Fraud rate: {y_train.mean()*100:.3f}%")

        # train all three models
        lr        = self.train_logistic_regression(X_train_scaled, y_train)
        rf        = self.train_random_forest(X_train, y_train)
        xgb_model = self.train_xgboost(X_train, y_train)

        # save models
        os.makedirs(self.config.model_path, exist_ok=True)

        save_object(Path(self.config.model_path) / 'logistic_regression.pkl', lr)
        save_object(Path(self.config.model_path) / 'random_forest.pkl', rf)
        save_object(Path(self.config.model_path) / 'xgboost.pkl', xgb_model)

        logger.info(f"Models saved to: {self.config.model_path}")
        logger.info("Model Training Complete ")

        return lr, rf, xgb_model