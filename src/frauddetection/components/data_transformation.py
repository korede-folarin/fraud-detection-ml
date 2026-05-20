import os 
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from optbinning import OptimalBinning
from frauddetection import logger
from frauddetection.entity.config_entity import DataTransformationConfig
from frauddetection.utils.common import save_json,save_object

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config 
    
    def create_features(self, df:pd.DataFrame) -> pd.DataFrame: 
    
        logger.info("Engineering features...")
        df = df.sort_values('Time').reset_index(drop=True)

        # log transform amount
        df['Amount_log'] = np.log1p(
            df['Amount'] * self.config.amount_log_base
            )

        # hour of day
        df['Hour'] = (
            (df['Time'] // self.config.seconds_per_hour) % 24
            ).astype(int)

        df['rolling_std_amount'] = (
            df['Amount']
            .rolling(window=self.config.rolling_window, 
                min_periods=self.config.rolling_min_periods)
            .std()
            .fillna(0)
        )
        
        df['rolling_fraud_density'] = (
            df['Class']
            .rolling(window=self.config.rolling_window, 
                     min_periods=self.config.rolling_min_periods)
            .mean()*100
        )

        # rolling transaction count — excluded later due to zero IV
        df['rolling_txn_count'] = (
            df['Amount']
            .rolling(window=self.config.rolling_window, min_periods=self.config.rolling_min_periods)
            .count()
        )

        # rolling average amount — excluded later due to low IV
        df['rolling_avg_amount'] = (
            df['Amount']
            .rolling(window=self.config.rolling_window, min_periods=self.config.rolling_min_periods)
            .mean()
        )


        logger.info(f"Feature engineered. Shape:{df.shape}")
        return df 


    def split_data(self, df:pd.DataFrame):
        logger.info("Splitting data using out-of-time split...")


        train_idx    = int(len(df) * self.config.train_ratio)
        validate_idx = int(len(df) * self.config.validate_ratio)

        train    = df.iloc[:train_idx]
        validate = df.iloc[train_idx:validate_idx]
        test     = df.iloc[validate_idx:]

        logger.info(f"Train:    {train.shape} | Fraud rate: {train['Class'].mean()*100:.3f}%")
        logger.info(f"Validate: {validate.shape} | Fraud rate: {validate['Class'].mean()*100:.3f}%")
        logger.info(f"Test:     {test.shape} | Fraud rate: {test['Class'].mean()*100:.3f}%")

        return train, validate, test

    def calculate_iv(self, X_train: pd.DataFrame, 
                     y_train: pd.Series) -> pd.DataFrame:
        """
        Calculates Information Value for all features
        Fitted on TRAIN only — prevents leakage into validate/test
        IV used for feature selection
        """
        logger.info("Calculating Information Value on train set only...")

        iv_results = {}
        for feature in X_train.columns:
            try:
                optb = OptimalBinning(
                    name=feature, dtype='numerical', solver='cp'
                )
                optb.fit(X_train[feature].values, y_train.values)
                iv = optb.binning_table.build()['IV'].iloc[-1]
                iv_results[feature] = iv
            except Exception as e:
                logger.warning(f"{feature}: IV calculation failed - {e}")
                iv_results[feature] = 0.0

        iv_df = pd.DataFrame.from_dict(
            iv_results, orient='index', columns=['IV']
        ).sort_values('IV', ascending=False)

        logger.info("IV calculation complete")
        return iv_df

    def select_features(self, iv_df: pd.DataFrame) -> list:
        """
        Selects features based on IV threshold from params.yaml
        Also removes predefined columns (Time, Amount etc)
        Returns final feature list
        """
        # features to always drop regardless of IV
        always_drop = self.config.features_to_drop + ['Class']

        # features above IV threshold
        strong_features = iv_df[
            iv_df['IV'] >= self.config.iv_threshold
        ].index.tolist()

        # remove always_drop from strong features
        final_features = [
            f for f in strong_features if f not in always_drop
        ]

        logger.info(f"Features selected: {len(final_features)}")
        logger.info(f"Features dropped (always drop): {always_drop}")
        logger.info(f"Features dropped (low IV): {set(iv_df.index) - set(final_features) - set(always_drop)}")

        return final_features

    def scale_features(self, X_train: pd.DataFrame,
                       X_validate: pd.DataFrame,
                       X_test: pd.DataFrame):
        """
        Fits StandardScaler on train only
        Transforms validate and test using train statistics
        Never refit on validate or test — prevents leakage
        Required for Logistic Regression
        """
        logger.info("Scaling features — fit on train only...")

        scaler = StandardScaler()
        X_train_scaled    = scaler.fit_transform(X_train)
        X_validate_scaled = scaler.transform(X_validate)
        X_test_scaled     = scaler.transform(X_test)

        logger.info("Scaling complete ✅")
        return scaler, X_train_scaled, X_validate_scaled, X_test_scaled

    def run(self):
        """
        Runs full data transformation pipeline:
        engineer features → split → IV → select → scale → save
        """
        logger.info("="*50)
        logger.info("Starting Data Transformation")
        logger.info("="*50)

        # load raw data
        # load raw data saved by data ingestion component
        raw_data_path = Path(self.config.root_dir).parent / 'data_ingestion' / 'raw_data.csv'
        logger.info(f"Loading raw data from: {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        logger.info(f"Raw data loaded: {df.shape}")
                        
            
        

        # engineer features
        df = self.create_features(df)

        # split data
        train, validate, test = self.split_data(df)

        # separate features and target
        X_train    = train.drop(columns=['Class'])
        X_validate = validate.drop(columns=['Class'])
        X_test     = test.drop(columns=['Class'])
        y_train    = train['Class']
        y_validate = validate['Class']
        y_test     = test['Class']

        # drop always-drop features
        always_drop = [
            f for f in self.config.features_to_drop
            if f in X_train.columns
        ]
        X_train    = X_train.drop(columns=always_drop, errors='ignore')
        X_validate = X_validate.drop(columns=always_drop, errors='ignore')
        X_test     = X_test.drop(columns=always_drop, errors='ignore')

        # calculate IV on train only
        iv_df = self.calculate_iv(X_train, y_train)

        # select features based on IV
        final_features = self.select_features(iv_df)
        X_train    = X_train[final_features]
        X_validate = X_validate[final_features]
        X_test     = X_test[final_features]

        # scale features
        scaler, X_train_sc, X_validate_sc, X_test_sc = self.scale_features(
            X_train, X_validate, X_test
        )

        # save everything
        os.makedirs(self.config.root_dir, exist_ok=True)

        save_object(Path(self.config.root_dir) / 'scaler.pkl', scaler)
        save_object(Path(self.config.root_dir) / 'X_train.pkl', X_train)
        save_object(Path(self.config.root_dir) / 'X_validate.pkl', X_validate)
        save_object(Path(self.config.root_dir) / 'X_test.pkl', X_test)
        save_object(Path(self.config.root_dir) / 'X_train_scaled.pkl', X_train_sc)
        save_object(Path(self.config.root_dir) / 'X_validate_scaled.pkl', X_validate_sc)
        save_object(Path(self.config.root_dir) / 'X_test_scaled.pkl', X_test_sc)
        save_object(Path(self.config.root_dir) / 'y_train.pkl', y_train)
        save_object(Path(self.config.root_dir) / 'y_validate.pkl', y_validate)
        save_object(Path(self.config.root_dir) / 'y_test.pkl', y_test)

        save_json(
            Path(self.config.root_dir) / 'final_features.json',
            {'features': final_features}
        )

        iv_df.to_csv(
            Path(self.config.root_dir) / 'iv_rankings.csv'
        )

        logger.info(f"All transformed data saved to: {self.config.root_dir}")
        logger.info("Data Transformation Complete ✅")

        return (X_train, X_validate, X_test,
                X_train_sc, X_validate_sc, X_test_sc,
                y_train, y_validate, y_test,
                final_features, iv_df)





