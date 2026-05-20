import os
import pandas as pd
from pathlib import Path
from frauddetection import logger
from frauddetection.entity.config_entity import DataIngestionConfig
from frauddetection.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_data(self) -> pd.DataFrame:
        """
        Loads raw creditcard.csv from local data folder
        Validates file exists before loading
        In production this would connect to a database or S3
        """
        logger.info(f"Loading data from: {self.config.local_data_file}")

        if not os.path.exists(self.config.local_data_file):
            raise FileNotFoundError(
                f"Data file not found at: {self.config.local_data_file}"
            )

        df = pd.read_csv(self.config.local_data_file)

        logger.info(f"Data loaded successfully")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Fraud rate: {df['Class'].mean()*100:.3f}%")
        logger.info(f"File size: {get_size(Path(self.config.local_data_file))}")

        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validates that loaded data matches expected schema
        Checks column names and data types
        Critical for production — catches upstream data issues early
        """
        logger.info("Validating data schema...")

        expected_columns = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
            'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
            'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23',
            'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class'
        ]

        # check all expected columns present
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # check no missing values
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Missing values found: {df.isnull().sum().sum()}")

        # check Class column is binary
        if not set(df['Class'].unique()).issubset({0, 1}):
            raise ValueError("Class column must be binary (0 or 1)")

        logger.info("Schema validation passed ✅")
        return True

    def save_raw_data(self, df: pd.DataFrame):
        """
        Saves validated raw data to artifacts folder
        Keeps a copy of raw data before transformation
        """
        os.makedirs(self.config.root_dir, exist_ok=True)
        save_path = os.path.join(self.config.root_dir, 'raw_data.csv')
        df.to_csv(save_path, index=False)
        logger.info(f"Raw data saved at: {save_path}")

    def run(self) -> pd.DataFrame:
        """
        Runs full data ingestion pipeline:
        load → validate → save
        Returns validated dataframe for next stage
        """
        logger.info("="*50)
        logger.info("Starting Data Ingestion")
        logger.info("="*50)

        df = self.load_data()
        self.validate_schema(df)
        self.save_raw_data(df)

        logger.info("Data Ingestion Complete ")
        return df