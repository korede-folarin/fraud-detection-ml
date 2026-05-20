from frauddetection.constant import *
from frauddetection.utils.common import read_yaml, create_directories
from frauddetection.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from pathlib import Path


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(config.local_data_file)
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.data_transformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            raw_data_path=Path(config.raw_data_path),
            train_path=Path(config.train_path),
            validate_path=Path(config.validate_path),
            test_path=Path(config.test_path),
            train_ratio=params.train_ratio,
            validate_ratio=params.validate_ratio,
            rolling_window=params.rolling_window,
            rolling_min_periods=params.rolling_min_periods,
            iv_threshold=params.iv_threshold,
            amount_log_base=params.amount_log_base,
            seconds_per_hour=params.seconds_per_hour,
            features_to_drop=params.features_to_drop

        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_data_path=Path(config.train_data_path),
            validate_data_path=Path(config.validate_data_path),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path)
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.business_cost
        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            validate_data_path=Path(config.validate_data_path),
            test_data_path=Path(config.test_data_path),
            evaluation_results=Path(config.evaluation_results),
            mlflow_uri=config.mlflow_uri,
            cost_false_negative=params.cost_false_negative,
            cost_false_positive=params.cost_false_positive,
            business_threshold=params.business_threshold
        )