from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    raw_data_path: Path          
    train_path: Path             
    validate_path: Path          
    test_path: Path              
    train_ratio: float
    validate_ratio: float
    rolling_window: int
    rolling_min_periods: int
    iv_threshold: float
    amount_log_base: int
    seconds_per_hour: int
    features_to_drop: list


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    validate_data_path: Path
    test_data_path: Path
    model_path: Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    validate_data_path: Path
    test_data_path: Path
    evaluation_results: Path
    mlflow_uri: str
    cost_false_negative: int
    cost_false_positive: int
    business_threshold: float