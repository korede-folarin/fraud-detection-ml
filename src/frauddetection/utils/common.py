import os
import yaml
import joblib
import json
import logging
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations


logger = logging.getLogger(__name__)


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns ConfigBox object
    ConfigBox allows dot notation access: config.key instead of config['key']
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates list of directories
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves dict as JSON file
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads JSON file and returns ConfigBox
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON loaded from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_object(path: Path, obj: object):
    """
    Saves Python object using joblib
    Used for saving models, scalers, transformers
    """
    joblib.dump(obj, path)
    logger.info(f"Object saved at: {path}")


@ensure_annotations
def load_object(path: Path) -> object:
    """
    Loads Python object using joblib
    """
    obj = joblib.load(path)
    logger.info(f"Object loaded from: {path}")
    return obj


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns file size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"