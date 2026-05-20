from frauddetection import logger
from frauddetection.config.configuration import ConfigurationManager
from frauddetection.components.data_ingestion import DataIngestion

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        config     = ConfigurationManager()
        ing_config = config.get_data_ingestion_config()
        ingestion  = DataIngestion(config=ing_config)
        ingestion.run()


if __name__ == '__main__':
    try:
        logger.info("="*50)
        logger.info(f"Starting {STAGE_NAME}")
        logger.info("="*50)
        obj = DataIngestionTrainingPipeline()
        obj.run()
        logger.info(f"{STAGE_NAME} Complete ✅")
    except Exception as e:
        logger.error(f"{STAGE_NAME} Failed: {e}")
        raise e