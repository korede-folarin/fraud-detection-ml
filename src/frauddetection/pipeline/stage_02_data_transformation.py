from frauddetection import logger
from frauddetection.config.configuration import ConfigurationManager
from frauddetection.components.data_transformation import DataTransformation

STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        config            = ConfigurationManager()
        transform_config  = config.get_data_transformation_config()
        transformation    = DataTransformation(config=transform_config)
        transformation.run()


if __name__ == '__main__':
    try:
        logger.info("="*50)
        logger.info(f"Starting {STAGE_NAME}")
        logger.info("="*50)
        obj = DataTransformationTrainingPipeline()
        obj.run()
        logger.info(f"{STAGE_NAME} Complete ")
    except Exception as e:
        logger.error(f"{STAGE_NAME} Failed: {e}")
        raise e