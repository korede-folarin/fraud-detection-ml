from frauddetection import logger
from frauddetection.config.configuration import ConfigurationManager
from frauddetection.components.model_trainer import ModelTrainer

STAGE_NAME = "Model Trainer Stage"


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        config        = ConfigurationManager()
        trainer_config = config.get_model_trainer_config()
        params        = config.params
        trainer       = ModelTrainer(config=trainer_config, params=params)
        trainer.run()


if __name__ == '__main__':
    try:
        logger.info("="*50)
        logger.info(f"Starting {STAGE_NAME}")
        logger.info("="*50)
        obj = ModelTrainerTrainingPipeline()
        obj.run()
        logger.info(f"{STAGE_NAME} Complete ✅")
    except Exception as e:
        logger.error(f"{STAGE_NAME} Failed: {e}")
        raise e