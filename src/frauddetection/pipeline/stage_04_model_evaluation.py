from frauddetection import logger
from frauddetection.config.configuration import ConfigurationManager
from frauddetection.components.model_evaluation import ModelEvaluation

STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def run(self):
        config     = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()
        evaluation = ModelEvaluation(config=eval_config)
        evaluation.run()


if __name__ == '__main__':
    try:
        logger.info("="*50)
        logger.info(f"Starting {STAGE_NAME}")
        logger.info("="*50)
        obj = ModelEvaluationTrainingPipeline()
        obj.run()
        logger.info(f"{STAGE_NAME} Complete ✅")
    except Exception as e:
        logger.error(f"{STAGE_NAME} Failed: {e}")
        raise e