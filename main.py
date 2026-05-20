from frauddetection import logger
from frauddetection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from frauddetection.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from frauddetection.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline
from frauddetection.pipeline.stage_04_model_evaluation import ModelEvaluationTrainingPipeline


# Stage 1 — Data Ingestion
try:
    logger.info("="*50)
    logger.info("Stage 1: Data Ingestion")
    logger.info("="*50)
    obj = DataIngestionTrainingPipeline()
    obj.run()
    logger.info("Stage 1 Complete ")
except Exception as e:
    logger.error(e)
    raise e


# Stage 2 — Data Transformation
try:
    logger.info("="*50)
    logger.info("Stage 2: Data Transformation")
    logger.info("="*50)
    obj = DataTransformationTrainingPipeline()
    obj.run()
    logger.info("Stage 2 Complete ")
except Exception as e:
    logger.error(e)
    raise e


# Stage 3 — Model Training
try:
    logger.info("="*50)
    logger.info("Stage 3: Model Training")
    logger.info("="*50)
    obj = ModelTrainerTrainingPipeline()
    obj.run()
    logger.info("Stage 3 Complete ")
except Exception as e:
    logger.error(e)
    raise e


# Stage 4 — Model Evaluation
try:
    logger.info("="*50)
    logger.info("Stage 4: Model Evaluation")
    logger.info("="*50)
    obj = ModelEvaluationTrainingPipeline()
    obj.run()
    logger.info("Stage 4 Complete ")
except Exception as e:
    logger.error(e)
    raise e