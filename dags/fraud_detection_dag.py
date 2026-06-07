from datetime import datetime, timedelta
from airflow.decorators import dag, task 
import sys

sys.path.insert(0, '/usr/local/airflow/src')

sys.path.insert(0, 'opt/local/airflow/src')

#define default argumnents for all task 
default_args ={
    'owner': 'korede-folarin',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}


@dag(
    dag_id = 'fraud_detection_dag',
    default_args = default_args,
    description='Weekly fraud model retraining pipeline with drift detection',
    schedule_interval= '@weekly',
    start_date = datetime(2024, 6, 1),
    catchup = False,
    tags = ['fraud', 'banking', 'mlops']
)

def fraud_detection_pipeline():

    @task()
    def data_ingestion():
        from frauddetection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
        pipeline = DataIngestionTrainingPipeline()
        pipeline.run()
    
    @task()
    def data_transformation():
        from frauddetection.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
        pipeline = DataTransformationTrainingPipeline()
        pipeline.run()

    @task()
    def model_training():
        from frauddetection.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline
        pipeline = ModelTrainerTrainingPipeline()
        pipeline.run()
        
    @task()
    def model_evaluation():
        from frauddetection.pipeline.stage_04_model_evaluation import ModelEvaluationTrainingPipeline
        pipeline = ModelEvaluationTrainingPipeline()
        pipeline.run()


    
    #schedule
    @task()
    def drift_check():
        """
        Checks if new model outperforms champion
        Compares KS statistic from latest run vs threshold
        Triggers alert if performance drops below 0.75
        """
        import json 
        import os
        from frauddetection import logger
        result_path= 'artifacts/model_evaluation/evaluation_results.json'
        if not os.path.exists(result_path):
            logger.warning("No evaluation results found for drift check")
            return
            
        with open (result_path)as f:
            results = json.load(f)

        xgb_ks =results['test']['XGBoost']['KS']

        if xgb_ks <0.75:
            logger.warning(f"Model drift detected! KS dropped to {xgb_ks} below threshold 0.75")
            logger.warning("Champion- challenger swap recommended")
        else:
            logger.info(f"No drift detected. KS is {xgb_ks}")
            
    #set dependencies: upstream >> downstream
    ingestion = data_ingestion()
    transformation = data_transformation()
    training = model_training()
    evaluation = model_evaluation()
    drift = drift_check()

    ingestion >> transformation >> training >> evaluation >> drift

fraud_detection_dag = fraud_detection_pipeline()
        