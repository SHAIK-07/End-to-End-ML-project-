import sys
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.logger import logger
from src.mlproject.exception import CustomException

def run_training_pipeline():
    try:
        logger.info(">>>>> Training Pipeline Started <<<<<")

        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        model_score = model_trainer.initiate_model_trainer(train_array, test_array)

        logger.info(f">>>>> Training Pipeline Completed Successfully with R2 Score: {model_score} <<<<<")
    
    except Exception as e:
        logger.error(f"Training Pipeline Failed: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_training_pipeline()
