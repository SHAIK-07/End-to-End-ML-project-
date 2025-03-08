from src.mlproject.logger import logger  
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer

import sys

if __name__ == "__main__":
    logger.info("Execution has started")

    try:
        data_ingestion = DataIngestion()
        
        train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()  # ✅ Updated function name
        
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))

    except Exception as e:
        logger.error("Custom Exception raised")
        raise CustomException(e, sys)

    logger.info("Execution has ended")
