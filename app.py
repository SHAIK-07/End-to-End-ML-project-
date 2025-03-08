from src.mlproject.logger import logger  
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation

import sys

if __name__ == "__main__":
    logger.info("Execution has started")

    try:
        data_ingestion = DataIngestion()
        
        train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()  # ✅ Updated function name
        
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    except Exception as e:
        logger.error("Custom Exception raised")
        raise CustomException(e, sys)

    logger.info("Execution has ended")
