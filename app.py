from src.mlproject.logger import logger  
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
import sys

if __name__ == "__main__":
    logger.info("Execution has started")

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()  # ✅ Updated function name

    except Exception as e:
        logger.error("Custom Exception raised")
        raise CustomException(e, sys)

    logger.info("Execution has ended")
