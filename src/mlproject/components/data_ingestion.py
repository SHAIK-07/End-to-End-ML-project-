import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.mlproject.logger import logger
from src.mlproject.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logger.info("Starting Data Ingestion Process...")

            # Ensure artifact directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            csv_path = os.path.abspath(os.path.join('notebooks\data', 'stud.csv'))
            logger.info(f"CSV file path: {csv_path}")

            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found at {csv_path}")
                raise FileNotFoundError(f"CSV file not found at {csv_path}")

            logger.info("Reading data from CSV file...")
            df = pd.read_csv(csv_path)
            logger.info("Data successfully read from CSV file.")

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logger.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # Split data into train and test
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logger.info(f"Train data saved at {self.ingestion_config.train_data_path}")
            logger.info(f"Test data saved at {self.ingestion_config.test_data_path}")
            logger.info("Data Ingestion Process Completed Successfully!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logger.error(f"Error during Data Ingestion: {str(e)}")
            raise CustomException(e, sys)



# if you want to read from mysql database use this code
#====================================================
# import os
# import sys
# import pandas as pd
# from dataclasses import dataclass
# from sklearn.model_selection import train_test_split

# from src.mlproject.logger import logger
# from src.mlproject.exception import CustomException
# from src.mlproject.utils import read_mysql_data  # Function to fetch data from MySQL

# @dataclass
# class DataIngestionConfig:
#     train_data_path: str = os.path.join("artifacts", "train.csv")
#     test_data_path: str = os.path.join("artifacts", "test.csv")
#     raw_data_path: str = os.path.join("artifacts", "raw.csv")

# class DataIngestion:
#     def __init__(self):
#         self.ingestion_config = DataIngestionConfig()

#     def initiate_data_ingestion(self):
#         try:
#             logger.info("Starting Data Ingestion Process...")

#             # Ensure artifact directory exists
#             os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

#             logger.info("Trying to fetch data from MySQL database...")
#             df = read_mysql_data()  # Fetch data from MySQL
#             logger.info("Data successfully read from MySQL database.")

#             # Save raw data
#             df.to_csv(self.ingestion_config.raw_data_path, index=False)
#             logger.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

#             # Split data into train and test
#             train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

#             # Save train and test datasets
#             train_set.to_csv(self.ingestion_config.train_data_path, index=False)
#             test_set.to_csv(self.ingestion_config.test_data_path, index=False)

#             logger.info(f"Train data saved at {self.ingestion_config.train_data_path}")
#             logger.info(f"Test data saved at {self.ingestion_config.test_data_path}")
#             logger.info("Data Ingestion Process Completed Successfully!")

#             return (
#                 self.ingestion_config.train_data_path,
#                 self.ingestion_config.test_data_path
#             )

#         except Exception as e:
#             logger.error(f"Error during Data Ingestion: {str(e)}")
#             raise CustomException(e, sys)
