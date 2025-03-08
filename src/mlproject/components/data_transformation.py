import sys
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.mlproject.logger import logger
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transform_object(self):
        """
        Create and return the preprocessing pipeline for numerical and categorical features.
        """
        try:
            logger.info("Initializing Data Transformation pipeline.")

            # Define columns
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender", "race_ethnicity", "parental_level_of_education",
                "lunch", "test_preparation_course"
            ]

            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")

            # Define pipelines
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False)),
            ])

            logger.info("Creating ColumnTransformer with numerical and categorical pipelines.")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns),
            ])

            logger.info("Data transformation pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logger.error(f"Error in data transformation pipeline: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Read train and test data, apply preprocessing, and save the preprocessor.
        """
        try:
            logger.info(f"Reading training data from: {train_path}")
            logger.info(f"Reading testing data from: {test_path}")

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError("Train or test file not found.")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(f"Train Data Shape: {train_df.shape}, Columns: {list(train_df.columns)}")
            logger.info(f"Test Data Shape: {test_df.shape}, Columns: {list(test_df.columns)}")

            preprocessor_obj = self.get_data_transform_object()

            # Define target variable
            target_column_name = "math_score"
            numerical_columns = ['writing_score', 'reading_score']

            if target_column_name not in train_df.columns:
                raise ValueError(f"Target column '{target_column_name}' not found in training data.")

            if target_column_name not in test_df.columns:
                raise ValueError(f"Target column '{target_column_name}' not found in testing data.")

            # Splitting input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying preprocessing to training and testing data.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logger.info(f"Transformed Train Data Shape: {input_feature_train_arr.shape}")
            logger.info(f"Transformed Test Data Shape: {input_feature_test_arr.shape}")

            # Combine input features and target variable
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.to_numpy()]

            # Ensure artifacts directory exists before saving
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)

            logger.info(f"Saving preprocessing object to: {self.data_transformation_config.preprocessor_obj_file_path}")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logger.info("Preprocessing object saved successfully.")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise CustomException(e, sys)

        except Exception as e:
            logger.error(f"Error in initiate_data_transformation: {str(e)}")
            raise CustomException(e, sys)
