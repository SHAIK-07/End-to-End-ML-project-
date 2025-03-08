import sys 
from dataclasses import dataclass
import os 

import numpy as np
import pandas as pd


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.mlproject.logger import logger
from src.mlproject.exception import CustomException

from src.mlproject.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_confing = DataTransformationConfig()
    
    def get_data_transfor_object(self):
        '''
        this function is responsible for data transformation 
        
        '''
        try:
            logger.info("Data Transformation initiated")
            
            
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )      
            
            logger.info(f"Categorical columns: {categorical_columns}")
            logger.info(f"numerical columns: {numerical_columns}")
            
            logger.info("data transformation Pipeline Initiated")      

            preprocessor = ColumnTransformer(    
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )         
            
            logger.info("data transformation Pipeline Completed")   

            return preprocessor

        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)            
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info("Read train and test data completed")

            preprocessor_obj = self.get_data_transfor_object()
            
            target_column_name = "math_score"
            numerical_columns = ['writing_score', 'reading_score']    
            
            # devide the train_data set into input features and target features  

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            #devide the test_data set into input features and target features

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df) 
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df) 
                
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info("Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_confing.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_confing.preprocessor_obj_file_path,
            )

        except Exception as e:
            logger.error(f"Error in initiate_data_transformation: {str(e)}")
            raise CustomException(e, sys)
            