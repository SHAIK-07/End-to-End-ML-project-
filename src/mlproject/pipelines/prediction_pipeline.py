import sys
import os
import pandas as pd
import pickle
import numpy as np

from src.mlproject.utils import load_object
from src.mlproject.logger import logger
from src.mlproject.exception import CustomException

class PredictionPipeline:
    def __init__(self, model_path: str, preprocessor_path: str):
        try:
            logger.info("Loading trained model and preprocessor...")
            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
            logger.info("Model and preprocessor loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model or preprocessor: {str(e)}")
            raise CustomException(e, sys)

    def predict(self, input_data: pd.DataFrame):
        try:
            logger.info("Applying preprocessing to input data...")
            transformed_data = self.preprocessor.transform(input_data)
            
            logger.info("Generating predictions...")
            predictions = self.model.predict(transformed_data)

            return predictions
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise CustomException(e, sys)

# if __name__ == "__main__":
#     try:
#         # Example Input Data (Modify this based on your dataset)
#         sample_data = {
#             "gender": ["male"],
#             "race_ethnicity": ["group A"],
#             "parental_level_of_education": ["bachelor's degree"],
#             "lunch": ["standard"],
#             "test_preparation_course": ["none"],
#             "writing_score": [74],
#             "reading_score": [72]
#         }

#         input_df = pd.DataFrame(sample_data)

#         # Define paths (Ensure correct paths before running)
#         model_path = "artifacts/best_model.pkl"
#         preprocessor_path = "artifacts/preprocessor.pkl"

#         prediction_pipeline = PredictionPipeline(model_path, preprocessor_path)
#         predictions = prediction_pipeline.predict(input_df)

#         print(f"Predicted Math Score: {predictions[0]}")

#     except Exception as e:
#         print(f"Prediction failed: {str(e)}")
