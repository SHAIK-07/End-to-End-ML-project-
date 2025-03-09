import os
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.mlproject.utils import load_object, save_object
from src.mlproject.logger import logger
from src.mlproject.exception import CustomException
from src.mlproject.pipelines.training_pipeline import ModelTrainer
from mlflow.models.signature import infer_signature
import sys

class ModelMonitoring:
    def __init__(self, model_path="artifacts/best_model.pkl", preprocessor_path="artifacts/preprocessor.pkl"):
        try:
            self.model_path = model_path
            self.preprocessor_path = preprocessor_path
            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
            logger.info("Model and preprocessor loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model or preprocessor: {str(e)}")
            raise CustomException(e, sys)

        # ✅ Set MLflow tracking
        mlflow.set_registry_uri("https://dagshub.com/SHAIK-07/practice.mlflow")
        mlflow.set_experiment("Model Monitoring")

    def evaluate_model(self, new_data_path):
        """Evaluates the model on new test data and logs metrics to MLflow."""
        if self.model is None:
            logger.error("No trained model found. Please train the model first.")
            return None

        try:
            df = pd.read_csv(new_data_path)
            if "math_score" not in df.columns:
                logger.error("Target column 'math_score' not found in new data!")
                return None

            X_new = df.drop(columns=["math_score"])
            y_true = df["math_score"]

            # ✅ Apply preprocessing
            X_processed = self.preprocessor.transform(X_new)

            # ✅ Make predictions
            y_pred = self.model.predict(X_processed)

            # ✅ Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            logger.info(f"Model Evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, R2 Score={r2:.4f}")

            # ✅ Log metrics to MLflow
            with mlflow.start_run(run_name="Model Monitoring Run"):
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

            # ✅ If R² drops below 0.6, trigger retraining
            if r2 < 0.6:
                logger.warning("🚨 Model performance has degraded! Retraining model...")
                self.retrain_model()

            return {"rmse": rmse, "mae": mae, "r2": r2}

        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise CustomException(e, sys)

    # def retrain_model(self):
    #     """Triggers model retraining, registers it in MLflow, and updates the best model."""
    #     try:
    #         trainer = ModelTrainer()
    #         train_path = "artifacts/train.csv"
    #         test_path = "artifacts/test.csv"

    #         # ✅ Load training data
    #         train_df = pd.read_csv(train_path)
    #         test_df = pd.read_csv(test_path)

    #         train_array = train_df.to_numpy()
    #         test_array = test_df.to_numpy()

    #         # ✅ Train new model
    #         new_r2 = trainer.initiate_model_trainer(train_array, test_array)

    #         if new_r2 > 0.6:
    #             # ✅ Load new model
    #             new_model = load_object(self.model_path)

    #             # ✅ Log new model to MLflow and register it
    #             with mlflow.start_run(run_name="New Best Model"):
    #                 input_example = pd.DataFrame(test_array[:, :-1][:5])  # Sample 5 rows
    #                 signature = infer_signature(test_array[:, :-1], new_model.predict(test_array[:, :-1]))

    #                 mlflow.sklearn.log_model(
    #                     new_model,
    #                     "best_model",
    #                     registered_model_name="Best_Maths_Model",  # ✅ Registering model in MLflow
    #                     signature=signature,
    #                     input_example=input_example
    #                 )

    #             # ✅ Save the new model locally
    #             save_object(self.model_path, new_model)
    #             logger.info("✅ New model trained, registered in MLflow, and saved successfully!")
    #         else:
    #             logger.warning("🚨 Retrained model did not improve performance!")

    #     except Exception as e:
    #         logger.error(f"Error during model retraining: {str(e)}")
    #         raise CustomException(e, sys)
