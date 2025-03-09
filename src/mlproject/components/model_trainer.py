import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from mlflow.models.signature import infer_signature

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "best_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

        # ✅ Set DagsHub as MLflow tracking server
        mlflow.set_registry_uri("https://dagshub.com/SHAIK-07/practice.mlflow")
        mlflow.set_experiment("Maths_Score_Prediction")

    def eval_metrics(self, actual, pred):
        """Calculate RMSE, MAE, and R2 Score."""
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # ✅ Define models as a tuple list
            models = [
                ("Random Forest", {"n_estimators": 100, "max_depth": None}, RandomForestRegressor(), (X_train, y_train), (X_test, y_test)),
                ("Decision Tree", {"criterion": "squared_error", "max_depth": None}, DecisionTreeRegressor(), (X_train, y_train), (X_test, y_test)),
                ("Gradient Boosting", {"learning_rate": 0.1, "n_estimators": 100, "subsample": 0.8}, GradientBoostingRegressor(), (X_train, y_train), (X_test, y_test)),
                ("Linear Regression", {}, LinearRegression(), (X_train, y_train), (X_test, y_test)),
                ("XGBRegressor", {"learning_rate": 0.1, "n_estimators": 100, "use_label_encoder": False, "eval_metric": "logloss"}, XGBRegressor(), (X_train, y_train), (X_test, y_test)),
                ("CatBoosting Regressor", {"depth": 6, "learning_rate": 0.05, "iterations": 100}, CatBoostRegressor(verbose=False), (X_train, y_train), (X_test, y_test)),
                ("AdaBoost Regressor", {"learning_rate": 0.1, "n_estimators": 100}, AdaBoostRegressor(), (X_train, y_train), (X_test, y_test)),
            ]

            # ✅ Track model performances
            model_scores = {}
            best_model = None
            best_model_name = None
            best_model_r2 = -np.inf  # Initialize with a low value

            # ✅ Train & Evaluate each model
            for model_name, params, model, (X_train, y_train), (X_test, y_test) in models:
                logging.info(f"Training {model_name} with params: {params}")

                # Set parameters & train
                model.set_params(**params)
                model.fit(X_train, y_train)

                # Predictions
                predictions = model.predict(X_test)

                # Evaluate
                rmse, mae, r2 = self.eval_metrics(y_test, predictions)
                model_scores[model_name] = r2  # Store R2 score

                logging.info(f"{model_name} - RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

                # ✅ Log to MLflow (but DO NOT register yet)
                with mlflow.start_run(run_name=f"{model_name} Run"):
                    mlflow.log_params(params)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("mae", mae)
                    mlflow.log_metric("r2", r2)
                    mlflow.sklearn.log_model(model, "model")  # ✅ Logs model but does NOT register

                # ✅ Select the best model
                if r2 > best_model_r2:
                    best_model_r2 = r2
                    best_model_name = model_name
                    best_model = model

            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_r2}")

            # ✅ Ensure the best model has a good score
            if best_model_r2 < 0.6:
                raise CustomException("No best model found with an acceptable R2 score.", sys)

            # ✅ Fix MLflow Warning (Add Model Signature & Input Example)
            input_example = pd.DataFrame(X_test[:5])  # Take first 5 rows as example input
            signature = infer_signature(X_test, best_model.predict(X_test))

            # ✅ Register Only the Best Model in MLflow Model Registry
            with mlflow.start_run(run_name=f"Best Model: {best_model_name}"):
                mlflow.sklearn.log_model(
                    best_model,
                    "best_model",
                    registered_model_name=best_model_name,
                    signature=signature,  # ✅ Adds model signature
                    input_example=input_example  # ✅ Adds input example
                )

            # ✅ Save the Best Model Locally
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Best model saved to {self.model_trainer_config.trained_model_file_path}")

            return best_model_r2

        except Exception as e:
            raise CustomException(e, sys)
