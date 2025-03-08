import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")


