import os
import sys
import pandas as pd
from flask import Flask, request, jsonify, render_template
from src.mlproject.pipelines.training_pipeline import ModelTrainer
from src.mlproject.pipelines.prediction_pipeline import PredictionPipeline
from src.mlproject.components.model_monitoring import ModelMonitoring  # ✅ Import model monitoring
from src.mlproject.utils import load_object
from src.mlproject.logger import logger
from src.mlproject.exception import CustomException

app = Flask(__name__)


# ✅ Load the trained model using the utility function
MODEL_PATH = "artifacts/best_model.pkl"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = load_object(MODEL_PATH)
        logger.info("Model loaded successfully.")
    else:
        logger.warning("Model not found. Training is required.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise CustomException(e, sys)  # ✅ Correctly raising CustomException

@app.route('/')
def home():
    """Render the homepage with a form for predictions."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions based on user input and log model performance."""
    try:
        if not model:
            raise CustomException("Model not found. Please train the model first.", sys)  # ✅ Raising CustomException

        # ✅ Ensure request contains JSON data
        if not request.is_json:
            raise CustomException("Invalid request format. Expected JSON.", sys)

        input_data = request.get_json()  # ✅ Get JSON input safely

        # ✅ Validate that JSON is not empty
        if not input_data:
            raise CustomException("Received empty input data.", sys)

        # ✅ Convert JSON to DataFrame
        df = pd.DataFrame([input_data])
        print(df)  # ✅ Debugging step: Check what the input looks like

        model_path = "artifacts/best_model.pkl"
        preprocessor_path = "artifacts/preprocessor.pkl"

        # ✅ Load the prediction pipeline
        predictor = PredictionPipeline(model_path, preprocessor_path)
        prediction = predictor.predict(df)

        # ✅ Log performance metrics to MLflow after prediction
        monitor = ModelMonitoring()
        metrics = monitor.evaluate_model("artifacts/test.csv")  # Evaluate on test data

        return jsonify({
            "prediction": prediction.tolist(),
            "metrics": metrics  # ✅ Return monitoring metrics
        }), 200

    except CustomException as e:
        logger.error(f"Prediction Error: {str(e)}")
        return jsonify({"error": str(e)}), 400  # Return error response

    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred."}), 500  # Internal Server Error


# @app.route('/monitor', methods=['GET'])
# def monitor_model():
#     """Endpoint to manually check model performance."""
#     try:
#         monitor = ModelMonitoring()
#         metrics = monitor.evaluate_model("artifacts/test.csv")
#         return jsonify({"metrics": metrics}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

