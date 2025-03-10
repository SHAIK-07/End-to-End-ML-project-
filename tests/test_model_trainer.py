import os
import pytest
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.utils import load_object
from src.mlproject.exception import CustomException

@pytest.fixture
def model_trainer():
    """Fixture to initialize ModelTrainer"""
    return ModelTrainer()

def test_model_trainer_initialization(model_trainer):
    """Test if ModelTrainer initializes correctly."""
    assert model_trainer.model_trainer_config.trained_model_file_path.endswith("best_model.pkl")

def test_best_model_exists():
    """Test if a trained model already exists and is valid."""
    model_path = "artifacts/best_model.pkl"

    # ✅ Ensure the model file exists
    assert os.path.exists(model_path), "No trained model found!"

    # ✅ Load the existing model
    model = load_object(model_path)
    assert model is not None, "Failed to load the best model!"

