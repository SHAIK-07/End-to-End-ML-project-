import os
import pytest
import pandas as pd
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.exception import CustomException

@pytest.fixture
def data_ingestion():
    """Fixture to initialize DataIngestion"""
    return DataIngestion()

def test_data_ingestion_initialization(data_ingestion):
    """Test if DataIngestion initializes correctly."""
    assert data_ingestion.ingestion_config.train_data_path.endswith("train.csv")
    assert data_ingestion.ingestion_config.test_data_path.endswith("test.csv")

def test_data_ingestion_process(data_ingestion):
    """Test the full data ingestion process and check for .csv.dvc files."""
    try:
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # ✅ Check if the DVC-tracked files exist in artifacts/
        train_dvc_path = train_path + ".dvc"
        test_dvc_path = test_path + ".dvc"
        raw_dvc_path = os.path.join(os.path.dirname(train_path), "raw.csv.dvc")

        assert os.path.exists(train_dvc_path), "Train CSV .dvc file was not created!"
        assert os.path.exists(test_dvc_path), "Test CSV .dvc file was not created!"
        assert os.path.exists(raw_dvc_path), "Raw CSV .dvc file was not created!"

    except CustomException as e:
        pytest.fail(f"Data Ingestion failed with CustomException: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error during Data Ingestion: {str(e)}")

