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
    """Test the full data ingestion process."""
    try:
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # ✅ Check if the files are created
        assert os.path.exists(train_path), "Train CSV file was not created!"
        assert os.path.exists(test_path), "Test CSV file was not created!"

        # ✅ Check if CSV files are not empty
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        assert not train_df.empty, "Train CSV file is empty!"
        assert not test_df.empty, "Test CSV file is empty!"

    except CustomException as e:
        pytest.fail(f"Data Ingestion failed with CustomException: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error during Data Ingestion: {str(e)}")

# def test_missing_csv_file():
#     """Test the case when the input CSV file is missing."""
#     data_ingestion = DataIngestion()

#     # ✅ Temporarily rename the expected CSV file (if it exists)
#     csv_path = os.path.join('notebooks/data', 'stud.csv')
#     temp_path = csv_path + ".backup"

#     if os.path.exists(csv_path):
#         os.rename(csv_path, temp_path)  # Rename to simulate missing file

#     try:
#         with pytest.raises(FileNotFoundError):
#             data_ingestion.initiate_data_ingestion()

#     finally:
#         # ✅ Restore the original file after the test
#         if os.path.exists(temp_path):
#             os.rename(temp_path, csv_path)
