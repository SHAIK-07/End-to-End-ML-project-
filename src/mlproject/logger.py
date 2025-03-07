import logging
import os
from datetime import datetime
from pathlib import Path

# Define base logs directory
LOGS_DIR = "logs"

# Get current date and time
current_date = datetime.now().strftime("%Y-%m-%d")  # Folder name (YYYY-MM-DD)
current_time = datetime.now().strftime("%H-%M-%S")  # File name (HH-MM-SS.log)

# Create a folder for today's logs (logs/YYYY-MM-DD/)
log_folder = os.path.join(LOGS_DIR, current_date)
Path(log_folder).mkdir(parents=True, exist_ok=True)  # Ensure directory exists

# Define log file path (logs/YYYY-MM-DD/HH-MM-SS.log)
LOG_FILE_PATH = os.path.join(log_folder, f"{current_time}.log")

# Create logger (use a separate variable instead of overwriting `logging`)
logger = logging.getLogger("mlproject_logger")
logger.setLevel(logging.INFO)  # Set logging level

# Prevent duplicate logs
if not logger.hasHandlers():
    # Create file handler
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"))

    # Add handler to logger
    logger.addHandler(file_handler)

    # Prevent log propagation to root logger
    logger.propagate = False
