import sys
from src.mlproject.logger import logger  # Use the fixed logger

def error_message_detail(error, error_detail: sys):
    """
    Capture detailed error information including file name, line number, and stack trace.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename  # File where error occurred
    line_number = exc_tb.tb_lineno  # Line number where error occurred

    # Structured error message
    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_number}] error message [{str(error)}]"
    )

    return error_message

class CustomException(Exception):
    """
    Custom Exception class to handle errors with detailed logging.
    """
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)

        # Log the error using the fixed logger
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message
