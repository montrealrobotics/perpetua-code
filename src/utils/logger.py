# my_logger.py
import logging

logger = logging.getLogger(__name__)  # A named logger to be shared across files

def setup_logger(log_file_path):
    """Sets up the logger to log to both console and a file."""
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Create file handler for logging to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    # Add handlers if they aren't already present
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

