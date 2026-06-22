"""Logging configuration module."""

import os
import logging
import json
from datetime import datetime
from pathlib import Path
from pythonjsonlogger import jsonlogger


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: str = "logs",
    use_json: bool = True,
) -> logging.Logger:
    """
    Configure and return a logger instance.

    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        use_json: Whether to use JSON format for logs

    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))

    if use_json:
        file_formatter = jsonlogger.JsonFormatter()
    else:
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


# Module-level logger
logger = setup_logger(
    __name__,
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=os.getenv("LOG_DIR", "logs"),
)
