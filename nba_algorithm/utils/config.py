#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Prediction System Configuration Module

This module handles configuration settings, logging setup, and global constants
for the NBA prediction system.

Author: Cascade
Date: April 2025
"""

import os
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Get the project root directory
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"  # Use the models directory in the project root
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure logging
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# API Keys and external service configurations
DEFAULT_API_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Suppress all warnings
warnings.filterwarnings("ignore")


def config_logging(log_file: Path, verbose: bool = False) -> None:
    """
    Configure logging for the prediction system
    
    Args:
        log_file: Path to log file
        verbose: Whether to enable verbose logging
    """
    # Set up root logger
    root_logger = logging.getLogger()
    
    # Set logging level based on verbosity
    if verbose:
        root_logger.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)
    
    # Create file handler for log file
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Only add stream handler if verbose is True
    if verbose:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter(
            "%(levelname)s: %(message)s"
        )
        stream_handler.setFormatter(stream_formatter)
        root_logger.addHandler(stream_handler)
    
    # Silence other loggers that might be noisy
    for logger_name in ['matplotlib', 'urllib3', 'requests', 'sklearn']:
        other_logger = logging.getLogger(logger_name)
        other_logger.setLevel(logging.WARNING)
    
    # Redirect warnings to logging
    logging.captureWarnings(True)


# Initialize logger
def initialize_logging(verbose: bool = False) -> None:
    """
    Initialize the logging system
    
    Args:
        verbose: Whether to enable verbose logging
    """
    config_logging(LOG_FILE, verbose=verbose)
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info(f"Starting NBA prediction system at {datetime.now()}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Base directory: {BASE_DIR}")
