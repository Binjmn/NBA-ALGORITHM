# -*- coding: utf-8 -*-
"""
Logger Module

This module provides logging utilities for the NBA prediction system.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

def setup_logging(app_name, log_level=logging.INFO):
    """
    Set up logging for the application
    
    Args:
        app_name: Name of the application for the log file
        log_level: Logging level (default: INFO)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create a logger
    logger = logging.getLogger(app_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create a unique log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{app_name}_{timestamp}.log"
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log initialization info
    logger.info(f"Logging initialized for {app_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger
