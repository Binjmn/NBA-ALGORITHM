#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standardized Logging Configuration

Provides standardized logging configuration for the entire NBA prediction system.
Implements rotating file handlers, proper log formatting, and configurable log levels.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List

# Default log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Max log size for rotating file handler (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024

# Max log backup count
MAX_LOG_BACKUPS = 5

# Default log format
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Default log level
DEFAULT_LOG_LEVEL = logging.INFO


def configure_logging(module_name: str, log_level: Union[int, str] = DEFAULT_LOG_LEVEL,
                    log_to_console: bool = True, log_to_file: bool = True,
                    log_format: str = DEFAULT_LOG_FORMAT) -> logging.Logger:
    """
    Configure standardized logging for a module
    
    Args:
        module_name: Name of the module (used for the logger name and log file)
        log_level: Logging level (default: INFO)
        log_to_console: Whether to log to console
        log_to_file: Whether to log to a file
        log_format: Log message format
        
    Returns:
        Configured logger instance
    """
    # Convert string log level to int if necessary
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:  
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        # Use module name in log file name
        module_slug = module_name.split('.')[-1]  # Get last part of module name
        log_file = LOG_DIR / f"{module_slug}_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=MAX_LOG_SIZE, 
            backupCount=MAX_LOG_BACKUPS
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_exception(logger: logging.Logger, exception: Exception, context: str = "An error occurred") -> None:
    """
    Log an exception with full traceback and context
    
    Args:
        logger: Logger instance
        exception: The exception to log
        context: Context message explaining where the exception occurred
    """
    exc_info = sys.exc_info()
    stack_trace = ''.join(traceback.format_exception(*exc_info))
    logger.error(
        f"{context}: {type(exception).__name__}: {str(exception)}\n" 
        f"Stack trace: \n{stack_trace}"
    )


def configure_all_loggers(log_level: Union[int, str] = DEFAULT_LOG_LEVEL) -> None:
    """
    Configure all loggers in the system to use the same level
    
    Args:
        log_level: Logging level to set for all loggers
    """
    # Convert string log level to int if necessary
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(level=log_level, format=DEFAULT_LOG_FORMAT)
    
    # Update existing loggers
    for logger_name in logging.root.manager.loggerDict:  
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        
        # Update handler levels
        for handler in logger.handlers:  
            handler.setLevel(log_level)
