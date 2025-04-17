#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Production Readiness Utilities

This module contains utilities to ensure the prediction system is production-ready:
1. System health checks
2. Model validation
3. Data quality verification
4. Dependency checks
5. Logging configuration
"""

import os
import sys
import logging
import importlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

def setup_production_logging(log_dir: str = "logs", app_name: str = "nba_algorithm"):
    """
    Configure production-quality logging with rotation and appropriate levels
    
    Args:
        log_dir: Directory to store log files
        app_name: Name of the application for the log file prefix
    """
    import logging.handlers
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"{app_name}_{timestamp}.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB max size, keep 5 backups
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Create console handler with a different format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging configured. Log file: {log_file}")
    return log_file

def check_required_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required dependencies are installed
    
    Returns:
        Tuple containing success status and list of missing dependencies
    """
    required_packages = [
        "pandas", "numpy", "scikit-learn", "matplotlib", 
        "requests", "joblib", "tqdm", "xgboost"
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.warning(f"Missing required dependencies: {', '.join(missing)}")
        return False, missing
    else:
        logger.info("All required dependencies are installed")
        return True, []

def check_api_keys() -> Tuple[bool, List[str]]:
    """
    Check if all required API keys are available
    
    Returns:
        Tuple containing success status and list of missing API keys
    """
    required_keys = [
        "BALLDONTLIE_API_KEY",
        "THEODDS_API_KEY"
    ]
    
    # Import the API key module
    try:
        from ..config.api_keys import API_KEYS
    except ImportError:
        logger.error("Failed to import API keys module")
        return False, required_keys
    
    missing = [key for key in required_keys if key not in API_KEYS or not API_KEYS[key]]
    
    if missing:
        logger.warning(f"Missing required API keys: {', '.join(missing)}")
        return False, missing
    else:
        logger.info("All required API keys are available")
        return True, []

def verify_model_files() -> Tuple[bool, Dict[str, bool]]:
    """
    Verify that all required model files exist and are valid
    
    Returns:
        Tuple containing overall success status and dictionary of model statuses
    """
    model_base_dir = Path("models")
    required_models = {
        "game": ["moneyline", "spread", "total"],
        "player_props": ["points", "rebounds", "assists"]
    }
    
    model_status = {}
    overall_status = True
    
    for category, types in required_models.items():
        for model_type in types:
            model_dir = model_base_dir / category / model_type
            model_path = model_dir / f"{model_type}_model.pkl"
            
            # Check if model file exists
            model_status[f"{category}/{model_type}"] = model_path.exists()
            
            if not model_path.exists():
                logger.warning(f"Missing model file: {model_path}")
                overall_status = False
    
    # Log overall status
    if overall_status:
        logger.info("All required model files are available")
    else:
        logger.warning("Some required model files are missing - training may be needed")
    
    return overall_status, model_status

def system_health_check() -> Dict[str, Any]:
    """
    Perform a comprehensive system health check
    
    Returns:
        Dictionary with health check results
    """
    health_report = {
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "checks": {}
    }
    
    # Check dependencies
    deps_ok, missing_deps = check_required_dependencies()
    health_report["checks"]["dependencies"] = {
        "status": "pass" if deps_ok else "fail",
        "missing": missing_deps
    }
    
    # Check API keys
    keys_ok, missing_keys = check_api_keys()
    health_report["checks"]["api_keys"] = {
        "status": "pass" if keys_ok else "fail",
        "missing": missing_keys
    }
    
    # Check model files
    models_ok, model_status = verify_model_files()
    health_report["checks"]["model_files"] = {
        "status": "pass" if models_ok else "warning",
        "details": model_status
    }
    
    # Check directories
    dirs_to_check = ["data", "logs", "predictions"]
    dirs_status = {}
    for directory in dirs_to_check:
        path = Path(directory)
        exists = path.exists()
        if not exists:
            try:
                path.mkdir(parents=True, exist_ok=True)
                dirs_status[directory] = "created"
            except Exception as e:
                dirs_status[directory] = f"error: {str(e)}"
        else:
            dirs_status[directory] = "exists"
    
    health_report["checks"]["directories"] = {
        "status": "pass",
        "details": dirs_status
    }
    
    # Determine overall status
    critical_checks = ["dependencies", "api_keys"]
    if all(health_report["checks"][check]["status"] == "pass" for check in critical_checks):
        if models_ok:
            health_report["status"] = "healthy"
        else:
            health_report["status"] = "warning"
    else:
        health_report["status"] = "unhealthy"
    
    logger.info(f"System health check completed. Status: {health_report['status']}")
    return health_report

def prepare_production_environment():
    """
    Prepare the environment for production use
    
    Returns:
        bool: True if preparation was successful, False otherwise
    """
    try:
        # Setup production logging
        setup_production_logging()
        
        # Run system health check
        health_report = system_health_check()
        
        # Log detailed health report
        logger.info(f"System health: {health_report['status']}")
        for check, result in health_report["checks"].items():
            logger.info(f"  {check}: {result['status']}")
        
        # Create necessary directories if they don't exist
        for directory in ["data", "logs", "predictions", "models"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        return health_report["status"] != "unhealthy"
    
    except Exception as e:
        logger.error(f"Failed to prepare production environment: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
