# -*- coding: utf-8 -*-
"""
Model Training Utility

This module provides functions to check if models need retraining
and delegates the training process to the production_ready_training script.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def check_model_age(model_path, max_age_days=7):
    """
    Check if a model file is older than max_age_days
    
    Args:
        model_path: Path to the model file
        max_age_days: Maximum age in days before retraining is needed
        
    Returns:
        bool: True if model is older than max_age_days or doesn't exist, False otherwise
    """
    if not os.path.exists(model_path):
        logger.info(f"Model file does not exist: {model_path}")
        return True
        
    file_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    current_time = datetime.now()
    age_days = (current_time - file_time).days
    
    if age_days > max_age_days:
        logger.info(f"Model is {age_days} days old (older than {max_age_days} days): {model_path}")
        return True
    
    return False


def check_performance_metrics():
    """
    Check if models need retraining based on recent performance metrics
    
    Returns:
        bool: True if retraining is needed, False otherwise
    """
    # This is a placeholder for more sophisticated performance-based retraining logic
    # In a production system, this would analyze recent prediction accuracy
    # and trigger retraining if accuracy drops below a threshold
    
    # For now, we'll simply return False to indicate no performance-based retraining is needed
    return False


def train_models_if_needed():
    """
    Check if models need to be retrained and run the training if necessary
    
    Returns:
        bool: True if training was performed, False otherwise
    """
    try:
        logger.info("Checking if models need retraining")
        
        # Check if main game prediction models need retraining
        models_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        main_model_path = os.path.join(models_dir, "production_gradient_boosting.pkl")
        props_model_dir = os.path.join(models_dir, "player_props")
        
        # Check if any model files are missing or too old
        models_need_training = check_model_age(main_model_path)
        
        # Check if player props models need retraining
        if os.path.exists(props_model_dir):
            props_files = os.listdir(props_model_dir)
            if not props_files:
                logger.info("No player props model files found")
                models_need_training = True
            else:
                # Check the age of one representative player props model
                for file in props_files:
                    if file.endswith('.pkl'):
                        props_model_path = os.path.join(props_model_dir, file)
                        if check_model_age(props_model_path):
                            models_need_training = True
                        break
        else:
            logger.info("Player props model directory does not exist")
            models_need_training = True
            
        # Also check if performance metrics indicate retraining is needed
        if check_performance_metrics():
            logger.info("Performance metrics indicate retraining is needed")
            models_need_training = True
        
        # Run training if needed
        if models_need_training:
            logger.info("Models need retraining. Running enhanced training pipeline")
            
            # Use the enhanced training_pipeline.py instead of production_ready_training.py
            training_script = os.path.join(PROJECT_ROOT, "src", "models", "training_pipeline.py")
            
            if os.path.exists(training_script):
                try:
                    logger.info(f"Running enhanced training pipeline: {training_script}")
                    result = subprocess.run(
                        [sys.executable, training_script],
                        capture_output=True,
                        text=True,
                        cwd=PROJECT_ROOT
                    )
                    
                    if result.returncode == 0:
                        logger.info("Model training completed successfully")
                        return True
                    else:
                        logger.error(f"Model training failed with return code {result.returncode}")
                        logger.error(f"Error output: {result.stderr}")
                except Exception as e:
                    logger.error(f"Error running training script: {str(e)}")
            else:
                logger.error(f"Training script not found: {training_script}")
            
            return False
        else:
            logger.info("Models are up-to-date. No retraining needed.")
            return False
    except Exception as e:
        logger.error(f"Error checking if models need retraining: {str(e)}")
        return False
