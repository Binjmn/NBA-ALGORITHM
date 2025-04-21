#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Loader Module

This module handles loading of trained prediction models for NBA games and player props.

Author: Cascade
Date: April 2025
"""

import os
import pickle
import logging
import traceback
import glob
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime

from ..utils.config import MODELS_DIR

# Configure logger
logger = logging.getLogger(__name__)


def load_model_from_file(model_path: str) -> Optional[Any]:
    """
    Load a model from a pickle file with proper error handling
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model object or None if loading fails
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file does not exist: {model_path}")
        return None
        
    try:
        # First try standard pickle loading
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e1:
        # If that fails, try with a different pickle protocol
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f, encoding='latin1')
            logger.info(f"Successfully loaded model using latin1 encoding from {model_path}")
            return model
        except Exception as e2:
            # Check if the file might be text instead of binary
            try:
                with open(model_path, 'r') as f:
                    content = f.read(1000)  # Read first 1000 chars
                    if content.strip().startswith('{') and '}' in content:
                        logger.error(f"Model file {model_path} appears to be JSON, not a valid pickle file")
                    elif "sklearn" in content or "joblib" in content:
                        logger.error(f"Model file {model_path} appears to be a text file with model info, not binary")
            except Exception:
                pass  # Ignore errors from text inspection
            
            # Provide detailed error information
            error_details = str(e1)
            if "invalid load key" in error_details:
                logger.error(f"Model file {model_path} appears to be corrupted: {error_details}")
            else:
                logger.error(f"Error loading model from {model_path}: {error_details}")
                logger.error(traceback.format_exc())
            return None


def find_latest_model_files(pattern: str, models_dir: str) -> List[Tuple[str, str]]:
    """
    Find the latest versions of model files that match a pattern
    
    Args:
        pattern: Pattern to match for model files
        models_dir: Directory to search for models
        
    Returns:
        List of (model_type, filepath) tuples
    """
    model_files = []
    
    # Get all files matching the pattern
    matching_files = glob.glob(os.path.join(models_dir, pattern))
    
    # Skip metadata.json files when dealing with player prop models
    matching_files = [f for f in matching_files if "_metadata.json" not in f]
    
    if not matching_files:
        logger.warning(f"No model files found matching pattern: {pattern} in {models_dir}")
        return []
    
    # Group by model type and find the latest one of each type
    model_types = {}
    
    # First try to get today's models (priority)
    today = datetime.now().strftime("%Y%m%d")
    today_models = [f for f in matching_files if today in f]
    
    # Use today's models if available, otherwise use all models
    models_to_check = today_models if today_models else matching_files
    
    for file_path in models_to_check:
        file_name = os.path.basename(file_path)
        
        # Extract model type (removing timestamp portions)
        parts = file_name.split('_')
        if len(parts) >= 3:  # Should have at least 3 parts
            if 'player' in file_name:
                # For player models: player_player_points_gradient_boosting_20250420_212744.pkl
                if len(parts) >= 4:
                    model_type = '_'.join(parts[2:4])  # e.g. 'points_gradient_boosting'
                else:
                    model_type = parts[2]  # Just get the prop type if structure is different
            else:
                # For game models: game_home_win_gradient_boosting_20250420_212027.pkl
                model_type = parts[2]  # e.g. 'gradient_boosting'
                
            # Check if this is newer than what we already have or if we don't have this type yet
            if model_type not in model_types or os.path.getmtime(file_path) > os.path.getmtime(model_types[model_type]):
                model_types[model_type] = file_path
                
    # Convert to list of tuples
    for model_type, file_path in model_types.items():
        model_files.append((model_type, file_path))
        logger.debug(f"Found model type {model_type}: {file_path}")
    
    if model_files:
        logger.info(f"Found {len(model_files)} model files for pattern: {pattern}")
    else:
        logger.warning(f"No usable model files found for pattern: {pattern}")
        
    return model_files


def load_ensemble_models() -> Dict[str, Any]:
    """
    Load ensemble of all available model types for enhanced prediction accuracy
    
    This function loads all variants of our trained models and creates ensembles:
    - game_outcome models (moneyline, spread, total)
    - player_props models
    
    Returns:
        Dictionary of loaded ensemble models by prediction type
    """
    ensembles = {
        "moneyline": {},
        "spread": {},
        "total": {},
        "player_props": {}
    }
    
    # Results dir is one level up from MODELS_DIR
    results_dir = str(Path(MODELS_DIR).parent)
    models_dir = os.path.join(results_dir, "models")
    
    logger.info(f"Scanning for models in: {models_dir}")
    
    # Load game outcome models
    game_moneyline_models = find_latest_model_files("game_home_win_*.pkl", models_dir)
    game_spread_models = find_latest_model_files("game_spread_diff_*.pkl", models_dir)
    game_total_models = find_latest_model_files("game_total_points_*.pkl", models_dir)
    
    # Load the models for each prediction type
    for model_type, file_path in game_moneyline_models:
        model = load_model_from_file(file_path)
        if model is not None:
            ensembles["moneyline"][model_type] = model
            logger.info(f"Loaded moneyline model: {model_type}")
    
    for model_type, file_path in game_spread_models:
        model = load_model_from_file(file_path)
        if model is not None:
            ensembles["spread"][model_type] = model
            logger.info(f"Loaded spread model: {model_type}")
    
    for model_type, file_path in game_total_models:
        model = load_model_from_file(file_path)
        if model is not None:
            ensembles["total"][model_type] = model
            logger.info(f"Loaded total model: {model_type}")
    
    # Load player props models
    player_prop_types = ["points", "rebounds", "assists", "threes", "steals", "blocks"]
    player_props_models = {}
    
    for prop_type in player_prop_types:
        player_props_models[prop_type] = find_latest_model_files(f"player_player_{prop_type}_*.pkl", models_dir)
        
        # Load models for this prop type
        prop_models = {}
        for model_type, file_path in player_props_models[prop_type]:
            model = load_model_from_file(file_path)
            if model is not None:
                prop_models[model_type] = model
                logger.info(f"Loaded {prop_type} model: {model_type}")
        
        # Add to the player_props ensemble
        if prop_models:
            ensembles["player_props"][prop_type] = prop_models
    
    # Report loaded model counts
    for prediction_type, models in ensembles.items():
        if prediction_type != "player_props":
            model_count = len(models)
            logger.info(f"Loaded {model_count} {prediction_type} models")
        else:
            total_models = sum(len(prop_models) for prop_models in models.values())
            logger.info(f"Loaded {total_models} player props models across {len(models)} prop types")
    
    return ensembles


# Add backward compatibility alias for existing code
def load_models() -> Dict[str, Any]:
    """
    Backward compatibility alias for load_ensemble_models
    
    Returns:
        Dictionary of loaded models, same as load_ensemble_models
    """
    logger.info("Using load_models() - this is an alias for load_ensemble_models()")
    return load_ensemble_models()


def check_player_props_ready() -> bool:
    """
    Check if player props models are available and ready to use
    
    Returns:
        bool: True if player props models are ready, False otherwise
    """
    # Results dir is one level up from MODELS_DIR
    results_dir = str(Path(MODELS_DIR).parent)
    models_dir = os.path.join(results_dir, "models")
    
    # Check for at least one model file for each player prop type
    prop_types = ["points", "rebounds", "assists", "threes", "steals", "blocks"]
    
    for prop_type in prop_types:
        model_files = glob.glob(os.path.join(models_dir, f"player_player_{prop_type}_*.pkl"))
        if not model_files:
            return False
    
    return True
