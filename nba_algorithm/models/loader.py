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
from pathlib import Path
from typing import Dict, Any, Optional, Union

from ..utils.config import MODELS_DIR

# Configure logger
logger = logging.getLogger(__name__)


def load_model_from_file(model_path: str) -> Optional[Any]:
    """
    Load a model from a pickle file with proper error handling
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model or None if loading failed
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except (FileNotFoundError, pickle.PickleError, EOFError, AttributeError) as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading model from {model_path}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def load_models() -> Dict[str, Any]:
    """
    Load all available trained models from the models directory
    with proper error handling and validation
    
    Returns:
        Dictionary of loaded models
    """
    models = {}
    model_files = {
        "spread": "spread_model.pkl",
        "moneyline": "moneyline_model.pkl",
        "total": "total_model.pkl",
        "ensemble": "ensemble_model.pkl"
    }
    
    for model_type, filename in model_files.items():
        model_path = str(MODELS_DIR / filename)
        model = load_model_from_file(model_path)
        
        if model is not None:
            models[model_type] = model
            logger.info(f"Loaded {model_type} model from {filename}")
        else:
            logger.warning(f"Could not load {model_type} model from {filename}")
    
    if not models:
        logger.error("No models could be loaded. Prediction capability will be limited.")
    
    return models


def load_enhanced_models() -> Dict[str, Dict[str, Any]]:
    """
    Load all available enhanced models for game and player prop predictions
    
    This loads both the game prediction models (spread, moneyline) and
    the player prop prediction models (points, assists, rebounds)
    
    Returns:
        Dictionary of loaded models by type and target
    """
    enhanced_models = {
        "game": {},
        "player_props": {}
    }
    
    # Game prediction models
    game_model_files = {
        "spread": ["gradient_boost_spread.pkl", "random_forest_spread.pkl"],
        "moneyline": ["gradient_boost_moneyline.pkl", "random_forest_moneyline.pkl"],
        "total": ["gradient_boost_total.pkl", "random_forest_total.pkl"],
        "ensemble": ["stacked_ensemble_spread.pkl", "stacked_ensemble_moneyline.pkl"]
    }
    
    # Load game prediction models
    for model_type, filenames in game_model_files.items():
        enhanced_models["game"][model_type] = {}
        
        for filename in filenames:
            model_name = filename.split("_")[0]  # e.g., gradient_boost, random_forest
            model_path = str(MODELS_DIR / filename)
            model = load_model_from_file(model_path)
            
            if model is not None:
                enhanced_models["game"][model_type][model_name] = model
                logger.info(f"Loaded {model_name} model for {model_type} from {filename}")
            else:
                logger.warning(f"Could not load {model_name} model for {model_type} from {filename}")
    
    # Load player prop models using our new dedicated loader
    player_props_models = load_player_props_models()
    if player_props_models:
        enhanced_models["player_props"] = player_props_models
    
    # Validate that we have at least some models
    if not any(enhanced_models["game"].values()):
        logger.error("No game prediction models could be loaded. Game predictions will be unavailable.")
    
    if not any(enhanced_models["player_props"].values()):
        logger.warning("No player prop models could be loaded. Player prop predictions will be unavailable.")
        # We'll still continue and just show a message in the predictions
    
    return enhanced_models


def load_player_props_models() -> Dict[str, Dict[str, Any]]:
    """
    Load all available player props models with dynamic discovery and validation
    
    This function looks for player props models in the dedicated player_props directory
    and loads the most recent version of each model type for each prop type.
    
    Returns:
        Dictionary of loaded models by prop type and model type
    """
    try:
        # Import from our dedicated player props module
        from .player_props_loader import load_player_props_models as load_impl
        
        # Call the implementation
        player_models = load_impl()
        
        if not player_models:
            logger.warning("No player props models could be loaded.")
            return {}
        
        # Validate the returned models
        total_models = sum(len(models) for models in player_models.values())
        if total_models == 0:
            logger.warning("Player props loader returned empty model dictionary")
        else:
            logger.info(f"Successfully loaded {total_models} player props models")
            
            # Log details of loaded models
            for prop_type, models in player_models.items():
                if models:
                    logger.info(f"  - {prop_type}: {', '.join(models.keys())}")
                    
        return player_models
    except ImportError as e:
        logger.error(f"Failed to import player_props_loader module: {str(e)}")
        logger.error("Ensure the player_props_loader.py file is properly installed")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error loading player props models: {str(e)}")
        logger.error(traceback.format_exc())
        return {}


def check_player_props_ready() -> bool:
    """
    Check if player props models are available and ready to use
    
    Returns:
        bool: True if player props models are ready, False otherwise
    """
    try:
        # Load the player props models
        player_models = load_player_props_models()
        
        # Check if we have any models
        total_models = sum(len(models) for models in player_models.values())
        if total_models > 0:
            logger.info(f"Player props ready with {total_models} trained models")
            return True
        else:
            logger.warning("No player props models available, player props not ready")
            return False
    except Exception as e:
        logger.error(f"Error checking player props readiness: {str(e)}")
        return False
