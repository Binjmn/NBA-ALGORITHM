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
    
    # Player prop prediction models
    prop_model_files = {
        "points": ["gradient_boost_points.pkl", "random_forest_points.pkl", "stacked_ensemble_points.pkl"],
        "assists": ["gradient_boost_assists.pkl", "random_forest_assists.pkl", "stacked_ensemble_assists.pkl"],
        "rebounds": ["gradient_boost_rebounds.pkl", "random_forest_rebounds.pkl", "stacked_ensemble_rebounds.pkl"]
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
    
    # Load player prop prediction models
    for prop_type, filenames in prop_model_files.items():
        enhanced_models["player_props"][prop_type] = {}
        
        for filename in filenames:
            model_name = filename.split("_")[0]  # e.g., gradient_boost, random_forest
            model_path = str(MODELS_DIR / filename)
            model = load_model_from_file(model_path)
            
            if model is not None:
                enhanced_models["player_props"][prop_type][model_name] = model
                logger.info(f"Loaded {model_name} model for {prop_type} from {filename}")
            else:
                logger.warning(f"Could not load {model_name} model for {prop_type} from {filename}")
    
    # Validate that we have at least some models
    if not any(enhanced_models["game"].values()):
        logger.error("No game prediction models could be loaded. Game predictions will be unavailable.")
    
    if not any(enhanced_models["player_props"].values()):
        logger.error("No player prop models could be loaded. Player prop predictions will be unavailable.")
    
    return enhanced_models
