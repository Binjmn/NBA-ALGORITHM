#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Player Props Model Loading Module

This module provides functionality for loading trained player props models
and making predictions for player performance.

Author: Cascade
Date: April 2025
"""

import os
import sys
import logging
import traceback
import pickle
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_DIR = Path("models/player_props")
MIN_SAMPLES_FOR_PREDICTION = 10


def load_player_props_models():
    """
    Load all available player props models with enhanced error handling and logging
    
    Returns:
        Dict: Dictionary containing models for different prop types
    """
    models = {
        "points": {},
        "rebounds": {},
        "assists": {}
    }
    
    try:
        logger.info(f"Loading player props models from {DEFAULT_MODEL_DIR}")
        
        # Ensure directory exists
        model_dir = Path(DEFAULT_MODEL_DIR)
        if not model_dir.exists():
            logger.warning(f"Player props model directory not found: {model_dir}")
            logger.warning("Creating directory for future model storage")
            model_dir.mkdir(parents=True, exist_ok=True)
            return models
        
        # Find all model files
        model_files = list(model_dir.glob("*.pkl"))
        if not model_files:
            logger.warning(f"No player props model files found in {model_dir}")
            return models
        
        logger.info(f"Found {len(model_files)} player props model files")
        
        # Group models by type and get most recent version of each
        model_groups = {}
        for model_file in model_files:
            # Parse model name pattern: ModelType_PropType_Date.pkl
            name_parts = model_file.stem.split('_')
            if len(name_parts) < 3:
                logger.warning(f"Skipping model file with unexpected name format: {model_file}")
                continue
                
            model_type = name_parts[0]  # e.g., GradientBoosting, RandomForest
            prop_type = name_parts[1]   # e.g., points, rebounds, assists
            date_str = name_parts[2]    # e.g., 20250417
            
            # Create a unique key for this model type and prop type
            key = f"{model_type}_{prop_type}"
            
            # Check if we already have a model of this type and if this one is newer
            if key in model_groups and model_groups[key][1] > date_str:
                # Existing model is newer, skip this one
                continue
                
            # Add or update model in groups
            model_groups[key] = (model_file, date_str, model_type, prop_type)
        
        # Load each of the most recent models
        for key, (model_file, date_str, model_type, prop_type) in model_groups.items():
            try:
                logger.info(f"Loading model from {model_file}")
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    
                # Verify the model has a predict method
                if not hasattr(model, 'predict') or not callable(model.predict):
                    logger.warning(f"Model from {model_file} does not have a predict method, skipping")
                    continue
                    
                # Add model to the appropriate prop type dict
                if prop_type in models:
                    models[prop_type][model_type] = model
                    logger.info(f"Successfully loaded {model_type} model for {prop_type} prediction")
                else:
                    logger.warning(f"Unknown prop type: {prop_type}, skipping model")
            except Exception as e:
                logger.error(f"Error loading model from {model_file}: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Log summary of loaded models
        for prop_type, prop_models in models.items():
            logger.info(f"Loaded {len(prop_models)} models for {prop_type} prediction")
        
        return models
    except Exception as e:
        logger.error(f"Error loading player props models: {str(e)}")
        logger.debug(traceback.format_exc())
        return models


def predict_player_props(player_features, player_models=None):
    """
    Generate player prop predictions using loaded models
    
    Args:
        player_features: DataFrame with player features
        player_models: Dictionary of player models (optional, will load if not provided)
        
    Returns:
        DataFrame with player prop predictions
    """
    if player_features is None or player_features.empty:
        logger.error("No player features provided for prediction")
        return pd.DataFrame()
    
    try:
        logger.info(f"Generating prop predictions for {len(player_features)} players")
        
        # Load models if not provided
        if player_models is None:
            player_models = load_player_props_models()
            
        # Check if we have any models
        model_counts = {prop: len(models) for prop, models in player_models.items()}
        if all(count == 0 for count in model_counts.values()):
            logger.error("No player props models loaded, cannot make predictions")
            logger.info("Please run training: python scripts/train_player_props.py")
            return pd.DataFrame()
        
        # Make predictions for each prop type
        predictions = []
        
        for _, player_row in player_features.iterrows():
            player_id = player_row.get('player_id')
            player_name = player_row.get('player_name', f"Player {player_id}")
            
            if not player_id:
                logger.warning(f"Skipping player with no ID: {player_name}")
                continue
                
            # Create prediction row
            pred_row = {
                'player_id': player_id,
                'player_name': player_name,
                'team_id': player_row.get('team_id'),
                'team_name': player_row.get('team_name', ''),
                'position': player_row.get('position', ''),
                'game_id': player_row.get('game_id'),
                'confidence_level': None,  # No default value
                'confidence_score': None,  # No default value
                'prediction_status': 'pending',  # Track prediction status
                'data_quality': 'unknown'  # Track data quality
            }
            
            # Make predictions for each prop type
            for prop_type, models in player_models.items():
                if not models:
                    # No models for this prop type
                    pred_row[f'predicted_{prop_type}'] = None
                    pred_row[f'{prop_type}_prediction_status'] = 'no_model'
                    logger.warning(f"No models available for {prop_type} prediction for {player_name}")
                    continue
                    
                # Get features as a DataFrame
                player_df = pd.DataFrame([player_row])
                
                # Make predictions with each model
                prop_predictions = []
                model_names_used = []
                
                for model_name, model in models.items():
                    try:
                        # Try to make prediction
                        pred = model.predict(player_df)[0]
                        if not np.isnan(pred) and pred >= 0:  # Ensure valid predictions
                            prop_predictions.append(pred)
                            model_names_used.append(model_name)
                    except Exception as e:
                        logger.warning(f"Error predicting {prop_type} for {player_name} with {model_name}: {str(e)}")
                
                # Calculate final prediction (average of all models)
                if prop_predictions:
                    final_pred = sum(prop_predictions) / len(prop_predictions)
                    pred_row[f'predicted_{prop_type}'] = round(final_pred, 1)
                    pred_row[f'{prop_type}_prediction_status'] = 'success'
                    pred_row[f'{prop_type}_models_used'] = ','.join(model_names_used)
                    pred_row['prediction_status'] = 'partial_success' if pred_row['prediction_status'] == 'pending' else pred_row['prediction_status']
                    logger.debug(f"Predicted {prop_type} for {player_name}: {final_pred:.1f}")
                else:
                    pred_row[f'predicted_{prop_type}'] = None
                    pred_row[f'{prop_type}_prediction_status'] = 'prediction_error'
                    logger.warning(f"No valid {prop_type} predictions for {player_name}")
                    
            # Add player prediction to results
            predictions.append(pred_row)
            
        # Create DataFrame from predictions
        predictions_df = pd.DataFrame(predictions)
        
        if predictions_df.empty:
            logger.warning("No valid player prop predictions generated")
        else:
            logger.info(f"Generated prop predictions for {len(predictions_df)} players")
            
        return predictions_df
    except Exception as e:
        logger.error(f"Error generating player prop predictions: {str(e)}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()
