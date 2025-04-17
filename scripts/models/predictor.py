#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Predictor Module

This module handles the core prediction logic for NBA games and player props.

Author: Cascade
Date: April 2025
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

# Configure logger
logger = logging.getLogger(__name__)


def run_predictions(models: Dict[str, Any], features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for each game using loaded models
    
    Args:
        models: Dictionary of loaded models
        features_df: DataFrame with game features
        
    Returns:
        DataFrame with predictions
    """
    if not models:
        logger.error("No models available for prediction. Aborting prediction process.")
        return pd.DataFrame()
    
    if features_df.empty:
        logger.error("No features provided for prediction. Aborting prediction process.")
        return pd.DataFrame()
    
    # Create a copy of the input DataFrame to add predictions
    results_df = features_df.copy()
    
    # Initialize prediction columns
    results_df['predicted_spread'] = np.nan
    results_df['predicted_total'] = np.nan
    results_df['home_win_probability'] = np.nan
    results_df['prediction_confidence'] = np.nan
    
    # Track available models
    available_models = [m for m in ['spread', 'moneyline', 'total', 'ensemble'] if m in models]
    
    if not available_models:
        logger.error("No prediction models found in the provided models dictionary.")
        return results_df
    
    logger.info(f"Making predictions with available models: {available_models}")
    
    try:
        # Get only the numeric features for prediction
        numeric_features = features_df.select_dtypes(include=['int64', 'float64']).fillna(0)
        
        # Predict point spread if the model is available
        if 'spread' in models:
            try:
                results_df['predicted_spread'] = models['spread'].predict(numeric_features)
                logger.info("Generated point spread predictions")
            except Exception as e:
                logger.error(f"Error predicting spread: {str(e)}")
                results_df['predicted_spread'] = 0.0
        
        # Predict over/under total if the model is available
        if 'total' in models:
            try:
                results_df['predicted_total'] = models['total'].predict(numeric_features)
                logger.info("Generated over/under total predictions")
            except Exception as e:
                logger.error(f"Error predicting total: {str(e)}")
                results_df['predicted_total'] = 220.0  # Reasonable NBA default
        
        # Predict win probability if the moneyline model is available
        if 'moneyline' in models:
            try:
                win_probs = models['moneyline'].predict_proba(numeric_features)
                results_df['home_win_probability'] = win_probs[:, 1]  # Probability of class 1 (home win)
                logger.info("Generated win probability predictions")
            except Exception as e:
                logger.error(f"Error predicting win probability: {str(e)}")
                # Convert spread to win probability if direct prediction fails
                if 'predicted_spread' in results_df.columns and not results_df['predicted_spread'].isna().all():
                    from ..utils.math_utils import spread_to_win_probability
                    results_df['home_win_probability'] = results_df['predicted_spread'].apply(spread_to_win_probability)
                    logger.info("Used point spread to estimate win probabilities")
                else:
                    results_df['home_win_probability'] = 0.5
        elif 'predicted_spread' in results_df.columns and not results_df['predicted_spread'].isna().all():
            # If no moneyline model but we have spread predictions, convert them
            from ..utils.math_utils import spread_to_win_probability
            results_df['home_win_probability'] = results_df['predicted_spread'].apply(spread_to_win_probability)
            logger.info("Used point spread to estimate win probabilities")
        
        # Use ensemble model if available
        if 'ensemble' in models and not numeric_features.empty:
            try:
                ensemble_preds = models['ensemble'].predict(numeric_features)
                ensemble_probs = models['ensemble'].predict_proba(numeric_features)[:, 1]
                
                # The ensemble may provide better predictions, so we'll update our estimates
                results_df['predicted_spread'] = ensemble_preds
                results_df['home_win_probability'] = ensemble_probs
                logger.info("Updated predictions using ensemble model")
            except Exception as e:
                logger.error(f"Error using ensemble model: {str(e)}")
                # Keep the existing predictions if ensemble fails
        
        # Calculate prediction confidence based on model agreement
        results_df['prediction_confidence'] = 0.7  # Default moderate confidence
        
        if 'spread' in models and 'moneyline' in models:
            from ..utils.math_utils import calculate_prediction_confidence
            
            # For each game, calculate confidence based on agreement between models
            for idx, row in results_df.iterrows():
                # Gather all model predictions for this game
                model_predictions = {}
                
                if 'predicted_spread' in row and not pd.isna(row['predicted_spread']):
                    model_predictions['spread'] = row['predicted_spread']
                
                if 'home_win_probability' in row and not pd.isna(row['home_win_probability']):
                    model_predictions['win_prob'] = row['home_win_probability']
                
                if model_predictions:
                    results_df.at[idx, 'prediction_confidence'] = calculate_prediction_confidence(model_predictions)
        
        logger.info("Prediction process completed successfully")
        return results_df
    
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        logger.error("Returning original DataFrame without predictions")
        return features_df


def predict_todays_games(games: List[Dict], team_stats: Dict, models: Dict) -> Dict[str, Any]:
    """
    Make predictions for today's NBA games using the enhanced models
    
    Args:
        games: List of game dictionaries
        team_stats: Dictionary of team statistics
        models: Dictionary of loaded prediction models
        
    Returns:
        Dictionary of game predictions
    """
    if not games:
        logger.error("No games provided for prediction. Cannot make game predictions.")
        return {}
    
    if not team_stats:
        logger.error("No team statistics provided. Cannot make game predictions.")
        return {}
    
    if not models or not models.get('game', {}):
        logger.error("No game prediction models loaded. Cannot make game predictions.")
        return {}
    
    from ..features.game_features import extract_game_features
    from ..utils.math_utils import spread_to_win_probability, calculate_prediction_confidence
    
    predictions = {}
    
    try:
        for game in games:
            game_id = game.get('id', f"game_{len(predictions) + 1}")
            home_team_id = game.get('home_team', {}).get('id')
            away_team_id = game.get('away_team', {}).get('id')
            
            if not home_team_id or not away_team_id:
                logger.warning(f"Missing team information for game {game_id}. Skipping prediction.")
                continue
            
            home_stats = team_stats.get(home_team_id, {})
            away_stats = team_stats.get(away_team_id, {})
            
            if not home_stats or not away_stats:
                logger.warning(f"Missing team stats for game {game_id}. Skipping prediction.")
                continue
            
            # Extract features for this game
            features = extract_game_features(home_stats, away_stats)
            
            # Convert to DataFrame for model prediction
            features_df = pd.DataFrame([features])
            
            # Make predictions with each available model
            model_predictions = {
                'spread': {},
                'moneyline': {},
                'total': {}
            }
            
            # Predict point spread
            for model_name, model in models['game'].get('spread', {}).items():
                try:
                    model_predictions['spread'][model_name] = model.predict(features_df)[0]
                except Exception as e:
                    logger.error(f"Error predicting spread with {model_name} model: {str(e)}")
            
            # Predict moneyline (win probability)
            for model_name, model in models['game'].get('moneyline', {}).items():
                try:
                    probs = model.predict_proba(features_df)[0]
                    model_predictions['moneyline'][model_name] = probs[1]  # Probability of home win
                except Exception as e:
                    logger.error(f"Error predicting moneyline with {model_name} model: {str(e)}")
            
            # Predict total (over/under)
            for model_name, model in models['game'].get('total', {}).items():
                try:
                    model_predictions['total'][model_name] = model.predict(features_df)[0]
                except Exception as e:
                    logger.error(f"Error predicting total with {model_name} model: {str(e)}")
            
            # Use ensemble models if available for more accurate predictions
            for model_name, model in models['game'].get('ensemble', {}).items():
                if 'stacked_ensemble_spread' in model_name:
                    try:
                        model_predictions['spread']['ensemble'] = model.predict(features_df)[0]
                    except Exception as e:
                        logger.error(f"Error predicting spread with ensemble model: {str(e)}")
                
                if 'stacked_ensemble_moneyline' in model_name:
                    try:
                        probs = model.predict_proba(features_df)[0]
                        model_predictions['moneyline']['ensemble'] = probs[1]  # Probability of home win
                    except Exception as e:
                        logger.error(f"Error predicting moneyline with ensemble model: {str(e)}")
            
            # Aggregate predictions across models
            point_spread = 0.0
            win_probability = 0.0
            over_under = 220.0  # Default reasonable value
            
            if model_predictions['spread']:
                # Prefer ensemble model if available, otherwise average all models
                if 'ensemble' in model_predictions['spread']:
                    point_spread = model_predictions['spread']['ensemble']
                else:
                    point_spread = sum(model_predictions['spread'].values()) / len(model_predictions['spread'])
            
            if model_predictions['moneyline']:
                # Prefer ensemble model if available, otherwise average all models
                if 'ensemble' in model_predictions['moneyline']:
                    win_probability = model_predictions['moneyline']['ensemble']
                else:
                    win_probability = sum(model_predictions['moneyline'].values()) / len(model_predictions['moneyline'])
            elif point_spread != 0.0:
                # Convert spread to win probability if no direct prediction
                win_probability = spread_to_win_probability(point_spread)
            
            if model_predictions['total']:
                over_under = sum(model_predictions['total'].values()) / len(model_predictions['total'])
            
            # Calculate prediction confidence
            confidence = calculate_prediction_confidence({
                'spread': point_spread,
                'win_prob': win_probability
            })
            
            # Store the final prediction
            predictions[game_id] = {
                'game_id': game_id,
                'home_team': game['home_team']['name'],
                'away_team': game['away_team']['name'],
                'predicted_spread': point_spread,
                'home_win_probability': win_probability,
                'predicted_total': over_under,
                'confidence': confidence,
                'date': game.get('date'),
                'details': game,
                'model_predictions': model_predictions
            }
            
            logger.info(f"Prediction for {game['away_team']['name']} @ {game['home_team']['name']} completed")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Unexpected error during game predictions: {str(e)}")
        logger.error("Returning empty predictions dictionary")
        return {}
