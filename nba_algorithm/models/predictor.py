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

# Add new imports
from ..models.injury_analysis import compare_team_injuries
from ..models.advanced_metrics import get_team_efficiency_comparison

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
    results_df['total_prediction_status'] = np.nan
    results_df['total_prediction_error'] = np.nan
    results_df['win_probability_status'] = np.nan
    results_df['win_probability_error'] = np.nan
    
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
                results_df['total_prediction_status'] = 'model_based'  # Mark as model-based prediction
                logger.info("Generated over/under total predictions")
            except Exception as e:
                logger.error(f"Error predicting total: {str(e)}")
                # Instead of using a default value, mark as error
                results_df['predicted_total'] = None
                results_df['total_prediction_status'] = 'prediction_error'
                results_df['total_prediction_error'] = str(e)
                logger.error("Total prediction failed - explicitly marking as error instead of using default value")
        else:
            logger.warning("No 'total' model available for predictions")
            results_df['predicted_total'] = None
            results_df['total_prediction_status'] = 'model_missing'
        
        # Predict win probability if the moneyline model is available
        if 'moneyline' in models:
            try:
                win_probs = models['moneyline'].predict_proba(numeric_features)
                results_df['home_win_probability'] = win_probs[:, 1]  # Probability of class 1 (home win)
                results_df['win_probability_status'] = 'model_based'  # Mark as model-based prediction
                logger.info("Generated win probability predictions")
            except Exception as e:
                logger.error(f"Error predicting win probability: {str(e)}")
                # Instead of using a default value, mark as error
                results_df['home_win_probability'] = None
                results_df['win_probability_status'] = 'prediction_error'
                results_df['win_probability_error'] = str(e)
                logger.error("Win probability prediction failed - explicitly marking as error instead of using default value")
        else:
            logger.warning("No 'moneyline' model available for predictions")
            results_df['home_win_probability'] = None
            results_df['win_probability_status'] = 'model_missing'
        
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
        
        # Adjust confidence for games with missing historical data
        if 'missing_historical_data' in results_df.columns:
            for idx, row in results_df.iterrows():
                if row.get('missing_historical_data', 0) == 1.0:
                    logger.warning(f"Reducing confidence for game {row.get('game_id')} due to missing historical data")
                    # Reduce confidence by half for games with missing historical data
                    results_df.at[idx, 'prediction_confidence'] = results_df.at[idx, 'prediction_confidence'] * 0.5
                    # Mark the prediction as based on limited data
                    results_df.at[idx, 'prediction_method'] = 'limited_historical_data'
        
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
    games_data_quality = {}
    
    try:
        for game in games:
            try:
                game_id = game['id']
                home_team_id = game['home_team']['id']
                away_team_id = game['away_team']['id']
                
                # Log game being processed
                logger.info(f"Processing game {game_id}: {game['away_team']['abbreviation']} @ {game['home_team']['abbreviation']}")
                
                # Get team statistics for home and away teams
                home_team_stats = team_stats.get(home_team_id, None)
                away_team_stats = team_stats.get(away_team_id, None)
                
                # Check for missing team data and log warning
                if home_team_stats is None:
                    logger.warning(f"No statistics found for home team {home_team_id} ({game['home_team']['abbreviation']}). Skipping game {game_id}.")
                    games_data_quality[game_id] = 'missing_home_team_stats'
                    continue
                    
                if away_team_stats is None:
                    logger.warning(f"No statistics found for away team {away_team_id} ({game['away_team']['abbreviation']}). Skipping game {game_id}.")
                    games_data_quality[game_id] = 'missing_away_team_stats'
                    continue
                
                # Create features for this game using the FeatureEngineering class
                features = extract_game_features(home_team_stats, away_team_stats)
                
                # Convert to a format suitable for the model
                features['game_id'] = game_id
                features['home_team'] = game['home_team']['full_name']
                features['away_team'] = game['away_team']['full_name']
                features['home_team_id'] = home_team_id
                features['away_team_id'] = away_team_id
                features['game_date'] = game['date']
                
                # Add quality tracking
                games_data_quality[game_id] = 'complete'
                
                # Check for any None values in required feature fields and log warnings
                none_features = [k for k, v in features.items() if v is None and k not in ['data_quality']]
                if none_features:
                    logger.warning(f"Game {game_id} has missing features: {', '.join(none_features)}")
                    games_data_quality[game_id] = 'incomplete_features'
                
                # Make predictions with each available model
                model_predictions = {
                    'spread': {},
                    'moneyline': {},
                    'total': {}
                }
                
                # Predict point spread
                for model_name, model in models['game'].get('spread', {}).items():
                    try:
                        model_predictions['spread'][model_name] = model.predict(pd.DataFrame([features]))[0]
                    except Exception as e:
                        logger.error(f"Error predicting spread with {model_name} model: {str(e)}")
                
                # Predict moneyline (win probability)
                for model_name, model in models['game'].get('moneyline', {}).items():
                    try:
                        probs = model.predict_proba(pd.DataFrame([features]))[0]
                        model_predictions['moneyline'][model_name] = probs[1]  # Probability of home win
                    except Exception as e:
                        logger.error(f"Error predicting moneyline with {model_name} model: {str(e)}")
                
                # Predict total (over/under)
                for model_name, model in models['game'].get('total', {}).items():
                    try:
                        model_predictions['total'][model_name] = model.predict(pd.DataFrame([features]))[0]
                    except Exception as e:
                        logger.error(f"Error predicting total with {model_name} model: {str(e)}")
                
                # Use ensemble models if available for more accurate predictions
                for model_name, model in models['game'].get('ensemble', {}).items():
                    if 'stacked_ensemble_spread' in model_name:
                        try:
                            model_predictions['spread']['ensemble'] = model.predict(pd.DataFrame([features]))[0]
                        except Exception as e:
                            logger.error(f"Error predicting spread with ensemble model: {str(e)}")
                    
                    if 'stacked_ensemble_moneyline' in model_name:
                        try:
                            probs = model.predict_proba(pd.DataFrame([features]))[0]
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
            
            except Exception as e:
                logger.error(f"Error creating features for game {game.get('id', 'unknown')}: {str(e)}")
                games_data_quality[game.get('id', 'unknown')] = 'feature_creation_error'
                continue
        
        return predictions
    
    except Exception as e:
        logger.error(f"Unexpected error during game predictions: {str(e)}")
        logger.error("Returning empty predictions dictionary")
        return {}


def ensemble_prediction(models: Dict[str, Any], features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate ensemble predictions by combining multiple models
    
    Args:
        models: Dictionary of models for ensemble prediction
        features: DataFrame with game features
        
    Returns:
        DataFrame with ensemble predictions
    """
    if not models or features.empty:
        logger.warning("No models or features for ensemble prediction")
        return pd.DataFrame()
    
    # Create a copy of the input DataFrame for results
    results = features.copy()
    
    # Track individual model predictions for ensemble voting
    model_predictions = {}
    model_probabilities = {}
    
    # Get predictions from each model
    for model_name, model in models.items():
        try:
            if model_name.endswith('_win_predictor'):
                # Classification model (win prediction)
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(features)
                    model_probabilities[model_name] = probas[:, 1]  # Probability of class 1 (home win)
                
                preds = model.predict(features)
                model_predictions[model_name] = preds
                logger.debug(f"Model {model_name} predictions: {preds}")
            
            # Add other model types here if needed
            
        except Exception as e:
            logger.error(f"Error making prediction with {model_name}: {str(e)}")
    
    # Calculate ensemble probabilities using weighted averaging
    if model_probabilities:
        # Simple average of probabilities
        results['ensemble_win_probability'] = np.mean(list(model_probabilities.values()), axis=0)
        
        # Make a binary prediction based on the probability
        results['ensemble_win_prediction'] = (results['ensemble_win_probability'] > 0.5).astype(int)
    
    return results


def predict_game_outcomes(models: Dict[str, Any], games: List[Dict], team_stats: Dict) -> pd.DataFrame:
    """
    Main prediction function for game outcomes using all available models
    
    Args:
        models: Dictionary of loaded prediction models
        games: List of game dictionaries from the API
        team_stats: Dictionary of team statistics
        
    Returns:
        DataFrame with comprehensive game predictions
    """
    logger.info(f"Predicting outcomes for {len(games)} games")
    predictions = []
    
    # Even if feature preparation fails, we should provide basic predictions
    for game in games:
        try:
            home_team = game.get('home_team', {})
            visitor_team = game.get('visitor_team', {})
            
            if not home_team or not visitor_team:
                logger.warning(f"Missing team data for game {game.get('id')}")
                continue
                
            home_id = home_team.get('id')
            visitor_id = visitor_team.get('id')
            
            if not home_id or not visitor_id:
                logger.warning(f"Missing team IDs for game {game.get('id')}")
                continue
            
            # Extract basic team statistics
            home_stats = team_stats.get(home_id, {})
            visitor_stats = team_stats.get(visitor_id, {})
            
            if not home_stats or not visitor_stats:
                logger.warning(f"Missing team statistics for game {game.get('id')}")
                continue
            
            # Calculate home win probability based on simple win percentage comparison
            home_win_pct = home_stats.get('win_pct', 0.5)
            visitor_win_pct = visitor_stats.get('win_pct', 0.5)
            
            # Apply home court advantage (typically around 3-5% in NBA)
            home_advantage = 0.04  # 4% home court advantage
            adjusted_home_win_pct = min(1.0, home_win_pct + home_advantage)
            
            # Simple formula for win probability
            if adjusted_home_win_pct + visitor_win_pct > 0:
                home_win_probability = adjusted_home_win_pct / (adjusted_home_win_pct + visitor_win_pct)
            else:
                home_win_probability = 0.5  # Default to 50% if no data
            
            # Estimate point spread (negative means home favorite)
            est_home_points = home_stats.get('points_pg', 110.0)
            est_visitor_points = visitor_stats.get('points_pg', 110.0)
            est_home_defense = home_stats.get('defensive_rating', 110.0)
            est_visitor_defense = visitor_stats.get('defensive_rating', 110.0)
            
            # Simple point differential estimate
            point_diff = (est_home_points - est_visitor_defense/100*est_home_points) - \
                         (est_visitor_points - est_home_defense/100*est_visitor_points)
            
            # Add home court advantage (typically 2-4 points)
            point_diff += 3.0
            
            # Create prediction entry
            prediction = {
                'game_id': game.get('id'),
                'date': game.get('date', ''),
                'home_team': home_team.get('full_name', ''),
                'visitor_team': visitor_team.get('full_name', ''),
                'home_win_probability': home_win_probability,
                'visitor_win_probability': 1.0 - home_win_probability,
                'predicted_winner': home_team.get('full_name') if home_win_probability > 0.5 else visitor_team.get('full_name'),
                'spread': -point_diff,  # Negative means home favorite in betting odds
                'prediction_method': 'fallback',
                'confidence': 'low'  # Fallback predictions should have low confidence
            }
            
            predictions.append(prediction)
            logger.info(f"Created fallback prediction for {visitor_team.get('full_name')} @ {home_team.get('full_name')}")
            
        except Exception as e:
            logger.warning(f"Error creating fallback prediction for game {game.get('id')}: {str(e)}")
            continue
    
    # Convert to DataFrame
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        logger.info(f"Created fallback predictions for {len(predictions)} games")
        return predictions_df
    else:
        logger.error("Failed to create any predictions, returning empty DataFrame")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['game_id', 'date', 'home_team', 'visitor_team', 'home_win_probability', 'spread'])
