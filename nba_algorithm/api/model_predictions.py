#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Model Predictions API Integration

This module connects trained prediction models to the API endpoints,
ensuring that real-time predictions are available for the frontend.
It handles loading models, preprocessing features, and generating
predictions with appropriate confidence scores.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, timezone

# Import data access modules
from src.api.direct_data_access import DirectDataAccess

# Import feature engineering
from src.data.feature_engineering import NBAFeatureEngineer

# Import model deployer
from src.models.model_deployer import ModelDeployer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelPredictionService:
    """
    Service for generating NBA game predictions using trained models
    """
    
    def __init__(self):
        """
        Initialize the prediction service
        """
        self.data_access = DirectDataAccess()
        self.feature_engineer = NBAFeatureEngineer()
        self.model_deployer = ModelDeployer()
        
        # Cache for loaded models to avoid reloading
        self._models_cache = {}
    
    def _get_model(self, prediction_type: str) -> Optional[Any]:
        """
        Get the appropriate model for a prediction type
        
        Args:
            prediction_type: Type of prediction ('moneyline', 'spread', 'total')
            
        Returns:
            Loaded model instance or None if not available
        """
        # Check cache first
        if prediction_type in self._models_cache:
            return self._models_cache[prediction_type]
        
        # Load model from production deployment
        model = self.model_deployer.load_production_model(prediction_type)
        
        if model:
            # Cache the model
            self._models_cache[prediction_type] = model
            return model
        else:
            logger.warning(f"No model available for {prediction_type} predictions")
            return None
    
    def prepare_game_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare features for a single game
        
        Args:
            game: Game data dictionary
            
        Returns:
            Dictionary of engineered features
        """
        try:
            # First, get historical games for feature context
            today = datetime.now(timezone.utc).date()
            start_date = (today - timedelta(days=90)).isoformat()  # 90 days of history
            end_date = today.isoformat()
            
            # Get historical games
            historical_games = self.data_access.get_games(start_date=start_date, end_date=end_date)
            
            if not historical_games:
                logger.warning("No historical games found for feature engineering")
                return {}
            
            # Convert to DataFrame
            games_df = pd.DataFrame(historical_games)
            
            # Add the current game to the DataFrame
            current_game_df = pd.DataFrame([game])
            all_games_df = pd.concat([games_df, current_game_df], ignore_index=True)
            
            # Use the feature engineer to create features
            game_id = str(game['id'])
            features = self.feature_engineer.create_game_features(all_games_df, game_id)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing game features: {str(e)}")
            return {}
    
    def predict_game_outcome(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive predictions for a single game
        
        Args:
            game: Game data dictionary
            
        Returns:
            Dictionary with predictions for moneyline, spread, and total
        """
        try:
            # Prepare features
            features = self.prepare_game_features(game)
            if not features:
                logger.error("Failed to prepare features for prediction")
                return {}
            
            # Convert to DataFrame for prediction
            features_df = pd.DataFrame([features])
            
            # Initialize results
            predictions = {
                'game_id': str(game['id']),
                'home_team': game.get('home_team', {}).get('name', 'Unknown'),
                'away_team': game.get('visitor_team', {}).get('name', 'Unknown'),
                'game_date': game.get('date', 'Unknown'),
                'predictions': {},
                'features_used': list(features_df.columns)[:10],  # Show first 10 features
                'prediction_time': datetime.now(timezone.utc).isoformat()
            }
            
            # Generate moneyline prediction (classification)
            moneyline_model = self._get_model('moneyline')
            if moneyline_model and moneyline_model.is_trained:
                try:
                    # Get win probability
                    win_proba = moneyline_model.predict_proba(features_df)
                    home_win_prob = float(win_proba[0][1]) if win_proba.shape[1] > 1 else float(win_proba[0])
                    
                    # Make prediction (1 for home win, 0 for away win)
                    home_win_pred = 1 if home_win_prob > 0.5 else 0
                    
                    predictions['predictions']['moneyline'] = {
                        'home_win_probability': home_win_prob,
                        'away_win_probability': 1.0 - home_win_prob,
                        'predicted_winner': predictions['home_team'] if home_win_pred == 1 else predictions['away_team'],
                        'confidence': abs(home_win_prob - 0.5) * 2  # Scale to 0-1 confidence
                    }
                except Exception as e:
                    logger.error(f"Error generating moneyline prediction: {str(e)}")
            
            # Generate spread prediction (regression)
            spread_model = self._get_model('spread')
            if spread_model and spread_model.is_trained:
                try:
                    # Predict point spread (positive for home advantage)
                    spread_pred = spread_model.predict(features_df)
                    predicted_spread = float(spread_pred[0])
                    
                    predictions['predictions']['spread'] = {
                        'predicted_spread': predicted_spread,
                        'home_cover_probability': 0.5 + (predicted_spread / 20),  # Simple mapping to probability
                        'away_cover_probability': 0.5 - (predicted_spread / 20)
                    }
                except Exception as e:
                    logger.error(f"Error generating spread prediction: {str(e)}")
            
            # Generate total points prediction (regression)
            total_model = self._get_model('total')
            if total_model and total_model.is_trained:
                try:
                    # Predict total points
                    total_pred = total_model.predict(features_df)
                    predicted_total = float(total_pred[0])
                    
                    predictions['predictions']['total'] = {
                        'predicted_total': predicted_total,
                        'over_probability': 0.5,  # Need game-specific line for actual probability
                        'under_probability': 0.5
                    }
                except Exception as e:
                    logger.error(f"Error generating total prediction: {str(e)}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting game outcome: {str(e)}")
            return {}
    
    def predict_upcoming_games(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Generate predictions for all upcoming games
        
        Args:
            days_ahead: Number of days ahead to predict
            
        Returns:
            List of game predictions
        """
        try:
            # Get upcoming games
            today = datetime.now(timezone.utc).date()
            end_date = (today + timedelta(days=days_ahead)).isoformat()
            
            upcoming_games = self.data_access.get_games(
                start_date=today.isoformat(),
                end_date=end_date,
                per_page=100  # Get a good number of games
            )
            
            if not upcoming_games:
                logger.warning(f"No upcoming games found in the next {days_ahead} days")
                return []
            
            # Generate predictions for each game
            predictions = []
            for game in upcoming_games:
                game_prediction = self.predict_game_outcome(game)
                if game_prediction:
                    predictions.append(game_prediction)
            
            logger.info(f"Generated predictions for {len(predictions)} upcoming games")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting upcoming games: {str(e)}")
            return []
    
    def evaluate_prediction_accuracy(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Evaluate the accuracy of past predictions against actual results
        
        Args:
            days_back: Number of days to look back for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Get completed games
            today = datetime.now(timezone.utc).date()
            start_date = (today - timedelta(days=days_back)).isoformat()
            end_date = today.isoformat()
            
            completed_games = self.data_access.get_games(
                start_date=start_date,
                end_date=end_date,
                per_page=100
            )
            
            # Filter to games with scores
            completed_games = [
                game for game in completed_games 
                if game.get('home_team_score') is not None and game.get('visitor_team_score') is not None
            ]
            
            if not completed_games:
                logger.warning(f"No completed games found in the past {days_back} days")
                return {}
            
            # Evaluate predictions against actual outcomes
            correct_ml = 0  # Moneyline
            correct_ats = 0  # Against the spread
            correct_ou = 0  # Over/Under
            total_evaluated = 0
            
            for game in completed_games:
                try:
                    # Get actual outcome
                    home_score = int(game['home_team_score'])
                    away_score = int(game['visitor_team_score'])
                    actual_winner = 'home' if home_score > away_score else 'away'
                    actual_spread = home_score - away_score
                    actual_total = home_score + away_score
                    
                    # Get predictions (make retroactive prediction)
                    # This simulates what we would have predicted before the game
                    game_copy = game.copy()
                    game_copy['home_team_score'] = None
                    game_copy['visitor_team_score'] = None
                    prediction = self.predict_game_outcome(game_copy)
                    
                    if not prediction or 'predictions' not in prediction:
                        continue
                    
                    # Evaluate moneyline
                    if 'moneyline' in prediction['predictions']:
                        ml_pred = prediction['predictions']['moneyline']
                        pred_winner = 'home' if ml_pred['home_win_probability'] > 0.5 else 'away'
                        if pred_winner == actual_winner:
                            correct_ml += 1
                    
                    # Evaluate spread (using 0 as the market spread for retro analysis)
                    if 'spread' in prediction['predictions']:
                        spread_pred = prediction['predictions']['spread']
                        pred_spread = spread_pred['predicted_spread']
                        # Predicted spread > 0 means home covers, < 0 means away covers
                        pred_cover = 'home' if pred_spread > 0 else 'away'
                        actual_cover = 'home' if actual_spread > 0 else 'away'
                        if pred_cover == actual_cover:
                            correct_ats += 1
                    
                    # Evaluate total (using predicted total as the line)
                    if 'total' in prediction['predictions']:
                        total_pred = prediction['predictions']['total']
                        pred_total = total_pred['predicted_total']
                        # Simple over/under on our predicted line
                        if (actual_total > pred_total and total_pred.get('over_probability', 0.5) > 0.5) or \
                           (actual_total < pred_total and total_pred.get('under_probability', 0.5) > 0.5):
                            correct_ou += 1
                    
                    total_evaluated += 1
                    
                except Exception as e:
                    logger.error(f"Error evaluating game {game.get('id')}: {str(e)}")
            
            # Calculate accuracy metrics
            metrics = {
                'period': f"{days_back} days",
                'games_evaluated': total_evaluated,
                'moneyline_accuracy': correct_ml / total_evaluated if total_evaluated > 0 else 0,
                'spread_accuracy': correct_ats / total_evaluated if total_evaluated > 0 else 0,
                'total_accuracy': correct_ou / total_evaluated if total_evaluated > 0 else 0
            }
            
            logger.info(f"Evaluated {total_evaluated} games with moneyline accuracy: {metrics['moneyline_accuracy']:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating prediction accuracy: {str(e)}")
            return {}


# Main function for testing
if __name__ == "__main__":
    import json
    
    # Create prediction service
    service = ModelPredictionService()
    
    # Get predictions for upcoming games
    predictions = service.predict_upcoming_games(days_ahead=7)
    
    if predictions:
        print(f"Generated predictions for {len(predictions)} upcoming games")
        # Print sample prediction
        print("\nSample prediction:")
        print(json.dumps(predictions[0], indent=2))
    else:
        print("No predictions generated")
    
    # Evaluate prediction accuracy
    metrics = service.evaluate_prediction_accuracy(days_back=30)
    
    if metrics:
        print("\nPrediction Accuracy:")
        print(f"Period: {metrics['period']}")
        print(f"Games Evaluated: {metrics['games_evaluated']}")
        print(f"Moneyline Accuracy: {metrics['moneyline_accuracy']:.2f}")
        print(f"Spread Accuracy: {metrics['spread_accuracy']:.2f}")
        print(f"Total Accuracy: {metrics['total_accuracy']:.2f}")
