#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Player Features Module

This module handles feature extraction and preparation for NBA player prop predictions.

Author: Cascade
Date: April 2025
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
import traceback

from ..utils.math_utils import calculate_matchup_difficulty, calculate_prop_confidence
from ..utils.string_utils import position_to_numeric, parse_height
from ..models.injury_analysis import fetch_team_injuries, get_injury_impact_score
from ..models.advanced_metrics import fetch_player_advanced_metrics, get_player_efficiency_rating

# Configure logger
logger = logging.getLogger(__name__)


def extract_player_features(player: Dict, game: Dict, is_home: bool) -> Dict[str, Any]:
    """
    Extract relevant features for player prop prediction with comprehensive error handling
    
    Args:
        player: Player data dictionary
        game: Game data dictionary
        is_home: Whether the player is on the home team
        
    Returns:
        Dictionary of features for prediction
    """
    if not player or not game:
        logger.error("Missing player or game data for feature extraction")
        raise ValueError("Missing player or game data for feature extraction")
    
    try:
        features = {}
        
        # Basic player information
        features['player_id'] = player.get('id', 0)
        features['player_name'] = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        
        # Team context
        team_type = 'home_team' if is_home else 'away_team'
        opponent_type = 'away_team' if is_home else 'home_team'
        
        features['team_id'] = game.get(team_type, {}).get('id', 0)
        features['opponent_id'] = game.get(opponent_type, {}).get('id', 0)
        features['is_home'] = int(is_home)  # Convert boolean to integer for model compatibility
        
        # Player demographics and position
        position = player.get('position', '')
        features['position'] = position
        features['position_value'] = position_to_numeric(position)
        
        height_str = player.get('height', '')
        features['height_inches'] = parse_height(height_str)
        
        # Weight might not be available for all players
        weight = player.get('weight', 0)
        features['weight'] = float(weight) if weight else 0.0
        
        # Player experience
        # Years pro might not be available or might be in different formats
        years_pro = player.get('years_pro', 0)
        features['years_pro'] = int(years_pro) if years_pro else 0
        
        # Season statistics (if available)
        season_stats = player.get('season_stats', {})
        
        # Minutes played
        features['minutes_per_game'] = float(season_stats.get('min', 0))
        
        # Scoring stats
        features['points_per_game'] = float(season_stats.get('pts', 0))
        features['field_goal_pct'] = float(season_stats.get('fg_pct', 0))
        features['three_point_pct'] = float(season_stats.get('fg3_pct', 0))
        features['free_throw_pct'] = float(season_stats.get('ft_pct', 0))
        
        # Rebounding stats
        features['rebounds_per_game'] = float(season_stats.get('reb', 0))
        features['offensive_rebounds'] = float(season_stats.get('oreb', 0))
        features['defensive_rebounds'] = float(season_stats.get('dreb', 0))
        
        # Assist stats
        features['assists_per_game'] = float(season_stats.get('ast', 0))
        
        # Other stats
        features['steals_per_game'] = float(season_stats.get('stl', 0))
        features['blocks_per_game'] = float(season_stats.get('blk', 0))
        features['turnovers_per_game'] = float(season_stats.get('turnover', 0))
        features['fouls_per_game'] = float(season_stats.get('pf', 0))
        
        # Advanced stats (if available)
        features['effective_fg_pct'] = float(season_stats.get('efg_pct', 0))
        features['true_shooting_pct'] = float(season_stats.get('ts_pct', 0))
        
        # NEW: Add injury context to player features
        try:
            # Get team injuries
            team_injuries = fetch_team_injuries(features['team_id'])
            opponent_injuries = fetch_team_injuries(features['opponent_id'])
            
            # Process team injuries
            team_injury_data = get_injury_impact_score(team_injuries, features['team_id'])
            opponent_injury_data = get_injury_impact_score(opponent_injuries, features['opponent_id'])
            
            # Check if any key defensive players are injured on opponent team
            # Filter for defensive positions (typically centers, power forwards, defensive specialists)
            defensive_player_injuries = []
            for injury in opponent_injury_data.get('detail', []):
                position = injury.get('position', '')
                # Centers and power forwards typically provide rim protection
                if position in ['C', 'PF', 'F-C', 'C-F']:
                    defensive_player_injuries.append(injury)
            
            features['opponent_injury_impact'] = opponent_injury_data['overall_impact']
            features['opponent_defensive_injuries'] = len(defensive_player_injuries) > 0
            features['team_injury_impact'] = team_injury_data['overall_impact']
            
            # Calculate opportunity increase due to teammate injuries
            # If teammates at same position are injured, player may get more minutes/usage
            same_position_injuries = []
            for injury in team_injury_data.get('detail', []):
                if injury.get('position', '') == position and injury.get('player_id') != features['player_id']:
                    same_position_injuries.append(injury)
            
            # Calculate opportunity boost (0 to 1 scale)
            if same_position_injuries:
                # Sum importance of injured teammates at same position
                importance_sum = sum(inj.get('importance', 0) for inj in same_position_injuries)
                features['opportunity_boost'] = min(importance_sum, 1.0)  # Cap at 1.0
            else:
                features['opportunity_boost'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error adding injury context for player {features['player_id']}: {str(e)}")
            # Set default values if error occurs
            features['opponent_injury_impact'] = 0.0
            features['opponent_defensive_injuries'] = False
            features['team_injury_impact'] = 0.0
            features['opportunity_boost'] = 0.0
        
        # NEW: Add advanced metrics
        try:
            # Get player advanced metrics
            advanced_metrics = fetch_player_advanced_metrics(features['player_id'])
            efficiency_data = get_player_efficiency_rating(features['player_id'])
            
            if advanced_metrics:
                # Add key advanced metrics
                features['offensive_rating'] = float(advanced_metrics.get('offensive_rating', 0))
                features['defensive_rating'] = float(advanced_metrics.get('defensive_rating', 0))
                features['player_efficiency'] = float(advanced_metrics.get('efficiency_score', 0))
                features['usage_percentage'] = float(advanced_metrics.get('usage_percentage', 0))
                features['assist_percentage'] = float(advanced_metrics.get('assist_percentage', 0))
                
                # Add efficiency trend (positive = improving, negative = declining)
                features['efficiency_trend'] = float(efficiency_data.get('trend', 0))
            else:
                # Set default values if no advanced metrics available
                features['offensive_rating'] = 110.0  # League average approximation
                features['defensive_rating'] = 110.0  # League average approximation
                features['player_efficiency'] = 0.5    # Average efficiency
                features['usage_percentage'] = 0.2     # Average usage
                features['assist_percentage'] = 0.15   # Average assist rate
                features['efficiency_trend'] = 0.0     # Neutral trend
        except Exception as e:
            logger.warning(f"Error adding advanced metrics for player {features['player_id']}: {str(e)}")
            # Set default values if error occurs
            features['offensive_rating'] = 110.0
            features['defensive_rating'] = 110.0
            features['player_efficiency'] = 0.5
            features['usage_percentage'] = 0.2
            features['assist_percentage'] = 0.15
            features['efficiency_trend'] = 0.0
        
        # Team context features
        team_stats = game.get(team_type, {}).get('stats', {})
        opponent_stats = game.get(opponent_type, {}).get('stats', {})
        
        if team_stats and opponent_stats:
            # Team pace and style
            features['team_pace'] = float(team_stats.get('pace', 100.0))
            features['team_off_rating'] = float(team_stats.get('offensive_rating', 110.0))
            
            # Opponent defense
            features['opponent_def_rating'] = float(opponent_stats.get('defensive_rating', 110.0))
            
            # Matchup difficulty
            features['matchup_difficulty'] = calculate_matchup_difficulty(player, opponent_stats)
            
            # Expected game environment
            features['expected_pace'] = (features['team_pace'] + float(opponent_stats.get('pace', 100.0))) / 2.0
            features['expected_points'] = float(team_stats.get('points_pg', 110.0))
            
            # Is team favored?
            point_spread = game.get('odds', {}).get('spread', 0.0)
            features['team_favored'] = 1 if (is_home and point_spread > 0) or (not is_home and point_spread < 0) else 0
            
            # Expected game total
            features['game_total'] = game.get('odds', {}).get('total', 220.0)
        else:
            # Default values if team stats are not available
            features.update({
                'team_pace': 100.0,
                'team_off_rating': 110.0,
                'opponent_def_rating': 110.0,
                'matchup_difficulty': 5.0,
                'expected_pace': 100.0,
                'expected_points': 110.0,
                'team_favored': 0,
                'game_total': 220.0
            })
        
        # Rest and schedule factors - get actual data from the team_data module
        try:
            from ..data.team_data import is_back_to_back, calculate_rest_days
            from datetime import datetime
            
            # Get game date (default to today if not provided)
            game_date = player.get('game_date')
            if not game_date:
                game_date = datetime.now()
            elif isinstance(game_date, str):
                game_date = datetime.fromisoformat(game_date)
            
            # Get team ID
            team_id = player.get('team', {}).get('id') if isinstance(player.get('team'), dict) else player.get('team_id')
            if not team_id:
                raise ValueError("Team ID not available for player")
            
            # Calculate rest days and back-to-back status
            rest_days = calculate_rest_days(team_id, game_date)
            features['rest_days'] = rest_days
            features['is_back_to_back'] = 1 if rest_days == 0 else 0
            
            logger.info(f"Player {player.get('first_name')} {player.get('last_name')} rest days: {rest_days}, B2B: {features['is_back_to_back']}")
        except Exception as e:
            logger.error(f"Error calculating rest days for player: {str(e)}")
            # Don't use default values - propagate the error to ensure data quality
            raise ValueError(f"Failed to determine rest days and back-to-back status: {str(e)}")
        
        return features
    
    except Exception as e:
        logger.error(f"Error extracting player features: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't return an empty dict - in production code we should propagate errors
        # so calling functions can handle them appropriately
        raise ValueError(f"Failed to extract player features: {str(e)}")


def predict_props_for_player(player: Dict, is_home_team: bool, team_name: str, opponent_name: str, 
                             prop_models: Dict, prop_types: List[str]) -> List[Dict[str, Any]]:
    """
    Predict props for a single player using comprehensive prop models
    
    Args:
        player: Player data dictionary
        is_home_team: Whether player is on home team
        team_name: Name of player's team
        opponent_name: Name of opposing team
        prop_models: Dictionary of loaded prediction models
        prop_types: List of prop types to predict
        
    Returns:
        List of prop prediction dictionaries
    """
    if not player or not prop_models or not prop_types:
        logger.error("Missing required data for player prop prediction")
        return []
    
    player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
    logger.info(f"Predicting props for {player_name} ({team_name})")
    
    prop_predictions = []
    
    try:
        # Get player features
        import pandas as pd
        features = extract_player_features(player, {'home_team': {'name': team_name}, 'away_team': {'name': opponent_name}}, is_home_team)
        features_df = pd.DataFrame([features])
        
        # For each prop type, make predictions using all available models
        for prop_type in prop_types:
            if prop_type not in prop_models:
                logger.warning(f"No models available for {prop_type} prediction")
                continue
            
            # Get season average for this prop type (if available)
            season_avg = get_player_season_average(player, prop_type)
            
            # Make predictions with each model
            model_predictions = {}
            for model_name, model in prop_models[prop_type].items():
                try:
                    prediction = model.predict(features_df)[0]
                    model_predictions[model_name] = prediction
                except Exception as e:
                    logger.error(f"Error predicting {prop_type} with {model_name} model: {str(e)}")
            
            if not model_predictions:
                logger.warning(f"No successful predictions for {player_name}'s {prop_type}")
                continue
            
            # Aggregate predictions (prefer ensemble if available)
            if 'stacked_ensemble' in model_predictions:
                predicted_value = model_predictions['stacked_ensemble']
            else:
                # Average available model predictions
                predicted_value = sum(model_predictions.values()) / len(model_predictions)
            
            # Calculate confidence in this prediction
            confidence = calculate_prop_confidence(player, prop_type, predicted_value)
            
            # Get betting recommendation
            recommendation = get_prop_recommendation(predicted_value, season_avg, prop_type)
            
            # Create the prediction object
            prop_prediction = {
                'player_name': player_name,
                'team': team_name,
                'opponent': opponent_name,
                'is_home': is_home_team,
                'prop_type': prop_type,
                'prediction': round(predicted_value, 1),
                'season_average': round(season_avg, 1),
                'confidence': confidence,
                'recommendation': recommendation,
                'models_used': list(model_predictions.keys()),
                'model_predictions': model_predictions
            }
            
            prop_predictions.append(prop_prediction)
            logger.info(f"Predicted {prop_type} for {player_name}: {predicted_value:.1f} (confidence: {confidence:.2f})")
        
        return prop_predictions
    
    except Exception as e:
        logger.error(f"Error predicting props for {player_name}: {str(e)}")
        return []


def get_player_season_average(player: Dict, prop_type: str) -> float:
    """
    Get player's season average for a specific prop type with proper error handling
    
    Args:
        player: Player data dictionary
        prop_type: Prop type (points, assists, rebounds)
        
    Returns:
        Season average for the specified prop
    """
    try:
        # Get season stats if available
        season_stats = player.get('season_stats', {})
        
        if not season_stats:
            logger.warning(f"No season stats available for {player.get('first_name', '')} {player.get('last_name', '')}")
            return 0.0
        
        # Return the appropriate stat based on prop type
        if prop_type == 'points':
            return float(season_stats.get('pts', 0.0))
        elif prop_type == 'rebounds':
            return float(season_stats.get('reb', 0.0))
        elif prop_type == 'assists':
            return float(season_stats.get('ast', 0.0))
        else:
            logger.warning(f"Unknown prop type: {prop_type}")
            return 0.0
    
    except Exception as e:
        logger.error(f"Error getting season average for {prop_type}: {str(e)}")
        return 0.0


def get_prop_recommendation(prediction: float, season_avg: float, prop_type: str) -> Dict[str, Any]:
    """
    Generate over/under recommendation based on prediction vs season average and other factors
    
    Args:
        prediction: Predicted prop value
        season_avg: Season average for this prop
        prop_type: Type of prop
        
    Returns:
        Dictionary with recommendation details
    """
    try:
        # Calculate percentage difference from season average
        if season_avg > 0:
            pct_diff = (prediction - season_avg) / season_avg
        else:
            pct_diff = 0.0
        
        # Determine recommendation strength based on difference
        strength = 'strong' if abs(pct_diff) > 0.15 else 'moderate' if abs(pct_diff) > 0.05 else 'weak'
        
        # Determine recommendation direction (over/under)
        direction = 'over' if prediction > season_avg else 'under'
        
        # Create explanation based on prop type and direction
        prop_name = {'points': 'scoring', 'rebounds': 'rebounding', 'assists': 'assist'}[prop_type]
        
        if direction == 'over':
            explanation = f"Projected {prop_name} is {abs(pct_diff):.1%} higher than season average"
            if strength == 'strong':
                explanation += f". Player's {prop_name} is expected to significantly outperform typical production."
            elif strength == 'moderate':
                explanation += f". Favorable matchup conditions for {prop_name} performance."
        else:  # under
            explanation = f"Projected {prop_name} is {abs(pct_diff):.1%} lower than season average"
            if strength == 'strong':
                explanation += f". Player's {prop_name} is expected to be notably limited in this matchup."
            elif strength == 'moderate':
                explanation += f". Matchup presents challenges for {prop_name} production."
        
        return {
            'direction': direction,
            'strength': strength,
            'explanation': explanation,
            'pct_diff_from_average': pct_diff
        }
    
    except Exception as e:
        logger.error(f"Error generating prop recommendation: {str(e)}")
        return {
            'direction': 'neutral',
            'strength': 'weak',
            'explanation': 'Insufficient data to generate recommendation',
            'pct_diff_from_average': 0.0
        }
