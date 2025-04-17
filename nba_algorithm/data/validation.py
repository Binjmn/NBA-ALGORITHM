# -*- coding: utf-8 -*-
"""
Data Validation Module

This module provides functions for validating API data and ensuring
sufficient quality for making predictions.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)


def validate_api_data(games, team_stats, odds, historical_games=None):
    """
    Validate that we have sufficient API data to make predictions
    
    Args:
        games: List of games to predict
        team_stats: Dictionary of team statistics
        odds: Dictionary of betting odds
        historical_games: Optional list of historical games
        
    Raises:
        ValueError: If data is insufficient for predictions
    """
    # Validate games data
    if not games:
        raise ValueError("No games available for prediction")
    
    # Validate team stats
    if not team_stats:
        raise ValueError("No team statistics available for prediction")
    
    # Check that we have stats for all teams in games
    missing_teams = []
    for game in games:
        home_team_id = game.get('home_team', {}).get('id')
        visitor_team_id = game.get('visitor_team', {}).get('id')
        
        if home_team_id and str(home_team_id) not in team_stats:
            home_team_name = game.get('home_team', {}).get('full_name', f"ID: {home_team_id}")
            missing_teams.append(home_team_name)
            
        if visitor_team_id and str(visitor_team_id) not in team_stats:
            visitor_team_name = game.get('visitor_team', {}).get('full_name', f"ID: {visitor_team_id}")
            missing_teams.append(visitor_team_name)
    
    if missing_teams:
        logger.warning(f"Missing team statistics for: {', '.join(missing_teams)}")
        # Don't fail completely, as we may have fallback data
    
    # Validate historical games if provided
    if historical_games is not None and not historical_games:
        logger.warning("No historical games available. Some features may be limited.")
        # Don't fail completely, as we can still make predictions without historical data
    
    # Validate betting odds
    if not odds:
        logger.warning("No betting odds available. Predictions will not include market insights.")
        # Don't fail for missing odds, as we can still make predictions without them


def check_api_keys():
    """
    Check for required API keys
    
    Returns:
        List[str]: List of missing API keys
    """
    import os
    missing_keys = []
    
    # Check for BallDontLie API key
    if not os.environ.get('BALLDONTLIE_API_KEY'):
        missing_keys.append('BALLDONTLIE_API_KEY')
    
    # Check for Odds API key
    if not os.environ.get('ODDS_API_KEY'):
        missing_keys.append('ODDS_API_KEY')
    
    return missing_keys


def validate_prediction_features(features_df):
    """
    Validate that features dataframe has all required columns for prediction
    
    Args:
        features_df: DataFrame of features for prediction
        
    Returns:
        bool: True if features are valid, False otherwise
    """
    # Define required feature columns
    required_features = [
        'home_team_win_pct', 'visitor_team_win_pct',
        'home_team_pts_per_game', 'visitor_team_pts_per_game',
        'home_team_pts_allowed_per_game', 'visitor_team_pts_allowed_per_game'
    ]
    
    # Check if all required features are present
    missing_features = [f for f in required_features if f not in features_df.columns]
    
    if missing_features:
        logger.error(f"Missing required features: {', '.join(missing_features)}")
        return False
    
    # Check for missing values in required features
    null_counts = features_df[required_features].isnull().sum()
    features_with_nulls = null_counts[null_counts > 0].index.tolist()
    
    if features_with_nulls:
        logger.error(f"Features with missing values: {', '.join(features_with_nulls)}")
        return False
    
    return True


def validate_player_data(players_data, player_stats):
    """
    Validate player data for player predictions
    
    Args:
        players_data: List of players
        player_stats: Dictionary of player statistics
        
    Returns:
        bool: True if player data is valid, False otherwise
    """
    if not players_data:
        logger.error("No player data available")
        return False
    
    if not player_stats:
        logger.error("No player statistics available")
        return False
    
    # Check that we have stats for at least some players
    player_ids_with_stats = set(player_stats.keys())
    player_ids = {str(p.get('id')) for p in players_data if p.get('id')}
    
    if not player_ids:
        logger.error("No valid player IDs found")
        return False
    
    # Check overlap between players and stats
    overlap = player_ids.intersection(player_ids_with_stats)
    overlap_percentage = len(overlap) / len(player_ids) if player_ids else 0
    
    if overlap_percentage < 0.5:  # Less than 50% of players have stats
        logger.warning(f"Only {overlap_percentage:.1%} of players have statistics")
        # Don't fail completely, as we can still make predictions for the available players
    
    return True
