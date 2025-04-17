#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Game Features Module

This module handles feature extraction and preparation for NBA game prediction.

Author: Cascade
Date: April 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..data.team_data import calculate_rest_days, is_back_to_back

# Configure logger
logger = logging.getLogger(__name__)


def prepare_game_features(games: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generate features for the provided games with comprehensive error handling
    
    Args:
        games: List of game dictionaries
        
    Returns:
        DataFrame with game features
    """
    if not games:
        logger.error("No games provided for feature preparation")
        return pd.DataFrame()
    
    logger.info(f"Preparing features for {len(games)} games")
    
    features_list = []
    
    try:
        for game in games:
            try:
                # Extract basic game information
                game_id = game.get('id')
                home_team = game.get('home_team', {})
                away_team = game.get('away_team', {})
                
                if not game_id or not home_team or not away_team:
                    logger.warning(f"Missing critical data for game {game_id}. Skipping feature preparation.")
                    continue
                
                # Get team stats
                home_team_stats = home_team.get('stats', {})
                away_team_stats = away_team.get('stats', {})
                
                # Get game date
                game_date_str = game.get('date')
                game_date = None
                if game_date_str:
                    try:
                        game_date = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to parse game date {game_date_str}: {str(e)}")
                        game_date = datetime.now()  # Use current date as fallback
                else:
                    game_date = datetime.now()  # Use current date as fallback
                
                # Initialize features dictionary
                game_features = {
                    'game_id': game_id,
                    'home_team_id': home_team.get('id'),
                    'away_team_id': away_team.get('id'),
                    'home_team_name': home_team.get('name'),
                    'away_team_name': away_team.get('name'),
                    'game_date': game_date,
                }
                
                # Add team performance stats
                if home_team_stats and away_team_stats:
                    # Home team stats
                    game_features['home_win_pct'] = home_team_stats.get('win_pct', 0.5)
                    game_features['home_pts_pg'] = home_team_stats.get('points_pg', 110.0)
                    game_features['home_opp_pts_pg'] = home_team_stats.get('opp_points_pg', 110.0)
                    game_features['home_off_rtg'] = home_team_stats.get('offensive_rating', 110.0)
                    game_features['home_def_rtg'] = home_team_stats.get('defensive_rating', 110.0)
                    game_features['home_net_rtg'] = home_team_stats.get('net_rating', 0.0)
                    game_features['home_pace'] = home_team_stats.get('pace', 100.0)
                    
                    # Away team stats
                    game_features['away_win_pct'] = away_team_stats.get('win_pct', 0.5)
                    game_features['away_pts_pg'] = away_team_stats.get('points_pg', 110.0)
                    game_features['away_opp_pts_pg'] = away_team_stats.get('opp_points_pg', 110.0)
                    game_features['away_off_rtg'] = away_team_stats.get('offensive_rating', 110.0)
                    game_features['away_def_rtg'] = away_team_stats.get('defensive_rating', 110.0)
                    game_features['away_net_rtg'] = away_team_stats.get('net_rating', 0.0)
                    game_features['away_pace'] = away_team_stats.get('pace', 100.0)
                    
                    # Calculate differential features
                    game_features['win_pct_diff'] = game_features['home_win_pct'] - game_features['away_win_pct']
                    game_features['pts_diff'] = game_features['home_pts_pg'] - game_features['away_pts_pg']
                    game_features['opp_pts_diff'] = game_features['home_opp_pts_pg'] - game_features['away_opp_pts_pg']
                    game_features['off_rtg_diff'] = game_features['home_off_rtg'] - game_features['away_off_rtg']
                    game_features['def_rtg_diff'] = game_features['home_def_rtg'] - game_features['away_def_rtg']
                    game_features['net_rtg_diff'] = game_features['home_net_rtg'] - game_features['away_net_rtg']
                    game_features['pace_diff'] = game_features['home_pace'] - game_features['away_pace']
                    
                    # Calculate expected pace of game (average of both teams)
                    game_features['expected_pace'] = (game_features['home_pace'] + game_features['away_pace']) / 2.0
                    
                    # Add advanced interaction features
                    game_features['home_off_vs_away_def'] = game_features['home_off_rtg'] - game_features['away_def_rtg']
                    game_features['away_off_vs_home_def'] = game_features['away_off_rtg'] - game_features['home_def_rtg']
                    
                    # Calculate expected scoring based on team stats
                    home_exp_score = (game_features['home_off_rtg'] * game_features['expected_pace'] / 100.0)
                    away_exp_score = (game_features['away_off_rtg'] * game_features['expected_pace'] / 100.0)
                    game_features['expected_home_score'] = home_exp_score
                    game_features['expected_away_score'] = away_exp_score
                    game_features['expected_total'] = home_exp_score + away_exp_score
                    game_features['expected_spread'] = home_exp_score - away_exp_score
                else:
                    logger.warning(f"Missing team stats for game {game_id}, using default values")
                    # Use reasonable default values
                    game_features.update({
                        'home_win_pct': 0.5, 'away_win_pct': 0.5, 'win_pct_diff': 0.0,
                        'home_pts_pg': 110.0, 'away_pts_pg': 110.0, 'pts_diff': 0.0,
                        'home_opp_pts_pg': 110.0, 'away_opp_pts_pg': 110.0, 'opp_pts_diff': 0.0,
                        'home_off_rtg': 110.0, 'away_off_rtg': 110.0, 'off_rtg_diff': 0.0,
                        'home_def_rtg': 110.0, 'away_def_rtg': 110.0, 'def_rtg_diff': 0.0,
                        'home_net_rtg': 0.0, 'away_net_rtg': 0.0, 'net_rtg_diff': 0.0,
                        'home_pace': 100.0, 'away_pace': 100.0, 'pace_diff': 0.0,
                        'expected_pace': 100.0,
                        'home_off_vs_away_def': 0.0, 'away_off_vs_home_def': 0.0,
                        'expected_home_score': 110.0, 'expected_away_score': 110.0,
                        'expected_total': 220.0, 'expected_spread': 0.0
                    })
                
                # Rest and schedule features
                home_team_id = home_team.get('id')
                away_team_id = away_team.get('id')
                
                if game_date and home_team_id and away_team_id:
                    game_features['home_rest_days'] = calculate_rest_days(home_team_id, game_date)
                    game_features['away_rest_days'] = calculate_rest_days(away_team_id, game_date)
                    game_features['home_b2b'] = 1 if is_back_to_back(home_team_id, game_date) else 0
                    game_features['away_b2b'] = 1 if is_back_to_back(away_team_id, game_date) else 0
                    game_features['rest_advantage'] = game_features['home_rest_days'] - game_features['away_rest_days']
                else:
                    # Default values for rest and schedule
                    game_features.update({
                        'home_rest_days': 2, 'away_rest_days': 2,
                        'home_b2b': 0, 'away_b2b': 0, 'rest_advantage': 0
                    })
                
                # Home court advantage
                game_features['home_court'] = 1  # Always 1 since we're preparing features for home vs away
                
                # Add betting odds features if available
                if 'odds' in game:
                    odds = game['odds']
                    game_features['vegas_spread'] = odds.get('spread', 0.0)
                    game_features['vegas_total'] = odds.get('total', 220.0)
                    
                    # Calculate implied probabilities from moneyline if available
                    home_ml = odds.get('home_moneyline')
                    away_ml = odds.get('away_moneyline')
                    
                    if home_ml and away_ml:
                        # Convert American odds to implied probability
                        home_prob = (1 / (home_ml / 100 + 1)) if home_ml > 0 else (abs(home_ml) / (abs(home_ml) + 100))
                        away_prob = (1 / (away_ml / 100 + 1)) if away_ml > 0 else (abs(away_ml) / (abs(away_ml) + 100))
                        
                        # Normalize probabilities (remove the vig)
                        total_prob = home_prob + away_prob
                        game_features['implied_home_win_prob'] = home_prob / total_prob if total_prob > 0 else 0.5
                        game_features['implied_away_win_prob'] = away_prob / total_prob if total_prob > 0 else 0.5
                    else:
                        # If moneyline odds aren't available, derive from spread
                        from ..utils.math_utils import spread_to_win_probability
                        game_features['implied_home_win_prob'] = spread_to_win_probability(game_features.get('vegas_spread', 0.0))
                        game_features['implied_away_win_prob'] = 1.0 - game_features['implied_home_win_prob']
                else:
                    # Default values for odds
                    game_features.update({
                        'vegas_spread': 0.0, 'vegas_total': 220.0,
                        'implied_home_win_prob': 0.5, 'implied_away_win_prob': 0.5
                    })
                
                # Add this game's features to the list
                features_list.append(game_features)
                logger.debug(f"Prepared features for game {game_id}: {home_team.get('name')} vs {away_team.get('name')}")
            
            except Exception as e:
                logger.error(f"Error preparing features for game {game.get('id')}: {str(e)}")
                continue
        
        # Convert to DataFrame
        if features_list:
            features_df = pd.DataFrame(features_list)
            logger.info(f"Successfully prepared features for {len(features_list)} games")
            return features_df
        else:
            logger.warning("No valid features could be prepared")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error in prepare_game_features: {str(e)}")
        return pd.DataFrame()


def extract_game_features(home_stats: Dict[str, Any], away_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features for game prediction with robust error handling
    
    Args:
        home_stats: Home team statistics
        away_stats: Away team statistics
        
    Returns:
        Dictionary of features for prediction
    """
    if not home_stats or not away_stats:
        logger.error("Missing team statistics for feature extraction")
        return {}
    
    try:
        features = {}
        
        # Team identification
        features['home_team_id'] = home_stats.get('id', 0)
        features['away_team_id'] = away_stats.get('id', 0)
        
        # Basic team performance metrics
        features['home_win_pct'] = home_stats.get('win_pct', 0.5)
        features['away_win_pct'] = away_stats.get('win_pct', 0.5)
        features['win_pct_diff'] = features['home_win_pct'] - features['away_win_pct']
        
        # Scoring metrics
        features['home_pts_pg'] = home_stats.get('points_pg', 110.0)
        features['away_pts_pg'] = away_stats.get('points_pg', 110.0)
        features['pts_diff'] = features['home_pts_pg'] - features['away_pts_pg']
        
        features['home_opp_pts_pg'] = home_stats.get('opp_points_pg', 110.0)
        features['away_opp_pts_pg'] = away_stats.get('opp_points_pg', 110.0)
        features['opp_pts_diff'] = features['home_opp_pts_pg'] - features['away_opp_pts_pg']
        
        # Advanced metrics
        features['home_off_rtg'] = home_stats.get('offensive_rating', 110.0)
        features['away_off_rtg'] = away_stats.get('offensive_rating', 110.0)
        features['off_rtg_diff'] = features['home_off_rtg'] - features['away_off_rtg']
        
        features['home_def_rtg'] = home_stats.get('defensive_rating', 110.0)
        features['away_def_rtg'] = away_stats.get('defensive_rating', 110.0)
        features['def_rtg_diff'] = features['home_def_rtg'] - features['away_def_rtg']
        
        features['home_net_rtg'] = home_stats.get('net_rating', 0.0)
        features['away_net_rtg'] = away_stats.get('net_rating', 0.0)
        features['net_rtg_diff'] = features['home_net_rtg'] - features['away_net_rtg']
        
        # Pace and tempo
        features['home_pace'] = home_stats.get('pace', 100.0)
        features['away_pace'] = away_stats.get('pace', 100.0)
        features['pace_diff'] = features['home_pace'] - features['away_pace']
        features['expected_pace'] = (features['home_pace'] + features['away_pace']) / 2.0
        
        # Matchup-specific features
        features['home_off_vs_away_def'] = features['home_off_rtg'] - features['away_def_rtg']
        features['away_off_vs_home_def'] = features['away_off_rtg'] - features['home_def_rtg']
        
        # Expected scoring
        home_exp_score = (features['home_off_rtg'] * features['expected_pace'] / 100.0)
        away_exp_score = (features['away_off_rtg'] * features['expected_pace'] / 100.0)
        features['expected_home_score'] = home_exp_score
        features['expected_away_score'] = away_exp_score
        features['expected_total'] = home_exp_score + away_exp_score
        features['expected_spread'] = home_exp_score - away_exp_score
        
        # Rest and schedule factors (placeholders)
        features['home_rest_days'] = 2  # Default value
        features['away_rest_days'] = 2  # Default value
        features['home_b2b'] = 0  # Default value
        features['away_b2b'] = 0  # Default value
        features['rest_advantage'] = 0  # Default value
        
        # Home court advantage
        features['home_court'] = 1  # Always 1 for home vs away
        
        return features
    
    except Exception as e:
        logger.error(f"Error extracting game features: {str(e)}")
        return {}
