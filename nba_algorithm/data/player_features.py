#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Player Features Module

This module provides functions for extracting player features for prop predictions.

Author: Cascade
Date: April 2025
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import traceback

from .player_data import fetch_player_data, fetch_players_for_game

logger = logging.getLogger(__name__)


def get_player_features(games, team_stats, historical_games=None):
    """
    Extract player features for prop predictions
    
    Args:
        games: List of games to extract players from
        team_stats: Dictionary of team statistics
        historical_games: Optional list of historical games
        
    Returns:
        DataFrame with player features for prediction
    """
    try:
        logger.info(f"Extracting player features for {len(games)} games")
        
        # Fetch player data for all games
        player_data = fetch_player_data(games)
        
        if not player_data:
            logger.error("Failed to fetch player data")
            return pd.DataFrame()
            
        logger.info(f"Fetched data for {len(player_data)} players")
        
        # Convert to DataFrame for processing
        players_df = pd.DataFrame(player_data)
        
        # Add derived features
        if not players_df.empty:
            # Add team context from team_stats
            if team_stats:
                for team_id, stats in team_stats.items():
                    # Add team stats as features for all players on this team
                    team_players = players_df[players_df['team_id'] == team_id]
                    if not team_players.empty:
                        for stat_name, stat_value in stats.items():
                            # Don't overwrite player stats with team stats
                            if f'team_{stat_name}' not in players_df.columns:
                                players_df.loc[players_df['team_id'] == team_id, f'team_{stat_name}'] = stat_value
            
            # Add contextual features
            for i, game in enumerate(games):
                game_id = game.get('id')
                home_team_id = game.get('home_team', {}).get('id')
                visitor_team_id = game.get('visitor_team', {}).get('id')
                
                if game_id and home_team_id and visitor_team_id:
                    # Add home/away indicator
                    players_df.loc[players_df['team_id'] == home_team_id, 'is_home'] = 1
                    players_df.loc[players_df['team_id'] == visitor_team_id, 'is_home'] = 0
                    
                    # Add game_id to players in this game
                    players_df.loc[(players_df['team_id'] == home_team_id) | 
                                   (players_df['team_id'] == visitor_team_id), 'game_id'] = game_id
                    
                    # Add opponent team info
                    players_df.loc[players_df['team_id'] == home_team_id, 'opponent_team_id'] = visitor_team_id
                    players_df.loc[players_df['team_id'] == visitor_team_id, 'opponent_team_id'] = home_team_id
                    
                    # Add opponent team name
                    home_team_name = game.get('home_team', {}).get('full_name', '')
                    away_team_name = game.get('visitor_team', {}).get('full_name', '')
                    players_df.loc[players_df['team_id'] == home_team_id, 'opponent_team'] = away_team_name
                    players_df.loc[players_df['team_id'] == visitor_team_id, 'opponent_team'] = home_team_name
                    
                    # Add home/away team names to all players in this game
                    players_df.loc[(players_df['team_id'] == home_team_id) | 
                                   (players_df['team_id'] == visitor_team_id), 'home_team'] = home_team_name
                    players_df.loc[(players_df['team_id'] == home_team_id) | 
                                   (players_df['team_id'] == visitor_team_id), 'visitor_team'] = away_team_name
        
            # Process historical game data if available
            if historical_games:
                logger.info(f"Processing {len(historical_games)} historical games for player context")
                # Here you would add logic to extract relevant historical performance metrics
                # for each player based on their recent games
            
            # Handle missing values
            numeric_columns = players_df.select_dtypes(include=['number']).columns
            players_df[numeric_columns] = players_df[numeric_columns].fillna(0)
            
            # Ensure all required columns exist
            required_columns = ['player_id', 'player_name', 'team_id', 'team_name', 'position', 'game_id']
            for col in required_columns:
                if col not in players_df.columns:
                    logger.warning(f"Missing required column: {col}")
                    players_df[col] = ''
            
            logger.info(f"Prepared features for {len(players_df)} players")
            return players_df
        else:
            logger.warning("No player data available for feature extraction")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error extracting player features: {str(e)}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()
