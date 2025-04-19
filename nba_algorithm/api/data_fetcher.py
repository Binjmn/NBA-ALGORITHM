#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Fetcher Module

This module adapts the new enhanced caching system to work with existing data fetching functions.
It implements the hybrid approach to use cached data for model training but fresh data for game-day predictions.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, date, timedelta
import pandas as pd

# Import the direct API clients
from ...src.api.balldontlie_client import BallDontLieClient
from ...src.api.theodds_client import TheOddsApiClient

# Import our enhanced data access module
from .data_access import (
    get_teams, get_players, get_player_status, get_upcoming_games,
    get_game_odds, get_team_stats, get_player_stats, get_historical_games
)

# Import caching utilities
from ..utils.cache_helpers import is_game_day, cache_aware_fetch, fetch_with_fallback

# Configure logging
logger = logging.getLogger(__name__)


def fetch_nba_games(date_str: str, force_refresh: bool = False) -> List[Dict]:
    """
    Fetch NBA games for a specific date with enhanced caching
    
    Args:
        date_str: Date string in format YYYY-MM-DD
        force_refresh: Whether to bypass cache completely
        
    Returns:
        List of game dictionaries
    """
    logger.info(f"Fetching NBA games for {date_str}")
    
    # Check if it's game day (today)
    game_day = is_game_day(date_str)
    
    # On game day, we need the freshest data
    # For historical days, cached data is fine
    games = cache_aware_fetch(
        'upcoming_games',
        lambda: BallDontLieClient().get_games_by_date(date_str),
        identifier=f"date_{date_str}",
        force_refresh=force_refresh,
        is_prediction_day=game_day
    )
    
    if games:
        logger.info(f"Found {len(games)} NBA games for {date_str}")
    else:
        logger.warning(f"No NBA games found for {date_str}")
        
    return games or []


def fetch_team_stats(team_id: str, force_refresh: bool = False) -> Dict:
    """
    Fetch team statistics with enhanced caching
    
    Args:
        team_id: Team ID
        force_refresh: Whether to bypass cache completely
        
    Returns:
        Team statistics dictionary
    """
    logger.info(f"Fetching stats for team {team_id}")
    
    # For team stats, we use the hybrid approach:
    # - On prediction day, get fresh data for the most accurate predictions
    # - For training or analysis, cached data is sufficient if available
    stats = get_team_stats(team_id, force_refresh=force_refresh)
    
    if stats:
        logger.info(f"Got stats for team {team_id}")
    else:
        logger.warning(f"Failed to fetch stats for team {team_id}")
        
    return stats or {}


def fetch_betting_odds(games: List[Dict], force_refresh: bool = False) -> Dict[str, Dict]:
    """
    Fetch betting odds for games with enhanced caching
    
    Args:
        games: List of game dictionaries
        force_refresh: Whether to bypass cache completely
        
    Returns:
        Dictionary mapping game IDs to odds dictionaries
    """
    logger.info(f"Fetching betting odds for {len(games)} games")
    
    # Betting odds are highly volatile, especially on game day
    # Always use fresh data for today's games
    odds = {}
    
    for game in games:
        game_id = game['id']
        game_date_str = game.get('date', '')
        game_day = is_game_day(game_date_str)
        
        # Always force refresh odds on game day
        should_force_refresh = force_refresh or game_day
        
        game_odds = get_game_odds(game_id, force_refresh=should_force_refresh)
        if game_odds:
            odds[game_id] = game_odds
    
    logger.info(f"Fetched odds for {len(odds)} games")
    return odds


def fetch_historical_games(days: int = 90, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch historical games for model training
    
    Args:
        days: Number of days of historical data to fetch
        force_refresh: Whether to bypass cache completely
        
    Returns:
        DataFrame of historical games
    """
    logger.info(f"Fetching historical games for the past {days} days")
    
    # Historical data is stable and perfect for caching
    # Only force refresh periodically or when explicitly requested
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    # Use the cache for historical data unless forced to refresh
    cache_id = f"days_{days}_to_{end_date.isoformat()}"
    
    games = cache_aware_fetch(
        'historical_games',
        lambda: BallDontLieClient().get_games_by_date_range(
            start_date.isoformat(), end_date.isoformat()
        ),
        identifier=cache_id,
        force_refresh=force_refresh
    )
    
    if games:
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(games)
        logger.info(f"Fetched {len(df)} historical games")
        return df
    else:
        logger.warning("No historical games found")
        return pd.DataFrame()


def fetch_player_data(games: List[Dict], force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch player data for games with enhanced caching
    
    Args:
        games: List of game dictionaries
        force_refresh: Whether to bypass cache completely
        
    Returns:
        DataFrame of player data
    """
    logger.info(f"Fetching player data for {len(games)} games")
    
    # Player lineup and status data is highly volatile
    # Always use fresh data on game day
    players_data = []
    
    for game in games:
        game_id = game['id']
        game_date_str = game.get('date', '')
        game_day = is_game_day(game_date_str)
        
        # Get players for home team
        home_team_id = game['home_team']['id']
        home_players = cache_aware_fetch(
            'player_lineup',
            lambda: BallDontLieClient().get_team_players(home_team_id),
            identifier=f"team_{home_team_id}_game_{game_id}",
            force_refresh=force_refresh or game_day,
            is_prediction_day=game_day
        )
        
        if home_players:
            for player in home_players:
                players_data.append({
                    'player_id': player['id'],
                    'player_name': player['full_name'],
                    'team_id': home_team_id,
                    'team_name': game['home_team']['name'],
                    'position': player.get('position', ''),
                    'game_id': game_id,
                    'is_home': True
                })
        
        # Get players for away team
        away_team_id = game['away_team']['id']
        away_players = cache_aware_fetch(
            'player_lineup',
            lambda: BallDontLieClient().get_team_players(away_team_id),
            identifier=f"team_{away_team_id}_game_{game_id}",
            force_refresh=force_refresh or game_day,
            is_prediction_day=game_day
        )
        
        if away_players:
            for player in away_players:
                players_data.append({
                    'player_id': player['id'],
                    'player_name': player['full_name'],
                    'team_id': away_team_id,
                    'team_name': game['away_team']['name'],
                    'position': player.get('position', ''),
                    'game_id': game_id,
                    'is_home': False
                })
    
    if players_data:
        df = pd.DataFrame(players_data)
        logger.info(f"Fetched data for {len(df)} players")
        return df
    else:
        logger.warning("No player data found")
        return pd.DataFrame()


def get_player_features(games: List[Dict], team_stats: Dict, historical_games: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """
    Get player features for player props predictions
    
    Args:
        games: List of game dictionaries
        team_stats: Dictionary of team statistics
        historical_games: DataFrame of historical games
        force_refresh: Whether to bypass cache completely
        
    Returns:
        DataFrame of player features
    """
    logger.info("Building player features for prediction")
    
    # Get player data
    player_df = fetch_player_data(games, force_refresh=force_refresh)
    
    if player_df.empty:
        logger.warning("No player data available for feature creation")
        return pd.DataFrame()
    
    # Fetch individual player statistics
    # Always use fresh data on game day
    enhanced_players = []
    
    for _, player in player_df.iterrows():
        player_id = player['player_id']
        game_id = player['game_id']
        game_date = next((g['date'] for g in games if g['id'] == game_id), None)
        game_day = is_game_day(game_date) if game_date else False
        
        # Get player stats
        player_stats = get_player_stats(player_id, force_refresh=force_refresh or game_day)
        
        if player_stats:
            # Create a new player record with enhanced features
            player_record = player.to_dict()
            
            # Add the team defensive metrics
            opponent_team_id = next(
                (g['home_team']['id'] if player['team_id'] == g['away_team']['id'] else g['away_team']['id']
                 for g in games if g['id'] == game_id),
                None
            )
            
            if opponent_team_id and opponent_team_id in team_stats:
                opponent_stats = team_stats[opponent_team_id]
                player_record['opponent_defensive_rating'] = opponent_stats.get('defensive_rating', 100)
            else:
                player_record['opponent_defensive_rating'] = 100  # League average as fallback
            
            # Add player stats
            player_record['ppg'] = player_stats.get('points_per_game', 0)
            player_record['rpg'] = player_stats.get('rebounds_per_game', 0)
            player_record['apg'] = player_stats.get('assists_per_game', 0)
            player_record['minutes_per_game'] = player_stats.get('minutes_per_game', 0)
            player_record['games_played'] = player_stats.get('games_played', 0)
            
            # Add to our enhanced players list
            enhanced_players.append(player_record)
    
    if enhanced_players:
        df = pd.DataFrame(enhanced_players)
        logger.info(f"Created features for {len(df)} players")
        return df
    else:
        logger.warning("No player features created")
        return pd.DataFrame()
