#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Data Access Module

This module provides access to NBA data with advanced caching capabilities:
- Tiered caching based on data volatility
- Smart cache invalidation
- Selective caching of stable vs. volatile metrics
- Force refresh option
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Import the cache manager
from ..utils.cache_manager import (
    read_cache, write_cache, clear_cache, CacheTier, 
    get_cache_stats, generate_cache_key
)

# Import API clients
from ...src.api.balldontlie_client import BallDontLieClient
from ...src.api.theodds_client import TheOddsApiClient

# Configure logging
logger = logging.getLogger(__name__)

# Initialize API clients
ball_client = None
odds_client = None

# Use dependency injection to allow for easier testing
def get_ball_client() -> BallDontLieClient:
    """Get or initialize the BallDontLie API client"""
    global ball_client
    if ball_client is None:
        ball_client = BallDontLieClient()
    return ball_client

def get_odds_client() -> TheOddsApiClient:
    """Get or initialize the Odds API client"""
    global odds_client
    if odds_client is None:
        odds_client = TheOddsApiClient()
    return odds_client

# Data access functions with enhanced caching
def get_teams(force_refresh: bool = False) -> List[Dict]:
    """Get all NBA teams with tiered caching
    
    Args:
        force_refresh: Whether to bypass cache and force a fresh request
        
    Returns:
        List of team dictionaries
    """
    logger.info("Getting NBA teams")
    
    # Try to read from cache (teams are very stable data, long TTL)
    cached_data = read_cache('teams', force_refresh=force_refresh)
    if cached_data:
        logger.info(f"Found {len(cached_data)} teams in cache")
        return cached_data
    
    # Fetch from API
    client = get_ball_client()
    teams = client.get_all_teams()
    
    if teams:
        # Store additional metadata with the cache
        metadata = {
            'source': 'balldontlie',
            'team_count': len(teams),
            'retrieval_time': datetime.now().isoformat()
        }
        
        # Cache the data (teams are stable, so they go in the stable tier)
        write_cache('teams', teams, metadata=metadata)
        logger.info(f"Cached {len(teams)} teams")
    
    return teams

def get_players(limit: int = 100, force_refresh: bool = False) -> List[Dict]:
    """Get NBA players with tiered caching
    
    Args:
        limit: Maximum number of players to retrieve
        force_refresh: Whether to bypass cache and force a fresh request
        
    Returns:
        List of player dictionaries
    """
    logger.info(f"Getting up to {limit} NBA players")
    
    # Generate a cache identifier based on the limit
    cache_id = f"limit_{limit}"
    
    # Try to read from cache
    cached_data = read_cache('players', identifier=cache_id, force_refresh=force_refresh)
    if cached_data:
        logger.info(f"Found {len(cached_data)} players in cache")
        return cached_data
    
    # Fetch from API
    client = get_ball_client()
    players = client.get_all_players(limit=limit)
    
    if players:
        # Store additional metadata with the cache
        metadata = {
            'source': 'balldontlie',
            'player_count': len(players),
            'limit': limit,
            'retrieval_time': datetime.now().isoformat()
        }
        
        # Cache the data
        write_cache('players', players, identifier=cache_id, metadata=metadata)
        logger.info(f"Cached {len(players)} players")
    
    return players

def get_player_status(player_id: str, force_refresh: bool = False) -> Dict:
    """Get current status for a player (high volatility data)
    
    Args:
        player_id: The player ID
        force_refresh: Whether to bypass cache and force a fresh request
        
    Returns:
        Player status dictionary
    """
    logger.info(f"Getting status for player {player_id}")
    
    # Player status is volatile - use short TTL and always refresh on game day
    cached_data = read_cache('player_status', identifier=player_id, force_refresh=force_refresh)
    if cached_data:
        logger.info(f"Found status for player {player_id} in cache")
        return cached_data
    
    # Fetch from API
    client = get_ball_client()
    status = client.get_player_status(player_id)
    
    if status:
        # Cache the data with a short TTL (volatile tier)
        write_cache('player_status', status, identifier=player_id)
        logger.info(f"Cached status for player {player_id}")
    
    return status or {}

def get_upcoming_games(days: int = 7, force_refresh: bool = False) -> List[Dict]:
    """Get upcoming NBA games
    
    Args:
        days: Number of days to look ahead
        force_refresh: Whether to bypass cache and force a fresh request
        
    Returns:
        List of game dictionaries
    """
    logger.info(f"Getting upcoming games for next {days} days")
    
    # Generate a cache identifier based on the days parameter
    cache_id = f"days_{days}"
    
    # Upcoming games are medium volatility - use regular tier
    cached_data = read_cache('upcoming_games', identifier=cache_id, force_refresh=force_refresh)
    if cached_data:
        logger.info(f"Found {len(cached_data)} upcoming games in cache")
        return cached_data
    
    # Fetch from API
    client = get_ball_client()
    start_date = datetime.now().date()
    end_date = start_date + timedelta(days=days)
    games = client.get_games_by_date_range(start_date.isoformat(), end_date.isoformat())
    
    if games:
        # Store additional metadata with the cache
        metadata = {
            'source': 'balldontlie',
            'game_count': len(games),
            'date_range': f"{start_date} to {end_date}",
            'retrieval_time': datetime.now().isoformat()
        }
        
        # Cache the data
        write_cache('upcoming_games', games, identifier=cache_id, metadata=metadata)
        logger.info(f"Cached {len(games)} upcoming games")
    
    return games

def get_game_odds(game_id: str, force_refresh: bool = False) -> Dict:
    """Get betting odds for a specific game (high volatility data)
    
    Args:
        game_id: The game ID
        force_refresh: Whether to bypass cache and force a fresh request
        
    Returns:
        Odds dictionary
    """
    logger.info(f"Getting odds for game {game_id}")
    
    # Game odds are highly volatile, especially as game time approaches
    # Always force refresh if game is today
    game_today = False  # TODO: Check if game is today
    should_refresh = force_refresh or game_today
    
    cached_data = read_cache('odds', identifier=game_id, force_refresh=should_refresh)
    if cached_data:
        logger.info(f"Found odds for game {game_id} in cache")
        return cached_data
    
    # Fetch from API
    client = get_odds_client()
    odds = client.get_game_odds('basketball_nba', game_id)
    
    if odds:
        # Cache the data with a short TTL
        write_cache('odds', odds, identifier=game_id)
        logger.info(f"Cached odds for game {game_id}")
    
    return odds or {}

def get_team_stats(team_id: str, force_refresh: bool = False) -> Dict:
    """Get team statistics
    
    Args:
        team_id: The team ID
        force_refresh: Whether to bypass cache and force a fresh request
        
    Returns:
        Team stats dictionary
    """
    logger.info(f"Getting stats for team {team_id}")
    
    # Team stats change after each game, medium volatility
    cached_data = read_cache('team_stats', identifier=team_id, force_refresh=force_refresh)
    if cached_data:
        logger.info(f"Found stats for team {team_id} in cache")
        return cached_data
    
    # Fetch from API
    client = get_ball_client()
    stats = client.get_team_stats(team_id)
    
    if stats:
        # Cache the data
        write_cache('team_stats', stats, identifier=team_id)
        logger.info(f"Cached stats for team {team_id}")
    
    return stats or {}

def get_player_stats(player_id: str, force_refresh: bool = False) -> Dict:
    """Get player statistics
    
    Args:
        player_id: The player ID
        force_refresh: Whether to bypass cache and force a fresh request
        
    Returns:
        Player stats dictionary
    """
    logger.info(f"Getting stats for player {player_id}")
    
    # Player stats change after each game, medium volatility
    cached_data = read_cache('player_stats', identifier=player_id, force_refresh=force_refresh)
    if cached_data:
        logger.info(f"Found stats for player {player_id} in cache")
        return cached_data
    
    # Fetch from API
    client = get_ball_client()
    stats = client.get_player_stats(player_id)
    
    if stats:
        # Cache the data
        write_cache('player_stats', stats, identifier=player_id)
        logger.info(f"Cached stats for player {player_id}")
    
    return stats or {}

def get_historical_games(team_id: str, days: int = 30, force_refresh: bool = False) -> List[Dict]:
    """Get historical games for a team
    
    Args:
        team_id: The team ID
        days: Number of days to look back
        force_refresh: Whether to bypass cache and force a fresh request
        
    Returns:
        List of game dictionaries
    """
    logger.info(f"Getting historical games for team {team_id} over past {days} days")
    
    # Generate a cache identifier
    cache_id = f"{team_id}_days_{days}"
    
    # Historical data is stable
    cached_data = read_cache('historical_games', identifier=cache_id, force_refresh=force_refresh)
    if cached_data:
        logger.info(f"Found {len(cached_data)} historical games for team {team_id} in cache")
        return cached_data
    
    # Fetch from API
    client = get_ball_client()
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    games = client.get_team_games(team_id, start_date.isoformat(), end_date.isoformat())
    
    if games:
        # Cache the data
        write_cache('historical_games', games, identifier=cache_id)
        logger.info(f"Cached {len(games)} historical games for team {team_id}")
    
    return games

def clear_volatile_cache():
    """Clear all volatile cache data"""
    count = clear_cache(tier=CacheTier.VOLATILE)
    logger.info(f"Cleared {count} volatile cache entries")
    return count

def clear_all_cache():
    """Clear all cache data"""
    count = clear_cache()
    logger.info(f"Cleared {count} cache entries")
    return count

def get_cache_information() -> Dict[str, Any]:
    """Get information about the current cache state"""
    stats = get_cache_stats()
    logger.info(f"Cache contains {stats['total_count']} entries totaling {stats['total_size_kb']} KB")
    return stats
