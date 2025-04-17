#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct Data Access Module

This module provides direct access to real NBA data from the BallDontLie API
without requiring a database connection. It implements caching to reduce API calls
and provides a consistent interface for accessing NBA data.
"""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import API client
from src.api.balldontlie_client import BallDontLieClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path('data/cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache TTLs in seconds
TEAM_CACHE_TTL = 86400 * 7  # 1 week
PLAYER_CACHE_TTL = 86400 * 3  # 3 days
GAME_CACHE_TTL = 3600  # 1 hour
STATS_CACHE_TTL = 3600 * 6  # 6 hours

# Initialize API client
api_client = None


def _get_api_client() -> BallDontLieClient:
    """Get or initialize the API client"""
    global api_client
    if api_client is None:
        api_client = BallDontLieClient()
    return api_client


def _get_cache_path(cache_type: str, identifier: Optional[str] = None) -> Path:
    """Get the path for a cache file"""
    if identifier:
        return CACHE_DIR / f"{cache_type}_{identifier}.json"
    return CACHE_DIR / f"{cache_type}.json"


def _read_cache(cache_path: Path, ttl: int) -> Optional[Dict]:
    """Read data from cache if it exists and is not expired"""
    try:
        if not cache_path.exists():
            return None
        
        # Check cache age
        file_time = cache_path.stat().st_mtime
        if time.time() - file_time > ttl:
            logger.debug(f"Cache expired for {cache_path}")
            return None
        
        # Read cache
        with cache_path.open('r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Error reading cache {cache_path}: {str(e)}")
        return None


def _write_cache(cache_path: Path, data: Any) -> bool:
    """Write data to cache"""
    try:
        with cache_path.open('w') as f:
            json.dump(data, f)
        return True
    except Exception as e:
        logger.warning(f"Error writing cache {cache_path}: {str(e)}")
        return False


def get_teams() -> List[Dict]:
    """Get all NBA teams"""
    logger.info("Getting NBA teams")
    cache_path = _get_cache_path('teams')
    
    # Try to read from cache
    cached_data = _read_cache(cache_path, TEAM_CACHE_TTL)
    if cached_data:
        logger.info(f"Found {len(cached_data)} teams in cache")
        return cached_data
    
    # Fetch from API
    client = _get_api_client()
    try:
        teams = client.get_teams()
        if teams:
            _write_cache(cache_path, teams)
            logger.info(f"Fetched {len(teams)} teams from API")
            return teams
    except Exception as e:
        logger.error(f"Error fetching teams from API: {str(e)}")
    
    # Return empty list if all else fails
    return []


def get_players(limit: int = 100) -> List[Dict]:
    """Get NBA players"""
    logger.info(f"Getting NBA players (limit: {limit})")
    cache_path = _get_cache_path('players')
    
    # Try to read from cache
    cached_data = _read_cache(cache_path, PLAYER_CACHE_TTL)
    if cached_data:
        logger.info(f"Found {len(cached_data)} players in cache")
        return cached_data[:limit] if limit else cached_data
    
    # Fetch from API
    client = _get_api_client()
    try:
        players = client.get_players(limit=limit)
        if players:
            _write_cache(cache_path, players)
            logger.info(f"Fetched {len(players)} players from API")
            return players
    except Exception as e:
        logger.error(f"Error fetching players from API: {str(e)}")
    
    # Return empty list if all else fails
    return []


def get_upcoming_games(days: int = 7) -> List[Dict]:
    """Get upcoming NBA games"""
    logger.info(f"Getting upcoming NBA games for next {days} days")
    cache_path = _get_cache_path('upcoming_games')
    
    # Try to read from cache
    cached_data = _read_cache(cache_path, GAME_CACHE_TTL)
    if cached_data:
        logger.info(f"Found {len(cached_data)} upcoming games in cache")
        return cached_data
    
    # Fetch from API
    client = _get_api_client()
    try:
        # Calculate date range
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
        
        games = client.get_games(start_date=start_date, end_date=end_date)
        if games:
            _write_cache(cache_path, games)
            logger.info(f"Fetched {len(games)} upcoming games from API")
            return games
    except Exception as e:
        logger.error(f"Error fetching upcoming games from API: {str(e)}")
    
    # Return empty list if all else fails
    return []


def get_recent_games(days: int = 7) -> List[Dict]:
    """Get recent NBA games"""
    logger.info(f"Getting recent NBA games from past {days} days")
    cache_path = _get_cache_path('recent_games')
    
    # Try to read from cache
    cached_data = _read_cache(cache_path, GAME_CACHE_TTL)
    if cached_data:
        logger.info(f"Found {len(cached_data)} recent games in cache")
        return cached_data
    
    # Fetch from API
    client = _get_api_client()
    try:
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        games = client.get_games(start_date=start_date, end_date=end_date)
        if games:
            _write_cache(cache_path, games)
            logger.info(f"Fetched {len(games)} recent games from API")
            return games
    except Exception as e:
        logger.error(f"Error fetching recent games from API: {str(e)}")
    
    # Return empty list if all else fails
    return []


def get_game_stats(game_id: str) -> Dict:
    """Get statistics for a specific game"""
    logger.info(f"Getting stats for game {game_id}")
    cache_path = _get_cache_path('game_stats', game_id)
    
    # Try to read from cache
    cached_data = _read_cache(cache_path, STATS_CACHE_TTL)
    if cached_data:
        logger.info(f"Found stats for game {game_id} in cache")
        return cached_data
    
    # Fetch from API
    client = _get_api_client()
    try:
        stats = client.get_game_stats(game_id)
        if stats:
            _write_cache(cache_path, stats)
            logger.info(f"Fetched stats for game {game_id} from API")
            return stats
    except Exception as e:
        logger.error(f"Error fetching stats for game {game_id} from API: {str(e)}")
    
    # Return empty dict if all else fails
    return {}


def get_player_stats(player_id: str) -> Dict:
    """Get statistics for a specific player"""
    logger.info(f"Getting stats for player {player_id}")
    cache_path = _get_cache_path('player_stats', player_id)
    
    # Try to read from cache
    cached_data = _read_cache(cache_path, STATS_CACHE_TTL)
    if cached_data:
        logger.info(f"Found stats for player {player_id} in cache")
        return cached_data
    
    # Fetch from API
    client = _get_api_client()
    try:
        stats = client.get_player_stats(player_id)
        if stats:
            _write_cache(cache_path, stats)
            logger.info(f"Fetched stats for player {player_id} from API")
            return stats
    except Exception as e:
        logger.error(f"Error fetching stats for player {player_id} from API: {str(e)}")
    
    # Return empty dict if all else fails
    return {}


def save_prediction(game_id: str, prediction: Dict) -> bool:
    """Save a prediction for a game"""
    logger.info(f"Saving prediction for game {game_id}")
    predictions_dir = Path('data/predictions')
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    prediction_path = predictions_dir / f"{game_id}.json"
    
    try:
        # Add timestamp to prediction
        prediction['timestamp'] = datetime.now().isoformat()
        
        with prediction_path.open('w') as f:
            json.dump(prediction, f)
        logger.info(f"Saved prediction for game {game_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving prediction for game {game_id}: {str(e)}")
        return False


def get_prediction(game_id: str) -> Optional[Dict]:
    """Get a saved prediction for a game"""
    logger.info(f"Getting prediction for game {game_id}")
    prediction_path = Path(f'data/predictions/{game_id}.json')
    
    try:
        if prediction_path.exists():
            with prediction_path.open('r') as f:
                prediction = json.load(f)
            logger.info(f"Found prediction for game {game_id}")
            return prediction
    except Exception as e:
        logger.error(f"Error reading prediction for game {game_id}: {str(e)}")
    
    logger.info(f"No prediction found for game {game_id}")
    return None


def get_all_predictions() -> List[Dict]:
    """Get all saved predictions"""
    logger.info("Getting all predictions")
    predictions_dir = Path('data/predictions')
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = []
    try:
        for prediction_file in predictions_dir.glob('*.json'):
            try:
                with prediction_file.open('r') as f:
                    prediction = json.load(f)
                    game_id = prediction_file.stem
                    prediction['game_id'] = game_id
                    predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Error reading prediction file {prediction_file}: {str(e)}")
    except Exception as e:
        logger.error(f"Error reading predictions directory: {str(e)}")
    
    logger.info(f"Found {len(predictions)} predictions")
    return predictions


# Main function for testing
if __name__ == "__main__":
    print("Testing direct data access module...")
    
    # Test getting teams
    teams = get_teams()
    print(f"Found {len(teams)} teams")
    
    # Test getting upcoming games
    upcoming_games = get_upcoming_games(days=7)
    print(f"Found {len(upcoming_games)} upcoming games")
    
    # Test getting recent games
    recent_games = get_recent_games(days=7)
    print(f"Found {len(recent_games)} recent games")
    
    # Test making a prediction
    if upcoming_games:
        game = upcoming_games[0]
        game_id = str(game['id'])
        
        # Simple prediction
        prediction = {
            'home_team': game['home_team']['name'],
            'away_team': game['visitor_team']['name'],
            'predicted_winner': game['home_team']['name'],
            'confidence': 0.75,
            'model': 'DirectAccess'
        }
        
        # Save the prediction
        save_prediction(game_id, prediction)
        
        # Read it back
        saved_prediction = get_prediction(game_id)
        print(f"Saved prediction: {saved_prediction}")
    
    print("Direct data access module test complete")
