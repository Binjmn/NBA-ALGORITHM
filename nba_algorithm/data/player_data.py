#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Player Data Module

This module handles fetching and processing NBA player data from external APIs.

Author: Cascade
Date: April 2025
"""

import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests

from ..utils.config import DATA_DIR, DEFAULT_API_TIMEOUT, MAX_RETRIES, RETRY_DELAY

# Configure logger
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = "https://api.balldontlie.io/v1"


def fetch_players_for_game(game_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch real players for a specific game from BallDontLie API with robust error handling
    
    Args:
        game_data: Game data dictionary with home and away team information
        
    Returns:
        Dictionary with 'home' and 'away' keys, each containing lists of player dictionaries
    """
    if not game_data or 'home_team' not in game_data or 'away_team' not in game_data:
        logger.error("Invalid game data provided to fetch_players_for_game")
        return {'home': [], 'away': []}
    
    home_team_id = game_data['home_team'].get('id')
    away_team_id = game_data['away_team'].get('id')
    
    if not home_team_id or not away_team_id:
        logger.error("Missing team IDs in game data")
        return {'home': [], 'away': []}
    
    logger.info(f"Fetching players for game: {game_data['away_team'].get('name', 'Away')} @ {game_data['home_team'].get('name', 'Home')}")
    
    # API endpoints
    players_endpoint = f"{API_BASE_URL}/players"
    
    # Get API key from environment or configuration
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    headers = {}
    
    if api_key:
        headers["Authorization"] = api_key
        logger.info("Using BallDontLie API key from environment")
    else:
        logger.warning("No BallDontLie API key found. Requests may be rate limited.")
    
    # Initialize result
    result = {
        'home': [],
        'away': []
    }
    
    try:
        # Fetch home team players
        home_players = []
        away_players = []
        
        # Make separate API requests for each team
        for team_type, team_id in [('home', home_team_id), ('away', away_team_id)]:
            # API query parameters
            params = {
                "team_ids[]": team_id,
                "per_page": 100  # Get as many players as possible
            }
            
            # Make API request with retries
            response = None
            retry_count = 0
            
            while retry_count < MAX_RETRIES:
                try:
                    response = requests.get(
                        players_endpoint, 
                        params=params, 
                        headers=headers,
                        timeout=DEFAULT_API_TIMEOUT
                    )
                    response.raise_for_status()  # Raise exception for HTTP errors
                    break  # Exit the retry loop if successful
                except requests.RequestException as e:
                    retry_count += 1
                    logger.warning(f"API request failed for {team_type} team (attempt {retry_count}): {str(e)}")
                    
                    if retry_count >= MAX_RETRIES:
                        logger.error(f"Failed to fetch {team_type} team players after {MAX_RETRIES} attempts")
                        raise
                    
                    # Wait before retrying
                    import time
                    time.sleep(RETRY_DELAY * retry_count)  # Exponential backoff
            
            # Parse the response
            if response and response.status_code == 200:
                data = response.json()
                players = data.get('data', [])
                
                if not players:
                    logger.warning(f"No players found for {team_type} team (ID: {team_id})")
                else:
                    logger.info(f"Successfully fetched {len(players)} players for {team_type} team")
                    
                    # Add players to the appropriate list
                    if team_type == 'home':
                        home_players = players
                    else:
                        away_players = players
                    
                    # Cache the response for future use
                    cache_file = DATA_DIR / f"players_{team_id}.json"
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(players, f)
                        logger.info(f"Cached player data to {cache_file}")
                    except Exception as e:
                        logger.warning(f"Failed to cache player data: {str(e)}")
            else:
                logger.error(f"Failed to fetch {team_type} team players: HTTP {response.status_code if response else 'No response'}")
                
                # Try to load from cache as fallback
                cache_file = DATA_DIR / f"players_{team_id}.json"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            cached_players = json.load(f)
                        logger.info(f"Loaded {len(cached_players)} {team_type} team players from cache")
                        
                        # Add cached players to the appropriate list
                        if team_type == 'home':
                            home_players = cached_players
                        else:
                            away_players = cached_players
                    except Exception as e:
                        logger.error(f"Failed to load {team_type} team players from cache: {str(e)}")
        
        # Add all the players to the result dictionary
        result['home'] = home_players
        result['away'] = away_players
        
        # Get player season averages for all players
        all_players = home_players + away_players
        player_ids = [p['id'] for p in all_players if 'id' in p]
        
        # Fetch stats for each player
        for player in all_players:
            player_id = player.get('id')
            if player_id:
                try:
                    stats = get_player_stats(player_id)
                    player['season_stats'] = stats
                except Exception as e:
                    logger.warning(f"Failed to fetch stats for player {player.get('first_name', '')} {player.get('last_name', '')}: {str(e)}")
                    raise ValueError(f"Unable to fetch stats for player {player.get('first_name', '')} {player.get('last_name', '')}: {str(e)}")
        
        logger.info("Successfully fetched and processed all player data")
        return result
    
    except Exception as e:
        logger.error(f"Error fetching players for game: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to fetch players for game: {str(e)}")


def fetch_player_data(games: List[Dict]) -> List[Dict]:
    """
    Fetch player data for all games with comprehensive error handling
    
    Args:
        games: List of game dictionaries
        
    Returns:
        List of player dictionaries with stats
    """
    if not games:
        logger.error("No games provided to fetch_player_data")
        return []
    
    logger.info(f"Fetching player data for {len(games)} games")
    all_players = []
    
    try:
        # Process each game
        for game in games:
            try:
                # Fetch players for this game
                game_players = fetch_players_for_game(game)
                
                # Add team and game information to each player
                for team_type in ['home', 'away']:
                    team_info = game.get(f"{team_type}_team", {})
                    team_name = team_info.get('name', team_type.capitalize())
                    
                    for player in game_players.get(team_type, []):
                        # Add team and game context to player
                        player['team_name'] = team_name
                        player['team_id'] = team_info.get('id')
                        player['game_id'] = game.get('id')
                        player['is_home_team'] = (team_type == 'home')
                        player['opponent_team'] = game.get('away_team' if team_type == 'home' else 'home_team', {}).get('name')
                        player['opponent_id'] = game.get('away_team' if team_type == 'home' else 'home_team', {}).get('id')
                        
                        # Add this player to the combined list
                        all_players.append(player)
            
            except Exception as e:
                logger.error(f"Error processing game {game.get('id')}: {str(e)}")
                continue
        
        logger.info(f"Successfully processed player data for {len(all_players)} players")
        return all_players
    
    except Exception as e:
        logger.error(f"Error in fetch_player_data: {str(e)}")
        logger.error(traceback.format_exc())
        return []


def get_season_averages(player_id: int) -> Dict[str, Any]:
    """
    Get season averages for a player with proper error handling and retries
    
    Args:
        player_id: Player ID to get stats for
        
    Returns:
        Dictionary of player season stats or empty dict if not available
    """
    logger.info(f"Fetching season averages for player {player_id}")
    
    # Skip very large player IDs which are likely invalid in the BallDontLie API
    # The BallDontLie API typically uses smaller IDs (under 1000000) for active players
    if player_id > 1000000:
        logger.warning(f"Player ID {player_id} appears to be outside of expected range for BallDontLie API")
        return {}
    
    # API endpoint
    stats_endpoint = f"{API_BASE_URL}/season_averages"
    
    # API query parameters - simple version first
    params = {
        "season": datetime.now().year if datetime.now().month > 9 else datetime.now().year - 1
    }
    
    # Try to load from cache first
    cache_file = DATA_DIR / f"player_stats_{player_id}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_stats = json.load(f)
            
            # Check if cache is recent (less than 24 hours old)
            cache_timestamp = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_timestamp < timedelta(hours=24):
                logger.info(f"Using cached stats for player {player_id} (from {cache_timestamp})")
                return cached_stats
            else:
                logger.info(f"Cached stats for player {player_id} are outdated, fetching fresh data")
        except Exception as e:
            logger.warning(f"Failed to load cached stats for player {player_id}: {str(e)}")
    
    # Get API key from environment or configuration
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    headers = {}
    
    if api_key:
        headers["Authorization"] = api_key
        
    # Make API request with retries
    response = None
    retry_count = 0
    
    while retry_count < MAX_RETRIES:
        try:
            # Direct approach - use the requests library's params support
            response = requests.get(
                stats_endpoint,
                params={
                    "season": params["season"],
                    "player_ids[]": player_id
                },
                headers=headers,
                timeout=DEFAULT_API_TIMEOUT
            )
            
            # For debugging
            logger.debug(f"Request URL: {response.url}")
            
            # Check for client errors (4xx) - these won't be fixed by retrying
            if response.status_code >= 400 and response.status_code < 500:
                logger.warning(f"Client error {response.status_code} for player {player_id}. Player may not exist in API.")
                return {}  # Return empty dict instead of raising an error
            
            response.raise_for_status()  # Raise exception for other HTTP errors
            break  # Exit the retry loop if successful
        except requests.RequestException as e:
            retry_count += 1
            logger.warning(f"API request failed for player {player_id} (attempt {retry_count}): {str(e)}")
            
            if retry_count >= MAX_RETRIES:
                logger.error(f"Failed to fetch stats for player {player_id} after {MAX_RETRIES} attempts")
                return {}  # Return empty dict instead of raising
            
            # Wait before retrying
            import time
            time.sleep(RETRY_DELAY * retry_count)  # Exponential backoff
    
    # Parse the response
    try:
        data = response.json()
        
        if not data or "data" not in data or not data["data"]:
            logger.warning(f"No season stats available for player {player_id}")
            return {}
        
        # Extract the player stats
        stats = data["data"][0]  # Just get the first entry
        
        # Cache the stats
        try:
            with open(cache_file, 'w') as f:
                json.dump(stats, f)
            logger.info(f"Cached stats for player {player_id}")
        except Exception as e:
            logger.warning(f"Failed to cache stats for player {player_id}: {str(e)}")
        
        return stats
    except Exception as e:
        logger.error(f"Error parsing stats response for player {player_id}: {str(e)}")
        return {}  # Return empty dict instead of raising


def get_player_stats(player_id: int) -> Dict[str, Any]:
    """
    Get comprehensive player statistics with enhanced error handling and data enrichment
    
    This function fetches both current season stats and recent game performance to
    provide a more complete statistical profile for each player.
    
    Args:
        player_id: Player ID to get stats for
        
    Returns:
        Dictionary of player stats with enriched data or empty dict if not available
    """
    logger.info(f"Fetching comprehensive stats for player {player_id}")
    
    # Skip very large player IDs which are likely invalid in the BallDontLie API
    if player_id > 1000000:
        logger.warning(f"Player ID {player_id} appears to be outside of expected range for BallDontLie API")
        return {}
    
    try:
        # Get season averages
        season_stats = get_season_averages(player_id)
        
        if not season_stats:
            logger.warning(f"No season stats available for player {player_id}")
            return {}
        
        # Add additional derived metrics
        enhanced_stats = season_stats.copy()
        
        # Calculate effective field goal percentage (eFG%)
        fg = enhanced_stats.get('fg_pct', 0)
        fg3 = enhanced_stats.get('fg3_pct', 0)
        fg3a = enhanced_stats.get('fg3a', 0)
        fga = enhanced_stats.get('fga', 0)
        
        if fga > 0:
            enhanced_stats['efg_pct'] = (enhanced_stats.get('fgm', 0) + 0.5 * enhanced_stats.get('fg3m', 0)) / fga
        else:
            enhanced_stats['efg_pct'] = 0
        
        # Calculate true shooting percentage (TS%)
        pts = enhanced_stats.get('pts', 0)
        fta = enhanced_stats.get('fta', 0)
        
        if fga > 0 or fta > 0:
            enhanced_stats['ts_pct'] = pts / (2 * (fga + 0.44 * fta))
        else:
            enhanced_stats['ts_pct'] = 0
        
        # Calculate usage rate (approximate)
        # A true usage rate would require team data, but we can approximate
        min = enhanced_stats.get('min', 0)
        fga = enhanced_stats.get('fga', 0)
        fta = enhanced_stats.get('fta', 0)
        to = enhanced_stats.get('turnover', 0)
        
        if min > 0:
            enhanced_stats['usage_rate'] = (fga + 0.44 * fta + to) / min
        else:
            enhanced_stats['usage_rate'] = 0
        
        # Calculate assist-to-turnover ratio
        ast = enhanced_stats.get('ast', 0)
        to = enhanced_stats.get('turnover', 0)
        
        if to > 0:
            enhanced_stats['ast_to_ratio'] = ast / to
        else:
            enhanced_stats['ast_to_ratio'] = ast  # If no turnovers, use assists directly
        
        # Calculate offensive rating (simplified)
        # A true offensive rating is complex, but we can approximate
        pts = enhanced_stats.get('pts', 0)
        fga = enhanced_stats.get('fga', 0)
        fta = enhanced_stats.get('fta', 0)
        to = enhanced_stats.get('turnover', 0)
        
        possessions = fga + 0.44 * fta + to
        if possessions > 0:
            enhanced_stats['off_rating'] = 100 * pts / possessions
        else:
            enhanced_stats['off_rating'] = 0
        
        # Get player position and convert to a standardized format
        position = enhanced_stats.get('position', '')
        if position:
            from ..utils.string_utils import position_to_numeric
            enhanced_stats['position_value'] = position_to_numeric(position)
        
        # Get player height and convert to inches
        height = enhanced_stats.get('height', '')
        if height:
            from ..utils.string_utils import parse_height
            enhanced_stats['height_inches'] = parse_height(height)
        
        # Add last updated timestamp
        enhanced_stats['last_updated'] = datetime.now().isoformat()
        
        logger.info(f"Successfully processed comprehensive stats for player {player_id}")
        return enhanced_stats
    
    except Exception as e:
        logger.error(f"Error getting comprehensive stats for player {player_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return {}
