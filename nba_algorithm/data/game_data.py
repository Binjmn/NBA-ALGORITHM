#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Game Data Module

This module handles fetching NBA game data from external APIs and processing it
for model consumption.

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
ODDS_API_URL = "https://api.the-odds-api.com/v4"


def fetch_nba_games(date_str: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch today's NBA games from the BallDontLie API with robust error handling
    
    Args:
        date_str: Optional date string in YYYY-MM-DD format. Defaults to today's date.
        
    Returns:
        List of game dictionaries
    """
    # Use today's date if not specified
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Fetching NBA games for {date_str}")
    
    # BallDontLie API endpoints
    games_endpoint = f"{API_BASE_URL}/games"
    
    # API query parameters
    params = {
        "dates[]": date_str,
        "leagues[]": "NBA",
        "per_page": 100,
    }
    
    # Get API key from environment or configuration
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    headers = {}
    
    if api_key:
        headers["Authorization"] = api_key
        logger.info("Using BallDontLie API key from environment")
    else:
        logger.warning("No BallDontLie API key found. Requests may be rate limited.")
    
    try:
        # Make API request with retries
        response = None
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            try:
                response = requests.get(
                    games_endpoint, 
                    params=params, 
                    headers=headers,
                    timeout=DEFAULT_API_TIMEOUT
                )
                response.raise_for_status()  # Raise exception for HTTP errors
                break  # Exit the retry loop if successful
            except requests.RequestException as e:
                retry_count += 1
                logger.warning(f"API request failed (attempt {retry_count}): {str(e)}")
                
                if retry_count >= MAX_RETRIES:
                    logger.error(f"Failed to fetch games after {MAX_RETRIES} attempts")
                    raise
                
                # Wait before retrying
                import time
                time.sleep(RETRY_DELAY * retry_count)  # Exponential backoff
        
        # Parse the response
        if response and response.status_code == 200:
            data = response.json()
            games = data.get('data', [])
            
            if not games:
                logger.warning(f"No NBA games found for {date_str}")
            else:
                logger.info(f"Successfully fetched {len(games)} games for {date_str}")
            
            # Cache the response for future use
            cache_file = DATA_DIR / f"games_{date_str}.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump(games, f)
                logger.info(f"Cached game data to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache game data: {str(e)}")
            
            # Enhance games with additional team info if available
            enhanced_games = []
            for game in games:
                game_copy = game.copy()
                
                # Add team abbreviations if missing
                if 'home_team' in game and 'abbreviation' not in game['home_team']:
                    game_copy['home_team']['abbreviation'] = get_team_abbreviation(game['home_team']['id'])
                
                if 'visitor_team' in game and 'abbreviation' not in game['visitor_team']:
                    game_copy['visitor_team']['abbreviation'] = get_team_abbreviation(game['visitor_team']['id'])
                
                enhanced_games.append(game_copy)
            
            return enhanced_games
        else:
            logger.error(f"Failed to fetch games: HTTP {response.status_code if response else 'No response'}")
            logger.error(f"Response: {response.text if response else 'None'}")
            
            # Try to load from cache as fallback
            cache_file = DATA_DIR / f"games_{date_str}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cached_games = json.load(f)
                    logger.info(f"Loaded {len(cached_games)} games from cache for {date_str}")
                    return cached_games
                except Exception as e:
                    logger.error(f"Failed to load games from cache: {str(e)}")
            
            # As a last resort, try to find any games in the cache directory
            try:
                cache_files = list(DATA_DIR.glob("games_*.json"))
                if cache_files:
                    # Use the most recent cache file
                    most_recent = max(cache_files, key=lambda f: f.stat().st_mtime)
                    with open(most_recent, 'r') as f:
                        cached_games = json.load(f)
                    logger.info(f"Loaded {len(cached_games)} games from most recent cache: {most_recent.name}")
                    return cached_games
            except Exception as e:
                logger.error(f"Failed to load games from any cache: {str(e)}")
            
            # If all else fails, return an empty list
            return []
    
    except Exception as e:
        logger.error(f"Error fetching NBA games: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try to load from cache as fallback
        cache_file = DATA_DIR / f"games_{date_str}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_games = json.load(f)
                logger.info(f"Loaded {len(cached_games)} games from cache for {date_str}")
                return cached_games
            except Exception as cache_e:
                logger.error(f"Failed to load games from cache: {str(cache_e)}")
        
        # If all else fails, return an empty list
        return []


def get_team_abbreviation(team_id: int) -> str:
    """
    Get team abbreviation from team ID using a lookup table
    
    Args:
        team_id: The team ID to look up
        
    Returns:
        Team abbreviation or a placeholder if not found
    """
    # Team ID to abbreviation mapping (could be expanded to a proper database/API call)
    team_abbr = {
        1: "ATL",  # Atlanta Hawks
        2: "BOS",  # Boston Celtics
        3: "BKN",  # Brooklyn Nets
        4: "CHA",  # Charlotte Hornets
        5: "CHI",  # Chicago Bulls
        6: "CLE",  # Cleveland Cavaliers
        7: "DAL",  # Dallas Mavericks
        8: "DEN",  # Denver Nuggets
        9: "DET",  # Detroit Pistons
        10: "GSW",  # Golden State Warriors
        11: "HOU",  # Houston Rockets
        12: "IND",  # Indiana Pacers
        13: "LAC",  # Los Angeles Clippers
        14: "LAL",  # Los Angeles Lakers
        15: "MEM",  # Memphis Grizzlies
        16: "MIA",  # Miami Heat
        17: "MIL",  # Milwaukee Bucks
        18: "MIN",  # Minnesota Timberwolves
        19: "NOP",  # New Orleans Pelicans
        20: "NYK",  # New York Knicks
        21: "OKC",  # Oklahoma City Thunder
        22: "ORL",  # Orlando Magic
        23: "PHI",  # Philadelphia 76ers
        24: "PHX",  # Phoenix Suns
        25: "POR",  # Portland Trail Blazers
        26: "SAC",  # Sacramento Kings
        27: "SAS",  # San Antonio Spurs
        28: "TOR",  # Toronto Raptors
        29: "UTA",  # Utah Jazz
        30: "WAS",  # Washington Wizards
    }
    
    return team_abbr.get(team_id, f"TEAM{team_id}")
