#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Season Data Collector

This module manages the rolling window of NBA seasons used for training data.
It ensures the system always maintains exactly 4 seasons of data, automatically
updating when a new season begins by adding the newest season and removing
the oldest one.

Author: Cascade
Date: April 2025
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import json
import shutil

from ..api.balldontlie_client import BallDontLieClient
from ..utils.cache_manager import CacheManager
from ..utils.season_manager import get_season_manager
from ..config.season_config import get_season_display_name

# Configure logger
logger = logging.getLogger(__name__)

# Set up cache for season data
cache = CacheManager(cache_name="season_data", ttl_seconds=86400)  # Cache for 24 hours

# Data directories
DATA_DIR = Path('data/historical')
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Season data subdirectories
SEASONS_DIR = DATA_DIR / 'seasons'
SEASONS_DIR.mkdir(parents=True, exist_ok=True)

# Archive directory for old seasons
ARCHIVE_DIR = DATA_DIR / 'archive'
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Number of seasons to maintain for training
TRAINING_SEASONS_COUNT = 4


def get_available_seasons() -> List[int]:
    """
    Get list of available seasons in the data directory
    
    Returns:
        List of season years for which data is available (sorted newest to oldest)
    """
    available_seasons = []
    
    # Check for season directories
    for item in SEASONS_DIR.iterdir():
        if item.is_dir() and item.name.isdigit():
            available_seasons.append(int(item.name))
    
    # Sort newest to oldest
    return sorted(available_seasons, reverse=True)


def get_training_seasons() -> List[int]:
    """
    Get the seasons that should be used for training (up to TRAINING_SEASONS_COUNT)
    
    Returns:
        List of season years to use for training (sorted newest to oldest)
    """
    available = get_available_seasons()
    
    # Return up to TRAINING_SEASONS_COUNT seasons
    return available[:TRAINING_SEASONS_COUNT]


def collect_season_data(season_year: int) -> bool:
    """
    Collect all data for a specific NBA season
    
    Args:
        season_year: Year of the NBA season to collect
        
    Returns:
        Boolean indicating success
    """
    season_dir = SEASONS_DIR / str(season_year)
    season_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize API client
        client = BallDontLieClient()
        
        # Get all regular season games for this season
        logger.info(f"Collecting regular season games for {get_season_display_name(season_year)}")
        
        params = {
            'seasons': [season_year],
            'per_page': 100
        }
        
        all_games = []
        page = 1
        
        while True:
            params['page'] = page
            response = client.get_games(**params)
            
            if not response or 'data' not in response or not response['data']:
                break
                
            games = response['data']
            all_games.extend(games)
            
            # Check if we've retrieved all pages
            if len(games) < params['per_page']:
                break
                
            page += 1
        
        # Save regular season games
        if all_games:
            games_file = season_dir / 'regular_season_games.json'
            with open(games_file, 'w') as f:
                json.dump(all_games, f)
        
        # Get team stats for this season
        logger.info(f"Collecting team stats for {get_season_display_name(season_year)}")
        
        team_stats = client.get_team_stats(season=season_year)
        if team_stats and 'data' in team_stats:
            teams_file = season_dir / 'team_stats.json'
            with open(teams_file, 'w') as f:
                json.dump(team_stats['data'], f)
        
        # Get player stats for this season
        logger.info(f"Collecting player stats for {get_season_display_name(season_year)}")
        
        all_players = []
        player_params = {
            'season': season_year,
            'per_page': 100
        }
        
        page = 1
        while True:
            player_params['page'] = page
            player_response = client.get_player_stats(**player_params)
            
            if not player_response or 'data' not in player_response or not player_response['data']:
                break
                
            players = player_response['data']
            all_players.extend(players)
            
            # Check if we've retrieved all pages
            if len(players) < player_params['per_page']:
                break
                
            page += 1
        
        # Save player stats
        if all_players:
            players_file = season_dir / 'player_stats.json'
            with open(players_file, 'w') as f:
                json.dump(all_players, f)
        
        logger.info(f"Successfully collected data for {get_season_display_name(season_year)}")
        return True
        
    except Exception as e:
        logger.error(f"Error collecting season data for {season_year}: {str(e)}")
        return False


def archive_season(season_year: int) -> bool:
    """
    Archive a season by moving it from active seasons to the archive directory
    
    Args:
        season_year: Year of the NBA season to archive
        
    Returns:
        Boolean indicating success
    """
    season_dir = SEASONS_DIR / str(season_year)
    archive_season_dir = ARCHIVE_DIR / str(season_year)
    
    try:
        if not season_dir.exists():
            logger.warning(f"Season directory {season_year} does not exist, nothing to archive")
            return False
        
        # Move season directory to archive
        shutil.move(str(season_dir), str(archive_season_dir))
        logger.info(f"Archived season {get_season_display_name(season_year)}")
        return True
        
    except Exception as e:
        logger.error(f"Error archiving season {season_year}: {str(e)}")
        return False


def maintain_training_window() -> bool:
    """
    Maintain the rolling window of training seasons (exactly TRAINING_SEASONS_COUNT)
    
    This function checks the available seasons and ensures we have exactly the
    most recent TRAINING_SEASONS_COUNT seasons, archiving older ones.
    
    Returns:
        Boolean indicating if any changes were made
    """
    try:
        # Get current season from the season manager
        season_manager = get_season_manager()
        current_season_year = season_manager.get_current_season_year()
        current_phase = season_manager.get_current_season_phase()
        
        # Get available seasons
        available_seasons = get_available_seasons()
        
        # Check if we need to collect the most recent completed season
        # We consider a season complete when the next season has started
        if available_seasons and current_season_year > available_seasons[0]:
            # The most recent season in our data is not the current one,
            # so the previous season is complete
            previous_season = current_season_year - 1
            
            if previous_season not in available_seasons:
                logger.info(f"Collecting data for completed season {get_season_display_name(previous_season)}")
                collect_season_data(previous_season)
                
                # Update available seasons list
                available_seasons = get_available_seasons()
        
        # Check if we have more than TRAINING_SEASONS_COUNT seasons
        if len(available_seasons) > TRAINING_SEASONS_COUNT:
            # Archive oldest seasons beyond the training window
            seasons_to_archive = available_seasons[TRAINING_SEASONS_COUNT:]
            
            for old_season in seasons_to_archive:
                logger.info(f"Archiving old season {get_season_display_name(old_season)}")
                archive_season(old_season)
                
            return True  # Changes were made
        
        return False  # No changes needed
        
    except Exception as e:
        logger.error(f"Error maintaining training window: {str(e)}")
        return False


def register_season_change_handlers():
    """
    Register handlers for season changes with the season manager
    """
    season_manager = get_season_manager()
    
    # Register callback for season changes
    season_manager.register_season_change_callback(lambda old, new: maintain_training_window())
    
    logger.info("Registered season change handlers")


def get_training_data_seasons() -> Dict[str, Any]:
    """
    Get consolidated training data from all seasons in the training window
    
    Returns:
        Dictionary with combined data from all training seasons
    """
    # Force maintenance of the training window
    maintain_training_window()
    
    # Get the seasons we should use for training
    training_seasons = get_training_seasons()
    
    if not training_seasons:
        logger.warning("No training seasons available")
        return {}
    
    # Check cache first
    cache_key = f"training_data_{'-'.join(map(str, training_seasons))}"
    cached_data = cache.get(cache_key)
    
    if cached_data is not None:
        logger.info(f"Using cached training data for seasons {training_seasons}")
        return cached_data
    
    # Combine data from all training seasons
    all_games = []
    all_team_stats = []
    all_player_stats = []
    
    for season_year in training_seasons:
        season_dir = SEASONS_DIR / str(season_year)
        
        # Check if season directory exists
        if not season_dir.exists():
            logger.warning(f"Season directory {season_year} does not exist, skipping")
            continue
        
        # Load games data
        games_file = season_dir / 'regular_season_games.json'
        if games_file.exists():
            try:
                with open(games_file, 'r') as f:
                    games = json.load(f)
                    for game in games:
                        game['season'] = season_year
                    all_games.extend(games)
            except Exception as e:
                logger.error(f"Error loading games for season {season_year}: {str(e)}")
        
        # Load team stats data
        teams_file = season_dir / 'team_stats.json'
        if teams_file.exists():
            try:
                with open(teams_file, 'r') as f:
                    teams = json.load(f)
                    for team in teams:
                        team['season'] = season_year
                    all_team_stats.extend(teams)
            except Exception as e:
                logger.error(f"Error loading team stats for season {season_year}: {str(e)}")
        
        # Load player stats data
        players_file = season_dir / 'player_stats.json'
        if players_file.exists():
            try:
                with open(players_file, 'r') as f:
                    players = json.load(f)
                    for player in players:
                        player['season'] = season_year
                    all_player_stats.extend(players)
            except Exception as e:
                logger.error(f"Error loading player stats for season {season_year}: {str(e)}")
    
    # Create the combined data dictionary
    training_data = {
        'seasons': training_seasons,
        'games': all_games,
        'team_stats': all_team_stats,
        'player_stats': all_player_stats
    }
    
    # Cache the combined data
    cache.set(cache_key, training_data)
    
    logger.info(f"Successfully compiled training data for {len(training_seasons)} seasons")
    return training_data


# Initialize by registering handlers
register_season_change_handlers()
