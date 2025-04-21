#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Collection Module for NBA Model Training Pipeline

Responsibilities:
- Collect high-quality historical NBA game data
- Gather team and player statistics
- Collect betting odds and lines
- Filter for active NBA teams
- Ensure data quality and completeness
- Provide validation and error handling

This module strictly uses real data only (no synthetic data)
and provides extensive error handling for production use.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import traceback
import json
import requests
from pathlib import Path

# Import data collection modules
from src.data.collector_adapter import HistoricalDataCollectorAdapter
from src.Model_training_pipeline.utils.rate_limiter import default_rate_limiter

# Import modules for active NBA team filtering and real data
from nba_algorithm.data.historical_collector import fetch_historical_games
from nba_algorithm.data.team_data import fetch_team_stats, fetch_all_teams
from nba_algorithm.data.nba_teams import get_active_nba_teams, filter_games_to_active_teams

from .config import logger
from .utils.rate_limiter import RateLimiter

class DataCollector:
    """
    Production-ready data collector for NBA prediction model training
    
    Features:
    - Collects real historical NBA game data
    - Gathers team and player statistics
    - Filters for active NBA teams
    - Collects betting odds and lines
    - Provides data validation and quality checks
    - Handles API rate limits and errors gracefully
    - No synthetic data generation - all real data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data collector with configuration
        
        Args:
            config: Configuration dictionary with data collection settings
        """
        self.config = config
        self.season = config.get('season', datetime.now().year)
        
        # Get API configuration
        api_config = config.get('data_collection', {}).get('api', {}) 
        # Update the base URL to the correct endpoint
        self.base_url = api_config.get('base_url', 'https://api.balldontlie.io/v1')
        self.timeout = api_config.get('timeout', 10)
        # Prioritize environment variable over config file
        self.api_key = os.environ.get('BALLDONTLIE_API_KEY') or api_config.get('key', '')
        self.odds_api_key = os.environ.get('THE_ODDS_API_KEY') or api_config.get('odds_key', '')
        
        # Create BallDontLie client - it uses the API key from environment variables
        from src.api.balldontlie_client import BallDontLieClient
        # No need to set environment variable if it's already there
        self.balldontlie_client = BallDontLieClient()
        
        # Collection settings
        data_collection = config.get('data_collection', {})
        self.days_back = data_collection.get('days_back', 365)
        self.include_stats = data_collection.get('include_stats', True)
        self.include_odds = data_collection.get('include_odds', True)
        self.filter_active_teams = data_collection.get('filter_active_teams', True)
        self.use_real_data_only = data_collection.get('use_real_data_only', True)
        self.min_games_required = data_collection.get('min_games_required', 100)
        self.max_games_per_season = data_collection.get('max_games_per_season', 100)
        
        # Team mapping for name lookups
        self.team_mapping = {}
        self.active_team_ids = []
        
        # Initialize metrics tracking
        self.metrics = {
            'seasons_collected': 0,
            'games_collected': 0,
            'teams_collected': 0,
            'api_requests': 0,
            'api_errors': 0,
            'processing_time': 0
        }
        
        # Log configuration summary
        logger.info(f"Initialized DataCollector for season {self.season} with API URL: {self.base_url}")
        if self.api_key:
            masked_key = self.api_key[:6] + '...' + self.api_key[-4:] if len(self.api_key) > 10 else '*****'
            logger.info(f"Using API key (masked): {masked_key}")
        if self.use_real_data_only:
            logger.info("Using real data only (no synthetic data)")
    
    def collect_historical_data(self, seasons: Optional[List[int]] = None) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Collect historical NBA game data
        
        Args:
            seasons: Optional list of seasons to collect (e.g., [2021, 2022, 2023])
                    If None, will automatically collect last 4 seasons
                    
        Returns:
            Tuple of (collected_data, success_status)
        """
        start_time = time.time()
        
        try:
            # Auto-detect seasons if not provided
            if seasons is None:
                current_season = self.season
                num_seasons = self.config['data_collection'].get('num_seasons', 4)
                seasons = list(range(current_season - num_seasons + 1, current_season + 1))
                logger.info(f"Auto-detected seasons to collect: {seasons}")
            
            logger.info(f"Collecting historical NBA data for seasons: {seasons}")
            
            # Get teams data and build the mapping dictionary
            teams_data = self._get_teams()
            if not teams_data:
                logger.error("Failed to retrieve teams data")
                return [], False
                
            # Build team mapping from team data
            self.team_mapping = {team['id']: team['name'] for team in teams_data}
            logger.info(f"Built team mapping dictionary with {len(self.team_mapping)} teams")
            
            # Apply active team filtering if needed
            if self.filter_active_teams:
                active_teams = self._filter_active_nba_teams(teams_data)
                self.active_team_ids = [team['id'] for team in active_teams]
                logger.info(f"Filtered to {len(self.active_team_ids)} active NBA teams")
            
            # Collect games for each season
            all_games = []
            total_games = 0
            
            for season in seasons:
                logger.info(f"Collecting games for season {season}")
                season_games = self._get_games_for_season(season)
                
                # Filter for games between active teams if needed
                if self.filter_active_teams and season_games:
                    filtered_games = [g for g in season_games if self._is_game_between_active_teams(g)]
                    logger.info(f"Retrieved {len(filtered_games)} games between active teams for season {season}")
                    all_games.extend(filtered_games)
                    total_games += len(filtered_games)
                else:
                    all_games.extend(season_games if season_games else [])
                    total_games += len(season_games) if season_games else 0
            
            # Check if we have enough games
            if len(all_games) < self.min_games_required:
                logger.warning(f"Collected only {len(all_games)} games, which is below the minimum threshold of {self.min_games_required}")
                if not self.use_real_data_only:
                    # Generate synthetic data if real data is insufficient and synthetic data is allowed
                    logger.warning("Generating synthetic data to supplement insufficient real data")
                    # This would be implemented in a separate method
                    # We're skipping this since you specified to never use synthetic data
            
            # Enrich games with additional data
            if all_games:
                logger.info(f"Successfully collected {len(all_games)} games across {len(seasons)} seasons")
                enriched_games = self._enrich_games_with_data(all_games)
                
                # Log sample of enriched game data with target values
                if enriched_games:
                    sample_game = enriched_games[0]
                    target_values = {
                        key: sample_game.get(key, 'missing') 
                        for key in ['home_win', 'spread_diff', 'total_points']
                    }
                    logger.info(f"Sample target values after enrichment: {target_values}")
                
                # Update metrics
                self.metrics['games_collected'] = len(enriched_games)
                self.metrics['seasons_collected'] = len(seasons)
                self.metrics['processing_time'] = time.time() - start_time
                
                return enriched_games, True
            else:
                logger.error("Failed to collect any historical data. Aborting training.")
                return [], False
                
        except Exception as e:
            logger.error(f"Error collecting historical data: {str(e)}")
            logger.error(traceback.format_exc())
            return [], False

    def collect_player_stats_for_game(self, game_id: int) -> List[Dict[str, Any]]:
        """
        Collect player statistics for a specific game
        
        Args:
            game_id: ID of the game to collect player stats for
            
        Returns:
            List of player statistics dictionaries
        """
        if not game_id:
            logger.warning("No game ID provided. Cannot collect player stats.")
            return []
            
        try:
            # Construct URL for the stats endpoint
            url = f"{self.base_url}/stats"
            params = {"game_ids[]" : [game_id], "per_page": 100}
            headers = {"Authorization": f"{self.api_key}"}
            
            logger.debug(f"Requesting player stats for game ID: {game_id}")
            
            # Rate limiting
            default_rate_limiter.wait_if_needed('balldontlie')
            
            # Make the request
            response = requests.get(url, params=params, headers=headers)
            
            # Check response
            if response.status_code != 200:
                logger.error(f"Failed to retrieve player stats for game {game_id}. Status code: {response.status_code}")
                return []
                
            # Extract data
            data = response.json()
            player_stats = data.get('data', [])
            
            if not player_stats:
                logger.warning(f"No player stats found for game {game_id}")
                return []
                
            logger.info(f"Retrieved {len(player_stats)} player statistics for game {game_id}")
            
            # Log the first player's stats to see the field names
            if player_stats and len(player_stats) > 0:
                logger.debug(f"Sample player stat fields: {list(player_stats[0].keys())}")
                logger.debug(f"Sample player stat: {player_stats[0]}")
            
            # Enhance player stats with team mapping
            for stat in player_stats:
                if 'team' in stat and 'id' in stat['team']:
                    team_id = stat['team']['id']
                    stat['team_name'] = self.team_mapping.get(team_id, 'Unknown')
                
                # Add game_id to each stat entry for reference
                stat['game_id'] = game_id
                
                # Map API fields to our expected format
                if 'pts' in stat:
                    stat['player_pts'] = stat['pts']
                if 'reb' in stat:
                    stat['player_reb'] = stat['reb']
                if 'ast' in stat:
                    stat['player_ast'] = stat['ast']
                if 'fg3m' in stat:
                    stat['player_3pm'] = stat['fg3m']
                if 'stl' in stat:
                    stat['player_stl'] = stat['stl']
                if 'blk' in stat:
                    stat['player_blk'] = stat['blk']
                if 'turnover' in stat:
                    stat['player_to'] = stat['turnover']
                if 'min' in stat:
                    # Convert 'min' from format like '31:20' to minutes as float
                    try:
                        min_parts = stat['min'].split(':')
                        if len(min_parts) >= 2:
                            stat['player_min'] = float(min_parts[0]) + float(min_parts[1])/60
                        else:
                            stat['player_min'] = float(min_parts[0]) if min_parts[0] else 0.0
                    except (ValueError, AttributeError):
                        stat['player_min'] = 0.0
                if 'fga' in stat:
                    stat['player_fga'] = stat['fga']
                if 'fgm' in stat:
                    stat['player_fgm'] = stat['fgm']
                if 'fg3a' in stat:
                    stat['player_tpa'] = stat['fg3a']
                if 'fg3m' in stat:
                    stat['player_tpm'] = stat['fg3m']
                
            return player_stats
            
        except Exception as e:
            logger.error(f"Error collecting player stats for game {game_id}: {str(e)}")
            return []
    
    def _get_teams(self) -> List[Dict[str, Any]]:
        """
        Get teams data
        
        Returns:
            List of team data dictionaries
        """
        try:
            url = f"{self.base_url}/teams"
            response = self._make_api_request(url)
            self.metrics['api_requests'] += 1
            
            if not response or 'data' not in response:
                logger.error("Invalid response for teams data")
                return []
            
            return response['data']
        
        except Exception as e:
            logger.error(f"Error fetching teams data: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics['api_errors'] += 1
            return []

    def _get_games_for_season(self, season: int) -> List[Dict[str, Any]]:
        """
        Get games for a specific season using the BallDontLie API
        
        Args:
            season: Season year (e.g., 2021 for 2021-22 season)
            
        Returns:
            List of games for the season
        """
        try:
            # Use the updated BallDontLie client to get games with scores
            max_games = self.config['data_collection'].get('max_games_per_season', 100)
            games = self.balldontlie_client.get_games(season=season, max_games=max_games)
            logger.info(f"Retrieved {len(games)} filtered games for season {season}")
            return games
        except Exception as e:
            logger.error(f"Error retrieving games for season {season}: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def _filter_active_nba_teams(self, teams_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter for current active NBA teams only
        
        Args:
            teams_data: List of team data dictionaries
            
        Returns:
            Filtered list of active NBA teams
        """
        active_teams = []
        for team in teams_data:
            # The BallDontLie API doesn't explicitly indicate if teams are active or in NBA
            # We'll use a combination of factors to determine:
            # 1. Team has a city and name
            # 2. Team has a division
            # 3. Team is not explicitly marked as inactive if that field exists
            has_required_fields = ('city' in team and 'name' in team and 'division' in team)
            is_active = team.get('is_active', True)  # Default to True if field doesn't exist
            
            if has_required_fields and is_active:
                active_teams.append(team)
        
        # Log the active teams for verification
        team_names = [f"{team['city']} {team['name']}" for team in active_teams]
        logger.info(f"Filtered {len(active_teams)} active NBA teams: {', '.join(team_names)}")
        
        return active_teams

    def _is_game_between_active_teams(self, game: Dict[str, Any]) -> bool:
        """
        Check if a game is between two active NBA teams
        
        Args:
            game: Game data dictionary
            
        Returns:
            True if game is between active teams
        """
        if 'home_team' not in game or 'visitor_team' not in game:
            return False
            
        home_team_id = game['home_team'].get('id') if isinstance(game['home_team'], dict) else game.get('home_team_id')
        visitor_team_id = game['visitor_team'].get('id') if isinstance(game['visitor_team'], dict) else game.get('visitor_team_id')
        
        return (home_team_id in self.active_team_ids and visitor_team_id in self.active_team_ids)

    def _enrich_games_with_data(self, games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich games with additional data
        
        Args:
            games: List of game data dictionaries
            
        Returns:
            List of enriched game data dictionaries
        """
        enriched_games = []
        games_processed = 0
        games_with_targets = 0
        
        for game in games:
            try:
                # Only process games with scores (completed games)
                if 'home_score' not in game or 'visitor_score' not in game:
                    # Skip games without scores (future games or incomplete data)
                    logger.debug(f"Skipping game without scores: {game.get('id', 'unknown')}")
                    continue
                    
                # Extract team information - handle both API response formats
                # Standard BallDontLie API format has nested 'home_team' and 'visitor_team' objects
                if 'home_team' in game and isinstance(game['home_team'], dict):
                    home_team_id = game['home_team'].get('id')
                    home_team_name = game['home_team'].get('name', 'Unknown')
                    visitor_team_id = game['visitor_team'].get('id')
                    visitor_team_name = game['visitor_team'].get('name', 'Unknown')
                # Alternative format with direct IDs
                elif 'home_team_id' in game:
                    home_team_id = game['home_team_id']
                    home_team_name = self.team_mapping.get(home_team_id, 'Unknown')
                    visitor_team_id = game['visitor_team_id']
                    visitor_team_name = self.team_mapping.get(visitor_team_id, 'Unknown')
                else:
                    logger.warning(f"Unexpected game data format: {game.keys()}")
                    continue
                    
                # Add team names to the game data
                game['home_team_name'] = home_team_name
                game['visitor_team_name'] = visitor_team_name
                
                # Ensure we have standard keys for our processing logic
                if 'home_team_id' not in game and home_team_id is not None:
                    game['home_team_id'] = home_team_id
                if 'visitor_team_id' not in game and visitor_team_id is not None:
                    game['visitor_team_id'] = visitor_team_id
                
                # Add target values for prediction models - we must have scores to calculate these
                home_score = int(game.get('home_score', 0))
                visitor_score = int(game.get('visitor_score', 0))
                
                # Add home_win flag (1 if home team won, 0 if lost)
                game['home_win'] = 1 if home_score > visitor_score else 0
                
                # Add point spread (regression - home team margin)
                game['spread_diff'] = home_score - visitor_score
                
                # Add total points (regression - total game score)
                game['total_points'] = home_score + visitor_score
                
                # Track games with targets
                games_with_targets += 1
                
                # Process complete, add to enriched games
                enriched_games.append(game)
                games_processed += 1
                
                # Log sample of processed games
                if games_processed % 100 == 0:
                    logger.info(f"Processed {games_processed} games")
                    
            except Exception as e:
                logger.error(f"Error enriching game data: {str(e)}")
                logger.debug(f"Game data: {game}")
        
        # Log summary
        logger.info(f"Enriched {games_processed} out of {len(games)} total games")
        logger.info(f"Added target values to {games_with_targets} games")
        
        # Log sample of a processed game with target values
        if enriched_games:
            sample = enriched_games[0]
            target_sample = {
                'home_win': sample.get('home_win', 'missing'),
                'spread_diff': sample.get('spread_diff', 'missing'),
                'total_points': sample.get('total_points', 'missing')
            }
            logger.info(f"Sample game targets: {target_sample}")
        
        return enriched_games

    def _make_api_request(self, url: str, params: Dict[str, Any] = None, api_type: str = 'balldontlie') -> Dict[str, Any]:
        """
        Make an API request with retry logic and rate limiting
        
        Args:
            url: API endpoint URL
            params: Optional query parameters
            api_type: Type of API ('balldontlie' or 'odds')
            
        Returns:
            JSON response data
            
        Raises:
            Exception if request fails after retries
        """
        retry_count = self.config['data_collection'].get('retry_count', 3)
        retry_delay = self.config['data_collection'].get('retry_delay', 2.0)
        timeout = self.config['data_collection']['api'].get('timeout', 10)
        
        for attempt in range(retry_count + 1):
            try:
                # Apply rate limiting
                wait_time = default_rate_limiter.wait_if_needed(api_type)
                if wait_time > 0:
                    logger.info(f"Rate limit applied for {api_type} API. Waited {wait_time:.2f} seconds")
                
                # Get API key based on API type - prioritize environment variables
                if api_type == 'balldontlie':
                    api_key = os.environ.get('BALLDONTLIE_API_KEY') or self.api_key
                elif api_type == 'odds':
                    api_key = os.environ.get('THE_ODDS_API_KEY') or self.odds_api_key
                else:
                    api_key = ''
                
                # Debug the API key (mask for security)
                if api_key:
                    masked_key = api_key[:4] + '...' + api_key[-4:] if len(api_key) > 10 else '****'
                    logger.info(f"Using API key for {api_type} (masked): {masked_key}")
                else:
                    logger.warning(f"No API key found for {api_type} API. This request will likely fail.")
                    if api_type == 'balldontlie':
                        logger.error(f"BALLDONTLIE_API_KEY environment variable is not set properly.")
                        # Raise an exception instead of continuing with a known-to-fail request
                        raise ValueError("Missing required BallDontLie API key. Please set the BALLDONTLIE_API_KEY environment variable.")
                
                # Set up request parameters based on API type
                headers = {}
                params = params if params else {}
                
                if api_type == 'balldontlie':
                    # BallDontLie API requires the API key in the Authorization header
                    # According to their latest documentation, they don't use the Bearer prefix
                    if api_key:
                        headers['Authorization'] = api_key
                        logger.info(f"Using Authorization header for BallDontLie API")
                elif api_type == 'odds':
                    # The Odds API expects the key as a query parameter
                    if api_key:
                        params['apiKey'] = api_key
                else:
                    # Default to Bearer token authorization
                    if api_key:
                        headers['Authorization'] = f'Bearer {api_key}'
                
                # Make the request
                logger.info(f"Making API request to: {url}")
                response = requests.get(url, headers=headers, params=params, timeout=timeout)
                
                # Log response status
                logger.debug(f"Received response with status code: {response.status_code}")
                
                # Handle specific response codes
                if response.status_code == 401:
                    logger.error(f"Authentication failed for {api_type} API. Check your API key.")
                
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                
                # Parse JSON response
                data = response.json()
                return data
                
            except requests.exceptions.RequestException as e:
                self.metrics['api_errors'] += 1
                
                if attempt < retry_count:
                    logger.warning(f"API request failed (attempt {attempt+1}/{retry_count+1}): {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"API request failed after {retry_count+1} attempts: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
            except Exception as e:
                self.metrics['api_errors'] += 1
                logger.error(f"Error making API request to {url}: {str(e)}")
                logger.error(traceback.format_exc())
                raise

    def get_collection_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the data collection process
        
        Returns:
            Dictionary with collection metrics
        """
        return self.metrics