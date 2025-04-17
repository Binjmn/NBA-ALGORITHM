#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Team Data Module

This module handles fetching and processing NBA team data from external APIs.

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
import numpy as np

from ..config import DATA_DIR, DEFAULT_API_TIMEOUT, MAX_RETRIES, RETRY_DELAY

# Configure logger
logger = logging.getLogger(__name__)

# Constants
API_BASE_URL = "https://api.balldontlie.io/v1"


def get_current_season() -> int:
    """
    Auto-detect the current NBA season based on the date
    
    NBA seasons are defined by the year they end in.
    The NBA season typically starts in October and ends in June of the following year.
    
    Returns:
        Current season year (e.g., 2024 for the 2023-2024 season)
    """
    today = datetime.now()
    
    # If we're after July 1st but before October, we're in the offseason
    # and should return the upcoming season
    if today.month > 7 and today.month < 10:
        return today.year + 1
    # If we're in October or later, the season has started for the next year
    elif today.month >= 10:
        return today.year + 1
    # If we're in January through June, we're in the current season
    else:
        return today.year


def get_season_for_date(date: datetime) -> int:
    """
    Determine the NBA season for a specific date
    
    Args:
        date: Date to get season for
        
    Returns:
        Season year (e.g., 2024 for the 2023-2024 season)
    """
    # If date is between Oct and Dec, season is next year
    if date.month >= 10:
        return date.year + 1
    # If date is between Jan and June, season is current year
    else:
        return date.year


def fetch_all_teams(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch all active NBA teams from the API with robust error handling
    
    Args:
        force_refresh: Force API fetch even if cached data exists
        
    Returns:
        List of team dictionaries with comprehensive team data
    """
    logger.info("Fetching all NBA teams from API")
    
    # Define cache file path with current season info for better organization
    current_season = get_current_season()
    cache_dir = DATA_DIR / f"teams_cache/{current_season}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = cache_dir / "all_teams.json"
    
    # Check if cache is fresh (less than 24 hours old) unless force_refresh
    if not force_refresh and cache_file.exists():
        cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        cache_age = datetime.now() - cache_mtime
        cache_is_fresh = cache_age.total_seconds() < 24 * 60 * 60  # 24 hours
        
        if cache_is_fresh:
            try:
                with open(cache_file, 'r') as f:
                    teams = json.load(f)
                logger.info(f"Loaded {len(teams)} teams from cache for season {current_season}")
                return teams
            except Exception as e:
                logger.warning(f"Failed to load teams from cache: {str(e)}")
    
    # API endpoint for teams
    teams_endpoint = f"{API_BASE_URL}/teams"
    
    # Get API key from environment
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if not api_key:
        raise ValueError("BALLDONTLIE_API_KEY environment variable not set")
    
    headers = {"Authorization": api_key}
    
    # Make API request with retries
    response = None
    retry_count = 0
    
    while retry_count < MAX_RETRIES:
        try:
            # Request all teams with a large per_page parameter
            params = {"per_page": 100}  # Request all teams, API will cap at max available
            response = requests.get(
                teams_endpoint,
                params=params,
                headers=headers,
                timeout=DEFAULT_API_TIMEOUT
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            break  # Exit the retry loop if successful
        except requests.RequestException as e:
            retry_count += 1
            logger.warning(f"API request for teams failed (attempt {retry_count}): {str(e)}")
            
            if retry_count >= MAX_RETRIES:
                logger.error(f"Failed to fetch teams after {MAX_RETRIES} attempts")
                raise ValueError(f"Failed to fetch teams from API: {str(e)}")
            
            # Wait before retrying
            import time
            time.sleep(RETRY_DELAY * retry_count)  # Exponential backoff
    
    # Parse the response
    if response and response.status_code == 200:
        data = response.json()
        teams = data.get('data', [])
        
        if not teams:
            logger.error("No teams returned from API")
            raise ValueError("API returned no teams")
        
        logger.info(f"Successfully fetched {len(teams)} teams from API")
        
        # Cache the response for future use
        try:
            with open(cache_file, 'w') as f:
                json.dump(teams, f)
            logger.info(f"Cached teams data to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache teams data: {str(e)}")
        
        return teams
    else:
        error_msg = f"Failed to fetch teams: HTTP {response.status_code if response else 'No response'}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def fetch_team_stats() -> Dict[int, Dict[str, Any]]:
    """
    Fetch comprehensive team statistics for all NBA teams with proper error handling
    
    Returns:
        Dictionary of team statistics by team ID
    """
    logger.info("Fetching team statistics for all NBA teams")
    
    # API endpoints
    teams_endpoint = f"{API_BASE_URL}/teams"
    stats_endpoint = f"{API_BASE_URL}/stats"
    
    # Get API key from environment or configuration
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    headers = {}
    
    if api_key:
        headers["Authorization"] = api_key
        logger.info("Using BallDontLie API key from environment")
    else:
        logger.warning("No BallDontLie API key found. Requests may be rate limited.")
    
    team_stats = {}
    active_teams = fetch_all_teams()
    
    try:
        # Now fetch stats for each team
        current_season = get_current_season()
        
        for team in active_teams:
            team_id = team.get('id')
            if not team_id:
                continue
            
            # Initialize team stats entry
            team_stats[team_id] = {
                'id': team_id,
                'name': team.get('name', 'Unknown'),
                'abbreviation': team.get('abbreviation', 'Unknown'),
                'conference': team.get('conference', 'Unknown'),
                'division': team.get('division', 'Unknown'),
            }
            
            try:
                # Fetch team's season stats
                params = {
                    "seasons[]": current_season,
                    "team_ids[]": team_id,
                    "per_page": 100
                }
                
                stats_response = None
                retry_count = 0
                
                while retry_count < MAX_RETRIES:
                    try:
                        stats_response = requests.get(
                            stats_endpoint, 
                            params=params,
                            headers=headers,
                            timeout=DEFAULT_API_TIMEOUT
                        )
                        stats_response.raise_for_status()
                        break
                    except requests.RequestException as e:
                        retry_count += 1
                        logger.warning(f"API request failed for team {team_id} stats (attempt {retry_count}): {str(e)}")
                        
                        if retry_count >= MAX_RETRIES:
                            logger.error(f"Failed to fetch stats for team {team_id} after {MAX_RETRIES} attempts")
                            raise
                        
                        import time
                        time.sleep(RETRY_DELAY * retry_count)
                
                if stats_response and stats_response.status_code == 200:
                    stats_data = stats_response.json()
                    team_games = stats_data.get('data', [])
                    
                    if team_games:
                        # Calculate aggregate stats from all games
                        games_played = len(team_games)
                        wins = sum(1 for g in team_games if g.get('team', {}).get('id') == team_id and g.get('win'))
                        losses = games_played - wins
                        win_pct = wins / games_played if games_played > 0 else 0.0
                        
                        # Calculate average stats
                        points_pg = sum(g.get('pts', 0) for g in team_games) / games_played if games_played > 0 else 0
                        opp_points_pg = sum(g.get('opp_pts', 0) for g in team_games) / games_played if games_played > 0 else 0
                        
                        # Calculate offensive and defensive ratings
                        # These are simplified versions - in a production environment you'd use more sophisticated formulas
                        possessions = sum((g.get('fga', 0) + 0.4 * g.get('fta', 0) - 1.07 * (g.get('orb', 0) / (g.get('orb', 0) + g.get('opp_drb', 0))) * (g.get('fga', 0) - g.get('fg', 0)) + g.get('tov', 0)) for g in team_games)
                        offensive_rating = sum(g.get('pts', 0) for g in team_games) * 100 / possessions if possessions > 0 else 0
                        defensive_rating = sum(g.get('opp_pts', 0) for g in team_games) * 100 / possessions if possessions > 0 else 0
                        
                        # Calculate pace (possessions per 48 minutes)
                        minutes = sum(g.get('min', 0) for g in team_games)
                        pace = 48 * possessions / minutes if minutes > 0 else 0
                        
                        # Store aggregated stats
                        stats_data = {
                            'games_played': games_played,
                            'wins': wins,
                            'losses': losses,
                            'win_pct': win_pct,
                            'points_pg': points_pg,
                            'opp_points_pg': opp_points_pg,
                            'offensive_rating': offensive_rating,
                            'defensive_rating': defensive_rating,
                            'pace': pace,
                            'net_rating': offensive_rating - defensive_rating,
                            'streak': 0  # Would need game-by-game data to calculate
                        }
                        
                        # Update team stats with the calculated values
                        team_stats[team_id].update(stats_data)
                        
                        # Cache the team's stats
                        cache_file = DATA_DIR / f"team_stats_{team_id}.json"
                        try:
                            with open(cache_file, 'w') as f:
                                json.dump(team_stats[team_id], f)
                            logger.info(f"Cached stats for team {team_id} to {cache_file}")
                        except Exception as e:
                            logger.warning(f"Failed to cache stats for team {team_id}: {str(e)}")
                    
                    logger.info(f"Loaded stats for {team.get('name')}")
                else:
                    logger.warning(f"Error fetching stats for {team.get('name')}: {stats_response.status_code if stats_response else 'No response'}")
                    continue
            
            except Exception as e:
                logger.warning(f"Error fetching stats for {team.get('name')}: {str(e)}")
                continue
        
        # Check if we have enough teams with stats
        if len(team_stats) < 5:
            error_msg = f"Insufficient team data: Only found stats for {len(team_stats)} teams. API may be experiencing issues."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Successfully processed stats for {len(team_stats)} teams")
        return team_stats
    
    except Exception as e:
        logger.error(f"Error fetching team stats: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to fetch team statistics: {str(e)}")


def calculate_rest_days(team_id: int, game_date: datetime) -> int:
    """
    Calculate the number of days rest for a team based on their schedule
    
    Args:
        team_id: Team ID to check
        game_date: Date of the current game
        
    Returns:
        Number of days rest (0 for back-to-back)
    """
    try:
        # Get the team's full schedule from the API
        api_key = os.environ.get('BALLDONTLIE_API_KEY')
        if not api_key:
            raise ValueError("BALLDONTLIE_API_KEY environment variable not set")
        
        # Format the game date for API query
        game_date_str = game_date.strftime("%Y-%m-%d")
        # Get season info based on the date
        season = get_season_for_date(game_date)
        
        # Define cache file path with season info for better organization
        cache_dir = DATA_DIR / f"schedule_cache/{season}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = cache_dir / f"team_schedule_{team_id}.json"
        
        # Check if cache is fresh (less than 24 hours old)
        cache_is_fresh = False
        if cache_file.exists():
            cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            cache_age = datetime.now() - cache_mtime
            cache_is_fresh = cache_age.total_seconds() < 24 * 60 * 60  # 24 hours
        
        schedule = []
        # Use cache if it exists and is fresh
        if cache_file.exists() and cache_is_fresh:
            try:
                with open(cache_file, 'r') as f:
                    schedule = json.load(f)
                logger.info(f"Loaded schedule from cache for team {team_id}")
            except Exception as e:
                logger.warning(f"Failed to load schedule from cache for team {team_id}: {str(e)}")
                # Cache exists but couldn't be loaded, we'll fetch from API
                cache_is_fresh = False
        
        # Fetch from API if no cache or cache is stale
        if not cache_is_fresh:
            # We need to fetch a date range to find games before our target date
            # Start 30 days before our target date to have enough history
            start_date = (game_date - timedelta(days=30)).strftime("%Y-%m-%d")
            
            # Build API request for team games
            team_schedule_url = f"https://api.balldontlie.io/v1/games"
            params = {
                "team_ids[]" : team_id,
                "start_date": start_date,
                "end_date": game_date_str,
                "per_page": 100
            }
            headers = {"Authorization": api_key}
            
            response = requests.get(team_schedule_url, params=params, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"API Error: Unable to fetch schedule. Status: {response.status_code}, Response: {response.text}")
                raise Exception(f"API error: {response.status_code}")
            
            data = response.json()
            
            # Extract game dates
            schedule = [{
                'date': game['date'], 
                'home_team_id': game['home_team']['id'],
                'away_team_id': game['visitor_team']['id']
            } for game in data.get('data', [])]            
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(schedule, f)
            logger.info(f"Fetched and cached schedule for team {team_id}")
        
        # Now find the previous game date before the target game date
        previous_games = [
            datetime.fromisoformat(g['date']) for g in schedule 
            if datetime.fromisoformat(g['date']) < game_date
        ]
        
        if previous_games:
            most_recent = max(previous_games)
            days_rest = (game_date - most_recent).days
            logger.info(f"Team {team_id} has {days_rest} days rest before game on {game_date_str}")
            return days_rest
        else:
            logger.warning(f"No previous games found for team {team_id} before {game_date_str}")
            # This might be the first game of the season or our API call didn't go back far enough
            # In a production system, you'd want to extend the search range or query a more comprehensive DB
            # For now we'll raise an exception to signal this issue
            raise ValueError(f"No previous games found for team {team_id} before {game_date_str}")
    
    except Exception as e:
        logger.error(f"Error calculating rest days for team {team_id}: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't return a default value - we want to be explicit about failures
        raise ValueError(f"Failed to calculate rest days: {str(e)}")


def is_back_to_back(team_id: int, game_date: datetime) -> bool:
    """
    Check if a team is playing on the second night of a back-to-back
    
    Args:
        team_id: Team ID to check
        game_date: Date of the current game
        
    Returns:
        True if team is on back-to-back, False otherwise
    """
    try:
        rest_days = calculate_rest_days(team_id, game_date)
        return rest_days == 0
    except Exception as e:
        logger.error(f"Error determining back-to-back status for team {team_id}: {str(e)}")
        # In production, we should propagate errors rather than guessing
        raise ValueError(f"Cannot determine back-to-back status: {str(e)}")
