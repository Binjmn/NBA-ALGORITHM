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

from ..utils.config import DATA_DIR, DEFAULT_API_TIMEOUT, MAX_RETRIES, RETRY_DELAY

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


def fetch_team_stats(valid_nba_teams: Optional[Dict[int, str]] = None, use_entire_season: bool = True, season_year: int = None) -> Dict[int, Dict[str, Any]]:
    """
    Fetch comprehensive team statistics for all NBA teams with proper error handling
    
    This function fetches team statistics using multiple API endpoints to ensure complete data,
    especially for defensive ratings. It uses game-level data to calculate advanced metrics
    rather than relying on default or fallback values. If data is insufficient, it returns None
    for those metrics to ensure predictions are only made with reliable data.
    
    Args:
        valid_nba_teams: Optional dictionary of valid NBA team IDs to team names for filtering
        use_entire_season: Whether to use data from the entire season (True) or recent games only
        season_year: Specific NBA season year to fetch data for (e.g., 2023 for 2022-2023 season)
        
    Returns:
        Dictionary of team statistics by team ID
    """
    logger.info("Fetching comprehensive team statistics for NBA teams")
    team_stats = {}
    
    # API endpoints
    teams_endpoint = f"{API_BASE_URL}/teams"
    stats_endpoint = f"{API_BASE_URL}/stats"
    games_endpoint = f"{API_BASE_URL}/games"
    standings_endpoint = f"{API_BASE_URL}/standings"
    
    # Get API key from environment or configuration
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    headers = {}
    
    if api_key:
        headers["Authorization"] = api_key
        logger.info("Using BallDontLie API key from environment")
    else:
        logger.error("No BallDontLie API key found. Required for reliable data collection.")
        raise ValueError("Missing API key for BallDontLie API. Cannot fetch complete team statistics.")
    
    # Get the current season or use provided season year
    current_season = season_year if season_year is not None else get_current_season()
    
    # In production, we always want the most current season data
    logger.info(f"Using data from NBA season: {current_season-1}-{current_season} (Season ID: {current_season})")
    
    # Use test call to verify we can get data for this season
    test_params = {
        "seasons[]": current_season,
        "per_page": 1
    }
    
    try:
        test_response = requests.get(
            games_endpoint,
            params=test_params,
            headers=headers,
            timeout=DEFAULT_API_TIMEOUT
        )
        test_response.raise_for_status()
        test_data = test_response.json()
        
        if not test_data.get('data'):
            logger.warning(f"No data found for season {current_season}, trying previous season")
            current_season -= 1
            logger.info(f"Falling back to season: {current_season-1}-{current_season} (Season ID: {current_season})")
    except Exception as e:
        logger.warning(f"Error testing season data availability: {str(e)}")
        # Continue with original season - the full data collection will handle errors
    
    # Now fetch stats for each team
    try:
        # Get all teams
        teams = fetch_all_teams()
        
        if len(teams) == 0:
            logger.error("Failed to fetch any teams from the API")
            return {}
        
        # If valid_nba_teams is provided, filter to just those teams
        filtered_teams = []
        if valid_nba_teams:
            logger.info(f"Filtering team stats to only include {len(valid_nba_teams)} valid NBA teams")
            for team in teams:
                team_id = team.get('id')
                if team_id in valid_nba_teams:
                    filtered_teams.append(team)
            logger.info(f"Found {len(filtered_teams)} teams after filtering")
        else:
            filtered_teams = teams
            logger.info(f"Processing all {len(filtered_teams)} teams without filtering")
        
        # First, fetch standings data for all teams to get season records
        standings_params = {
            "season": current_season,
            "per_page": 100
        }
        
        standings_response = None
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            try:
                standings_response = requests.get(
                    standings_endpoint,
                    params=standings_params,
                    headers=headers,
                    timeout=DEFAULT_API_TIMEOUT
                )
                standings_response.raise_for_status()
                break
            except requests.RequestException as e:
                retry_count += 1
                logger.warning(f"API request failed for standings (attempt {retry_count}): {str(e)}")
                
                if retry_count >= MAX_RETRIES:
                    logger.error(f"Failed to fetch standings after {MAX_RETRIES} attempts")
                    raise
                
                import time
                time.sleep(RETRY_DELAY * retry_count)
        
        standings_data = {}
        if standings_response and standings_response.status_code == 200:
            all_standings = standings_response.json().get('data', [])
            for standing in all_standings:
                team_info = standing.get('team', {})
                team_id = team_info.get('id')
                if team_id:
                    standings_data[team_id] = standing
            
            logger.info(f"Fetched standings data for {len(standings_data)} teams")
        else:
            logger.warning("Failed to fetch standings data")
        
        # Get historical games for each team for more complete stats (entire season)
        games_by_team = {}
        if use_entire_season:
            # Fetch games for the entire season
            start_date = datetime(current_season - 1, 10, 1)  # Season starts in October of previous year
            end_date = datetime.now()
        else:
            # Fetch games for the last 90 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        logger.info(f"Fetching games between {start_date_str} and {end_date_str}")
        
        # Process each team
        for team in filtered_teams:
            team_id = team.get('id')
            if not team_id:
                continue
                
            team_name = team.get('name', 'Unknown')
            logger.info(f"Processing stats for {team_name} (ID: {team_id})")
            
            # Initialize team stats entry with base info
            team_stats[team_id] = {
                'id': team_id,
                'name': team_name,
                'abbreviation': team.get('abbreviation', 'Unknown'),
                'conference': team.get('conference', 'Unknown'),
                'division': team.get('division', 'Unknown'),
            }
            
            # Add standings data if available
            if team_id in standings_data:
                standing = standings_data[team_id]
                team_stats[team_id].update({
                    'wins': standing.get('wins', 0),
                    'losses': standing.get('losses', 0),
                    'win_pct': standing.get('wins', 0) / max(standing.get('wins', 0) + standing.get('losses', 0), 1),
                    'home_record': standing.get('home_record', '0-0'),
                    'road_record': standing.get('road_record', '0-0'),
                    'conference_record': standing.get('conference_record', '0-0'),
                    'conference_rank': standing.get('conference_rank', 0),
                    'division_record': standing.get('division_record', '0-0'),
                    'division_rank': standing.get('division_rank', 0),
                })
            
            # Fetch team's games for advanced stats calculation
            games_params = {
                "seasons[]": current_season,
                "team_ids[]": team_id,
                "per_page": 100,
                "start_date": start_date_str,
                "end_date": end_date_str,
            }
            
            # Get games for this team
            all_team_games = []
            page = 1
            while True:
                games_params["page"] = page
                
                games_response = None
                retry_count = 0
                
                while retry_count < MAX_RETRIES:
                    try:
                        games_response = requests.get(
                            games_endpoint,
                            params=games_params,
                            headers=headers,
                            timeout=DEFAULT_API_TIMEOUT
                        )
                        games_response.raise_for_status()
                        break
                    except requests.RequestException as e:
                        retry_count += 1
                        logger.warning(f"API request failed for team {team_id} games (attempt {retry_count}): {str(e)}")
                        
                        if retry_count >= MAX_RETRIES:
                            logger.error(f"Failed to fetch games for team {team_id} after {MAX_RETRIES} attempts")
                            break
                        
                        import time
                        time.sleep(RETRY_DELAY * retry_count)
                
                if games_response and games_response.status_code == 200:
                    games_data = games_response.json()
                    games = games_data.get('data', [])
                    all_team_games.extend(games)
                    
                    # Check if we have more pages
                    meta = games_data.get('meta', {})
                    total_pages = meta.get('total_pages', 1)
                    
                    if page >= total_pages or not games:
                        break
                    
                    page += 1
                else:
                    logger.warning(f"Failed to fetch games for team {team_id}")
                    break
            
            logger.info(f"Fetched {len(all_team_games)} games for {team_name}")
            games_by_team[team_id] = all_team_games
            
            # Now fetch stats for each game to get detailed metrics
            team_game_stats = []
            for game in all_team_games:
                game_id = game.get('id')
                if not game_id:
                    continue
                
                # Fetch stats for this specific game
                stats_params = {
                    "game_ids[]": game_id,
                    "team_ids[]": team_id,
                    "per_page": 100
                }
                
                stats_response = None
                retry_count = 0
                
                while retry_count < MAX_RETRIES:
                    try:
                        stats_response = requests.get(
                            stats_endpoint,
                            params=stats_params,
                            headers=headers,
                            timeout=DEFAULT_API_TIMEOUT
                        )
                        stats_response.raise_for_status()
                        break
                    except requests.RequestException as e:
                        retry_count += 1
                        logger.warning(f"API request failed for game {game_id} stats (attempt {retry_count}): {str(e)}")
                        
                        if retry_count >= MAX_RETRIES:
                            logger.error(f"Failed to fetch stats for game {game_id} after {MAX_RETRIES} attempts")
                            break
                        
                        import time
                        time.sleep(RETRY_DELAY * retry_count)
                
                if stats_response and stats_response.status_code == 200:
                    stats_data = stats_response.json()
                    game_stats = stats_data.get('data', [])
                    team_game_stats.extend(game_stats)
            
            # Aggregate stats from all games
            if all_team_games:
                # Calculate wins, losses, and basic stats
                games_played = len(all_team_games)
                wins = 0
                losses = 0
                team_pts = 0
                opp_pts = 0
                
                for game in all_team_games:
                    home_team = game.get('home_team', {})
                    visitor_team = game.get('visitor_team', {})
                    
                    is_home = home_team.get('id') == team_id
                    team_score = game.get('home_team_score', 0) if is_home else game.get('visitor_team_score', 0)
                    opp_score = game.get('visitor_team_score', 0) if is_home else game.get('home_team_score', 0)
                    
                    # Count wins and losses
                    if team_score > opp_score:
                        wins += 1
                    else:
                        losses += 1
                    
                    # Sum points
                    team_pts += team_score
                    opp_pts += opp_score
                
                # Calculate per-game averages
                points_pg = team_pts / games_played if games_played > 0 else None
                opp_points_pg = opp_pts / games_played if games_played > 0 else None
                
                # Calculate advanced stats from aggregated game data
                # For proper defensive and offensive ratings, we need detailed possession data
                if team_game_stats:
                    # Calculate core stats
                    fgm = sum(g.get('fgm', 0) for g in team_game_stats)
                    fga = sum(g.get('fga', 0) for g in team_game_stats)
                    fg3m = sum(g.get('fg3m', 0) for g in team_game_stats)
                    fg3a = sum(g.get('fg3a', 0) for g in team_game_stats)
                    ftm = sum(g.get('ftm', 0) for g in team_game_stats)
                    fta = sum(g.get('fta', 0) for g in team_game_stats)
                    oreb = sum(g.get('oreb', 0) for g in team_game_stats)
                    dreb = sum(g.get('dreb', 0) for g in team_game_stats)
                    reb = sum(g.get('reb', 0) for g in team_game_stats)
                    ast = sum(g.get('ast', 0) for g in team_game_stats)
                    stl = sum(g.get('stl', 0) for g in team_game_stats)
                    blk = sum(g.get('blk', 0) for g in team_game_stats)
                    turnovers = sum(g.get('turnover', 0) for g in team_game_stats)
                    pf = sum(g.get('pf', 0) for g in team_game_stats)
                    pts = sum(g.get('pts', 0) for g in team_game_stats)
                    
                    # Calculate percentages
                    fg_pct = fgm / fga if fga > 0 else None
                    fg3_pct = fg3m / fg3a if fg3a > 0 else None
                    ft_pct = ftm / fta if fta > 0 else None
                    
                    # Calculate possessions (advanced formula) 
                    # Possession = 0.5 * ((Team FGA + 0.4 * Team FTA - 1.07 * (Team OREB / (Team OREB + Opp DREB)) * (Team FGA - Team FGM) + Team TOV) + 
                    #              (Opp FGA + 0.4 * Opp FTA - 1.07 * (Opp OREB / (Opp OREB + Team DREB)) * (Opp FGA - Opp FGM) + Opp TOV))
                    # Simplified version for our case
                    possessions = 0.5 * ((fga + 0.4 * fta - 1.07 * (oreb / (oreb + dreb) if (oreb + dreb) > 0 else 0) * (fga - fgm) + turnovers) * games_played)
                    
                    # Calculate minutes played (estimate)
                    total_mins = games_played * 48 * 5  # 5 players × 48 minutes per game
                    
                    # Calculate offensive and defensive ratings
                    offensive_rating = team_pts * 100 / possessions if possessions > 0 else None
                    defensive_rating = opp_pts * 100 / possessions if possessions > 0 else None
                    net_rating = offensive_rating - defensive_rating if offensive_rating is not None and defensive_rating is not None else None
                    
                    # Calculate pace (possessions per 48 minutes)
                    pace = 48 * (possessions / games_played) / (total_mins / games_played / 5) if games_played > 0 and total_mins > 0 else None
                    
                    # Update team stats with calculated real values - NO FALLBACKS
                    real_stats = {
                        'games_played': games_played,
                        'wins': wins,
                        'losses': losses,
                        'win_pct': wins / games_played if games_played > 0 else None,
                        'points_pg': points_pg,
                        'opp_points_pg': opp_points_pg,
                        'offensive_rating': offensive_rating,
                        'defensive_rating': defensive_rating,
                        'net_rating': net_rating,
                        'pace': pace,
                        'fg_pct': fg_pct,
                        'fg3_pct': fg3_pct,
                        'ft_pct': ft_pct,
                        'ast': ast / games_played if games_played > 0 else None,
                        'reb': reb / games_played if games_played > 0 else None,
                        'stl': stl / games_played if games_played > 0 else None,
                        'blk': blk / games_played if games_played > 0 else None,
                        'turnover': turnovers / games_played if games_played > 0 else None,
                        'pf': pf / games_played if games_played > 0 else None,
                        'total_points': team_pts,
                        'total_opp_points': opp_pts,
                        'possessions': possessions,
                        'data_quality': 'complete' if defensive_rating is not None else 'incomplete'
                    }
                    
                    # Add aliases to match feature engineering expectations
                    real_stats.update({
                        'pts': points_pg,
                        'opp_pts': opp_points_pg,
                        'w': wins,
                        'l': losses
                    })
                    
                    # Only include non-None values
                    filtered_stats = {k: v for k, v in real_stats.items() if v is not None}
                    
                    # Update team stats with the calculated values
                    team_stats[team_id].update(filtered_stats)
                    
                    # Validate if we have sufficient data for predictions
                    if 'defensive_rating' not in filtered_stats:
                        logger.warning(f"Missing defensive rating for team {team_name} (ID: {team_id}). Insufficient data for prediction.")
                    
                    # Cache the team's stats
                    cache_file = DATA_DIR / f"team_stats_{team_id}.json"
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(team_stats[team_id], f)
                        logger.info(f"Cached stats for team {team_id} to {cache_file}")
                    except Exception as e:
                        logger.warning(f"Failed to cache stats for team {team_id}: {str(e)}")
                
                logger.info(f"Loaded stats for {team_name}")
            else:
                logger.warning(f"No game data available for {team_name}")
        
        # Check if we have enough teams with stats
        if len(team_stats) < 5:
            logger.warning(f"Only found stats for {len(team_stats)} teams, which is suspiciously low")
            return {}
        
        logger.info(f"Successfully fetched stats for {len(team_stats)} teams")
        return team_stats
    
    except Exception as e:
        logger.error(f"Error fetching team stats: {str(e)}")
        return {}
        
    logger.info(f"Successfully fetched stats for {len(team_stats)} teams")
    return team_stats


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
