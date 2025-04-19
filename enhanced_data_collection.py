#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced NBA Data Collection

This script implements comprehensive data collection for the NBA prediction system
with a focus on ensuring complete defensive ratings for all teams. It uses the
entire season's worth of data and multiple API endpoints to ensure data completeness.

Specific improvements:
1. Collects data for the entire season rather than just recent games
2. Uses alternative API endpoints for more comprehensive team data
3. Calculates defensive ratings from raw game data for consistency
4. Implements strict validation to ensure data completeness

Usage:
    python enhanced_data_collection.py [--validate-only] [--full-season] [--force-season]

Options:
    --validate-only    Only validate existing data without fetching new data
    --full-season      Fetch data for the entire season (default: True)
    --force-season     Force specific season year (e.g., 2024 for 2024-2025 season)

Author: Cascade
Date: April 2025
"""

import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
import os

from nba_algorithm.data.team_data import fetch_team_stats
from nba_algorithm.data.historical_collector import fetch_historical_games
from nba_algorithm.utils.logger import setup_logging
from nba_algorithm.data.team_data import fetch_all_teams
from nba_algorithm.data.nba_teams import get_active_nba_teams

# Set up logging
logger = setup_logging("enhanced_data_collection")


def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Enhanced NBA Data Collection")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing data without fetching new data"
    )
    parser.add_argument(
        "--full-season",
        action="store_true",
        default=True,
        help="Fetch data for the entire season"
    )
    parser.add_argument(
        "--force-season",
        type=int,
        help="Force specific season year (e.g., 2024 for 2024-2025 season)"
    )
    return parser.parse_args()


def get_current_nba_season():
    """
    Get current NBA season with data availability verification
    
    Returns:
        int: Current NBA season year with data
    """
    try:
        today = datetime.now()
        current_year = today.year
        
        # Determine current season based on date
        if today.month >= 10:  # October-December
            season_year = current_year + 1
        else:  # January-September
            season_year = current_year
        
        # Test if we can get data for this season
        api_key = os.environ.get("BALLDONTLIE_API_KEY")
        if not api_key:
            logger.error("No BallDontLie API key found in environment variables")
            return season_year
        
        # Test API call to verify season data
        headers = {"Authorization": api_key}
        test_params = {"seasons[]": season_year, "per_page": 1}
        
        response = requests.get(
            "https://api.balldontlie.io/v1/games",
            params=test_params,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                logger.info(f"Verified data availability for NBA season {season_year-1}-{season_year}")
                return season_year
            else:
                logger.warning(f"No data found for season {season_year}, falling back to previous season")
                return season_year - 1
        else:
            logger.warning(f"API error checking season {season_year}: {response.status_code}")
            return season_year - 1
    except Exception as e:
        logger.error(f"Error determining current season: {e}")
        # Default to current season
        return current_year


def validate_team_data(team_stats):
    """
    Validate team statistics data for completeness
    
    Args:
        team_stats: Dictionary of team statistics by team ID
        
    Returns:
        tuple: (is_valid, missing_teams)
    """
    missing_teams = []
    for team_id, stats in team_stats.items():
        team_name = stats.get('name', f"ID: {team_id}")
        
        # Check for critical metrics
        if 'defensive_rating' not in stats:
            logger.warning(f"Team {team_name} is missing defensive rating")
            missing_teams.append(team_name)
        elif 'offensive_rating' not in stats:
            logger.warning(f"Team {team_name} is missing offensive rating")
            missing_teams.append(team_name)
        elif 'opp_points_pg' not in stats and 'opp_pts' not in stats:
            logger.warning(f"Team {team_name} is missing opponent points per game")
            missing_teams.append(team_name)
    
    if missing_teams:
        logger.error(f"Found {len(missing_teams)} teams with missing data: {', '.join(missing_teams)}")
        return False, missing_teams
    else:
        logger.info("All teams have complete data!")
        return True, []


def collect_enhanced_team_data(season_year=2023):
    """
    Collect enhanced team data with focus on defensive ratings
    
    Args:
        season_year: The NBA season year to collect data for (e.g., 2023 for the 2022-2023 season)
    
    Returns:
        dict: Team statistics by team ID
    """
    try:
        # Get valid NBA teams for filtering
        teams_data = fetch_all_teams()
        valid_nba_teams = get_active_nba_teams(teams_data)
        logger.info(f"Got {len(valid_nba_teams)} valid NBA teams")
        
        # Use our improved fetch_team_stats function with entire season data
        # Explicitly setting the season year to 2023 (2022-2023 NBA season) to ensure we have data
        team_stats = fetch_team_stats(valid_nba_teams, use_entire_season=True, season_year=season_year)
        
        if not team_stats or len(team_stats) < 5:
            logger.error("Failed to fetch sufficient team statistics")
            return None
        
        logger.info(f"Successfully collected enhanced data for {len(team_stats)} teams")
        return team_stats
    except Exception as e:
        logger.error(f"Error collecting enhanced team data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def collect_enhanced_game_data(season_year=2023):
    """
    Collect enhanced game data for the entire season
    
    Args:
        season_year: The NBA season year to collect data for (e.g., 2023 for the 2022-2023 season)
    
    Returns:
        list: Historical games
    """
    try:
        # Use our improved fetch_historical_games function with entire season data
        # Force the season year to be 2023 (2022-2023 NBA season) to ensure we have data
        # The start_date will be set to October 1st of the previous year (2022)
        historical_games = fetch_historical_games(fetch_full_season=True, season_year=season_year)
        
        if not historical_games:
            logger.error("Failed to fetch any historical games")
            return None
        
        logger.info(f"Successfully collected {len(historical_games)} historical games")
        return historical_games
    except Exception as e:
        logger.error(f"Error collecting enhanced game data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    """
    Main entry point for enhanced data collection
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    
    logger.info("Starting enhanced data collection")
    logger.info(f"Full season mode: {args.full_season}")
    
    if args.force_season:
        season_year = args.force_season
    else:
        season_year = get_current_nba_season()
    
    logger.info(f"Using season year: {season_year}")
    
    if not args.validate_only:
        # Collect enhanced team data
        logger.info("Collecting enhanced team data...")
        team_stats = collect_enhanced_team_data(season_year)
        
        if not team_stats:
            logger.error("Failed to collect enhanced team data")
            return 1
        
        # Collect enhanced game data
        logger.info("Collecting enhanced game data...")
        historical_games = collect_enhanced_game_data(season_year)
        
        if not historical_games:
            logger.error("Failed to collect enhanced game data")
            return 1
        
        # Validate the collected data
        logger.info("Validating collected data...")
        is_valid, missing_teams = validate_team_data(team_stats)
        
        if not is_valid:
            logger.warning("Data validation failed - some teams are missing critical metrics")
            logger.warning(f"Teams missing data: {', '.join(missing_teams)}")
            print("\nData collection completed but some teams are still missing critical metrics.")
            print(f"Teams missing data: {', '.join(missing_teams)}")
            print("This may be due to insufficient game data for these teams.")
            print("The prediction system will exclude these teams from predictions.\n")
            return 1
    else:
        # Only validate existing data
        logger.info("Validating existing team data...")
        
        # First load existing team stats from cached files
        from nba_algorithm.utils.config import DATA_DIR
        import os
        import json
        
        team_stats = {}
        team_files = [f for f in os.listdir(DATA_DIR) if f.startswith('team_stats_') and f.endswith('.json')]
        
        for file in team_files:
            try:
                with open(DATA_DIR / file, 'r') as f:
                    team_data = json.load(f)
                    team_id = int(file.replace('team_stats_', '').replace('.json', ''))
                    team_stats[team_id] = team_data
            except Exception as e:
                logger.error(f"Error loading team data from {file}: {str(e)}")
        
        if not team_stats:
            logger.error("No team data files found to validate")
            return 1
        
        # Validate the loaded data
        is_valid, missing_teams = validate_team_data(team_stats)
        
        if not is_valid:
            logger.warning("Data validation failed - some teams are missing critical metrics")
            logger.warning(f"Teams missing data: {', '.join(missing_teams)}")
            print("\nExisting data validation failed - some teams are missing critical metrics.")
            print(f"Teams missing data: {', '.join(missing_teams)}")
            print("Please run this script without --validate-only to collect enhanced data.\n")
            return 1
    
    logger.info("Enhanced data collection completed successfully")
    print("\nEnhanced data collection completed successfully")
    print("All teams now have complete defensive ratings and required metrics")
    print("The prediction system will now use this data for more accurate predictions.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
