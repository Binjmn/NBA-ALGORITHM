"""
Data Collection Tasks

This module provides task functions for collecting data from the APIs that can be
scheduled by the PredictionScheduler. Each function is designed to be called on a
specific schedule to collect different types of NBA data.

These tasks integrate with both the BallDontLie and The Odds API clients to retrieve
and store data for use in the prediction pipeline.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

import pytz

from src.api.balldontlie_client import BallDontLieClient
from src.api.theodds_client import TheOddsClient
from src.utils.data_processor import NBADataProcessor
from config.season_config import SeasonPhase

# Configure logging
logger = logging.getLogger(__name__)

# Initialize API clients (cached as module-level variables for reuse)
_balldontlie_client = None
_theodds_client = None
_data_processor = None


def _get_clients():
    """
    Get or initialize API clients
    
    Returns:
        tuple: (BallDontLieClient, TheOddsClient, NBADataProcessor)
    """
    global _balldontlie_client, _theodds_client, _data_processor
    
    if _balldontlie_client is None:
        _balldontlie_client = BallDontLieClient()
        
    if _theodds_client is None:
        _theodds_client = TheOddsClient()
        
    if _data_processor is None:
        _data_processor = NBADataProcessor(
            balldontlie_client=_balldontlie_client,
            theodds_client=_theodds_client
        )
        
    return _balldontlie_client, _theodds_client, _data_processor


def collect_daily_data(
    season_year: int,
    season_phase: SeasonPhase,
    force_refresh: bool = True
) -> Dict[str, int]:
    """
    Collect daily NBA data for teams, players, and games
    
    This function is designed to be called once per day in the morning
    to collect fresh data for the current day's analysis and predictions.
    
    Args:
        season_year (int): Current NBA season year
        season_phase (SeasonPhase): Current NBA season phase
        force_refresh (bool): Whether to force refresh cached data
        
    Returns:
        Dict[str, int]: Counts of collected data items
    """
    logger.info(f"Collecting daily NBA data for {season_year}")
    
    # Initialize data counts
    data_counts = {
        'teams': 0,
        'players': 0,
        'games': 0,
        'standings': 0
    }
    
    try:
        # Get API clients
        balldontlie, theodds, processor = _get_clients()
        
        # Create data directory for today
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern)
        daily_dir = os.path.join(
            'data', 
            'daily',
            f"{today.strftime('%Y-%m-%d')}"
        )
        os.makedirs(daily_dir, exist_ok=True)
        
        # Collect team data
        logger.info("Collecting team data")
        teams = balldontlie.get_teams(force_refresh=force_refresh)
        data_counts['teams'] = len(teams.get('data', []))
        processor.save_data("teams", teams)
        
        # Collect active players
        logger.info("Collecting active player data")
        all_players = []
        page = 1
        while True:
            players = balldontlie.get_active_players(
                page=page, 
                per_page=100,
                force_refresh=force_refresh
            )
            player_data = players.get('data', [])
            all_players.extend(player_data)
            
            if len(player_data) < 100:
                break
            page += 1
            
        data_counts['players'] = len(all_players)
        processor.save_data("active_players", {'data': all_players})
        
        # Collect today's games with odds
        logger.info("Collecting today's games with odds")
        games_with_odds = processor.get_todays_games_with_odds()
        data_counts['games'] = len(games_with_odds)
        processor.save_data("todays_games", {'data': games_with_odds})
        
        # Collect current standings
        logger.info("Collecting current standings")
        standings = balldontlie.get_standings(season=season_year)
        data_counts['standings'] = len(standings.get('data', []))
        processor.save_data("standings", standings)
        
        logger.info(f"Daily data collection complete: {data_counts}")
        return data_counts
    
    except Exception as e:
        logger.error(f"Error collecting daily data: {e}")
        raise


def update_odds_data(
    season_year: int,
    season_phase: SeasonPhase,
    markets: Optional[List[str]] = None
) -> int:
    """
    Update betting odds data
    
    This function is designed to be called multiple times per day
    to refresh odds data as they change.
    
    Args:
        season_year (int): Current NBA season year
        season_phase (SeasonPhase): Current NBA season phase
        markets (Optional[List[str]]): Specific markets to update
        
    Returns:
        int: Number of games with updated odds
    """
    logger.info(f"Updating odds data for {season_year}")
    
    try:
        # Get API clients
        balldontlie, theodds, processor = _get_clients()
        
        # Create data directory for today
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern)
        odds_dir = os.path.join(
            'data', 
            'odds',
            f"{today.strftime('%Y-%m-%d')}"
        )
        os.makedirs(odds_dir, exist_ok=True)
        
        # Get latest odds from BallDontLie
        bdl_odds = balldontlie.get_todays_odds()
        
        # Get latest odds from The Odds API (always force refresh)
        theodds_odds = theodds.get_todays_odds(markets=markets)
        
        # Save combined odds data with timestamp
        timestamp = today.strftime('%Y%m%d_%H%M%S')
        combined_odds = {
            'timestamp': timestamp,
            'balldontlie_odds': bdl_odds,
            'theodds_odds': theodds_odds
        }
        
        # Save to timestamped file
        odds_file = os.path.join(odds_dir, f"odds_{timestamp}.json")
        processor.save_data(f"odds_{timestamp}", combined_odds)
        
        # Also update the latest odds
        processor.save_data("latest_odds", combined_odds)
        
        num_games = len(theodds_odds)
        logger.info(f"Updated odds for {num_games} games")
        return num_games
    
    except Exception as e:
        logger.error(f"Error updating odds data: {e}")
        raise


def update_live_game_data(
    season_year: int,
    season_phase: SeasonPhase
) -> Dict[str, Any]:
    """
    Update live game data
    
    This function is designed to be called frequently during games
    to collect real-time updates.
    
    Args:
        season_year (int): Current NBA season year
        season_phase (SeasonPhase): Current NBA season phase
        
    Returns:
        Dict[str, Any]: Summary of collected live data
    """
    logger.info(f"Updating live game data for {season_year}")
    
    try:
        # Get API clients
        balldontlie, theodds, processor = _get_clients()
        
        # Create live data directory
        eastern = pytz.timezone('US/Eastern')
        today = datetime.now(eastern)
        live_dir = os.path.join(
            'data', 
            'live',
            f"{today.strftime('%Y-%m-%d')}"
        )
        os.makedirs(live_dir, exist_ok=True)
        
        # Get today's games
        games = balldontlie.get_todays_games()
        game_data = games.get('data', [])
        
        # Get live scores from The Odds API
        live_scores = theodds.get_live_scores()
        
        # Process each active game
        live_data = {}
        timestamp = today.strftime('%Y%m%d_%H%M%S')
        
        for game in game_data:
            game_id = game['id']
            status = game.get('status')
            
            # Only process active games
            if status in ['1st Qtr', '2nd Qtr', 'Halftime', '3rd Qtr', '4th Qtr', 'OT']:
                logger.info(f"Processing live data for game {game_id}")
                
                # Get live box score
                try:
                    box_score = processor.get_live_game_data(game_id)
                    
                    # Add to live data
                    game_key = f"game_{game_id}"
                    live_data[game_key] = box_score
                    
                    # Save individual game data
                    processor.save_data(f"live_{game_key}_{timestamp}", box_score)
                except Exception as e:
                    logger.error(f"Error getting live data for game {game_id}: {e}")
        
        # Save all live data together
        if live_data:
            all_live_data = {
                'timestamp': timestamp,
                'games': live_data,
                'live_scores': live_scores
            }
            processor.save_data(f"live_data_{timestamp}", all_live_data)
            processor.save_data("latest_live_data", all_live_data)
            
            logger.info(f"Updated live data for {len(live_data)} active games")
            return {
                'active_games': len(live_data),
                'timestamp': timestamp
            }
        else:
            logger.info("No active games found")
            return {
                'active_games': 0,
                'timestamp': timestamp
            }
    
    except Exception as e:
        logger.error(f"Error updating live game data: {e}")
        raise
