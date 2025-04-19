#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Historical NBA Data Collector

This module provides functionality to collect historical NBA data from multiple sources:
- BallDontLie API for game results, team stats, and player performance
- The Odds API for historical betting odds

The collected data is structured and stored for model training purposes.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# Import API clients
from src.api.balldontlie_client import BallDontLieClient
from src.api.theodds_client import TheOddsClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path('data/historical')
DATA_DIR.mkdir(parents=True, exist_ok=True)


class HistoricalDataCollector:
    """Collect and organize historical NBA data for model training"""
    
    def __init__(self, bdl_client=None, odds_client=None):
        """
        Initialize historical data collector
        
        Args:
            bdl_client: BallDontLie API client
            odds_client: The Odds API client
        """
        self.bdl_client = bdl_client or BallDontLieClient()
        self.odds_client = odds_client or TheOddsClient()
        self.odds_available = self.odds_client.api_key is not None and len(self.odds_client.api_key) > 0
        
        # Set up data directories
        self.data_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent / 'data'
        self.games_dir = self.data_dir / 'games'
        self.stats_dir = self.data_dir / 'stats'
        self.odds_dir = self.data_dir / 'odds'
        self.teams_dir = self.data_dir / 'teams'
        self.players_dir = self.data_dir / 'players'
        
        # Create directories if they don't exist
        os.makedirs(self.games_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        os.makedirs(self.odds_dir, exist_ok=True)
        os.makedirs(self.teams_dir, exist_ok=True)
        os.makedirs(self.players_dir, exist_ok=True)
        
        # Add the ensure_date_format method for standardizing date formats
        self._date_format_cache = {}  # Cache normalized dates to avoid repeated processing
        self.teams_cache = {}  # Initialize teams_cache attribute
        self.players_cache = {}  # Initialize players_cache attribute
        self.games_cache = {}  # Initialize games_cache attribute
        
    def ensure_date_format(self, date_str: str) -> str:
        """
        Ensure a date string is in YYYY-MM-DD format as required by The Odds API
        Caches results to avoid repeated processing of the same date
        
        Args:
            date_str: Date string in any format
            
        Returns:
            Date string in YYYY-MM-DD format or original string if can't be parsed
        """
        # Return from cache if already processed
        if date_str in self._date_format_cache:
            return self._date_format_cache[date_str]
            
        try:
            # First, clean the date string
            clean_date = date_str.strip()
            
            # If it contains a T (ISO format), extract just the date part
            if 'T' in clean_date:
                clean_date = clean_date.split('T')[0]
                
            # Try to parse with various formats
            parsed_date = None
            formats_to_try = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
            
            for fmt in formats_to_try:
                try:
                    parsed_date = datetime.strptime(clean_date, fmt)
                    break
                except ValueError:
                    continue
                    
            if parsed_date is None:
                logger.warning(f"Could not parse date: {date_str} - using original")
                self._date_format_cache[date_str] = date_str
                return date_str
                
            # Format to YYYY-MM-DD
            formatted_date = parsed_date.strftime('%Y-%m-%d')
            self._date_format_cache[date_str] = formatted_date
            return formatted_date
            
        except Exception as e:
            logger.error(f"Error formatting date {date_str}: {str(e)}")
            self._date_format_cache[date_str] = date_str
            return date_str
    
    def collect_teams(self) -> List[Dict]:
        """Collect all NBA teams"""
        logger.info("Collecting NBA teams data")
        
        response = self.bdl_client.get_teams()
        teams = []
        
        # Check if the response contains data in the expected format
        if isinstance(response, dict) and 'data' in response:
            teams = response['data']
        elif isinstance(response, list):
            teams = response
        
        if not teams:
            logger.error("Failed to collect teams data")
            return []
        
        # Save teams data
        teams_file = self.teams_dir / 'teams.json'
        with teams_file.open('w') as f:
            json.dump(teams, f, indent=2)
        
        # Cache teams by ID for quick lookup
        for team in teams:
            if isinstance(team, dict) and 'id' in team:
                self.teams_cache[str(team['id'])] = team
        
        logger.info(f"Collected {len(teams)} NBA teams")
        return teams
    
    def collect_players(self, limit: int = 500) -> List[Dict]:
        """Collect NBA players data"""
        logger.info(f"Collecting NBA players data (limit: {limit})")
        
        response = self.bdl_client.get_players(per_page=100)
        players = []
        
        # Check if the response contains data in the expected format
        if isinstance(response, dict) and 'data' in response:
            players = response['data']
        elif isinstance(response, list):
            players = response
            
        # If we need more players and have pagination support
        if isinstance(response, dict) and 'meta' in response and 'total_pages' in response['meta']:
            total_pages = min(response['meta']['total_pages'], limit // 100 + 1)
            
            # Collect from additional pages if needed
            for page in range(2, total_pages + 1):
                logger.info(f"Collecting players data - page {page}/{total_pages}")
                page_response = self.bdl_client.get_players(page=page, per_page=100)
                if isinstance(page_response, dict) and 'data' in page_response:
                    players.extend(page_response['data'])
                elif isinstance(page_response, list):
                    players.extend(page_response)
        
        if not players:
            logger.error("Failed to collect players data")
            return []
        
        # Limit the number of players if necessary
        if len(players) > limit:
            players = players[:limit]
        
        # Save players data
        players_file = self.players_dir / 'players.json'
        with players_file.open('w') as f:
            json.dump(players, f, indent=2)
        
        # Cache players by ID for quick lookup
        for player in players:
            if isinstance(player, dict) and 'id' in player:
                self.players_cache[str(player['id'])] = player
        
        logger.info(f"Collected {len(players)} NBA players")
        return players
    
    def collect_games_for_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Collect games for a specific date range"""
        logger.info(f"Collecting games from {start_date} to {end_date}")
        
        response = self.bdl_client.get_games(start_date=start_date, end_date=end_date)
        games = []
        
        # Check if the response contains data in the expected format
        if isinstance(response, dict) and 'data' in response:
            games = response['data']
            
            # Handle pagination if available
            if 'meta' in response and 'total_pages' in response['meta']:
                total_pages = response['meta']['total_pages']
                
                # Collect from additional pages
                for page in range(2, total_pages + 1):
                    logger.info(f"Collecting games - page {page}/{total_pages}")
                    page_response = self.bdl_client.get_games(
                        start_date=start_date, 
                        end_date=end_date, 
                        page=page
                    )
                    if isinstance(page_response, dict) and 'data' in page_response:
                        games.extend(page_response['data'])
        elif isinstance(response, list):
            games = response
        
        if not games:
            logger.warning(f"No games found from {start_date} to {end_date}")
            return []
        
        # Save games by date
        date_file = self.games_dir / f"games_{start_date}_to_{end_date}.json"
        with date_file.open('w') as f:
            json.dump(games, f, indent=2)
        
        # Cache games by ID
        for game in games:
            if isinstance(game, dict) and 'id' in game:
                game_id = str(game['id'])
                self.games_cache[game_id] = game
                
                # Also save individual game files for quick access
                game_file = self.games_dir / f"game_{game_id}.json"
                with game_file.open('w') as f:
                    json.dump(game, f, indent=2)
        
        logger.info(f"Collected {len(games)} games from {start_date} to {end_date}")
        return games
    
    def collect_game_stats(self, game_id: str) -> Dict:
        """Collect statistics for a specific game"""
        logger.info(f"Collecting stats for game {game_id}")
        
        response = self.bdl_client.get_stats(game_id=int(game_id))
        stats = {}
        
        # Check if the response contains data in the expected format
        if isinstance(response, dict):
            if 'data' in response:
                stats = response['data']
            else:
                stats = response
        elif isinstance(response, list):
            # If it's a list, we'll store it with a standard format
            stats = {'data': response}
        
        if not stats:
            logger.warning(f"No stats found for game {game_id}")
            return {}
        
        # Save game stats
        stats_file = self.stats_dir / f"game_stats_{game_id}.json"
        with stats_file.open('w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Collected stats for game {game_id}")
        return stats
    
    def collect_historical_odds(self, date_str: str) -> List[Dict[str, Any]]:
        """
        Collect historical odds data for a specific date
        
        Args:
            date_str: Date in any format (will be converted to YYYY-MM-DD)
            
        Returns:
            List of historical odds data
        """
        logger.info(f"Collecting historical odds for {date_str}")
        
        # Use our direct odds_helper module that properly formats dates for The Odds API
        from src.api.odds_helper import get_historical_nba_odds
        
        try:
            # Directly fetch historical odds data with the helper function
            historical_odds = get_historical_nba_odds(date_str=date_str)
            
            if historical_odds:
                # Count by market type to provide useful logging
                market_counts = {}
                for game in historical_odds:
                    if 'bookmakers' in game:
                        for bookmaker in game['bookmakers']:
                            if 'markets' in bookmaker:
                                for market in bookmaker['markets']:
                                    market_type = market.get('key', 'unknown')
                                    market_counts[market_type] = market_counts.get(market_type, 0) + 1
                
                # Log the results with market details
                if market_counts:
                    markets_str = ', '.join([f"{k}: {v}" for k, v in market_counts.items()])
                    logger.info(f"Successfully collected {len(historical_odds)} historical odds entries for {date_str} with markets: {markets_str}")
                else:
                    logger.info(f"Successfully collected {len(historical_odds)} historical odds entries for {date_str}")
                
                # Save to disk for future reference
                odds_file = self.odds_dir / f"odds_{date_str}.json"
                with odds_file.open('w') as f:
                    json.dump(historical_odds, f, indent=2)
                    
                return historical_odds
            else:
                logger.debug(f"No odds data found for {date_str} from The Odds API")
                return []
        except Exception as e:
            logger.error(f"Error fetching historical odds for {date_str}: {str(e)}")
            return []
    
    def collect_historical_data(self, start_date: str, end_date: str, include_stats: bool = True,
                                include_odds: bool = True) -> Dict[str, Any]:
        """Collect comprehensive historical data for a date range"""
        logger.info(f"Collecting comprehensive historical data from {start_date} to {end_date}")
        
        # Ensure teams and players are collected first
        if not self.teams_cache:
            self.collect_teams()
        
        if not self.players_cache:
            self.collect_players()
        
        # Collect games for date range
        games = self.collect_games_for_date_range(start_date, end_date)
        
        # Collect game stats in parallel if requested
        if include_stats and games:
            logger.info("Collecting game stats in parallel")
            game_ids = [str(game['id']) for game in games]
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit stats collection tasks
                futures = {executor.submit(self.collect_game_stats, game_id): game_id for game_id in game_ids}
                
                # Process results as they complete
                for future in as_completed(futures):
                    game_id = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error collecting stats for game {game_id}: {str(e)}")
        
        # Collect odds data if requested
        if include_odds and self.odds_available:
            logger.info("Collecting historical odds data")
            
            # Parse dates
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            date_range = [(start + timedelta(days=i)).strftime('%Y-%m-%d') 
                       for i in range((end - start).days + 1)]
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit odds collection tasks (limited concurrency to respect API limits)
                futures = {executor.submit(self.collect_historical_odds, date): date for date in date_range}
                
                # Process results as they complete
                for future in as_completed(futures):
                    date = futures[future]
                    try:
                        future.result()
                        # Sleep to respect API rate limits
                        time.sleep(1)
                    except Exception as e:
                        logger.error(f"Error collecting odds for date {date}: {str(e)}")
        
        # Return collection summary
        return {
            "date_range": f"{start_date} to {end_date}",
            "games_collected": len(games),
            "teams_collected": len(self.teams_cache),
            "players_collected": len(self.players_cache),
            "include_stats": include_stats,
            "include_odds": include_odds and self.odds_available
        }
    
    def collect_data_for_season(self, season: str = "2025", include_stats: bool = True,
                               include_odds: bool = True) -> Dict[str, Any]:
        """
        Collect all data needed for training for a specific season
        
        Args:
            season: NBA season (e.g., "2025" for 2024-2025 season)
            include_stats: Whether to include game stats
            include_odds: Whether to include betting odds
            
        Returns:
            Dictionary containing all collected data
        """
        logger.info(f"Collecting data for {season} season")
        
        # Calculate date range based on season
        season_year = int(season)
        start_date = f"{season_year-1}-10-01"  # NBA season typically starts in October
        end_date = f"{season_year}-06-30"    # Finals typically end in June
        
        # For current season, only collect up to yesterday
        today = datetime.now().strftime('%Y-%m-%d')
        if end_date > today:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info(f"Date range for {season} season: {start_date} to {end_date}")
        
        try:
            # Get games for this date range
            game_list = self.collect_games_for_date_range(start_date, end_date)
            games_count = len(game_list)
            
            if games_count == 0:
                logger.warning(f"No games found for {season} season. Trying alternative date range.")
                
                # Try a more recent range if no games found
                alt_start = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
                alt_end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                logger.info(f"Trying alternative date range: {alt_start} to {alt_end}")
                
                game_list = self.collect_games_for_date_range(alt_start, alt_end)
                games_count = len(game_list)
                
                if games_count == 0:
                    logger.error("Still no games found. Cannot proceed with data collection.")
                    return {}
            
            logger.info(f"Successfully collected {games_count} games for {season} season")
            
            # Create a grid of all dates in the range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            date_list = [(start_dt + timedelta(days=i)).strftime('%Y-%m-%d') 
                       for i in range((end_dt - start_dt).days + 1)]
            
            # Group games by date
            games_by_date = {}
            for game in game_list:
                game_date = game.get('date', '').split('T')[0]  # Extract date part
                if game_date:
                    if game_date not in games_by_date:
                        games_by_date[game_date] = []
                    games_by_date[game_date].append(game)
            
            # Check which dates have games
            dates_with_games = list(games_by_date.keys())
            logger.info(f"Found games on {len(dates_with_games)} dates in the season")
            
            # Process game data in batches with proper error handling
            processed_games = []
            
            # Process in smaller batches to avoid API rate limits
            batch_size = 7  # Process a week at a time
            date_batches = [date_list[i:i+batch_size] for i in range(0, len(date_list), batch_size)]
            
            for batch_idx, date_batch in enumerate(date_batches):
                logger.info(f"Processing batch {batch_idx+1}/{len(date_batches)} of dates")
                
                for date_str in date_batch:
                    if date_str in games_by_date and games_by_date[date_str]:
                        try:
                            # Process each date with games
                            date_games = self.process_date_data(date_str, games_by_date[date_str], 
                                                             include_stats, include_odds)
                            processed_games.extend(date_games)
                            logger.info(f"Processed {len(date_games)} games for {date_str}")
                        except Exception as e:
                            logger.error(f"Error processing data for {date_str}: {str(e)}")
                
                # Add a small delay between batches to avoid rate limits
                if batch_idx < len(date_batches) - 1:
                    time.sleep(2)
            
            # Final data collection result
            result = {
                'season': season,
                'start_date': start_date,
                'end_date': end_date,
                'games_count': len(processed_games),
                'games': processed_games
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error collecting data for {season} season: {str(e)}")
            return {}
    
    def collect_data_for_multiple_seasons(self, seasons: List[str], include_stats: bool = True,
                                 include_odds: bool = True) -> Dict[str, Any]:
        """
        Collect comprehensive NBA data from multiple seasons for more robust model training
        
        Args:
            seasons: List of season years to collect (e.g., ["2022", "2023", "2024"])
            include_stats: Whether to collect detailed stats for each game
            include_odds: Whether to collect betting odds (requires The Odds API key)
            
        Returns:
            Dict with collection summary statistics
        """
        logger.info(f"Collecting data for multiple NBA seasons: {seasons}")
        
        results = {}
        total_games = 0
        failed_seasons = []
        
        # Process each season sequentially
        for season in seasons:
            try:
                logger.info(f"Processing season {season}...")
                season_result = self.collect_data_for_season(
                    season=season,
                    include_stats=include_stats,
                    include_odds=include_odds
                )
                
                # Add to overall results
                results[season] = season_result
                if isinstance(season_result, dict) and 'games_count' in season_result:
                    total_games += season_result['games_count']
                    
                logger.info(f"Successfully collected data for season {season}")
                
                # Add a small delay between seasons to avoid overwhelming APIs
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to collect data for season {season}: {str(e)}")
                failed_seasons.append(season)
        
        # Return overall summary
        return {
            "seasons_collected": len(seasons) - len(failed_seasons),
            "seasons_failed": failed_seasons,
            "total_games": total_games,
            "teams": len(self.teams_cache),
            "players": len(self.players_cache),
            "season_details": results
        }
    
    def load_collected_data(self) -> Dict[str, Any]:
        """Load all collected data for analysis"""
        logger.info("Loading collected historical data")
        
        # Load teams
        teams_file = self.teams_dir / 'teams.json'
        teams = []
        if teams_file.exists():
            with teams_file.open('r') as f:
                teams = json.load(f)
            # Update cache
            for team in teams:
                self.teams_cache[str(team['id'])] = team
        
        # Load players
        players_file = self.players_dir / 'players.json'
        players = []
        if players_file.exists():
            with players_file.open('r') as f:
                players = json.load(f)
            # Update cache
            for player in players:
                self.players_cache[str(player['id'])] = player
        
        # Load games (from all date range files)
        games = []
        for game_file in self.games_dir.glob('games_*.json'):
            with game_file.open('r') as f:
                games_batch = json.load(f)
                games.extend(games_batch)
                # Update cache
                for game in games_batch:
                    self.games_cache[str(game['id'])] = game
        
        # Return data summary
        return {
            "teams": len(teams),
            "players": len(players),
            "games": len(games),
            "team_cache_size": len(self.teams_cache),
            "player_cache_size": len(self.players_cache),
            "game_cache_size": len(self.games_cache)
        }
    
    def get_games_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get games within a specific date range, either from cache or by fetching from API
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing all games in the date range
        """
        logger.info(f"Getting games from {start_date} to {end_date}")
        
        try:
            # First try to load from cache
            games = []
            cache_file = self.games_dir / f"games_{start_date}_to_{end_date}.json"
            
            if cache_file.exists():
                logger.info(f"Loading games from cache: {cache_file}")
                with cache_file.open('r') as f:
                    games = json.load(f)
            else:
                # If not in cache, collect from API
                logger.info(f"Games not in cache, collecting from API")
                games = self.collect_games_for_date_range(start_date, end_date)
            
            # Convert to DataFrame
            if not games:
                logger.warning(f"No games found for date range {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Extract relevant game information
            game_data = []
            for game in games:
                if isinstance(game, dict):
                    try:
                        game_id = game.get('id')
                        date = game.get('date', '').split('T')[0]  # Extract date part
                        home_team = game.get('home_team', {}).get('name', '')
                        home_team_id = game.get('home_team', {}).get('id', 0)
                        home_score = game.get('home_team_score', 0)
                        away_team = game.get('visitor_team', {}).get('name', '')
                        away_team_id = game.get('visitor_team', {}).get('id', 0)
                        away_score = game.get('visitor_team_score', 0)
                        period = game.get('period', 0)
                        status = game.get('status', '')
                        time = game.get('time', '')
                        postseason = game.get('postseason', False)
                        
                        # Determine winner
                        home_won = None
                        if status.lower() == 'final' and home_score != away_score:
                            home_won = home_score > away_score
                            
                        game_data.append({
                            'game_id': game_id,
                            'date': date,
                            'home_team': home_team,
                            'home_team_id': home_team_id,
                            'home_score': home_score,
                            'away_team': away_team,
                            'away_team_id': away_team_id,
                            'away_score': away_score,
                            'period': period,
                            'status': status,
                            'time': time,
                            'postseason': postseason,
                            'home_won': home_won
                        })
                    except Exception as e:
                        logger.error(f"Error processing game {game.get('id', 'unknown')}: {str(e)}")
                        continue
            
            # Create DataFrame
            games_df = pd.DataFrame(game_data)
            logger.info(f"Returning {len(games_df)} games from {start_date} to {end_date}")
            return games_df
            
        except Exception as e:
            logger.error(f"Error getting games by date range: {str(e)}")
            return pd.DataFrame()
    
    def collect_data_for_date(self, date_str: str, include_stats: bool = True,
                           include_odds: bool = True) -> List[Dict[str, Any]]:
        """
        Collect game data for a specific date
        
        Args:
            date_str: Date in YYYY-MM-DD format
            include_stats: Whether to include game stats
            include_odds: Whether to include betting odds
            
        Returns:
            List of game data dictionaries
        """
        logger.info(f"Collecting data for {date_str}")
        
        try:
            # Get games for this date
            games = self.get_games_for_date(date_str)
            if not games:
                logger.warning(f"No games found for {date_str}")
                return []
                
            logger.info(f"Found {len(games)} games for {date_str}")
            
            # Add team stats for each game
            if include_stats:
                games = self.add_stats_to_games(games, date_str)
            
            # Add betting odds for each game if requested
            if include_odds:
                try:
                    # Fetch historical odds using supported markets only
                    odds_data = self.collect_historical_odds(self.ensure_date_format(date_str))
                    if odds_data:
                        # Add odds to each game
                        for game in games:
                            game_id = game.get('id')
                            # Find matching odds entry
                            matching_odds = next((o for o in odds_data if str(o.get('game_id')) == str(game_id)), None)
                            if matching_odds:
                                game['odds'] = matching_odds
                            else:
                                game['odds'] = {}
                    else:
                        logger.debug(f"No odds data available for {date_str}")
                except Exception as e:
                    logger.warning(f"Error fetching odds data for {date_str}: {str(e)}")
            
            return games
        except Exception as e:
            logger.error(f"Error collecting data for {date_str}: {str(e)}")
            return []
    
    def get_games_for_date(self, date_str: str) -> List[Dict]:
        """
        Get games for a specific date
        
        Args:
            date_str: Date in YYYY-MM-DD format
            
        Returns:
            List of game data dictionaries
        """
        logger.info(f"Getting games for {date_str}")
        
        try:
            # First try to load from cache
            games = []
            cache_file = self.games_dir / f"games_{date_str}.json"
            
            if cache_file.exists():
                logger.info(f"Loading games from cache: {cache_file}")
                with cache_file.open('r') as f:
                    games = json.load(f)
            else:
                # If not in cache, collect from API
                logger.info(f"Games not in cache, collecting from API")
                games = self.collect_games_for_date_range(date_str, date_str)
            
            return games
        except Exception as e:
            logger.error(f"Error getting games for date: {str(e)}")
            return []
    
    def add_stats_to_games(self, games: List[Dict], date_str: str) -> List[Dict]:
        """
        Add team stats to each game
        
        Args:
            games: List of game data dictionaries
            date_str: Date in YYYY-MM-DD format
            
        Returns:
            List of game data dictionaries with added team stats
        """
        logger.info(f"Adding team stats to games for {date_str}")
        
        try:
            # Collect team stats for this date
            team_stats = self.collect_team_stats_for_date(date_str)
            
            # Add team stats to each game
            for game in games:
                game_id = game.get('id')
                home_team_id = game.get('home_team', {}).get('id', 0)
                away_team_id = game.get('visitor_team', {}).get('id', 0)
                
                # Find matching team stats
                home_team_stats = next((s for s in team_stats if s.get('team_id') == home_team_id), None)
                away_team_stats = next((s for s in team_stats if s.get('team_id') == away_team_id), None)
                
                if home_team_stats:
                    game['home_team_stats'] = home_team_stats
                else:
                    game['home_team_stats'] = {}
                
                if away_team_stats:
                    game['away_team_stats'] = away_team_stats
                else:
                    game['away_team_stats'] = {}
            
            return games
        except Exception as e:
            logger.error(f"Error adding team stats to games: {str(e)}")
            return games
    
    def collect_team_stats_for_date(self, date_str: str) -> List[Dict]:
        """
        Collect team stats for a specific date
        
        Args:
            date_str: Date in YYYY-MM-DD format
            
        Returns:
            List of team stats data dictionaries
        """
        logger.info(f"Collecting team stats for {date_str}")
        
        try:
            # Collect team stats from API
            team_stats = self.bdl_client.get_team_stats(date_str)
            
            # Save team stats to cache
            cache_file = self.stats_dir / f"team_stats_{date_str}.json"
            with cache_file.open('w') as f:
                json.dump(team_stats, f, indent=2)
            
            return team_stats
        except Exception as e:
            logger.error(f"Error collecting team stats: {str(e)}")
            return []
    
    def process_date_data(self, date_str: str, games: List[Dict], 
                       include_stats: bool = True, include_odds: bool = True) -> List[Dict]:
        """
        Process all data for a specific date
        
        Args:
            date_str: Date in YYYY-MM-DD format
            games: List of games for this date
            include_stats: Whether to include game stats
            include_odds: Whether to include betting odds
            
        Returns:
            List of processed games with stats and odds if requested
        """
        logger.info(f"Processing data for {date_str} ({len(games)} games)")
        
        # Format date for API use
        formatted_date = self.ensure_date_format(date_str)
        
        # Skip odds collection for these problematic dates that are causing errors
        problematic_dates = ['2024-11-04', '2024-11-06', '2024-11-07'] 
        skip_odds = formatted_date in problematic_dates
        
        if skip_odds and include_odds:
            logger.warning(f"Skipping odds collection for known problematic date {formatted_date}")
            include_odds = False
        
        processed_games = []
        
        # Process each game
        for game in games:
            game_id = str(game.get('id', ''))
            if not game_id:
                continue
            
            processed_game = game.copy()
            
            # Add stats if requested
            if include_stats:
                try:
                    stats = self.collect_game_stats(game_id)
                    if stats:
                        processed_game['stats'] = stats
                except Exception as e:
                    logger.error(f"Error collecting stats for game {game_id}: {str(e)}")
            
            # Add odds if requested
            if include_odds and self.odds_available and not skip_odds:
                try:
                    # Collect odds for this game date
                    odds = self.collect_historical_odds(formatted_date)
                    
                    if odds:
                        # Find matching odds for this game
                        matching_odds = next((o for o in odds if str(o.get('game_id', '')) == game_id), None)
                        if matching_odds:
                            processed_game['odds'] = matching_odds
                except Exception as e:
                    logger.error(f"Error collecting odds for game {game_id} on {formatted_date}: {str(e)}")
            
            processed_games.append(processed_game)
        
        logger.info(f"Processed {len(processed_games)} games for {formatted_date}")
        return processed_games


# Main function for testing
if __name__ == "__main__":
    # Test data collection
    collector = HistoricalDataCollector()
    
    # You can uncomment these lines to test collection for a specific date range
    # start_date = "2024-01-01"
    # end_date = "2024-01-31"
    # result = collector.collect_historical_data(start_date, end_date)
    # print(f"Collection result: {result}")
    
    # Or collect data for the current season
    # result = collector.collect_data_for_season("2024")
    # print(f"Season collection result: {result}")
    
    # Load previously collected data
    data_summary = collector.load_collected_data()
    print(f"Loaded data summary: {data_summary}")
