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
    
    def __init__(self):
        """Initialize the data collector"""
        self.balldontlie_client = BallDontLieClient()
        
        # Initialize The Odds API client if API key is available
        try:
            self.odds_client = TheOddsClient()
            self.odds_available = True
        except ValueError:
            logger.warning("The Odds API key not found, odds data will not be collected")
            self.odds_available = False
        
        # Create data directories
        self.games_dir = DATA_DIR / 'games'
        self.teams_dir = DATA_DIR / 'teams'
        self.players_dir = DATA_DIR / 'players'
        self.stats_dir = DATA_DIR / 'stats'
        self.odds_dir = DATA_DIR / 'odds'
        
        self.games_dir.mkdir(exist_ok=True)
        self.teams_dir.mkdir(exist_ok=True)
        self.players_dir.mkdir(exist_ok=True)
        self.stats_dir.mkdir(exist_ok=True)
        self.odds_dir.mkdir(exist_ok=True)
        
        # Cache for loaded data
        self.teams_cache = {}
        self.players_cache = {}
        self.games_cache = {}
    
    def collect_teams(self) -> List[Dict]:
        """Collect all NBA teams"""
        logger.info("Collecting NBA teams data")
        
        response = self.balldontlie_client.get_teams()
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
        
        response = self.balldontlie_client.get_players(per_page=100)
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
                page_response = self.balldontlie_client.get_players(page=page, per_page=100)
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
        
        response = self.balldontlie_client.get_games(start_date=start_date, end_date=end_date)
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
                    page_response = self.balldontlie_client.get_games(
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
        
        response = self.balldontlie_client.get_stats(game_id=int(game_id))
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
    
    def collect_historical_odds(self, date: str) -> List[Dict]:
        """Collect historical odds for a specific date"""
        if not self.odds_available:
            logger.warning("The Odds API not available, skipping odds collection")
            return []
        
        logger.info(f"Collecting historical odds for {date}")
        
        try:
            # Use The Odds API to get historical odds
            response = self.odds_client.get_historical_odds(date=date)
            odds = []
            
            # Check if the response contains data in the expected format
            if isinstance(response, dict) and 'data' in response:
                odds = response['data']
            elif isinstance(response, list):
                odds = response
            
            if not odds:
                logger.warning(f"No odds data found for {date}")
                return []
            
            # Save odds data
            odds_file = self.odds_dir / f"odds_{date}.json"
            with odds_file.open('w') as f:
                json.dump(odds, f, indent=2)
            
            logger.info(f"Collected odds data for {date}")
            return odds
        except Exception as e:
            logger.error(f"Error collecting historical odds for {date}: {str(e)}")
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
    
    def collect_data_for_season(self, season: str = "2024", include_stats: bool = True,
                               include_odds: bool = True) -> Dict[str, Any]:
        """Collect all data for a specific NBA season"""
        logger.info(f"Collecting data for NBA season {season}")
        
        # Use season dates based on season parameter
        # For simplicity, assuming October 1st to June 30th as the NBA season
        start_date = f"{int(season)-1}-10-01" if int(season) > 2000 else f"20{int(season)-1}-10-01"
        end_date = f"{season}-06-30" if int(season) > 2000 else f"20{season}-06-30"
        
        # Force include_odds to False if we don't have odds API access
        if include_odds and not self.odds_available:
            include_odds = False
            logger.warning("Odds API not available, skipping odds collection")
            
        try:
            result = self.collect_historical_data(
                start_date=start_date,
                end_date=end_date,
                include_stats=include_stats,
                include_odds=include_odds
            )
        except Exception as e:
            logger.error(f"Error collecting full season data: {str(e)}")
            # Continue with partial data collection instead of failing completely
            
            # First collect teams and players regardless of errors
            teams = self.collect_teams()
            players = self.collect_players()
            
            # Then try to collect games
            try:
                games = self.collect_games_for_date_range(start_date, end_date)
            except Exception as game_err:
                logger.error(f"Error collecting games: {str(game_err)}")
                games = []
            
            # Create partial result with what we have
            result = {
                "teams": len(teams),
                "players": len(players),
                "games": len(games),
                "stats": 0,
                "odds": 0,
                "start_date": start_date,
                "end_date": end_date,
                "include_stats": include_stats,
                "include_odds": include_odds,
                "season": season,
                "status": "partial",
                "error": str(e)
            }
            
            # If we have games and should collect stats, try that too
            if games and include_stats:
                stats_count = 0
                for game in games:
                    try:
                        game_id = str(game.get('id', ''))
                        if game_id:
                            stats = self.collect_game_stats(game_id)
                            if stats:
                                stats_count += 1
                    except Exception as stats_err:
                        logger.error(f"Error collecting stats for game {game_id}: {str(stats_err)}")
                        continue
                
                result["stats"] = stats_count
            
            # Skip odds collection as it's causing issues
            
        # Add season information
        if "season" not in result:
            result["season"] = season
            
        return result
    
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
                if isinstance(season_result, dict) and 'games_collected' in season_result:
                    total_games += season_result['games_collected']
                elif isinstance(season_result, dict) and 'games' in season_result:
                    total_games += season_result['games']
                    
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
                        date = game.get('date', '').split('T')[0]  # Extract just the date part
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
