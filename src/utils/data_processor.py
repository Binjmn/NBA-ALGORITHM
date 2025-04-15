#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Data Processor

This module provides a data processor for organizing and processing data
retrieved from the BallDontLie API and The Odds API.

Usage:
    from src.utils.data_processor import NBADataProcessor
    
    # Create a processor with our API clients
    processor = NBADataProcessor(balldontlie_client, theodds_client)
    
    # Get processed team data
    teams = processor.get_teams()
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

from src.api.balldontlie_client import BallDontLieClient
from src.api.theodds_client import TheOddsClient

# Configure logging
logger = logging.getLogger(__name__)


class NBADataProcessor:
    """
    A class for organizing and processing NBA data from multiple API sources
    
    This class combines and processes data from the BallDontLie API and The Odds API
    to provide a unified data source for the NBA prediction models.
    """
    
    def __init__(self, 
                balldontlie_client: BallDontLieClient, 
                theodds_client: TheOddsClient,
                data_dir: str = 'data'):
        """
        Initialize the NBA data processor
        
        Args:
            balldontlie_client (BallDontLieClient): BallDontLie API client
            theodds_client (TheOddsClient): The Odds API client
            data_dir (str): Directory for storing processed data
        """
        self.balldontlie_client = balldontlie_client
        self.theodds_client = theodds_client
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info("Initialized NBA data processor")
    
    def get_teams(self, force_refresh: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        Get all NBA teams
        
        Args:
            force_refresh (bool): Whether to force a refresh of the cached data
            
        Returns:
            Dict[int, Dict[str, Any]]: Dictionary of teams keyed by team ID
        """
        if not hasattr(self, '_teams_cache') or force_refresh:
            response = self.balldontlie_client.get_teams(per_page=100)
            teams = response.get('data', [])
            
            # Create a dictionary keyed by team ID
            self._teams_cache = {team['id']: team for team in teams}
            logger.info(f"Loaded {len(self._teams_cache)} teams from API")
        
        return self._teams_cache
    
    def get_players(self, force_refresh: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        Get all active NBA players
        
        Args:
            force_refresh (bool): Whether to force a refresh of the cached data
            
        Returns:
            Dict[int, Dict[str, Any]]: Dictionary of players keyed by player ID
        """
        if not hasattr(self, '_players_cache') or force_refresh:
            # Get all pages of active players
            all_players = []
            page = 1
            per_page = 100
            
            while True:
                response = self.balldontlie_client.get_active_players(page=page, per_page=per_page)
                players = response.get('data', [])
                all_players.extend(players)
                
                # Check if we've reached the last page
                meta = response.get('meta', {})
                total_pages = meta.get('total_pages', 1)
                
                if page >= total_pages:
                    break
                
                page += 1
            
            # Create a dictionary keyed by player ID
            self._players_cache = {player['id']: player for player in all_players}
            logger.info(f"Loaded {len(self._players_cache)} active players from API")
        
        return self._players_cache
    
    def get_game_stats(self, game_id: int) -> Dict[str, Any]:
        """
        Get comprehensive stats for a specific game
        
        Args:
            game_id (int): Game ID
            
        Returns:
            Dict[str, Any]: Comprehensive game statistics
        """
        # Get basic game information
        game_response = self.balldontlie_client.get_games()
        game_data = None
        
        # Find the specific game
        for game in game_response.get('data', []):
            if game['id'] == game_id:
                game_data = game
                break
        
        if not game_data:
            logger.warning(f"Game with ID {game_id} not found")
            return {}
        
        # Get box score for the game
        box_score = self.balldontlie_client.get_box_scores(game_id)
        
        # Get player stats for the game
        player_stats = self.balldontlie_client.get_stats(game_id=game_id, per_page=100)
        
        # Get advanced stats for the game
        advanced_stats = self.balldontlie_client.get_advanced_stats(game_id=game_id, per_page=100)
        
        # Get odds for the game
        odds = self.theodds_client.get_odds(game_id=game_id)
        
        # Combine all data
        result = {
            'game': game_data,
            'box_score': box_score,
            'player_stats': player_stats.get('data', []),
            'advanced_stats': advanced_stats.get('data', []),
            'odds': odds.get('data', [])
        }
        
        return result
    
    def get_player_season_stats(self, player_id: int, season: Optional[int] = None) -> Dict[str, Any]:
        """
        Get season stats for a player
        
        Args:
            player_id (int): Player ID
            season (Optional[int]): Season year (e.g., 2023 for 2023-24 season)
            
        Returns:
            Dict[str, Any]: Player season statistics
        """
        # Get basic player information
        players = self.get_players()
        player_data = players.get(player_id)
        
        if not player_data:
            logger.warning(f"Player with ID {player_id} not found")
            return {}
        
        # Get season averages for different categories
        categories = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'field_goals',
                     'three_points', 'free_throws', 'minutes', 'games_played']
        
        season_stats = {}
        for category in categories:
            try:
                stats = self.balldontlie_client.get_season_averages(
                    category=category,
                    player_id=player_id,
                    season=season
                )
                season_stats[category] = stats.get('data', [])
            except Exception as e:
                logger.error(f"Error getting {category} stats for player {player_id}: {e}")
                season_stats[category] = []
        
        # Get injury information
        try:
            injuries = self.balldontlie_client.get_player_injuries(player_id=player_id)
            season_stats['injuries'] = injuries.get('data', [])
        except Exception as e:
            logger.error(f"Error getting injury data for player {player_id}: {e}")
            season_stats['injuries'] = []
        
        # Combine all data
        result = {
            'player': player_data,
            'season_stats': season_stats
        }
        
        return result
    
    def get_team_season_stats(self, team_id: int, season: Optional[int] = None) -> Dict[str, Any]:
        """
        Get season stats for a team
        
        Args:
            team_id (int): Team ID
            season (Optional[int]): Season year (e.g., 2023 for 2023-24 season)
            
        Returns:
            Dict[str, Any]: Team season statistics
        """
        # Get basic team information
        teams = self.get_teams()
        team_data = teams.get(team_id)
        
        if not team_data:
            logger.warning(f"Team with ID {team_id} not found")
            return {}
        
        # Get season averages for different categories
        categories = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'field_goals',
                     'three_points', 'free_throws', 'offensive_rating', 'defensive_rating']
        
        season_stats = {}
        for category in categories:
            try:
                stats = self.balldontlie_client.get_season_averages(
                    category=category,
                    team_id=team_id,
                    season=season
                )
                season_stats[category] = stats.get('data', [])
            except Exception as e:
                logger.error(f"Error getting {category} stats for team {team_id}: {e}")
                season_stats[category] = []
        
        # Get standings
        try:
            standings = self.balldontlie_client.get_standings(season=season)
            # Filter for this team
            team_standings = [s for s in standings.get('data', []) if s.get('team_id') == team_id]
            season_stats['standings'] = team_standings
        except Exception as e:
            logger.error(f"Error getting standings for team {team_id}: {e}")
            season_stats['standings'] = []
        
        # Combine all data
        result = {
            'team': team_data,
            'season_stats': season_stats
        }
        
        return result
    
    def get_todays_games_with_odds(self) -> List[Dict[str, Any]]:
        """
        Get today's games with odds data
        
        Returns:
            List[Dict[str, Any]]: Today's games with odds data
        """
        # Get today's games from BallDontLie
        balldontlie_games = self.balldontlie_client.get_todays_games()
        games_list = balldontlie_games.get('data', [])
        
        # Get today's odds from The Odds API
        theodds_games = self.theodds_client.get_todays_odds()
        
        # Map BallDontLie game IDs to The Odds API event IDs
        # This is complex because the APIs use different formats and identifiers
        enriched_games = []
        
        for game in games_list:
            game_data = {
                'id': game['id'],
                'date': game['date'],
                'status': game['status'],
                'period': game.get('period', 0),
                'time': game.get('time', ''),
                'home_team': {
                    'id': game['home_team']['id'],
                    'name': game['home_team']['name'],
                    'abbreviation': game['home_team']['abbreviation'],
                    'city': game['home_team']['city'],
                    'score': game.get('home_team_score', 0)
                },
                'visitor_team': {
                    'id': game['visitor_team']['id'],
                    'name': game['visitor_team']['name'],
                    'abbreviation': game['visitor_team']['abbreviation'],
                    'city': game['visitor_team']['city'],
                    'score': game.get('visitor_team_score', 0)
                },
                'odds': None  # Will be populated if odds are found
            }
            
            # Try to find matching odds from The Odds API
            # Match based on team names
            home_name = game['home_team']['name'].lower()
            visitor_name = game['visitor_team']['name'].lower()
            
            for odds_game in theodds_games:
                odds_home = odds_game['home_team'].lower()
                odds_away = odds_game['away_team'].lower()
                
                # Check if this is the same game
                # We're using partial matching because team names might be formatted differently
                if (home_name in odds_home or odds_home in home_name) and \
                   (visitor_name in odds_away or odds_away in visitor_name):
                    # Found matching odds!
                    game_data['odds'] = {
                        'event_id': odds_game['id'],
                        'commence_time': odds_game['commence_time'],
                        'bookmakers': odds_game['bookmakers']
                    }
                    break
            
            enriched_games.append(game_data)
        
        return enriched_games
    
    def get_live_game_data(self, game_id: int) -> Dict[str, Any]:
        """
        Get live data for a specific game by combining BallDontLie and The Odds API data
        
        Args:
            game_id (int): BallDontLie game ID
            
        Returns:
            Dict[str, Any]: Combined live game data
        """
        # Get box score data from BallDontLie
        box_score = self.balldontlie_client.get_live_box_scores(game_id)
        
        # Get game details to find corresponding Odds API event
        game_details = self.balldontlie_client.get_games(game_id=game_id)
        game = game_details.get('data', [{}])[0]
        
        if not game:
            logger.warning(f"Game {game_id} not found")
            return box_score
        
        # Get live scores from The Odds API
        live_scores = self.theodds_client.get_live_scores()
        
        # Try to find matching game in live scores
        home_team_name = game['home_team']['name'].lower()
        visitor_team_name = game['visitor_team']['name'].lower()
        
        matching_score = None
        for score in live_scores:
            odds_home = score['home_team'].lower()
            odds_away = score['away_team'].lower()
            
            # Check for match using partial string matching
            if (home_team_name in odds_home or odds_home in home_team_name) and \
               (visitor_team_name in odds_away or odds_away in visitor_team_name):
                matching_score = score
                break
        
        # Combine the data
        result = box_score
        if matching_score:
            result['odds_api_data'] = matching_score
        
        return result
    
    def save_processed_data(self, data: Any, filename: str) -> str:
        """
        Save processed data to a JSON file
        
        Args:
            data (Any): Data to save
            filename (str): Filename (without extension)
            
        Returns:
            str: Path to saved file
        """
        filepath = os.path.join(self.data_dir, f"{filename}.json")
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved processed data to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving data to {filepath}: {e}")
            return ""
    
    def load_processed_data(self, filename: str) -> Optional[Any]:
        """
        Load processed data from a JSON file
        
        Args:
            filename (str): Filename (without extension)
            
        Returns:
            Optional[Any]: Loaded data or None if file doesn't exist or error occurs
        """
        filepath = os.path.join(self.data_dir, f"{filename}.json")
        
        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} not found")
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded processed data from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return None
    
    def close(self) -> None:
        """
        Close the API clients
        """
        self.balldontlie_client.close()
        self.theodds_client.close()
        logger.info("Closed NBA data processor")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
