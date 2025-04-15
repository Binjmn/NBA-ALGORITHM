#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BallDontLie API Client Module
Purpose: Provide a client for accessing the BallDontLie API endpoints with rate limiting,
caching, error handling, and retries.

Endpoints Implemented:
- GET /teams: Team information
- GET /players: Player information
- GET /games: Game schedules and results
- GET /stats: Game-level player statistics
- GET /players/active: Lists active players
- GET /player_injuries: Injury data for players
- GET /season_averages/{category}: Season averages
- GET /stats/advanced: Advanced game statistics
- GET /box_scores: Detailed box scores
- GET /box_scores/live: Live box scores
- GET /standings: Team standings
- GET /leaders: League leaders
- GET /odds: Betting odds for games
- GET /team_stats: Team statistics
- GET /season: Current season information
- GET /draft: NBA draft data
- GET /plays: Play-by-play data for games
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

from src.api.base_client import BaseAPIClient, APIError
from config.api_keys import get_api_key

logger = logging.getLogger(__name__)


class BallDontLieClient(BaseAPIClient):
    """Client for accessing the BallDontLie API"""
    
    def __init__(self):
        """
        Initialize the BallDontLie API client
        """
        api_key = get_api_key('balldontlie')
        if not api_key:
            raise ValueError("BallDontLie API key not found")
        
        # Define time-sensitive endpoints that should have a shorter cache TTL
        time_sensitive_endpoints = [
            'odds',              # Betting odds change frequently
            'box_scores/live',   # Live game data should be very fresh
            'player_injuries',   # Injury reports can change suddenly
            'games'              # Game statuses can change
        ]
        
        super().__init__(
            base_url="https://api.balldontlie.io/v1",
            api_key=api_key,
            rate_limit=60,  # GOAT plan has higher limits, but we'll be conservative
            rate_limit_period=60,  # 60 requests per minute
            cache_ttl=3600,  # Regular data cache for 1 hour
            time_sensitive_endpoints=time_sensitive_endpoints,
            time_sensitive_ttl=300  # Time-sensitive data cached for only 5 minutes
        )
        logger.info("Initialized BallDontLie API client")
    
    def get_teams(self, page: int = 1, per_page: int = 100) -> Dict[str, Any]:
        """
        Get team information
        
        Args:
            page (int): Page number for pagination
            per_page (int): Number of items per page
            
        Returns:
            Dict[str, Any]: Team information
        """
        params = {
            'page': page,
            'per_page': per_page
        }
        return self.request('teams', params=params)
    
    def get_players(self, page: int = 1, per_page: int = 100, search: Optional[str] = None) -> Dict[str, Any]:
        """
        Get player information
        
        Args:
            page (int): Page number for pagination
            per_page (int): Number of items per page
            search (Optional[str]): Search string for player name
            
        Returns:
            Dict[str, Any]: Player information
        """
        params = {
            'page': page,
            'per_page': per_page
        }
        
        if search:
            params['search'] = search
        
        return self.request('players', params=params)
    
    def get_active_players(self, page: int = 1, per_page: int = 100) -> Dict[str, Any]:
        """
        Get active players
        
        Args:
            page (int): Page number for pagination
            per_page (int): Number of items per page
            
        Returns:
            Dict[str, Any]: Active player information
        """
        params = {
            'page': page,
            'per_page': per_page
        }
        return self.request('players/active', params=params)
    
    def get_player_injuries(self, player_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get player injuries
        
        Args:
            player_id (Optional[int]): Player ID to filter by
            
        Returns:
            Dict[str, Any]: Player injury information
        """
        params = {}
        if player_id:
            params['player_id'] = player_id
        
        return self.request('player_injuries', params=params)
    
    def get_games(self, 
                  start_date: Optional[Union[str, datetime]] = None,
                  end_date: Optional[Union[str, datetime]] = None,
                  team_ids: Optional[List[int]] = None,
                  page: int = 1, 
                  per_page: int = 100) -> Dict[str, Any]:
        """
        Get games information
        
        Args:
            start_date (Optional[Union[str, datetime]]): Start date for filtering games (YYYY-MM-DD)
            end_date (Optional[Union[str, datetime]]): End date for filtering games (YYYY-MM-DD)
            team_ids (Optional[List[int]]): Team IDs to filter by
            page (int): Page number for pagination
            per_page (int): Number of items per page
            
        Returns:
            Dict[str, Any]: Games information
        """
        params = {
            'page': page,
            'per_page': per_page
        }
        
        # Convert datetime to string if needed
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            params['start_date'] = start_date
        
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            params['end_date'] = end_date
        
        if team_ids:
            params['team_ids'] = ','.join(str(team_id) for team_id in team_ids)
        
        return self.request('games', params=params)
    
    def get_stats(self, 
                 game_id: Optional[int] = None,
                 player_id: Optional[int] = None,
                 team_id: Optional[int] = None,
                 start_date: Optional[Union[str, datetime]] = None,
                 end_date: Optional[Union[str, datetime]] = None,
                 page: int = 1, 
                 per_page: int = 100) -> Dict[str, Any]:
        """
        Get player statistics for games
        
        Args:
            game_id (Optional[int]): Game ID to filter by
            player_id (Optional[int]): Player ID to filter by
            team_id (Optional[int]): Team ID to filter by
            start_date (Optional[Union[str, datetime]]): Start date for filtering stats (YYYY-MM-DD)
            end_date (Optional[Union[str, datetime]]): End date for filtering stats (YYYY-MM-DD)
            page (int): Page number for pagination
            per_page (int): Number of items per page
            
        Returns:
            Dict[str, Any]: Player statistics
        """
        params = {
            'page': page,
            'per_page': per_page
        }
        
        if game_id:
            params['game_id'] = game_id
        
        if player_id:
            params['player_id'] = player_id
        
        if team_id:
            params['team_id'] = team_id
        
        # Convert datetime to string if needed
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            params['start_date'] = start_date
        
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            params['end_date'] = end_date
        
        return self.request('stats', params=params)
    
    def get_advanced_stats(self, 
                          game_id: Optional[int] = None,
                          player_id: Optional[int] = None,
                          team_id: Optional[int] = None,
                          page: int = 1, 
                          per_page: int = 100) -> Dict[str, Any]:
        """
        Get advanced player statistics
        
        Args:
            game_id (Optional[int]): Game ID to filter by
            player_id (Optional[int]): Player ID to filter by
            team_id (Optional[int]): Team ID to filter by
            page (int): Page number for pagination
            per_page (int): Number of items per page
            
        Returns:
            Dict[str, Any]: Advanced player statistics
        """
        params = {
            'page': page,
            'per_page': per_page
        }
        
        if game_id:
            params['game_id'] = game_id
        
        if player_id:
            params['player_id'] = player_id
        
        if team_id:
            params['team_id'] = team_id
        
        return self.request('stats/advanced', params=params)
    
    def get_season_averages(self, 
                           category: str,
                           season: Optional[int] = None,
                           player_id: Optional[int] = None,
                           team_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get season averages for players or teams
        
        Args:
            category (str): Category to get averages for (e.g., 'points', 'rebounds')
            season (Optional[int]): Season year (e.g., 2023 for 2023-24 season)
            player_id (Optional[int]): Player ID to filter by
            team_id (Optional[int]): Team ID to filter by
            
        Returns:
            Dict[str, Any]: Season averages
        """
        params = {}
        
        if season:
            params['season'] = season
        
        if player_id:
            params['player_id'] = player_id
        
        if team_id:
            params['team_id'] = team_id
        
        return self.request(f'season_averages/{category}', params=params)
    
    def get_box_scores(self, game_id: int) -> Dict[str, Any]:
        """
        Get detailed box scores for a game
        
        Args:
            game_id (int): Game ID
            
        Returns:
            Dict[str, Any]: Box score data
        """
        params = {'game_id': game_id}
        return self.request('box_scores', params=params)
    
    def get_live_box_scores(self, game_id: int) -> Dict[str, Any]:
        """
        Get live box scores for an ongoing game
        
        Args:
            game_id (int): Game ID
            
        Returns:
            Dict[str, Any]: Live box score data
        """
        params = {'game_id': game_id}
        # Never use cache for live data to ensure maximum accuracy
        return self.request('box_scores/live', params=params, use_cache=False, force_refresh=True)
    
    def get_standings(self, season: Optional[int] = None) -> Dict[str, Any]:
        """
        Get team standings
        
        Args:
            season (Optional[int]): Season year (e.g., 2023 for 2023-24 season)
            
        Returns:
            Dict[str, Any]: Team standings
        """
        params = {}
        if season:
            params['season'] = season
        
        return self.request('standings', params=params)
    
    def get_leaders(self, category: str, season: Optional[int] = None) -> Dict[str, Any]:
        """
        Get league leaders in a statistical category
        
        Args:
            category (str): Statistical category (e.g., 'points', 'rebounds')
            season (Optional[int]): Season year (e.g., 2023 for 2023-24 season)
            
        Returns:
            Dict[str, Any]: League leaders
        """
        params = {}
        if season:
            params['season'] = season
        
        return self.request(f'leaders/{category}', params=params)
    
    def get_odds(self, 
                game_id: Optional[int] = None,
                date: Optional[Union[str, datetime]] = None) -> Dict[str, Any]:
        """
        Get betting odds for games
        
        Args:
            game_id (Optional[int]): Game ID to filter by
            date (Optional[Union[str, datetime]]): Date for filtering odds (YYYY-MM-DD)
            
        Returns:
            Dict[str, Any]: Betting odds
        """
        params = {}
        
        if game_id:
            params['game_id'] = game_id
        
        if date:
            if isinstance(date, datetime):
                date = date.strftime('%Y-%m-%d')
            params['date'] = date
        
        return self.request('odds', params=params)
    
    def get_todays_games(self) -> Dict[str, Any]:
        """
        Get games scheduled for today
        
        Returns:
            Dict[str, Any]: Today's games
        """
        today = datetime.now().strftime('%Y-%m-%d')
        return self.get_games(start_date=today, end_date=today)
    
    def get_todays_odds(self) -> Dict[str, Any]:
        """
        Get betting odds for today's games
        
        Returns:
            Dict[str, Any]: Today's betting odds
        """
        today = datetime.now().strftime('%Y-%m-%d')
        # Always force a refresh for today's odds to ensure maximum accuracy
        return self.get_odds(date=today, force_refresh=True)
    
    def get_team_stats(self,
                      team_id: int,
                      season: Optional[int] = None,
                      start_date: Optional[Union[str, datetime]] = None,
                      end_date: Optional[Union[str, datetime]] = None,
                      page: int = 1,
                      per_page: int = 100) -> Dict[str, Any]:
        """
        Get aggregated team statistics
        
        Args:
            team_id (int): Team ID to get stats for
            season (Optional[int]): Season year (e.g., 2023 for 2023-24 season)
            start_date (Optional[Union[str, datetime]]): Start date for filtering stats (YYYY-MM-DD)
            end_date (Optional[Union[str, datetime]]): End date for filtering stats (YYYY-MM-DD)
            page (int): Page number for pagination
            per_page (int): Number of items per page
            
        Returns:
            Dict[str, Any]: Team statistics
        """
        params = {
            'team_id': team_id,
            'page': page,
            'per_page': per_page
        }
        
        if season:
            params['season'] = season
            
        # Convert datetime to string if needed
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            params['start_date'] = start_date
        
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            params['end_date'] = end_date
        
        return self.request('team_stats', params=params)
    
    def get_current_season(self) -> Dict[str, Any]:
        """
        Get information about the current NBA season
        
        Returns:
            Dict[str, Any]: Current season information
        """
        return self.request('season', force_refresh=True)
    
    def get_draft_data(self, 
                      year: Optional[int] = None,
                      round_num: Optional[int] = None,
                      page: int = 1,
                      per_page: int = 100) -> Dict[str, Any]:
        """
        Get NBA draft data
        
        Args:
            year (Optional[int]): Draft year (e.g., 2023)
            round_num (Optional[int]): Draft round (1 or 2)
            page (int): Page number for pagination
            per_page (int): Number of items per page
            
        Returns:
            Dict[str, Any]: Draft data
        """
        params = {
            'page': page,
            'per_page': per_page
        }
        
        if year:
            params['year'] = year
            
        if round_num:
            params['round'] = round_num
            
        return self.request('draft', params=params)
    
    def get_game_plays(self, 
                      game_id: int,
                      period: Optional[int] = None,
                      page: int = 1,
                      per_page: int = 100) -> Dict[str, Any]:
        """
        Get play-by-play data for a specific game
        
        Args:
            game_id (int): Game ID
            period (Optional[int]): Filter by period/quarter (1-4 for regulation)
            page (int): Page number for pagination
            per_page (int): Number of items per page
            
        Returns:
            Dict[str, Any]: Play-by-play data
        """
        params = {
            'game_id': game_id,
            'page': page,
            'per_page': per_page
        }
        
        if period:
            params['period'] = period
            
        return self.request('plays', params=params)
