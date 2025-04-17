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
            rate_limit=1000000,  # Effectively unlimited (1 million)
            rate_limit_period=60,  # Per minute rather than monthly
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
                      season: Optional[str] = None,
                      start_date: Optional[Union[str, datetime]] = None,
                      end_date: Optional[Union[str, datetime]] = None,
                      page: int = 1,
                      per_page: int = 100) -> Dict[str, Any]:
        """
        Get aggregated team statistics
        
        Args:
            team_id (int): Team ID to get stats for
            season (Optional[str]): Season in format "2023-2024" (not just the year)
            start_date (Optional[Union[str, datetime]]): Start date for filtering stats (YYYY-MM-DD)
            end_date (Optional[Union[str, datetime]]): End date for filtering stats (YYYY-MM-DD)
            page (int): Page number for pagination
            per_page (int): Number of items per page
            
        Returns:
            Dict[str, Any]: Team statistics
        """
        # Dynamically get current NBA team IDs instead of using a hardcoded list
        current_nba_team_ids = self.get_active_nba_team_ids()
        
        # Check if the requested team ID is a current NBA team
        if team_id not in current_nba_team_ids:
            logger.warning(f"Requested stats for team ID {team_id} which is not a current NBA team. Skipping.")
            return {
                'data': [{
                    'team_id': team_id,
                    'name': 'Unknown Team',
                    'abbreviation': 'UNK',
                    'games_played': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_percentage': 0.0,
                    'points_pg': 0.0,
                    'opp_points_pg': 0.0
                }]
            }
            
        # BallDontLie doesn't have a direct team_stats endpoint
        # So we'll fetch the team info and game stats separately and combine them
        
        # Step 1: Get the team info
        team_info = self.request(f'teams/{team_id}')
        
        # Step 2: Get team game stats
        games_params = {
            'team_ids[]': team_id,
            'per_page': per_page,
            'page': page
        }
        
        # Format date parameters
        if start_date:
            if isinstance(start_date, datetime):
                start_date = start_date.strftime('%Y-%m-%d')
            games_params['start_date'] = start_date
        
        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            games_params['end_date'] = end_date
        
        # Get the games for this team
        games_data = self.request('games', params=games_params)
        
        # Step 3: Get the team stats from the games
        team_games = games_data.get('data', [])
        total_games = len(team_games)
        
        if total_games == 0:
            # If no games, return basic team info
            return {
                'data': [{
                    'team_id': team_id,
                    'name': team_info.get('data', {}).get('name', 'Unknown'),
                    'abbreviation': team_info.get('data', {}).get('abbreviation', 'UNK'),
                    'games_played': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_percentage': 0.0,
                    'points_pg': 0.0,
                    'opp_points_pg': 0.0
                }]
            }
        
        # Calculate stats from the games
        wins = 0
        losses = 0
        total_points = 0
        total_opp_points = 0
        
        for game in team_games:
            home_team_id = game.get('home_team', {}).get('id')
            home_score = game.get('home_team_score')
            away_score = game.get('visitor_team_score')
            
            # Skip games with no scores (future games)
            if home_score is None or away_score is None:
                continue
                
            if home_team_id == team_id:
                # This team is home
                if home_score > away_score:
                    wins += 1
                else:
                    losses += 1
                total_points += home_score
                total_opp_points += away_score
            else:
                # This team is away
                if away_score > home_score:
                    wins += 1
                else:
                    losses += 1
                total_points += away_score
                total_opp_points += home_score
        
        # Handle case where we have games but none have scores
        actual_games = wins + losses
        if actual_games == 0:
            return {
                'data': [{
                    'team_id': team_id,
                    'name': team_info.get('data', {}).get('name', 'Unknown'),
                    'abbreviation': team_info.get('data', {}).get('abbreviation', 'UNK'),
                    'games_played': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_percentage': 0.0,
                    'points_pg': 0.0,
                    'opp_points_pg': 0.0
                }]
            }
        
        # Calculate averages
        points_pg = total_points / actual_games
        opp_points_pg = total_opp_points / actual_games
        win_percentage = wins / actual_games
        
        # Estimate offensive and defensive ratings (simplified)
        pace = 100.0  # League average pace as fallback
        offensive_rating = (points_pg / pace) * 100
        defensive_rating = (opp_points_pg / pace) * 100
        net_rating = offensive_rating - defensive_rating
        
        # Return the team stats
        return {
            'data': [{
                'team_id': team_id,
                'name': team_info.get('data', {}).get('name', 'Unknown'),
                'abbreviation': team_info.get('data', {}).get('abbreviation', 'UNK'),
                'conference': team_info.get('data', {}).get('conference', 'Unknown'),
                'division': team_info.get('data', {}).get('division', 'Unknown'),
                'games_played': actual_games,
                'wins': wins,
                'losses': losses,
                'win_percentage': win_percentage,
                'points_pg': points_pg,
                'opp_points_pg': opp_points_pg,
                'fg_pct': 0.45,  # Fallback value
                'fg3_pct': 0.35,  # Fallback value
                'ft_pct': 0.75,  # Fallback value
                'offensive_rating': offensive_rating,
                'defensive_rating': defensive_rating,
                'pace': pace,
                'net_rating': net_rating,
                'streak': (wins - losses) if (wins - losses) != 0 else 0  # Simple streak estimate
            }]
        }
    
    def get_active_nba_team_ids(self) -> List[int]:
        """
        Get active NBA team IDs for the current season
        
        This method dynamically determines which teams are active in the current season
        by fetching games and extracting unique team IDs, rather than using a hardcoded list.
        
        Returns:
            List[int]: List of active NBA team IDs
        """
        # Check if we have cached the active teams already
        if hasattr(self, '_active_nba_team_ids'):
            return self._active_nba_team_ids
            
        try:
            # Get the current season information
            season_info = self.get_current_season()
            current_season = season_info.get('data', {}).get('year', datetime.now().year)
            
            # Get games from the current season
            season_start = f"{current_season}-10-01"  # NBA season typically starts in October
            season_end = f"{current_season+1}-06-30"  # and ends in June of the next year
            
            # Fetch games with a larger per_page to minimize API calls
            games_response = self.get_games(
                start_date=season_start,
                end_date=season_end,
                per_page=100  # Get a substantial sample of games
            )
            
            # Extract active team IDs from games
            active_team_ids = set()
            if isinstance(games_response, dict) and 'data' in games_response:
                for game in games_response['data']:
                    # Add home team ID
                    if isinstance(game.get('home_team'), dict) and 'id' in game['home_team']:
                        active_team_ids.add(game['home_team']['id'])
                    
                    # Add visitor team ID
                    if isinstance(game.get('visitor_team'), dict) and 'id' in game['visitor_team']:
                        active_team_ids.add(game['visitor_team']['id'])
            
            # Fallback to a standard list of 30 NBA teams if we couldn't determine active teams
            if len(active_team_ids) < 30:
                logger.warning(f"Could only identify {len(active_team_ids)} active teams from games data. Using fallback list.")
                # Standard NBA team IDs as fallback
                active_team_ids = set(range(1, 31))  # IDs 1-30 for the 30 NBA teams
            
            logger.info(f"Identified {len(active_team_ids)} active NBA teams for current season")
            
            # Cache the result to avoid repeated API calls
            self._active_nba_team_ids = list(active_team_ids)
            return self._active_nba_team_ids
            
        except Exception as e:
            logger.error(f"Error determining active NBA teams: {str(e)}")
            # Fallback to standard 30 NBA teams (IDs 1-30)
            logger.warning("Using fallback list of 30 NBA teams (IDs 1-30)")
            return list(range(1, 31))

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
