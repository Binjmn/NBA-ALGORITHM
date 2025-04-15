"""
The Odds API Client

This module provides a client for interacting with The Odds API,
which offers sports betting odds and related data.

Usage:
    from src.api.theodds_client import TheOddsClient
    
    # Create an instance
    client = TheOddsClient()
    
    # Get available sports
    sports = client.get_sports()
    
    # Get NBA odds
    nba_odds = client.get_nba_odds()
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union

from src.api.base_client import BaseAPIClient
from config.api_keys import get_api_key

# Configure logging
logger = logging.getLogger(__name__)


class TheOddsClient(BaseAPIClient):
    """Client for The Odds API"""
    
    # Constants for the API
    SPORT_KEY = "basketball_nba"
    DEFAULT_REGIONS = ["us"]
    DEFAULT_MARKETS = ["h2h", "spreads", "totals", "player_points", "player_rebounds", 
                       "player_assists", "player_threes", "player_double_double", 
                       "player_triple_double", "player_blocks", "player_steals"]
    DEFAULT_ODDS_FORMAT = "decimal"
    DEFAULT_DATE_FORMAT = "iso8601"
    
    def __init__(self):
        """Initialize The Odds API client"""
        api_key = get_api_key('theodds')
        if not api_key:
            raise ValueError("The Odds API key not found")
        
        # Define time-sensitive endpoints that should have a shorter cache TTL
        time_sensitive_endpoints = [
            'odds',    # Betting odds change frequently
            'scores',  # Game scores update live
            'events'   # Event status can change
        ]
        
        # Initialize base client
        super().__init__(
            base_url="https://api.the-odds-api.com/v4",
            api_key=api_key,
            rate_limit=500,     # Free plan has 500 requests per month
            rate_limit_period=2592000,  # 30 days in seconds
            cache_ttl=21600,    # Regular data cache for 6 hours
            time_sensitive_endpoints=time_sensitive_endpoints,
            time_sensitive_ttl=900  # Time-sensitive data cached for only 15 minutes
        )
        logger.info("Initialized The Odds API client")
    
    def _add_api_key_param(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add API key to parameters
        
        Args:
            params (Optional[Dict[str, Any]]): Query parameters
            
        Returns:
            Dict[str, Any]: Parameters with API key
        """
        if params is None:
            params = {}
        
        params['apiKey'] = self.api_key
        return params
    
    def request(self, endpoint: str, method: str = 'GET', params: Optional[Dict[str, Any]] = None,
                data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,
                use_cache: bool = True, force_refresh: bool = False, min_data_points: int = 1,
                max_retries: int = 3, retry_delay: int = 2) -> Dict[str, Any]:
        """
        Make a request to The Odds API
        
        For The Odds API, we need to pass the API key as a query parameter instead of
        in the headers as in the base class.
        
        Args:
            endpoint (str): API endpoint
            method (str): HTTP method (GET, POST, etc.)
            params (Optional[Dict[str, Any]]): Query parameters
            data (Optional[Dict[str, Any]]): Request body for POST/PUT requests
            headers (Optional[Dict[str, str]]): Additional headers
            use_cache (bool): Whether to use cache for GET requests
            force_refresh (bool): Whether to bypass cache and force a fresh request
            min_data_points (int): Minimum number of data points required for caching
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Delay between retries in seconds
            
        Returns:
            Dict[str, Any]: API response
        """
        # Add API key to parameters
        params = self._add_api_key_param(params)
        
        # The Odds API doesn't use Authorization header, so we'll override it
        if headers is None:
            headers = {}
        headers.pop('Authorization', None)
        
        return super().request(
            endpoint=endpoint,
            method=method,
            params=params,
            data=data,
            headers=headers,
            use_cache=use_cache,
            force_refresh=force_refresh,
            min_data_points=min_data_points,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
    
    def get_sports(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get available sports
        
        Args:
            force_refresh (bool): Whether to bypass cache and force a fresh request
            
        Returns:
            List[Dict[str, Any]]: Available sports
        """
        response = self.request('sports', force_refresh=force_refresh)
        return response
    
    def is_nba_available(self) -> bool:
        """
        Check if NBA is available
        
        Returns:
            bool: True if NBA is available, False otherwise
        """
        sports = self.get_sports()
        return any(sport['key'] == self.SPORT_KEY for sport in sports)
    
    def get_nba_odds(self, 
                     regions: Optional[List[str]] = None,
                     markets: Optional[List[str]] = None,
                     odds_format: str = DEFAULT_ODDS_FORMAT,
                     date_format: str = DEFAULT_DATE_FORMAT,
                     force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get NBA odds
        
        Args:
            regions (Optional[List[str]]): Regions to get odds for
            markets (Optional[List[str]]): Markets to get odds for
            odds_format (str): Format of odds (decimal, american, etc.)
            date_format (str): Format of dates
            force_refresh (bool): Whether to bypass cache and force a fresh request
            
        Returns:
            List[Dict[str, Any]]: NBA odds
        """
        regions = regions or self.DEFAULT_REGIONS
        markets = markets or self.DEFAULT_MARKETS
        
        params = {
            'regions': ','.join(regions),
            'markets': ','.join(markets),
            'oddsFormat': odds_format,
            'dateFormat': date_format
        }
        
        endpoint = f'sports/{self.SPORT_KEY}/odds'
        response = self.request(endpoint, params=params, force_refresh=force_refresh)
        return response
    
    def get_live_scores(self, 
                       days_from: int = 3,
                       date_format: str = DEFAULT_DATE_FORMAT,
                       force_refresh: bool = True) -> List[Dict[str, Any]]:
        """
        Get live scores for NBA games
        
        Always forces a refresh for live scores to ensure data accuracy
        
        Args:
            days_from (int): Number of days from today to include
            date_format (str): Format of dates
            force_refresh (bool): Whether to bypass cache and force a fresh request
            
        Returns:
            List[Dict[str, Any]]: Live scores
        """
        params = {
            'daysFrom': days_from,
            'dateFormat': date_format
        }
        
        endpoint = f'sports/{self.SPORT_KEY}/scores'
        # Always force a refresh for live scores
        response = self.request(endpoint, params=params, use_cache=False, force_refresh=True)
        return response
    
    def get_nba_events(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get NBA events (games)
        
        Args:
            force_refresh (bool): Whether to bypass cache and force a fresh request
            
        Returns:
            List[Dict[str, Any]]: NBA events
        """
        endpoint = f'sports/{self.SPORT_KEY}/events'
        response = self.request(endpoint, force_refresh=force_refresh)
        return response
    
    def get_game_odds(self, 
                     event_id: str,
                     regions: Optional[List[str]] = None,
                     markets: Optional[List[str]] = None,
                     odds_format: str = DEFAULT_ODDS_FORMAT,
                     date_format: str = DEFAULT_DATE_FORMAT,
                     force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get odds for a specific game
        
        Args:
            event_id (str): Event ID
            regions (Optional[List[str]]): Regions to get odds for
            markets (Optional[List[str]]): Markets to get odds for
            odds_format (str): Format of odds (decimal, american, etc.)
            date_format (str): Format of dates
            force_refresh (bool): Whether to bypass cache and force a fresh request
            
        Returns:
            Dict[str, Any]: Game odds
        """
        regions = regions or self.DEFAULT_REGIONS
        markets = markets or self.DEFAULT_MARKETS
        
        params = {
            'regions': ','.join(regions),
            'markets': ','.join(markets),
            'oddsFormat': odds_format,
            'dateFormat': date_format
        }
        
        endpoint = f'sports/{self.SPORT_KEY}/events/{event_id}/odds'
        response = self.request(endpoint, params=params, force_refresh=force_refresh)
        return response
    
    def get_historical_odds(self, 
                           date: Union[str, datetime],
                           regions: Optional[List[str]] = None,
                           markets: Optional[List[str]] = None,
                           odds_format: str = DEFAULT_ODDS_FORMAT,
                           date_format: str = DEFAULT_DATE_FORMAT) -> List[Dict[str, Any]]:
        """
        Get historical odds for NBA games
        
        Args:
            date (Union[str, datetime]): Date for historical odds (ISO format or datetime)
            regions (Optional[List[str]]): Regions to get odds for
            markets (Optional[List[str]]): Markets to get odds for
            odds_format (str): Format of odds (decimal, american, etc.)
            date_format (str): Format of dates
            
        Returns:
            List[Dict[str, Any]]: Historical odds
        """
        regions = regions or self.DEFAULT_REGIONS
        markets = markets or self.DEFAULT_MARKETS
        
        # Convert datetime to ISO string if needed
        if isinstance(date, datetime):
            date_str = date.isoformat()
        else:
            date_str = date
        
        params = {
            'regions': ','.join(regions),
            'markets': ','.join(markets),
            'date': date_str,
            'oddsFormat': odds_format,
            'dateFormat': date_format
        }
        
        endpoint = f'historical/sports/{self.SPORT_KEY}/odds'
        response = self.request(endpoint, params=params)
        return response
    
    def get_historical_events(self, date: Union[str, datetime]) -> List[Dict[str, Any]]:
        """
        Get historical events (games) for NBA
        
        Args:
            date (Union[str, datetime]): Date for historical events (ISO format or datetime)
            
        Returns:
            List[Dict[str, Any]]: Historical events
        """
        # Convert datetime to ISO string if needed
        if isinstance(date, datetime):
            date_str = date.isoformat()
        else:
            date_str = date
        
        params = {
            'date': date_str
        }
        
        endpoint = f'historical/sports/{self.SPORT_KEY}/events'
        response = self.request(endpoint, params=params)
        return response
    
    def get_historical_game_odds(self, 
                                event_id: str,
                                date: Union[str, datetime],
                                regions: Optional[List[str]] = None,
                                markets: Optional[List[str]] = None,
                                odds_format: str = DEFAULT_ODDS_FORMAT,
                                date_format: str = DEFAULT_DATE_FORMAT) -> Dict[str, Any]:
        """
        Get historical odds for a specific NBA game
        
        Args:
            event_id (str): Event ID
            date (Union[str, datetime]): Date for historical odds (ISO format or datetime)
            regions (Optional[List[str]]): Regions to get odds for
            markets (Optional[List[str]]): Markets to get odds for
            odds_format (str): Format of odds (decimal, american, etc.)
            date_format (str): Format of dates
            
        Returns:
            Dict[str, Any]: Historical game odds
        """
        regions = regions or self.DEFAULT_REGIONS
        markets = markets or self.DEFAULT_MARKETS
        
        # Convert datetime to ISO string if needed
        if isinstance(date, datetime):
            date_str = date.isoformat()
        else:
            date_str = date
        
        params = {
            'regions': ','.join(regions),
            'markets': ','.join(markets),
            'date': date_str,
            'oddsFormat': odds_format,
            'dateFormat': date_format
        }
        
        endpoint = f'historical/sports/{self.SPORT_KEY}/events/{event_id}/odds'
        response = self.request(endpoint, params=params)
        return response
    
    def get_todays_odds(self, 
                       regions: Optional[List[str]] = None,
                       markets: Optional[List[str]] = None,
                       odds_format: str = DEFAULT_ODDS_FORMAT) -> List[Dict[str, Any]]:
        """
        Get odds for today's NBA games
        
        Args:
            regions (Optional[List[str]]): Regions to get odds for
            markets (Optional[List[str]]): Markets to get odds for
            odds_format (str): Format of odds
            
        Returns:
            List[Dict[str, Any]]: Today's odds
        """
        # Always force a refresh for today's odds to ensure data accuracy
        return self.get_nba_odds(
            regions=regions,
            markets=markets,
            odds_format=odds_format,
            force_refresh=True
        )
    
    def convert_to_est(self, timestamp: Union[str, datetime]) -> datetime:
        """
        Convert a timestamp to Eastern Standard Time (EST)
        
        Args:
            timestamp (Union[str, datetime]): Timestamp to convert
            
        Returns:
            datetime: Timestamp in EST
        """
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = timestamp
            
        # Ensure dt is aware (has timezone info)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
            
        # Convert to EST (UTC-5)
        est = timezone(timedelta(hours=-5))
        return dt.astimezone(est)
