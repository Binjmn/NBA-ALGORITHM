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
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union

from src.api.base_client import BaseAPIClient

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
    # Limited markets supported for historical data
    HISTORICAL_MARKETS = ["h2h", "spreads", "totals"]
    DEFAULT_ODDS_FORMAT = "decimal"
    DEFAULT_DATE_FORMAT = "iso8601"
    
    def __init__(self):
        """
        Initialize The Odds API client
        
        Base URL: https://api.the-odds-api.com/v4
        Documentation: https://the-odds-api.com/liveapi/guides/v4/
        """
        # Get API key directly from environment variable first, then fall back to config
        api_key = os.environ.get('THE_ODDS_API_KEY')
        
        if api_key:
            logger.info("Using The Odds API key from environment variable")
        else:
            # Fall back to config if not in environment
            from config.api_keys import get_api_key
            api_key = get_api_key('theodds')
            logger.info("Using The Odds API key from configuration file")
        
        if not api_key:
            logger.warning("The Odds API key not found or empty")
        else:
            # Don't log the full key, just a masked version for security
            masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***masked***"
            logger.info(f"The Odds API key loaded (masked: {masked_key})")
        
        # Initialize base client
        super().__init__(
            base_url="https://api.the-odds-api.com/v4",
            api_key=api_key,
            rate_limit=1000000,  # Effectively unlimited (1 million)
            rate_limit_period=60,  # Per minute rather than monthly
            cache_ttl=3600 * 3,  # Cache for 3 hours
            time_sensitive_endpoints=[
                'odds',    # Betting odds change frequently
                'scores',  # Game scores update live
                'events'   # Event status can change
            ],
            time_sensitive_ttl=600  # 10 minutes for time-sensitive data
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
        
        # Check if we should try to use cache
        if method == 'GET' and use_cache and self.cache_enabled and not force_refresh:
            # Try to get from cache first
            cached_response = self._get_from_cache(endpoint, params, min_data_points)
            if cached_response is not None:
                return cached_response
        
        # Make the request using the base class method - without the cache parameters
        response = super().request(
            endpoint=endpoint,
            method=method,
            params=params,
            data=data,
            headers=headers
        )
        
        # Save GET responses to cache if caching is enabled
        if method == 'GET' and self.cache_enabled and response and use_cache:
            self._save_to_cache(endpoint, params, response)
        
        return response
    
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
                           date_format: str = DEFAULT_DATE_FORMAT,
                           force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get historical odds for NBA games on a specific date
        
        The Odds API offers historical odds with these constraints:
        - Core markets (moneyline, spreads, totals): Available from June 6, 2020
        - Player props markets: Available from May 3, 2023
        - Dates must be in YYYY-MM-DD format and in the past
        
        Args:
            date (Union[str, datetime]): Date for historical odds (will be converted to YYYY-MM-DD)
            regions (Optional[List[str]]): Regions to get odds for
            markets (Optional[List[str]]): Markets to get odds for
            odds_format (str): Format of odds (decimal, american, etc.)
            date_format (str): Format of dates in the response
            force_refresh (bool): Whether to bypass cache and force a fresh request
            
        Returns:
            List[Dict[str, Any]]: Historical odds
        """
        # Force date to YYYY-MM-DD format as strictly required by the API
        try:
            # Parse input date to ensure proper format for API
            if isinstance(date, datetime):
                parsed_date = date
            elif isinstance(date, str):
                # Clean the date string first
                clean_date = date.strip()
                if 'T' in clean_date:
                    clean_date = clean_date.split('T')[0]  # Remove time component
                
                # Try various date formats, but always output YYYY-MM-DD
                try:
                    parsed_date = datetime.strptime(clean_date, '%Y-%m-%d')
                except ValueError:
                    # Try other common formats
                    try:
                        parsed_date = datetime.strptime(clean_date, '%m/%d/%Y')
                    except ValueError:
                        try:
                            parsed_date = datetime.strptime(clean_date, '%d/%m/%Y')
                        except ValueError:
                            try:
                                parsed_date = datetime.strptime(clean_date, '%Y/%m/%d')
                            except ValueError:
                                logger.error(f"Cannot parse date: {date} - The Odds API requires YYYY-MM-DD format")
                                return []
            else:
                logger.error(f"Invalid date type: {type(date)}. Must be string or datetime.")
                return []
            
            # Format to YYYY-MM-DD as required by API - ensure no spaces
            date_str = parsed_date.strftime('%Y-%m-%d').strip()
            
            # Double-check formatting - The Odds API is extremely strict
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                logger.error(f"Date string {date_str} does not match required format YYYY-MM-DD")
                return []
            
            # Core markets available from June 6, 2020
            core_markets_start = datetime(2020, 6, 6)
            # Player props available from May 3, 2023
            props_markets_start = datetime(2023, 5, 3)
            
            # Skip future dates
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            if parsed_date >= today:
                logger.warning(f"Cannot request historical odds for current or future date: {date_str}")
                return []
            
            # Check if date is before any historical data is available
            if parsed_date < core_markets_start:
                logger.warning(f"Date {date_str} is before June 6, 2020. The Odds API doesn't provide historical data before this date.")
                return []
            
            # Determine which markets are available based on the date
            if markets:
                # User specified markets - check validity against date
                if parsed_date < props_markets_start:
                    # Before May 3, 2023 - only core markets available
                    valid_markets = [m for m in markets if m in self.HISTORICAL_MARKETS]
                    if len(valid_markets) < len(markets):
                        logger.warning(f"Some requested markets not available for {date_str} (before May 3, 2023). Using only core markets.")
                    historical_markets = valid_markets
                else:
                    # After May 3, 2023 - all markets available
                    historical_markets = markets
            else:
                # No markets specified - use appropriate defaults based on date
                if parsed_date < props_markets_start:
                    historical_markets = self.HISTORICAL_MARKETS
                    logger.info(f"Using core markets for {date_str} (before May 3, 2023)")
                else:
                    historical_markets = self.DEFAULT_MARKETS
                    logger.info(f"Using all markets including player props for {date_str}")
            
            # Set up params with required markets
            regions = regions or self.DEFAULT_REGIONS
            
            endpoint = f'historical/sports/{self.SPORT_KEY}/odds'
            
            try:
                logger.info(f"Requesting historical odds for date: {date_str} with markets: {', '.join(historical_markets)}")
                
                # Directly build the URL with date parameter to avoid formatting issues
                url_params = {
                    'apiKey': self.api_key,
                    'regions': ','.join(regions),
                    'markets': ','.join(historical_markets),
                    'oddsFormat': odds_format,
                    'dateFormat': date_format
                }
                
                # Create a copy of the params dictionary
                fixed_params = dict(url_params)
                # Add date parameter directly to avoid formatting issues
                fixed_params['date'] = date_str
                
                # Log the exact parameters being sent
                debug_params = {k: v for k, v in fixed_params.items() if k != 'apiKey'}
                debug_params['apiKey'] = '***masked***'
                logger.debug(f"Historical odds request parameters: {debug_params}")
                
                # Make the request with the fixed parameters
                response = self.request(endpoint, params=fixed_params, force_refresh=force_refresh)
                
                if response:
                    if isinstance(response, list):
                        odds_count = len(response)
                        logger.info(f"Successfully retrieved {odds_count} historical odds records for {date_str}")
                        return response
                    elif isinstance(response, dict) and 'data' in response:
                        odds_count = len(response['data'])
                        logger.info(f"Successfully retrieved {odds_count} historical odds records for {date_str}")
                        return response['data']
                    else:
                        logger.warning(f"Unexpected response format from The Odds API for {date_str}: {response}")
                        return []
                else:
                    logger.warning(f"No historical odds data received for {date_str} - API returned empty response")
                    return []
                    
            except Exception as api_err:
                logger.error(f"Error fetching historical odds: {str(api_err)}")
                return []
                
        except Exception as e:
            logger.error(f"Error in historical odds preparation: {str(e)}")
            return []
    
    def get_historical_events(self, date: Union[str, datetime]) -> List[Dict[str, Any]]:
        """
        Get historical events for a specific date
        
        Args:
            date (Union[str, datetime]): Date for historical events (YYYY-MM-DD format)
            
        Returns:
            List[Dict[str, Any]]: Historical events
        """
        # Use the same robust date parsing logic as get_historical_odds
        try:
            # Parse input date to ensure proper format for API
            if isinstance(date, datetime):
                parsed_date = date
            elif isinstance(date, str):
                # Clean the date string first
                clean_date = date.strip()
                if 'T' in clean_date:
                    clean_date = clean_date.split('T')[0]  # Remove time component
                
                # Try various date formats, but always output YYYY-MM-DD
                try:
                    parsed_date = datetime.strptime(clean_date, '%Y-%m-%d')
                except ValueError:
                    # Try other common formats
                    try:
                        parsed_date = datetime.strptime(clean_date, '%m/%d/%Y')
                    except ValueError:
                        try:
                            parsed_date = datetime.strptime(clean_date, '%d/%m/%Y')
                        except ValueError:
                            try:
                                parsed_date = datetime.strptime(clean_date, '%Y/%m/%d')
                            except ValueError:
                                logger.error(f"Cannot parse date: {date} - The Odds API requires YYYY-MM-DD format")
                                return []
            else:
                logger.error(f"Invalid date type: {type(date)}. Must be string or datetime.")
                return []
            
            # Format to YYYY-MM-DD as required by API
            date_str = parsed_date.strftime('%Y-%m-%d')
            
            # Skip future dates
            today = datetime.now().strftime('%Y-%m-%d')
            if date_str > today:
                logger.warning(f"Cannot request historical events for future date: {date_str}")
                return []
            
            params = {
                'date': date_str
            }
            
            endpoint = f'historical/sports/{self.SPORT_KEY}/events'
            
            try:
                logger.info(f"Requesting historical events for date: {date_str}")
                response = self.request(endpoint, params=params)
                
                if response:
                    if isinstance(response, list):
                        logger.info(f"Successfully retrieved {len(response)} historical events for {date_str}")
                        return response
                    elif isinstance(response, dict) and 'data' in response:
                        logger.info(f"Successfully retrieved {len(response['data'])} historical events for {date_str}")
                        return response['data']
                    else:
                        logger.warning(f"Unexpected response format from The Odds API for events on {date_str}")
                        return []
                else:
                    logger.warning(f"No historical events data received for {date_str}")
                    return []
                    
            except Exception as api_err:
                logger.error(f"Error fetching historical events: {str(api_err)}")
                return []
                
        except Exception as e:
            logger.error(f"Error in historical events preparation: {str(e)}")
            return []
    
    def get_historical_game_odds(self, 
                                  event_id: str,
                                  date: Union[str, datetime],
                                  regions: Optional[List[str]] = None,
                                  markets: Optional[List[str]] = None,
                                  odds_format: str = DEFAULT_ODDS_FORMAT,
                                  date_format: str = DEFAULT_DATE_FORMAT) -> Dict[str, Any]:
        """
        Get historical odds for a specific game on a date
        
        Args:
            event_id (str): ID of the event/game
            date (Union[str, datetime]): Date for historical odds (YYYY-MM-DD format)
            regions (Optional[List[str]]): Regions to get odds for
            markets (Optional[List[str]]): Markets to get odds for
            odds_format (str): Format of odds (decimal, american, etc.)
            date_format (str): Format of dates in the response
            
        Returns:
            Dict[str, Any]: Historical odds for the game
        """
        # Use the same robust date parsing logic as get_historical_odds
        try:
            # Parse input date to ensure proper format for API
            if isinstance(date, datetime):
                parsed_date = date
            elif isinstance(date, str):
                # Clean the date string first
                clean_date = date.strip()
                if 'T' in clean_date:
                    clean_date = clean_date.split('T')[0]  # Remove time component
                
                # Try various date formats, but always output YYYY-MM-DD
                try:
                    parsed_date = datetime.strptime(clean_date, '%Y-%m-%d')
                except ValueError:
                    # Try other common formats
                    try:
                        parsed_date = datetime.strptime(clean_date, '%m/%d/%Y')
                    except ValueError:
                        try:
                            parsed_date = datetime.strptime(clean_date, '%d/%m/%Y')
                        except ValueError:
                            try:
                                parsed_date = datetime.strptime(clean_date, '%Y/%m/%d')
                            except ValueError:
                                logger.error(f"Cannot parse date: {date} - The Odds API requires YYYY-MM-DD format")
                                return {}
            else:
                logger.error(f"Invalid date type: {type(date)}. Must be string or datetime.")
                return {}
            
            # Format to YYYY-MM-DD as required by API
            date_str = parsed_date.strftime('%Y-%m-%d')
            
            # Skip future dates
            today = datetime.now().strftime('%Y-%m-%d')
            if date_str > today:
                logger.warning(f"Cannot request historical game odds for future date: {date_str}")
                return {}
            
            # Set up params with required markets
            regions = regions or self.DEFAULT_REGIONS
            historical_markets = markets or self.HISTORICAL_MARKETS
            
            params = {
                'regions': ','.join(regions),
                'markets': ','.join(historical_markets),
                'date': date_str,  # This must be YYYY-MM-DD
                'oddsFormat': odds_format,
                'dateFormat': date_format
            }
            
            endpoint = f'historical/sports/{self.SPORT_KEY}/events/{event_id}/odds'
            
            try:
                logger.info(f"Requesting historical game odds for event {event_id} on date: {date_str}")
                response = self.request(endpoint, params=params)
                
                if response:
                    logger.info(f"Successfully retrieved historical odds for game {event_id} on {date_str}")
                    return response
                else:
                    logger.warning(f"No historical game odds data received for game {event_id} on {date_str}")
                    return {}
                    
            except Exception as api_err:
                logger.error(f"Error fetching historical game odds: {str(api_err)}")
                return {}
                
        except Exception as e:
            logger.error(f"Error in historical game odds preparation: {str(e)}")
            return {}
    
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
