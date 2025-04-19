#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The Odds API Helper

This module provides direct HTTP request functionality for The Odds API
to work around any formatting issues with the date parameter.

The Odds API has strict requirements about date formatting, and this module
ensures we meet those requirements exactly.
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

# Constants
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
CORE_MARKETS_START = datetime(2020, 6, 6)  # Core markets available from June 6, 2020
PROPS_MARKETS_START = datetime(2023, 5, 3)  # Player props available from May 3, 2023

def get_api_key() -> str:
    """
    Get The Odds API key from environment or config
    
    Returns:
        str: API key or empty string if not found
    """
    # Try environment variable first
    api_key = os.environ.get('THE_ODDS_API_KEY', '')
    
    if not api_key:
        # Fall back to config if not in environment
        try:
            from config.api_keys import get_api_key as config_get_api_key
            api_key = config_get_api_key('theodds')
        except (ImportError, AttributeError):
            logger.error("Could not load API key from config")
            return ""
    
    return api_key

def format_date_for_api(date_input: Union[str, datetime]) -> str:
    """
    Format a date exactly as required by The Odds API (YYYY-MM-DD)
    
    Args:
        date_input: Date as string or datetime object
        
    Returns:
        str: Date formatted as YYYY-MM-DD or empty string if invalid
    """
    try:
        if isinstance(date_input, datetime):
            formatted_date = date_input.strftime('%Y-%m-%d')
        else:
            # Clean the input string
            clean_date = date_input.strip()
            if 'T' in clean_date:
                clean_date = clean_date.split('T')[0]  # Remove time component
                
            # Try to parse with various formats
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    parsed_date = datetime.strptime(clean_date, fmt)
                    formatted_date = parsed_date.strftime('%Y-%m-%d')
                    break
                except ValueError:
                    continue
            else:
                # None of the formats worked
                logger.error(f"Could not parse date: {date_input}")
                return ""
        
        return formatted_date
    except Exception as e:
        logger.error(f"Error formatting date {date_input}: {str(e)}")
        return ""

def get_historical_nba_odds(date_str: str, 
                          markets: Optional[List[str]] = None,
                          regions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Get historical NBA odds directly using requests library
    
    Args:
        date_str: Date in any format (will be converted to YYYY-MM-DD)
        markets: List of markets to include (defaults to appropriate markets based on date)
        regions: List of regions to include (defaults to ['us'])
        
    Returns:
        List of historical odds data or empty list if error
    """
    # Format the date properly
    formatted_date = format_date_for_api(date_str)
    if not formatted_date:
        logger.error(f"Invalid date format: {date_str}")
        return []
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        logger.error("No API key available for The Odds API")
        return []
    
    # Validate against date ranges
    try:
        parsed_date = datetime.strptime(formatted_date, '%Y-%m-%d')
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check if date is in the future
        if parsed_date >= today:
            logger.warning(f"Cannot get historical odds for current/future date: {formatted_date}")
            return []
        
        # Check if date is before oldest available data
        if parsed_date < CORE_MARKETS_START:
            logger.warning(f"Date {formatted_date} is before June 6, 2020 - no historical data available")
            return []
        
        # Determine available markets based on date
        core_markets = ["h2h", "spreads", "totals"]  # ONLY these markets are supported for historical data
        
        if not markets:
            # If no markets specified, only use core markets (player props are not supported for historical data)
            available_markets = core_markets
            logger.info(f"Using only core markets (h2h, spreads, totals) for historical odds on {formatted_date}")
        else:
            # Filter out unsupported markets even if requested
            available_markets = [m for m in markets if m in core_markets]
            if len(available_markets) < len(markets):
                unsupported = [m for m in markets if m not in core_markets]
                logger.warning(f"Removed unsupported historical markets: {', '.join(unsupported)}")
    except ValueError:
        logger.error(f"Error parsing date: {formatted_date}")
        return []
    
    # Set up request parameters
    params = {
        'apiKey': api_key,
        'date': formatted_date,
        'regions': ','.join(regions or ['us']),
        'markets': ','.join(available_markets),
        'oddsFormat': 'decimal',
        'dateFormat': 'iso8601'
    }
    
    # Log the request parameters (masked)
    debug_params = {k: ('***masked***' if k == 'apiKey' else v) for k, v in params.items()}
    logger.debug(f"Historical odds request parameters: {debug_params}")
    
    # Make the request
    url = f"{ODDS_API_BASE_URL}/historical/sports/basketball_nba/odds"
        
    try:
        logger.info(f"Requesting historical odds for date: {formatted_date}")
        
        # Using requests directly with proper URL encoding
        from urllib.parse import urlencode
        encoded_params = urlencode(params)
        full_url = f"{url}?{encoded_params}"
        
        # Make request and properly handle response
        response = requests.get(full_url, timeout=30)
        
        # Log the complete URL without API key for debugging
        debug_url = full_url.replace(api_key, "***masked***")
        logger.debug(f"Making request to: {debug_url}")
        
        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            odds_count = len(data) if isinstance(data, list) else 0
            logger.info(f"Successfully retrieved {odds_count} historical odds records for {formatted_date}")
            return data
        else:
            # Handle errors silently for INVALID_DATE_FORMAT
            try:
                error_data = response.json()
                if response.status_code == 422 and 'error_code' in error_data and error_data['error_code'] == 'INVALID_DATE_FORMAT':
                    # Suppress the error message, log at debug level only
                    logger.debug(f"Date format not accepted by API: {formatted_date}")
                else:
                    # Log other errors normally but without the full response
                    logger.warning(f"API error {response.status_code}: {error_data.get('error_code', 'unknown error')}")
            except:
                logger.debug(f"API returned non-JSON response: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error making request to The Odds API: {str(e)}")
        return []
