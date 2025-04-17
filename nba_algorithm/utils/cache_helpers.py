#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cache Helper Utilities

This module provides helper functions to support the hybrid caching approach:
- Cached data for model training
- Fresh data for game-day predictions
- Selective caching for different data types
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
import functools

from .cache_manager import (
    read_cache, write_cache, clear_cache, CacheTier,
    get_cache_stats, CACHE_CONFIG
)

# Configure logging
logger = logging.getLogger(__name__)


def is_game_day(game_date: Union[str, date, datetime]) -> bool:
    """
    Check if the given date is today (game day)
    
    Args:
        game_date: Date to check, can be string in format 'YYYY-MM-DD' or date/datetime object
        
    Returns:
        Boolean indicating if the date is today
    """
    today = date.today()
    
    if isinstance(game_date, str):
        try:
            # Parse string date in format 'YYYY-MM-DD'
            game_date = datetime.strptime(game_date, '%Y-%m-%d').date()
        except ValueError:
            logger.warning(f"Invalid date format: {game_date}")
            return False
    
    if isinstance(game_date, datetime):
        game_date = game_date.date()
        
    return game_date == today


def cache_aware_fetch(data_type: str, fetch_func: Callable, 
                      identifier: Optional[str] = None,
                      force_refresh: bool = False,
                      is_prediction_day: bool = False,
                      **kwargs) -> Any:
    """
    Fetch data with smart caching depending on volatility and context
    
    This function implements the hybrid approach:
    - Always use fresh data on game day for time-sensitive data
    - Use cached data for model training and historical analysis
    - Respect the force_refresh flag to bypass cache
    
    Args:
        data_type: Type of data being fetched (used for determining cache policy)
        fetch_func: Function to call to fetch data if cache is invalid
        identifier: Optional identifier for the cached data
        force_refresh: If True, bypass cache completely
        is_prediction_day: If True, treat this as a real-time prediction day
        **kwargs: Additional arguments to pass to fetch_func
        
    Returns:
        The fetched data (from cache or fresh)
    """
    # Determine if we should force refresh based on data type and context
    config = CACHE_CONFIG.get(data_type, {'tier': CacheTier.REGULAR})
    tier = config.get('tier', CacheTier.REGULAR)
    
    # Always force refresh volatile data on prediction day
    should_force_refresh = force_refresh
    if is_prediction_day and tier == CacheTier.VOLATILE:
        should_force_refresh = True
        logger.debug(f"Force refreshing {data_type} data because it's volatile and prediction day")
    
    # Try to get from cache if not forcing refresh
    if not should_force_refresh:
        cached_data = read_cache(data_type, identifier=identifier, force_refresh=False)
        if cached_data is not None:
            logger.debug(f"Using cached {data_type} data")
            return cached_data
    
    # Cache miss or force refresh, fetch fresh data
    logger.debug(f"Fetching fresh {data_type} data")
    try:
        fresh_data = fetch_func(**kwargs)
        
        # Cache the fresh data if it's valid
        if fresh_data is not None:
            # Add metadata about the fetch context
            metadata = {
                'fetched_at': datetime.now().isoformat(),
                'force_refreshed': should_force_refresh,
                'is_prediction_day': is_prediction_day
            }
            
            # Add any additional metadata from kwargs
            if 'metadata' in kwargs:
                metadata.update(kwargs['metadata'])
                
            write_cache(data_type, fresh_data, identifier=identifier, metadata=metadata)
            
        return fresh_data
        
    except Exception as e:
        logger.error(f"Error fetching {data_type} data: {str(e)}")
        return None


def fetch_with_fallback(primary_data_type: str, fallback_data_type: str,
                        fetch_func: Callable, identifier: Optional[str] = None,
                        force_refresh: bool = False, **kwargs) -> Any:
    """
    Fetch data with a primary data type, falling back to a more stable data type if needed
    
    This supports the pattern of using the most recent volatile data when available,
    but falling back to more stable historical data when needed.
    
    Args:
        primary_data_type: The preferred, possibly more volatile data type
        fallback_data_type: The more stable fallback data type
        fetch_func: Function to call to fetch data
        identifier: Optional identifier for the cached data
        force_refresh: If True, bypass cache completely
        **kwargs: Additional arguments to pass to fetch_func
        
    Returns:
        The fetched data from either primary or fallback source
    """
    # Try primary data type first
    primary_data = cache_aware_fetch(
        primary_data_type, fetch_func, identifier=identifier,
        force_refresh=force_refresh, **kwargs
    )
    
    if primary_data is not None:
        logger.debug(f"Using primary {primary_data_type} data")
        return primary_data
    
    # Fall back to more stable data type
    logger.debug(f"Primary {primary_data_type} data not available, trying fallback {fallback_data_type}")
    fallback_data = cache_aware_fetch(
        fallback_data_type, fetch_func, identifier=identifier,
        force_refresh=force_refresh, **kwargs
    )
    
    if fallback_data is not None:
        logger.debug(f"Using fallback {fallback_data_type} data")
    else:
        logger.warning(f"Both primary and fallback data unavailable for {identifier or 'unknown'}")
        
    return fallback_data


def smart_cache_decorator(data_type: str):
    """
    Decorator to add smart caching to any function
    
    Args:
        data_type: The type of data being cached
        
    Returns:
        Decorated function with smart caching
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, force_refresh=False, is_prediction_day=False, **kwargs):
            # Generate a cache identifier from the args and kwargs
            arg_str = '_'.join(str(arg) for arg in args)
            kwarg_str = '_'.join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
            cache_id = f"{arg_str}_{kwarg_str}"
            
            return cache_aware_fetch(
                data_type, func, identifier=cache_id,
                force_refresh=force_refresh, is_prediction_day=is_prediction_day,
                args=args, kwargs=kwargs
            )
        return wrapper
    return decorator
