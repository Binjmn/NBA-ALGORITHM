#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Rate Limiter

Provides rate limiting functionality for API requests to prevent throttling.
Supports different rate limits for different API providers.
"""

import time
import threading
import logging
from typing import Dict, Any, Optional
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for API requests
    
    Provides token bucket implementation for rate limiting:
    - BallDontLie API: 6000 requests per minute
    - Odds API: 1500 requests per minute
    - Custom rate limits for other APIs
    """
    
    # Default rate limits (requests per minute)
    DEFAULT_RATE_LIMITS = {
        'balldontlie': 6000,
        'odds': 1500,
        'default': 100
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize rate limiter
        
        Args:
            config: Optional configuration dictionary with rate limits
        """
        self.config = config or {}
        self.rate_limits = self._get_rate_limits()
        
        # Create token buckets for each API
        self.buckets = {}
        self.request_times = {}
        self.locks = {}
        
        for api_name, rpm in self.rate_limits.items():
            # Convert requests per minute to seconds per request
            self.buckets[api_name] = rpm
            self.request_times[api_name] = deque(maxlen=rpm)
            self.locks[api_name] = threading.Lock()
            
        logger.info(f"Initialized rate limiter with limits: {self.rate_limits}")
    
    def _get_rate_limits(self) -> Dict[str, int]:
        """
        Get rate limits from config or use defaults
        
        Returns:
            Dictionary of API rate limits
        """
        rate_limits = self.DEFAULT_RATE_LIMITS.copy()
        
        # Override with config values if provided
        if self.config.get('api_rate_limits'):
            for api_name, limit in self.config.get('api_rate_limits', {}).items():
                rate_limits[api_name] = limit
        
        return rate_limits
    
    def wait_if_needed(self, api_name: str = 'default') -> float:
        """
        Wait if rate limit is exceeded
        
        Args:
            api_name: API provider name
            
        Returns:
            Wait time in seconds (0 if no wait)
        """
        # Use default if API name not recognized
        if api_name not in self.rate_limits:
            api_name = 'default'
            
        with self.locks[api_name]:
            now = time.time()
            
            # If bucket is not full, just add the request and return
            if len(self.request_times[api_name]) < self.buckets[api_name]:
                self.request_times[api_name].append(now)
                return 0
            
            # Check if the oldest request is more than a minute old
            if now - self.request_times[api_name][0] >= 60:
                self.request_times[api_name].append(now)
                return 0
            
            # Need to wait until oldest request is a minute old
            wait_time = 60 - (now - self.request_times[api_name][0])
            wait_time = max(0, wait_time)  # Ensure non-negative
            
            if wait_time > 0:
                logger.warning(f"Rate limit reached for {api_name} API. Waiting {wait_time:.2f} seconds.")
                time.sleep(wait_time)
                
            # Update with current time after waiting
            self.request_times[api_name].append(time.time())
            return wait_time


# Create a global instance for use throughout the application
default_rate_limiter = RateLimiter()
