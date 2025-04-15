#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base API Client Module
Purpose: Provide a base class for API clients with common functionality like rate limiting,
caching, error handling, and retries.
"""

import time
import logging
import json
from typing import Any, Dict, Optional, Union, List, Tuple
from datetime import datetime, timedelta
import os
import hashlib

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

logger = logging.getLogger(__name__)


class APIRateLimitExceeded(Exception):
    """Exception raised when API rate limit is exceeded"""
    pass


class APIError(Exception):
    """Exception raised for general API errors"""
    pass


class BaseAPIClient:
    """Base class for API clients with common functionality"""
    
    def __init__(self, base_url: str, api_key: str, rate_limit: int = 60,
                 rate_limit_period: int = 60, cache_dir: Optional[str] = None,
                 cache_ttl: int = 3600, cache_enabled: bool = True,
                 time_sensitive_endpoints: Optional[List[str]] = None,
                 time_sensitive_ttl: int = 300):
        """
        Initialize the API client
        
        Args:
            base_url (str): Base URL for the API
            api_key (str): API key for authentication
            rate_limit (int): Maximum number of requests allowed in rate_limit_period
            rate_limit_period (int): Time period in seconds for rate limiting
            cache_dir (Optional[str]): Directory to store cache files
            cache_ttl (int): Cache time-to-live in seconds (for normal data)
            cache_enabled (bool): Whether caching is enabled by default
            time_sensitive_endpoints (Optional[List[str]]): List of endpoints that contain time-sensitive data
            time_sensitive_ttl (int): Cache TTL for time-sensitive endpoints (in seconds)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.rate_limit_period = rate_limit_period
        self.cache_dir = cache_dir or os.path.join('data', 'api_cache')
        self.cache_ttl = cache_ttl
        self.cache_enabled = cache_enabled
        self.time_sensitive_endpoints = time_sensitive_endpoints or []
        self.time_sensitive_ttl = time_sensitive_ttl
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Rate limiting state
        self.request_timestamps: List[float] = []
        
        # Session for connection pooling
        self.session = requests.Session()
        
        logger.info(f"Initialized API client for {base_url}")
    
    def _wait_for_rate_limit(self) -> None:
        """
        Wait if necessary to respect rate limits
        """
        current_time = time.time()
        
        # Remove timestamps older than the rate limit period
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if current_time - ts < self.rate_limit_period]
        
        # Check if we've hit the rate limit
        if len(self.request_timestamps) >= self.rate_limit:
            oldest_timestamp = min(self.request_timestamps)
            sleep_time = self.rate_limit_period - (current_time - oldest_timestamp) + 0.1
            
            if sleep_time > 0:
                logger.warning(f"Rate limit approached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        # Add current request to timestamps
        self.request_timestamps.append(time.time())
    
    def _get_cache_path(self, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Get the cache file path for a request
        
        Args:
            endpoint (str): API endpoint
            params (Dict[str, Any]): Request parameters
            
        Returns:
            str: Path to cache file
        """
        # Create a unique cache key based on the endpoint and parameters
        cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
        hashed_key = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.json")
    
    def _validate_cached_data(self, data: Dict[str, Any], min_data_points: int = 1) -> bool:
        """
        Validate that cached data is complete and contains actual data
        
        Args:
            data (Dict[str, Any]): Data to validate
            min_data_points (int): Minimum number of data points required
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check if data exists
        if not data:
            return False
        
        # Check if the data field exists and has content
        data_field = data.get('data')
        if data_field is None:
            return False
        
        # If data is a list, ensure it has at least min_data_points
        if isinstance(data_field, list) and len(data_field) < min_data_points:
            return False
        
        # If we got here, the data is valid
        return True
    
    def _get_from_cache(self, endpoint: str, params: Dict[str, Any], min_data_points: int = 1) -> Optional[Dict[str, Any]]:
        """
        Get response from cache if available, not expired, and valid
        
        Args:
            endpoint (str): API endpoint
            params (Dict[str, Any]): Request parameters
            min_data_points (int): Minimum number of data points required to consider the data valid
            
        Returns:
            Optional[Dict[str, Any]]: Cached response or None if not available/valid
        """
        cache_path = self._get_cache_path(endpoint, params)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is expired
            if endpoint in self.time_sensitive_endpoints:
                ttl = self.time_sensitive_ttl
            else:
                ttl = self.cache_ttl
            
            if cached_data.get('cached_at', 0) + ttl < time.time():
                logger.debug(f"Cache expired for {endpoint}")
                return None
            
            # Validate the data
            data = cached_data.get('data')
            if not self._validate_cached_data(data, min_data_points):
                logger.debug(f"Cached data for {endpoint} is invalid or incomplete")
                return None
            
            logger.debug(f"Cache hit for {endpoint}")
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading cache for {endpoint}: {e}")
            return None
    
    def _save_to_cache(self, endpoint: str, params: Dict[str, Any], data: Dict[str, Any]) -> None:
        """
        Save response to cache
        
        Args:
            endpoint (str): API endpoint
            params (Dict[str, Any]): Request parameters
            data (Dict[str, Any]): Response data to cache
        """
        cache_path = self._get_cache_path(endpoint, params)
        
        try:
            cache_data = {
                'data': data,
                'cached_at': time.time(),
                'endpoint': endpoint,
                'params': params
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            
            logger.debug(f"Saved response to cache for {endpoint}")
        except IOError as e:
            logger.warning(f"Error saving cache for {endpoint}: {e}")
    
    def request(self, endpoint: str, method: str = 'GET', params: Optional[Dict[str, Any]] = None,
                data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,
                use_cache: bool = True, force_refresh: bool = False, min_data_points: int = 1,
                max_retries: int = 3, retry_delay: int = 2) -> Dict[str, Any]:
        """
        Make an API request with retries, rate limiting, and caching
        
        Args:
            endpoint (str): API endpoint (without base URL)
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
            
        Raises:
            APIRateLimitExceeded: If API rate limit is exceeded
            APIError: If API returns an error
            RequestException: If request fails after max retries
        """
        params = params or {}
        headers = headers or {}
        
        # Normalize endpoint
        endpoint = endpoint.lstrip('/')
        url = f"{self.base_url}/{endpoint}"
        
        # Add API key to headers
        headers['Authorization'] = f"{self.api_key}"
        
        # Try to get from cache for GET requests if caching is enabled and not forcing refresh
        if method == 'GET' and use_cache and self.cache_enabled and not force_refresh:
            cached_response = self._get_from_cache(endpoint, params, min_data_points)
            if cached_response is not None:
                return cached_response
        
        # Wait for rate limit if necessary
        self._wait_for_rate_limit()
        
        # Make request with retries
        retries = 0
        last_exception = None
        
        while retries <= max_retries:
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=headers,
                    timeout=30  # 30-second timeout
                )
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', retry_delay))
                    logger.warning(f"Rate limit exceeded, waiting for {retry_after} seconds")
                    time.sleep(retry_after)
                    retries += 1
                    continue
                
                # Handle other error status codes
                if response.status_code >= 400:
                    error_message = f"API error: {response.status_code} - {response.text}"
                    logger.error(error_message)
                    
                    if response.status_code >= 500:
                        # Server error, retry
                        retries += 1
                        time.sleep(retry_delay * (2 ** retries))  # Exponential backoff
                        continue
                    else:
                        # Client error, don't retry
                        raise APIError(error_message)
                
                # Parse response
                try:
                    result = response.json()
                except ValueError:
                    result = {"text": response.text}
                
                # Cache successful GET responses if they contain valid data
                if method == 'GET' and use_cache and self.cache_enabled and response.status_code == 200:
                    # Only cache if the response contains valid data
                    if self._validate_cached_data(result, min_data_points):
                        self._save_to_cache(endpoint, params, result)
                    else:
                        logger.warning(f"Not caching response for {endpoint} as it contains insufficient data")
                
                return result
            
            except (ConnectionError, Timeout) as e:
                last_exception = e
                logger.warning(f"Request failed (attempt {retries+1}/{max_retries+1}): {e}")
                retries += 1
                time.sleep(retry_delay * (2 ** retries))  # Exponential backoff
        
        # If we get here, all retries failed
        error_message = f"Max retries exceeded for {url}: {last_exception}"
        logger.error(error_message)
        raise APIError(error_message)
    
    def close(self) -> None:
        """
        Close the session
        """
        self.session.close()
        logger.debug("Closed API client session")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
