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
    
    def __init__(self, 
                 base_url: str, 
                 api_key: Optional[str] = None,
                 auth_header_prefix: Optional[str] = None,
                 rate_limit: int = 60,
                 rate_limit_period: int = 60,
                 cache_enabled: bool = True,
                 cache_dir: Optional[str] = None,
                 cache_ttl: int = 3600,
                 time_sensitive_endpoints: Optional[List[str]] = None,
                 time_sensitive_ttl: int = 300,
                 timeout: int = 30):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL for the API
            api_key: API key for authentication
            auth_header_prefix: Prefix for Authorization header (e.g., 'Bearer')
            rate_limit: Rate limit per minute
            rate_limit_period: Time period in seconds for rate limiting
            cache_enabled: Whether to cache responses
            cache_dir: Directory for cache files
            cache_ttl: Cache time-to-live in seconds (for normal data)
            time_sensitive_endpoints: List of endpoints that contain time-sensitive data
            time_sensitive_ttl: Cache TTL for time-sensitive endpoints (in seconds)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.auth_header_prefix = auth_header_prefix
        self.rate_limit = rate_limit
        self.rate_limit_period = rate_limit_period
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.time_sensitive_endpoints = time_sensitive_endpoints or []
        self.time_sensitive_ttl = time_sensitive_ttl
        self.timeout = timeout
        
        # Cache directory
        self.cache_dir = cache_dir or os.path.join('data', 'api_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Rate limiting state
        self.request_timestamps: List[float] = []
        
        # Set up requests session
        self.session = requests.Session()
        
        logger.info(f"Initialized API client for {base_url}")
    
    def _wait_for_rate_limit(self) -> None:
        """
        Wait if necessary to respect rate limits
        """
        # COMPLETELY DISABLED FOR TRAINING - NO RATE LIMITING
        # We'll clean up old timestamps but not enforce any waits
        current_time = time.time()
        
        # Clean up old timestamps beyond our window
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                if current_time - ts <= self.rate_limit_period]
        
        # Add current request to timestamps just for tracking
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
    
    def request(self, endpoint: str, method: str = 'GET', params: Optional[Dict] = None, data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Any:
        """
        Make a request to the API
        
        Args:
            endpoint: API endpoint to request
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Request body data
            headers: Request headers
            
        Returns:
            API response data
        """
        url = f"{self.base_url}/{endpoint}"
        headers = headers or {}
        
        # Add API key if available
        if self.api_key and not headers.get('Authorization'):
            if self.auth_header_prefix:
                headers['Authorization'] = f"{self.auth_header_prefix} {self.api_key}"
            else:
                # Add as a query param instead
                params = params or {}
                params['apiKey'] = self.api_key
        
        # Enhanced logging - log the full request details for debugging
        debug_info = {
            'url': url,
            'method': method,
            'params': params,
            'headers': {k: v for k, v in headers.items() if k.lower() != 'authorization'} if headers else None
        }
        logger.debug(f"API Request: {json.dumps(debug_info, indent=2)}")
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)  
            elif method == 'POST':
                response = self.session.post(url, params=params, json=data, headers=headers, timeout=self.timeout)  
            elif method == 'PUT':
                response = self.session.put(url, params=params, json=data, headers=headers, timeout=self.timeout)  
            elif method == 'DELETE':
                response = self.session.delete(url, params=params, headers=headers, timeout=self.timeout)  
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Try to parse the response as JSON
            try:
                result = response.json()
                return result
            except ValueError:
                # Not JSON, return the raw text
                return response.text
                
        except requests.exceptions.HTTPError as e:
            # For HTTP errors, try to get more detailed error info from the response
            error_detail = ""
            try:
                error_response = e.response.json()
                error_detail = json.dumps(error_response)
                
                # Specifically log the date format if that's the issue
                if 'INVALID_DATE_FORMAT' in error_detail:
                    date_param = None
                    if params and ('date' in params):
                        date_param = params.get('date')
                    logger.error(f"Invalid date format error - Date parameter: {date_param}")
                    logger.error(f"Full request that caused error: {json.dumps(debug_info)}")
                
            except ValueError:
                error_detail = e.response.text
                
            logger.error(f"API error: {e.response.status_code} - {error_detail}")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            return None
    
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
