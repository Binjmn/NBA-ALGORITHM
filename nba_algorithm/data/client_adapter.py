#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Client Adapter Module

This module provides compatibility adapters between different versions of API clients
in the codebase. It ensures that clients from the src directory can be used with code
that expects clients from the nba_algorithm directory, and vice versa.
"""

import logging
from typing import Any, Dict, List, Optional, Union

# Import the actual client
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.api.balldontlie_client import BallDontLieClient as SourceBallDontLieClient
from src.api.theodds_client import TheOddsClient as SourceTheOddsClient

logger = logging.getLogger(__name__)

class BallDontLieClientAdapter:
    """
    Adapter class for BallDontLieClient to ensure compatibility with all code that uses it
    This adds the 'client' attribute that some code might expect to find
    """
    
    def __init__(self):
        """
        Initialize the adapter with the actual client
        """
        self._source_client = SourceBallDontLieClient()
        # Add the 'client' attribute that some code expects
        self.client = self._source_client
        
    def __getattr__(self, name: str) -> Any:
        """
        Forward all attribute access to the source client
        
        Args:
            name: Attribute name to access
            
        Returns:
            The requested attribute from the source client
        """
        return getattr(self._source_client, name)
    
    def request(self, endpoint: str, method: str = 'GET', params: Optional[Dict] = None, data: Optional[Dict] = None, 
                use_cache: bool = True, force_refresh: bool = False, **kwargs) -> Dict:
        """
        Make a request to the API, safely handling parameters that may not be supported
        
        This wrapper ensures compatibility with code that might pass the 'force_refresh' parameter,
        which isn't supported by the underlying client.
        
        Args:
            endpoint: API endpoint to request
            method: HTTP method (GET, POST, etc.)
            params: Query parameters for the request
            data: Request body data
            use_cache: Whether to use cached responses
            force_refresh: Parameter that will be safely ignored
            **kwargs: Additional keyword arguments
            
        Returns:
            API response data
        """
        # Filter out parameters not supported by the source client
        filtered_params = params.copy() if params else {}
        if 'force_refresh' in filtered_params:
            logger.debug(f"Removing 'force_refresh' parameter from request to {endpoint}")
            del filtered_params['force_refresh']
            
        # Forward the request to the source client
        return self._source_client.request(endpoint, method=method, params=filtered_params, data=data, **kwargs)

class TheOddsClientAdapter:
    """
    Adapter class for TheOddsClient to ensure compatibility with all code that uses it
    """
    
    def __init__(self):
        """
        Initialize the adapter with the actual client
        """
        self._source_client = SourceTheOddsClient()
        # Add any attributes that some code might expect
        self.client = self._source_client
        
    def __getattr__(self, name: str) -> Any:
        """
        Forward all attribute access to the source client
        
        Args:
            name: Attribute name to access
            
        Returns:
            The requested attribute from the source client
        """
        return getattr(self._source_client, name)
