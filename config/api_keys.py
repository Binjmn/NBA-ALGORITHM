#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Keys Configuration Module
Purpose: Securely manage API keys for the NBA Betting Prediction System

WARNING: This file should NEVER be committed to version control.
This is a template and should be copied to api_keys.py.sample with sensitive values removed.
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# API Keys
BALLDONTLIE_API_KEY = "d0e93357-b9b0-4a62-bed1-920eeab5db50"  # GOAT plan
THE_ODDS_API_KEY = "12186f096bb2e6a9f9b472391323893d"  # Paid plan


def get_api_key(service: str) -> Optional[str]:
    """
    Get API key for the specified service
    
    Args:
        service (str): Service name (e.g., 'balldontlie', 'theodds')
        
    Returns:
        Optional[str]: API key if available, None otherwise
    """
    # Try to get from environment variables first (more secure)
    env_var_name = f"{service.upper()}_API_KEY"
    env_key = os.environ.get(env_var_name)
    
    if env_key:
        return env_key
    
    # Fall back to constants in this file
    keys_map: Dict[str, str] = {
        'balldontlie': BALLDONTLIE_API_KEY,
        'theodds': THE_ODDS_API_KEY
    }
    
    api_key = keys_map.get(service.lower())
    
    if not api_key:
        logger.warning(f"No API key found for service: {service}")
        return None
    
    return api_key


def validate_api_keys() -> bool:
    """
    Validate that required API keys are available
    
    Returns:
        bool: True if all required API keys are available, False otherwise
    """
    required_services = ['balldontlie']
    missing_keys = []
    
    for service in required_services:
        if not get_api_key(service):
            missing_keys.append(service)
    
    if missing_keys:
        logger.error(f"Missing API keys for services: {', '.join(missing_keys)}")
        return False
    
    return True
