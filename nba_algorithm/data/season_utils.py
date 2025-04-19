#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Season Utilities

This module provides utilities for detecting and managing NBA seasons, including:
- Auto-detection of current NBA season based on date
- Helper functions for season transitions
- Season phase detection

It serves as a bridge between the season configuration and the data collection modules.
"""

import os
import logging
from datetime import datetime
from typing import Optional, Union, Dict, List, Tuple, Any

# Import existing season utilities from the project
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from the existing season config and manager
from config.season_config import (
    SeasonPhase, 
    SEASON_PHASE_DATES,
    get_season_year_from_date,
    get_season_display_name
)

from src.utils.season_manager import SeasonManager

logger = logging.getLogger(__name__)

def get_current_season() -> str:
    """
    Get the current NBA season year
    This is the function that training_pipeline.py is trying to import
    
    Returns:
        str: Current season year as a string (e.g., "2025" for the 2024-25 season)
    """
    return detect_current_season()

def detect_current_season() -> str:
    """
    Auto-detect the current NBA season based on the current date
    
    Returns:
        str: Season year as a string (e.g., "2025" for the 2024-25 season)
    """
    try:
        # Use the existing SeasonManager to detect the season
        season_manager = SeasonManager()
        current_season = season_manager.get_current_season_year()
        logger.info(f"Auto-detected NBA season: {current_season}")
        return str(current_season)
    except Exception as e:
        logger.error(f"Error auto-detecting NBA season: {str(e)}")
        # Default to the current year + 1 as a fallback
        current_year = datetime.now().year
        if datetime.now().month >= 9:  # After September, use next year
            fallback_season = str(current_year + 1)
        else:
            fallback_season = str(current_year)
        logger.warning(f"Using fallback season: {fallback_season}")
        return fallback_season

def get_season_phase(date: Optional[datetime] = None) -> SeasonPhase:
    """
    Get the current NBA season phase based on date
    
    Args:
        date: Date to check (defaults to current date)
        
    Returns:
        SeasonPhase: The current phase of the NBA season
    """
    try:
        season_manager = SeasonManager()
        phase = season_manager.get_current_phase(reference_date=date)
        return phase
    except Exception as e:
        logger.error(f"Error determining season phase: {str(e)}")
        return SeasonPhase.UNKNOWN

def get_season_date_range(season: Union[str, int]) -> Tuple[str, str]:
    """
    Get the date range for a given NBA season
    
    Args:
        season: Season year (e.g., 2025 for the 2024-25 season)
        
    Returns:
        Tuple[str, str]: Start and end dates in YYYY-MM-DD format
    """
    season_year = int(season)
    # NBA season typically starts in October of previous year
    start_year = season_year - 1
    start_date = f"{start_year}-10-01"  # October 1st of previous year
    end_date = f"{season_year}-06-30"   # June 30th of season year
    return start_date, end_date

def is_season_active(season: Union[str, int]) -> bool:
    """
    Check if a given season is currently active
    
    Args:
        season: Season year (e.g., 2025 for the 2024-25 season)
        
    Returns:
        bool: True if the season is currently active
    """
    current_season = detect_current_season()
    return str(season) == current_season
