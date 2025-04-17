# -*- coding: utf-8 -*-
"""
Season Manager Module

This module provides the SeasonManager class for automatically detecting
and handling NBA season transitions. It ensures the prediction system
adapts to different seasons and phases automatically.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

from ..config.season_config import (
    SeasonPhase,
    IN_SEASON_PHASES,
    get_season_display_name, 
    get_current_season_info
)

logger = logging.getLogger(__name__)


class SeasonManager:
    """
    NBA Season Manager
    
    This class manages NBA season information, including auto-detection of the current
    season year, phase transitions, and season-specific configuration. It supports
    automatic adaptation of the prediction system to new seasons and phases.
    """
    
    def __init__(self, auto_update: bool = True):
        """
        Initialize the SeasonManager
        
        Args:
            auto_update: Whether to automatically update the season on initialization
        """
        self._current_season_info = None
        self._last_check_time = None
        self._check_interval = timedelta(hours=12)  # Check for season changes twice a day
        
        # Event callbacks for season transitions
        self._on_season_change_callbacks = []
        self._on_phase_change_callbacks = []
        
        if auto_update:
            self.update_current_season()
            
        logger.info(f"Season Manager initialized. Current season: {self.get_current_season_display()}")
    
    def update_current_season(self, reference_date: Optional[datetime] = None) -> bool:
        """
        Update the current season information based on the reference date
        
        Args:
            reference_date: Date to use for season determination (defaults to today)
            
        Returns:
            bool: True if the season or phase changed, False otherwise
        """
        try:
            # Get current season info
            new_season_info = get_current_season_info(reference_date)
            
            # Store the time we checked
            self._last_check_time = datetime.now()
            
            # If this is the first check, just store the info
            if self._current_season_info is None:
                self._current_season_info = new_season_info
                return False
                
            # Check if season year changed
            old_season_year = self._current_season_info["season_year"]
            new_season_year = new_season_info["season_year"]
            
            # Check if phase changed
            old_phase = self._current_season_info["phase"]
            new_phase = new_season_info["phase"]
            
            # Store new season info
            self._current_season_info = new_season_info
            
            # Check if either season or phase changed
            season_changed = old_season_year != new_season_year
            phase_changed = old_phase != new_phase
            
            # Handle season change callbacks
            if season_changed:
                logger.info(f"Season transition detected: {get_season_display_name(old_season_year)} → "
                           f"{get_season_display_name(new_season_year)}")
                self._handle_season_change(old_season_year, new_season_year)
                
            # Handle phase change callbacks
            if phase_changed:
                logger.info(f"Season phase transition detected: {old_phase.value} → {new_phase.value}")
                self._handle_phase_change(old_phase, new_phase)
                
            return season_changed or phase_changed
                
        except Exception as e:
            logger.error(f"Error updating current season: {str(e)}")
            # If we can't update, keep using the current season info
            return False
    
    def check_for_updates(self) -> bool:
        """
        Check for season updates if enough time has passed since the last check
        
        Returns:
            bool: True if an update occurred, False otherwise
        """
        # Skip if we checked recently
        if (self._last_check_time is not None and 
                datetime.now() - self._last_check_time < self._check_interval):
            return False
            
        return self.update_current_season()
    
    def get_current_season_year(self) -> int:
        """
        Get the current season year
        
        Returns:
            int: Current season year (e.g., 2025 for the 2024-25 season)
        """
        self.check_for_updates()
        return self._current_season_info["season_year"]
    
    def get_current_season_phase(self) -> SeasonPhase:
        """
        Get the current season phase
        
        Returns:
            SeasonPhase: Current phase of the season
        """
        self.check_for_updates()
        return self._current_season_info["phase"]
    
    def get_current_season_display(self) -> str:
        """
        Get a display name for the current season (e.g., "2024-25")
        
        Returns:
            str: Current season display name
        """
        self.check_for_updates()
        return self._current_season_info["display_name"]
    
    def get_current_season_info(self) -> Dict[str, Any]:
        """
        Get complete information about the current season
        
        Returns:
            Dict: Current season information
        """
        self.check_for_updates()
        return self._current_season_info.copy()
    
    def is_in_season(self) -> bool:
        """
        Check if we're currently in an active NBA season phase
        
        Returns:
            bool: True if currently in season, False otherwise
        """
        self.check_for_updates()
        return self._current_season_info["phase"] in IN_SEASON_PHASES
    
    def is_regular_season(self) -> bool:
        """
        Check if we're currently in the regular season
        
        Returns:
            bool: True if in regular season, False otherwise
        """
        self.check_for_updates()
        return self._current_season_info["phase"] == SeasonPhase.REGULAR_SEASON
    
    def is_playoffs(self) -> bool:
        """
        Check if we're currently in the playoffs
        
        Returns:
            bool: True if in playoffs, False otherwise
        """
        self.check_for_updates()
        return self._current_season_info["phase"] in [SeasonPhase.PLAYOFFS, SeasonPhase.FINALS]
    
    def register_season_change_callback(self, callback) -> None:
        """
        Register a callback function to be called when the season changes
        
        Args:
            callback: Function to call when season changes
        """
        if callback not in self._on_season_change_callbacks:
            self._on_season_change_callbacks.append(callback)
    
    def register_phase_change_callback(self, callback) -> None:
        """
        Register a callback function to be called when the season phase changes
        
        Args:
            callback: Function to call when season phase changes
        """
        if callback not in self._on_phase_change_callbacks:
            self._on_phase_change_callbacks.append(callback)
    
    def _handle_season_change(self, old_season_year: int, new_season_year: int) -> None:
        """
        Handle season change event
        
        Args:
            old_season_year: Previous season year
            new_season_year: New season year
        """
        try:
            for callback in self._on_season_change_callbacks:
                callback(old_season_year, new_season_year)
        except Exception as e:
            logger.error(f"Error in season change callback: {str(e)}")
    
    def _handle_phase_change(self, old_phase: SeasonPhase, new_phase: SeasonPhase) -> None:
        """
        Handle phase change event
        
        Args:
            old_phase: Previous season phase
            new_phase: New season phase
        """
        try:
            for callback in self._on_phase_change_callbacks:
                callback(old_phase, new_phase)
        except Exception as e:
            logger.error(f"Error in phase change callback: {str(e)}")


# Global season manager instance
_season_manager = None


def get_season_manager() -> SeasonManager:
    """
    Get or create the global season manager instance
    
    Returns:
        SeasonManager: Global season manager instance
    """
    global _season_manager
    if _season_manager is None:
        _season_manager = SeasonManager()
    return _season_manager
