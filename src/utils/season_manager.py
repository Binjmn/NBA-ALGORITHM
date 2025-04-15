"""
NBA Season Manager

This module provides the SeasonManager class for detecting NBA seasons, identifying
current season phases, and handling season transitions. It's designed to make the
entire application season-aware without requiring manual updates.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import pytz

from config.season_config import (
    SeasonPhase,
    SEASON_PHASE_DATES,
    SEASON_ADJUSTMENTS,
    get_season_year_from_date,
    get_season_display_name,
    IN_SEASON_PHASES
)

logger = logging.getLogger(__name__)

class SeasonManager:
    """
    Class for managing NBA season detection and transitions
    
    This class provides functionality to:
    - Detect the current NBA season based on date
    - Identify the current phase of the season
    - Handle transitions between seasons
    - Load and validate season data from APIs
    - Cache season information to avoid unnecessary API calls
    """
    
    def __init__(
        self,
        api_client=None,  # Optional API client for fetching real data
        data_dir: str = 'data',
        cache_ttl: int = 86400,  # 24 hours
        use_api_validation: bool = True
    ):
        """
        Initialize the season manager
        
        Args:
            api_client: BallDontLie API client for fetching real data
            data_dir (str): Directory for storing season data
            cache_ttl (int): Cache time-to-live in seconds
            use_api_validation (bool): Whether to validate with the API
        """
        self.api_client = api_client
        self.data_dir = data_dir
        self.cache_ttl = cache_ttl
        self.use_api_validation = use_api_validation
        
        # Create data directory if it doesn't exist
        self.seasons_dir = os.path.join(data_dir, 'seasons')
        os.makedirs(self.seasons_dir, exist_ok=True)
        
        # Initialize cache
        self._season_cache = {}
        self._season_cache_time = {}
        
        # Current season information
        self._current_season_year = None
        self._current_phase = None
        self._current_phase_start = None
        self._current_phase_end = None
        
        # Load current season information
        self.update_current_season()
        
        logger.info("Initialized NBA season manager")

    def update_current_season(self, reference_date: Optional[datetime] = None) -> None:
        """
        Update the current season information based on the given date
        
        Args:
            reference_date (Optional[datetime]): Reference date, defaults to current date
        """
        # Use current date if not provided
        if reference_date is None:
            # Use Eastern Time (EST) as per project requirements
            eastern = pytz.timezone('US/Eastern')
            reference_date = datetime.now(eastern)
        
        # Ensure date is timezone-aware
        if reference_date.tzinfo is None:
            eastern = pytz.timezone('US/Eastern')
            reference_date = eastern.localize(reference_date)
            
        # Determine season year
        season_year = get_season_year_from_date(reference_date)
        
        # Determine current phase
        phase, start_date, end_date = self.get_current_phase(reference_date)
        
        # Update current season information
        self._current_season_year = season_year
        self._current_phase = phase
        self._current_phase_start = start_date
        self._current_phase_end = end_date
        
        logger.info(f"Current NBA season: {get_season_display_name(season_year)}, "
                   f"Phase: {phase.name}")
        
        # Validate with API if requested and client is available
        if self.use_api_validation and self.api_client is not None:
            self._validate_season_with_api()
    
    def get_current_phase(self, reference_date: datetime) -> Tuple[SeasonPhase, datetime, datetime]:
        """
        Determine the current NBA season phase based on the given date
        
        Args:
            reference_date (datetime): Reference date
            
        Returns:
            Tuple[SeasonPhase, datetime, datetime]: Phase, phase start date, phase end date
        """
        # Extract month and day
        month = reference_date.month
        day = reference_date.day
        year = reference_date.year
        
        # Determine season year
        season_year = get_season_year_from_date(reference_date)
        
        # Check for season adjustments
        adjustments = SEASON_ADJUSTMENTS.get(season_year, {})
        
        # Check each phase
        for phase, (description, (start_month, start_day), (end_month, end_day)) in SEASON_PHASE_DATES.items():
            # Apply season adjustments if available
            if phase in adjustments:
                start_month, start_day, end_month, end_day = adjustments[phase]
            
            # Create start and end dates for comparison
            # Handle year transitions (e.g., season crosses calendar years)
            if start_month > end_month:  # Phase crosses calendar years
                if month >= start_month:  # We're in the first calendar year of the phase
                    phase_start = datetime(year, start_month, start_day, tzinfo=reference_date.tzinfo)
                    phase_end = datetime(year + 1, end_month, end_day, tzinfo=reference_date.tzinfo)
                else:  # We're in the second calendar year of the phase
                    phase_start = datetime(year - 1, start_month, start_day, tzinfo=reference_date.tzinfo)
                    phase_end = datetime(year, end_month, end_day, tzinfo=reference_date.tzinfo)
            else:  # Phase is within a single calendar year
                phase_start = datetime(year, start_month, start_day, tzinfo=reference_date.tzinfo)
                phase_end = datetime(year, end_month, end_day, tzinfo=reference_date.tzinfo)
            
            # Check if current date is in this phase
            if phase_start <= reference_date <= phase_end:
                return phase, phase_start, phase_end
        
        # If we get here, we couldn't identify the phase
        logger.warning(f"Could not identify NBA season phase for date {reference_date}")
        return SeasonPhase.UNKNOWN, reference_date, reference_date
    
    def _validate_season_with_api(self) -> None:
        """
        Validate the current season information with the API
        
        This method attempts to get the current season information from the
        BallDontLie API and compare it with our calculated values.
        """
        if self.api_client is None:
            logger.warning("Cannot validate season with API: No API client provided")
            return
        
        try:
            # Get current season from API
            season_data = self.api_client.get_current_season()
            
            if not season_data or 'data' not in season_data:
                logger.warning("Failed to get current season data from API")
                return
                
            api_season_year = season_data['data'].get('year')
            
            if api_season_year is not None:
                if api_season_year != self._current_season_year:
                    logger.warning(
                        f"Season year discrepancy: Calculated {self._current_season_year}, "
                        f"API reports {api_season_year}"
                    )
                    # Update to API value when there's a discrepancy
                    self._current_season_year = api_season_year
                else:
                    logger.debug("Season year validated with API")
        except Exception as e:
            logger.error(f"Error validating season with API: {e}")

    def get_current_season_year(self) -> int:
        """
        Get the current NBA season year
        
        Returns:
            int: Current season year (e.g., 2024 for the 2023-24 season)
        """
        return self._current_season_year
    
    def get_current_season_display(self) -> str:
        """
        Get the display name for the current NBA season
        
        Returns:
            str: Season display name (e.g., "2023-24")
        """
        return get_season_display_name(self._current_season_year)
    
    def get_current_phase(self) -> SeasonPhase:
        """
        Get the current NBA season phase
        
        Returns:
            SeasonPhase: Current season phase
        """
        return self._current_phase
    
    def is_in_season(self) -> bool:
        """
        Check if we are currently in an active NBA season
        
        Returns:
            bool: True if in season, False otherwise
        """
        return self._current_phase in IN_SEASON_PHASES
    
    def days_until_phase(self, phase: SeasonPhase) -> int:
        """
        Get the number of days until the specified phase begins
        
        Args:
            phase (SeasonPhase): Target phase
            
        Returns:
            int: Days until phase begins, or -1 if phase is current or in the past
        """
        # If we're already in this phase, return 0
        if self._current_phase == phase:
            return 0
            
        # Get current date in EST
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        # Determine target season year
        season_year = self._current_season_year
        
        # Get phase dates
        description, (start_month, start_day), (end_month, end_day) = SEASON_PHASE_DATES[phase]
        
        # Apply season adjustments if available
        adjustments = SEASON_ADJUSTMENTS.get(season_year, {})
        if phase in adjustments:
            start_month, start_day, end_month, end_day = adjustments[phase]
        
        # Create phase start date
        # Handle year transitions based on current date
        if start_month < now.month or (start_month == now.month and start_day < now.day):
            # Phase starts in the next year (we've passed it this year)
            if phase == SeasonPhase.OFFSEASON and self._current_phase in [
                SeasonPhase.PRESEASON, SeasonPhase.REGULAR_SEASON, 
                SeasonPhase.ALL_STAR_BREAK, SeasonPhase.PLAY_IN_TOURNAMENT, 
                SeasonPhase.PLAYOFFS, SeasonPhase.FINALS
            ]:
                # Special case: we're in the current season but checking for next offseason
                phase_start = datetime(now.year, start_month, start_day, tzinfo=eastern)
            else:
                phase_start = datetime(now.year + 1, start_month, start_day, tzinfo=eastern)
        else:
            # Phase starts later this year
            phase_start = datetime(now.year, start_month, start_day, tzinfo=eastern)
            
        # Calculate days until phase starts
        days = (phase_start - now).days
        
        return max(0, days)  # Don't return negative days
    
    def handle_season_transition(self) -> bool:
        """
        Check if a season transition has occurred and handle it
        
        This method checks if we've moved to a new season since the last check,
        and triggers any necessary data migrations or initializations.
        
        Returns:
            bool: True if a transition occurred, False otherwise
        """
        # Get the current date in EST
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        # Determine season year
        new_season_year = get_season_year_from_date(now)
        
        # Check if season has changed
        if self._current_season_year != new_season_year:
            logger.info(f"Season transition detected: {get_season_display_name(self._current_season_year)} "
                       f"-> {get_season_display_name(new_season_year)}")
            
            # Save current season data
            self._archive_season_data(self._current_season_year)
            
            # Update to new season
            self.update_current_season()
            
            # Initialize new season data
            self._initialize_season_data(new_season_year)
            
            return True
        
        return False
    
    def _archive_season_data(self, season_year: int) -> None:
        """
        Archive data for the specified season
        
        Args:
            season_year (int): Season year to archive
        """
        season_dir = os.path.join(self.seasons_dir, f"season_{season_year}")
        os.makedirs(season_dir, exist_ok=True)
        
        # Create a metadata file with season information
        metadata = {
            "season_year": season_year,
            "display_name": get_season_display_name(season_year),
            "archived_at": datetime.now().isoformat()
        }
        
        try:
            with open(os.path.join(season_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Archived season data for {get_season_display_name(season_year)}")
        except Exception as e:
            logger.error(f"Error archiving season data: {e}")
    
    def _initialize_season_data(self, season_year: int) -> None:
        """
        Initialize data for the specified season
        
        Args:
            season_year (int): Season year to initialize
        """
        season_dir = os.path.join(self.seasons_dir, f"season_{season_year}")
        os.makedirs(season_dir, exist_ok=True)
        
        # Create a metadata file with season information
        metadata = {
            "season_year": season_year,
            "display_name": get_season_display_name(season_year),
            "initialized_at": datetime.now().isoformat()
        }
        
        try:
            with open(os.path.join(season_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Initialized season data for {get_season_display_name(season_year)}")
        except Exception as e:
            logger.error(f"Error initializing season data: {e}")
