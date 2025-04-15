"""
NBA Season Configuration

This module provides configuration settings for NBA seasons, including calendar definitions,
season phase identification, and typical date ranges for different parts of the season.

The NBA typically follows this calendar pattern:
- Preseason: Late September to early October
- Regular Season: Late October to mid-April
- All-Star Break: Mid-February (about 4-5 days)
- Play-In Tournament: Mid-April (about 3-4 days after regular season)
- Playoffs: Mid-April to mid-June
- Draft: Late June
- Free Agency: Early July
- Summer League: Early-Mid July
- Offseason: Mid-July to late September

Note: These dates can vary slightly each year, so the system will validate against
the BallDontLie API when possible.
"""

import logging
from enum import Enum, auto
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)

class SeasonPhase(Enum):
    """Enumeration of NBA season phases"""
    PRESEASON = auto()
    REGULAR_SEASON = auto()
    ALL_STAR_BREAK = auto()
    PLAY_IN_TOURNAMENT = auto()
    PLAYOFFS = auto()
    FINALS = auto()
    DRAFT = auto()
    FREE_AGENCY = auto()
    SUMMER_LEAGUE = auto()
    OFFSEASON = auto()
    UNKNOWN = auto()

# Maps each phase to a description and typical start month/day
# Format: (description, (start_month, start_day), (end_month, end_day))
# Using 0 for day means "end of month"
SEASON_PHASE_DATES = {
    SeasonPhase.PRESEASON: (
        "NBA Preseason", 
        (9, 28),    # Sept 28
        (10, 17)    # Oct 17
    ),
    SeasonPhase.REGULAR_SEASON: (
        "NBA Regular Season",
        (10, 18),   # Oct 18
        (4, 10)     # April 10
    ),
    SeasonPhase.ALL_STAR_BREAK: (
        "NBA All-Star Break",
        (2, 16),    # Feb 16
        (2, 22)     # Feb 22
    ),
    SeasonPhase.PLAY_IN_TOURNAMENT: (
        "NBA Play-In Tournament",
        (4, 11),    # April 11
        (4, 14)     # April 14
    ),
    SeasonPhase.PLAYOFFS: (
        "NBA Playoffs",
        (4, 15),    # April 15
        (6, 10)     # June 10
    ),
    SeasonPhase.FINALS: (
        "NBA Finals",
        (6, 1),     # June 1
        (6, 18)     # June 18
    ),
    SeasonPhase.DRAFT: (
        "NBA Draft",
        (6, 22),    # June 22
        (6, 24)     # June 24
    ),
    SeasonPhase.FREE_AGENCY: (
        "NBA Free Agency",
        (6, 30),    # June 30
        (7, 15)     # July 15
    ),
    SeasonPhase.SUMMER_LEAGUE: (
        "NBA Summer League",
        (7, 7),     # July 7
        (7, 17)     # July 17
    ),
    SeasonPhase.OFFSEASON: (
        "NBA Offseason",
        (7, 18),    # July 18
        (9, 27)     # Sept 27
    )
}

# Define which phases are considered "in-season" (active games being played)
IN_SEASON_PHASES = [
    SeasonPhase.PRESEASON,
    SeasonPhase.REGULAR_SEASON,
    SeasonPhase.PLAY_IN_TOURNAMENT,
    SeasonPhase.PLAYOFFS,
    SeasonPhase.FINALS
]

# Phases where betting odds are available
BETTING_PHASES = [
    SeasonPhase.PRESEASON,
    SeasonPhase.REGULAR_SEASON,
    SeasonPhase.PLAY_IN_TOURNAMENT,
    SeasonPhase.PLAYOFFS,
    SeasonPhase.FINALS
]

# Calendar calibration - adjustments for exceptional seasons
# Format: {season_year: {phase: (new_start_month, new_start_day, new_end_month, new_end_day)}}
SEASON_ADJUSTMENTS = {
    # Example for a hypothetical future lockout or COVID-like situation
    # 2025: {
    #     SeasonPhase.REGULAR_SEASON: (12, 25, 5, 15),  # Start Dec 25, end May 15
    #     SeasonPhase.PLAYOFFS: (5, 16, 7, 20),         # Start May 16, end July 20
    # }
}

def get_season_year_from_date(date: datetime) -> int:
    """
    Determine the NBA season year from a given date.
    The NBA season spans two calendar years, and is identified by the year it ends in.
    For example, the 2023-24 season is identified as 2024.
    
    Args:
        date (datetime): The date to determine season for
        
    Returns:
        int: The season year (e.g., 2024 for the 2023-24 season)
    """
    month = date.month
    year = date.year
    
    # If we're in October through December, the season is identified by the next year
    if month >= 7:  # July or later
        return year + 1
    else:
        return year

def get_season_display_name(season_year: int) -> str:
    """
    Get the display name for a season year
    
    Args:
        season_year (int): The season year (e.g., 2024 for the 2023-24 season)
        
    Returns:
        str: The season display name (e.g., "2023-24" for 2024)
    """
    return f"{season_year-1}-{str(season_year)[-2:]}"
