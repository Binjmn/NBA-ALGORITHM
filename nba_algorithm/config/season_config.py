# -*- coding: utf-8 -*-
"""
NBA Season Configuration

This module defines constants and utilities for working with NBA seasons,
including season phase definitions and date ranges for current and future seasons.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any


class SeasonPhase(Enum):
    """
    Enumeration of NBA season phases
    """
    OFF_SEASON = "off_season"
    PRE_SEASON = "pre_season"
    REGULAR_SEASON = "regular_season"
    ALL_STAR_BREAK = "all_star_break"
    PLAYOFFS = "playoffs"
    FINALS = "finals"


# Phases considered part of the active NBA season
IN_SEASON_PHASES = [SeasonPhase.PRE_SEASON, SeasonPhase.REGULAR_SEASON, 
                   SeasonPhase.ALL_STAR_BREAK, SeasonPhase.PLAYOFFS, 
                   SeasonPhase.FINALS]


# Season date ranges (configurable for future years)
# Format: (season_year, phase, start_date, end_date)
# Note: dates are inclusive (start_date <= date <= end_date)
SEASON_DATE_RANGES = [
    # 2024-2025 Season
    (2025, SeasonPhase.PRE_SEASON, datetime(2024, 10, 4), datetime(2024, 10, 18)),
    (2025, SeasonPhase.REGULAR_SEASON, datetime(2024, 10, 19), datetime(2025, 2, 13)),
    (2025, SeasonPhase.ALL_STAR_BREAK, datetime(2025, 2, 14), datetime(2025, 2, 19)),
    (2025, SeasonPhase.REGULAR_SEASON, datetime(2025, 2, 20), datetime(2025, 4, 10)),
    (2025, SeasonPhase.PLAYOFFS, datetime(2025, 4, 11), datetime(2025, 5, 25)),
    (2025, SeasonPhase.FINALS, datetime(2025, 5, 26), datetime(2025, 6, 18)),
    (2025, SeasonPhase.OFF_SEASON, datetime(2025, 6, 19), datetime(2025, 10, 3)),
    
    # 2025-2026 Season (projected dates based on typical NBA schedule)
    (2026, SeasonPhase.PRE_SEASON, datetime(2025, 10, 3), datetime(2025, 10, 17)),
    (2026, SeasonPhase.REGULAR_SEASON, datetime(2025, 10, 18), datetime(2026, 2, 12)),
    (2026, SeasonPhase.ALL_STAR_BREAK, datetime(2026, 2, 13), datetime(2026, 2, 18)),
    (2026, SeasonPhase.REGULAR_SEASON, datetime(2026, 2, 19), datetime(2026, 4, 11)),
    (2026, SeasonPhase.PLAYOFFS, datetime(2026, 4, 12), datetime(2026, 5, 27)),
    (2026, SeasonPhase.FINALS, datetime(2026, 5, 28), datetime(2026, 6, 19)),
    (2026, SeasonPhase.OFF_SEASON, datetime(2026, 6, 20), datetime(2026, 10, 2)),
    
    # 2026-2027 Season (projected dates based on typical NBA schedule)
    (2027, SeasonPhase.PRE_SEASON, datetime(2026, 10, 2), datetime(2026, 10, 16)),
    (2027, SeasonPhase.REGULAR_SEASON, datetime(2026, 10, 17), datetime(2027, 2, 11)),
    (2027, SeasonPhase.ALL_STAR_BREAK, datetime(2027, 2, 12), datetime(2027, 2, 17)),
    (2027, SeasonPhase.REGULAR_SEASON, datetime(2027, 2, 18), datetime(2027, 4, 10)),
    (2027, SeasonPhase.PLAYOFFS, datetime(2027, 4, 11), datetime(2027, 5, 26)),
    (2027, SeasonPhase.FINALS, datetime(2027, 5, 27), datetime(2027, 6, 18)),
    (2027, SeasonPhase.OFF_SEASON, datetime(2027, 6, 19), datetime(2027, 10, 1)),
    
    # 2027-2028 Season (projected dates based on typical NBA schedule)
    (2028, SeasonPhase.PRE_SEASON, datetime(2027, 10, 1), datetime(2027, 10, 15)),
    (2028, SeasonPhase.REGULAR_SEASON, datetime(2027, 10, 16), datetime(2028, 2, 10)),
    (2028, SeasonPhase.ALL_STAR_BREAK, datetime(2028, 2, 11), datetime(2028, 2, 16)),
    (2028, SeasonPhase.REGULAR_SEASON, datetime(2028, 2, 17), datetime(2028, 4, 8)),
    (2028, SeasonPhase.PLAYOFFS, datetime(2028, 4, 9), datetime(2028, 5, 24)),
    (2028, SeasonPhase.FINALS, datetime(2028, 5, 25), datetime(2028, 6, 16)),
    (2028, SeasonPhase.OFF_SEASON, datetime(2028, 6, 17), datetime(2028, 9, 30)),
    
    # 2028-2029 Season (projected dates based on typical NBA schedule)
    (2029, SeasonPhase.PRE_SEASON, datetime(2028, 9, 30), datetime(2028, 10, 14)),
    (2029, SeasonPhase.REGULAR_SEASON, datetime(2028, 10, 15), datetime(2029, 2, 9)),
    (2029, SeasonPhase.ALL_STAR_BREAK, datetime(2029, 2, 10), datetime(2029, 2, 15)),
    (2029, SeasonPhase.REGULAR_SEASON, datetime(2029, 2, 16), datetime(2029, 4, 7)),
    (2029, SeasonPhase.PLAYOFFS, datetime(2029, 4, 8), datetime(2029, 5, 23)),
    (2029, SeasonPhase.FINALS, datetime(2029, 5, 24), datetime(2029, 6, 15)),
    (2029, SeasonPhase.OFF_SEASON, datetime(2029, 6, 16), datetime(2029, 9, 29)),
    
    # 2029-2030 Season (projected dates based on typical NBA schedule)
    (2030, SeasonPhase.PRE_SEASON, datetime(2029, 9, 29), datetime(2029, 10, 13)),
    (2030, SeasonPhase.REGULAR_SEASON, datetime(2029, 10, 14), datetime(2030, 2, 8)),
    (2030, SeasonPhase.ALL_STAR_BREAK, datetime(2030, 2, 9), datetime(2030, 2, 14)),
    (2030, SeasonPhase.REGULAR_SEASON, datetime(2030, 2, 15), datetime(2030, 4, 6)),
    (2030, SeasonPhase.PLAYOFFS, datetime(2030, 4, 7), datetime(2030, 5, 22)),
    (2030, SeasonPhase.FINALS, datetime(2030, 5, 23), datetime(2030, 6, 14)),
    (2030, SeasonPhase.OFF_SEASON, datetime(2030, 6, 15), datetime(2030, 9, 28)),
]


def get_season_display_name(season_year: int) -> str:
    """
    Convert a season year to a display name (e.g., 2025 -> "2024-25")
    
    Args:
        season_year: The season year (e.g., 2025 for the 2024-25 season)
        
    Returns:
        str: The season display name
    """
    prev_year = season_year - 1
    short_year = str(season_year)[-2:]
    return f"{prev_year}-{short_year}"


def get_current_season_info(reference_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Get information about the current NBA season based on the given date
    
    Args:
        reference_date: Date to check (defaults to today)
        
    Returns:
        Dict with season_year and phase information
    """
    if reference_date is None:
        reference_date = datetime.now()
        
    # Convert to date only (no time component) for comparison
    reference_date = datetime(reference_date.year, reference_date.month, reference_date.day)
    
    # Find the season and phase for the reference date
    for season_year, phase, start_date, end_date in SEASON_DATE_RANGES:
        if start_date <= reference_date <= end_date:
            return {
                "season_year": season_year,
                "phase": phase,
                "start_date": start_date,
                "end_date": end_date,
                "display_name": get_season_display_name(season_year)
            }
    
    # If no match found, use the most recent season's off-season
    # This is a fallback that should rarely be needed if date ranges are properly maintained
    latest_season = max(SEASON_DATE_RANGES, key=lambda x: x[0])[0]
    return {
        "season_year": latest_season,
        "phase": SeasonPhase.OFF_SEASON,
        "display_name": get_season_display_name(latest_season)
    }
