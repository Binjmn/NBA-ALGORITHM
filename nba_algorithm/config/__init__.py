# -*- coding: utf-8 -*-
"""
Configuration Package for NBA Algorithm

This package contains configuration modules for the NBA prediction system.
"""

from .season_config import (
    SeasonPhase,
    IN_SEASON_PHASES,
    get_season_display_name,
    get_current_season_info
)

__all__ = [
    # Season configuration
    'SeasonPhase',
    'IN_SEASON_PHASES',
    'get_season_display_name',
    'get_current_season_info'
]
