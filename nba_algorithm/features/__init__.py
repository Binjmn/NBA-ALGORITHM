#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Features Package for NBA Algorithm

This package contains modules for feature engineering and preparation
for the NBA prediction models.
"""

# Import the version from game_features.py with a different name to avoid conflicts
from .game_features import prepare_game_features as prepare_basic_game_features
from .game_features import extract_game_features

# Import the full-featured version from feature_engineering
from .feature_engineering import prepare_game_features

from .player_features import extract_player_features, predict_props_for_player, get_player_season_average

__all__ = [
    # Game features
    'prepare_game_features',
    'prepare_basic_game_features',  # Provide both versions
    'extract_game_features',
    
    # Player features
    'extract_player_features',
    'predict_props_for_player',
    'get_player_season_average'
]
