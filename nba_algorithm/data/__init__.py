#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Package for NBA Algorithm

This package contains modules for data collection, validation, and processing
for the NBA prediction system.
"""

from .game_data import fetch_nba_games, fetch_game_data
from .team_data import fetch_team_stats, calculate_rest_days, is_back_to_back
from .historical_collector import fetch_historical_games
from .validation import validate_api_data, validate_player_data, check_api_keys
from .player_data import fetch_players_for_game, get_player_stats, fetch_player_injuries

from .odds_data import fetch_betting_odds

__all__ = [
    # Game data functions
    'fetch_nba_games',
    'fetch_game_data',
    
    # Team data functions
    'fetch_team_stats',
    'calculate_rest_days',
    'is_back_to_back',
    
    # Historical data functions
    'fetch_historical_games',
    
    # Validation functions
    'validate_api_data',
    'validate_player_data',
    'check_api_keys',
    
    # Player data functions
    'fetch_players_for_game',
    'get_player_stats',
    'fetch_player_injuries',
    
    # Odds data functions
    'fetch_betting_odds'
]
