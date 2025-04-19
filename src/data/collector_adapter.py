#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Historical Data Collector Adapter

This module provides compatibility adapters between the src.data.HistoricalDataCollector
and the nba_algorithm.data.HistoricalDataCollector classes. It ensures proper functionality
when either collector is used by the training pipeline.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

from src.data.historical_collector import HistoricalDataCollector as SourceCollector
from src.api.balldontlie_client import BallDontLieClient
from src.api.theodds_client import TheOddsClient

# Configure logging
logger = logging.getLogger(__name__)

# Data directory paths
DATA_DIR = Path('data/historical')
DATA_DIR.mkdir(parents=True, exist_ok=True)

class HistoricalDataCollectorAdapter(SourceCollector):
    """
    Adapter class for HistoricalDataCollector that adds missing attributes
    and provides compatibility between different collector versions
    """
    
    def __init__(self, bdl_client=None, odds_client=None):
        """
        Initialize the historical data collector adapter with proper directory structure
        
        Args:
            bdl_client: BallDontLie API client (optional)
            odds_client: The Odds API client (optional)
        """
        # Initialize the base collector
        super().__init__(bdl_client=bdl_client, odds_client=odds_client)
        
        # Add directory attributes that might be expected by other code
        self.games_dir = DATA_DIR / 'games'
        self.teams_dir = DATA_DIR / 'teams'
        self.players_dir = DATA_DIR / 'players'
        self.stats_dir = DATA_DIR / 'stats'
        self.odds_dir = DATA_DIR / 'odds'
        
        # Create the directories
        self.games_dir.mkdir(exist_ok=True)
        self.teams_dir.mkdir(exist_ok=True)
        self.players_dir.mkdir(exist_ok=True)
        self.stats_dir.mkdir(exist_ok=True)
        self.odds_dir.mkdir(exist_ok=True)
        
        # Add cache dictionaries that might be expected
        self.teams_cache = {}
        self.players_cache = {}
        self.games_cache = {}
        
        logger.info("Initialized HistoricalDataCollector adapter with required attributes")
