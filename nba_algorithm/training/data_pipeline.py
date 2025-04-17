#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Production Data Pipeline Module

This module provides comprehensive data collection and validation for the NBA prediction system,
eliminating synthetic data in favor of complete, high-quality historical data.
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Import data collectors
from ..data.historical_collector import fetch_historical_games
from ..data.season_collector import get_training_seasons, get_training_data_seasons
from ..data.team_data import fetch_team_stats
from ..api.balldontlie_client import BallDontLieClient
from ..utils.validation import validate_data_completeness

# Configure logging
logger = logging.getLogger(__name__)


class ProductionDataPipeline:
    """Comprehensive data pipeline for production-quality model training"""
    
    def __init__(self, seasons: Optional[List[str]] = None):
        """
        Initialize the production data pipeline
        
        Args:
            seasons: Optional list of seasons to collect data for. If None, uses the 4-season rolling window.
        """
        self.seasons = seasons or get_training_seasons()
        self.client = BallDontLieClient()
        self.data_dir = Path('data')
        self.training_dir = self.data_dir / 'training'
        self.training_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Initialized production data pipeline for seasons: {self.seasons}")
    
    def collect_comprehensive_data(self) -> Dict[str, Any]:
        """
        Collect comprehensive data for model training without synthetic data
        
        Returns:
            Dict containing all data required for model training
        """
        logger.info("Starting comprehensive data collection")
        
        # Get data from the 4-season rolling window
        season_data = get_training_data_seasons()
        
        # Collect team statistics
        team_stats = fetch_team_stats()
        
        # Validate data completeness
        is_complete, missing_data = validate_data_completeness(season_data, team_stats)
        
        if not is_complete:
            logger.warning(f"Data is incomplete. Missing: {missing_data}")
            self._fill_data_gaps(season_data, missing_data)
        
        # Save the complete dataset
        training_file = self.training_dir / f"complete_training_data_{datetime.now().strftime('%Y%m%d')}.json"
        with open(training_file, 'w') as f:
            json.dump(season_data, f)
        
        logger.info(f"Saved complete training dataset to {training_file}")
        return season_data
    
    def _fill_data_gaps(self, data: Dict[str, Any], missing_data: Dict[str, Any]) -> None:
        """
        Fill gaps in the dataset with additional API calls instead of using synthetic data
        
        Args:
            data: Current dataset
            missing_data: Dictionary of missing data items
        """
        logger.info("Filling data gaps with additional API calls")
        
        # Fill in missing team statistics
        if 'team_stats' in missing_data:
            for team_id in missing_data['team_stats']:
                try:
                    team_stats = self.client.get_team_stats(team_id)
                    if team_id not in data['team_stats']:
                        data['team_stats'][team_id] = {}
                    data['team_stats'][team_id]['stats'] = team_stats
                    logger.info(f"Filled missing stats for team ID {team_id}")
                except Exception as e:
                    logger.error(f"Could not retrieve stats for team ID {team_id}: {str(e)}")
        
        # Fill in missing games
        if 'games' in missing_data:
            for game_id in missing_data['games']:
                try:
                    game_data = self.client.get_game(game_id)
                    if game_data:
                        data['games'].append(game_data)
                        logger.info(f"Filled missing data for game ID {game_id}")
                except Exception as e:
                    logger.error(f"Could not retrieve game ID {game_id}: {str(e)}")


def get_production_training_data() -> Dict[str, Any]:
    """
    Get complete, production-quality training data without synthetic values
    
    Returns:
        Dict containing all data required for model training
    """
    pipeline = ProductionDataPipeline()
    return pipeline.collect_comprehensive_data()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run the pipeline
    data = get_production_training_data()
    logger.info(f"Collected {len(data.get('games', []))} games and {len(data.get('team_stats', {}))} team stats")
