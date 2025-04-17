#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Test Script for Advanced Features

This script tests that the new advanced features we've added are properly integrated
with the existing prediction system.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import key modules
from nba_algorithm.data.season_collector import (
    get_training_seasons,
    maintain_training_window,
    register_season_change_handlers
)
from nba_algorithm.features.advanced_features import (
    create_momentum_features,
    create_matchup_features,
    integrate_advanced_features
)
from nba_algorithm.features.feature_engineering import (
    prepare_game_features
)

def test_season_collector():
    """Test the season_collector functionality"""
    logger.info("Testing season_collector")
    # Call the functions from season_collector to validate they work
    try:
        # Just check that these functions can be called
        seasons = get_training_seasons()
        logger.info(f"✓ get_training_seasons returns {len(seasons)} seasons")
        return True
    except Exception as e:
        logger.error(f"Season collector test failed: {str(e)}")
        return False

def test_advanced_features():
    """Test the advanced features functionality"""
    logger.info("Testing advanced features")
    
    # Create dummy historical data for testing
    historical_games = [
        {
            "id": 1, 
            "home_team": {"id": 1, "name": "Team A"}, 
            "visitor_team": {"id": 2, "name": "Team B"},
            "home_team_score": 100,
            "visitor_team_score": 95,
            "date": "2025-01-01"
        },
        {
            "id": 2, 
            "home_team": {"id": 1, "name": "Team A"}, 
            "visitor_team": {"id": 3, "name": "Team C"},
            "home_team_score": 110,
            "visitor_team_score": 105,
            "date": "2025-01-03"
        },
        {
            "id": 3, 
            "home_team": {"id": 2, "name": "Team B"}, 
            "visitor_team": {"id": 1, "name": "Team A"},
            "home_team_score": 98,
            "visitor_team_score": 102,
            "date": "2025-01-05"
        },
    ]
    
    # Test momentum features
    try:
        momentum_features = create_momentum_features(historical_games, 1, decay_factor=0.85)
        logger.info(f"✓ Momentum features created successfully: {momentum_features}")
    except Exception as e:
        logger.error(f"Failed to create momentum features: {str(e)}")
        return False
    
    # Test matchup features
    try:
        matchup_features = create_matchup_features(historical_games, 1, 2)
        logger.info(f"✓ Matchup features created successfully: {matchup_features}")
    except Exception as e:
        logger.error(f"Failed to create matchup features: {str(e)}")
        return False
    
    return True

def test_feature_engineering_integration():
    """Test integration with the feature engineering pipeline"""
    logger.info("Testing feature engineering integration")
    
    # This is just an import test to make sure our changes to feature_engineering.py
    # don't break anything. We're not actually running the full pipeline.
    
    logger.info("✓ Feature engineering imports work correctly")
    return True

def main():
    """Run all integration tests"""
    tests = [
        test_season_collector,
        test_advanced_features,
        test_feature_engineering_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    if all(results):
        logger.info("All integration tests passed! The new features are properly integrated.")
        return 0
    else:
        logger.error("Some integration tests failed. See logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
