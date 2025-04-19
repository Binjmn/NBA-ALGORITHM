#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the FeatureEngineering module
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple

# Add src directory to path if needed for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.Model_training_pipeline.feature_engineering import FeatureEngineering
from src.Model_training_pipeline.config import get_default_config


class TestFeatureEngineering(unittest.TestCase):
    """Test case for the FeatureEngineering class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.config = get_default_config()
        self.feature_engineer = FeatureEngineering(self.config)
        
        # Create sample game data
        self.sample_games = [
            {
                'id': 1,
                'date': '2023-01-01',
                'home_team': 'Boston Celtics',
                'away_team': 'Los Angeles Lakers',
                'home_team_id': 1,
                'away_team_id': 2,
                'home_score': 105,
                'away_score': 98,
                'home_win': 1,
                'away_win': 0,
                'season': 2022,
                # Team stats
                'home_pts': 105,
                'away_pts': 98,
                'home_fgm': 40,
                'away_fgm': 35,
                'home_fga': 85,
                'away_fga': 90,
                'home_tpm': 12,
                'away_tpm': 10,
                'home_reb': 45,
                'away_reb': 40,
                'home_ast': 25,
                'away_ast': 20,
                'home_stl': 8,
                'away_stl': 6,
                'home_blk': 5,
                'away_blk': 4,
                'home_to': 12,
                'away_to': 15,
                'home_pf': 18,
                'away_pf': 20,
                # Advanced stats
                'home_win_streak': 3,
                'away_win_streak': 1,
                'home_form': ['W', 'W', 'W', 'L'],
                'away_form': ['W', 'L', 'L', 'L'],
                'home_pts_last4': 102.5,
                'away_pts_last4': 96.0,
                'matchup_history': {
                    'home_wins': 3,
                    'away_wins': 2,
                    'avg_point_diff': 4.5,
                    'last_point_diff': 7.0
                }
            },
            {
                'id': 2,
                'date': '2023-01-02',
                'home_team': 'Golden State Warriors',
                'away_team': 'Chicago Bulls',
                'home_team_id': 3,
                'away_team_id': 4,
                'home_score': 110,
                'away_score': 112,
                'home_win': 0,
                'away_win': 1,
                'season': 2022,
                # Team stats
                'home_pts': 110,
                'away_pts': 112,
                'home_fgm': 42,
                'away_fgm': 43,
                'home_fga': 88,
                'away_fga': 86,
                'home_tpm': 14,
                'away_tpm': 15,
                'home_reb': 42,
                'away_reb': 44,
                'home_ast': 28,
                'away_ast': 24,
                'home_stl': 7,
                'away_stl': 9,
                'home_blk': 4,
                'away_blk': 3,
                'home_to': 14,
                'away_to': 12,
                'home_pf': 20,
                'away_pf': 18,
                # Advanced stats
                'home_win_streak': 0,
                'away_win_streak': 2,
                'home_form': ['L', 'W', 'W', 'L'],
                'away_form': ['W', 'W', 'L', 'L'],
                'home_pts_last4': 108.5,
                'away_pts_last4': 109.0,
                'matchup_history': {
                    'home_wins': 1,
                    'away_wins': 4,
                    'avg_point_diff': -3.0,
                    'last_point_diff': -2.0
                }
            }
        ]
    
    def test_init(self):
        """Test initialization of FeatureEngineering"""
        self.assertEqual(self.feature_engineer.config, self.config)
        self.assertEqual(self.feature_engineer.metrics['features_created'], 0)
        
    def test_engineer_features(self):
        """Test feature engineering from game data"""
        # Call the method under test
        games_df, features = self.feature_engineer.engineer_features(self.sample_games)
        
        # Verify the results
        self.assertEqual(len(features), 2)  # Two games
        self.assertIsInstance(games_df, pd.DataFrame)
        self.assertEqual(games_df.shape[0], 2)  # Two rows
        
        # Check that core features are created
        self.assertIn('home_team', features[0])
        self.assertIn('away_team', features[0])
        self.assertIn('home_win', features[0])
        self.assertIn('away_win', features[0])
        
        # Check advanced features
        self.assertIn('home_win_streak', features[0])
        self.assertIn('home_win_pct', features[0])
        self.assertIn('matchup_avg_point_diff', features[0])
        
    def test_prepare_features_for_target(self):
        """Test preparing features for specific prediction target"""
        # Engineer features first
        _, features = self.feature_engineer.engineer_features(self.sample_games)
        
        # Prepare for moneyline prediction
        X, y = self.feature_engineer.prepare_features_for_target(features, 'home_win', 'classification')
        
        # Verify results
        self.assertEqual(len(y), 2)  # Two target values
        self.assertEqual(y[0], 1)  # First game home win
        self.assertEqual(y[1], 0)  # Second game home loss
        
        # Should have multiple feature columns
        self.assertTrue(X.shape[1] > 5)
        
    def test_get_engineering_metrics(self):
        """Test retrieving engineering metrics"""
        # Set up some metrics
        self.feature_engineer.metrics['features_created'] = 50
        self.feature_engineer.metrics['advanced_features'] = 20
        
        # Get metrics
        metrics = self.feature_engineer.get_engineering_metrics()
        
        # Verify metrics are returned correctly
        self.assertEqual(metrics['features_created'], 50)
        self.assertEqual(metrics['advanced_features'], 20)
        
    def test_add_advanced_features(self):
        """Test adding advanced features to feature dictionary"""
        # Create a base feature dictionary
        feature_dict = {
            'home_team': 'Boston Celtics',
            'away_team': 'Los Angeles Lakers',
            'home_pts': 105,
            'away_pts': 98,
            'home_fgm': 40,
            'away_fgm': 35,
            'home_fga': 85,
            'away_fga': 90,
            'home_tpm': 12,
            'away_tpm': 10
        }
        
        # Add advanced features
        count = self.feature_engineer._add_advanced_features(self.sample_games[0], feature_dict)
        
        # Verify features were added
        self.assertTrue(count > 0)
        self.assertIn('home_win_streak', feature_dict)
        self.assertIn('home_win_pct', feature_dict)
        self.assertIn('home_pts_momentum', feature_dict)
        self.assertIn('matchup_avg_point_diff', feature_dict)


if __name__ == '__main__':
    unittest.main()
