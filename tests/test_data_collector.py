#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the DataCollector module
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional, Tuple

# Add src directory to path if needed for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.Model_training_pipeline.data_collector import DataCollector
from src.Model_training_pipeline.config import get_default_config


class TestDataCollector(unittest.TestCase):
    """Test case for the DataCollector class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.config = get_default_config()
        self.data_collector = DataCollector(self.config)
        
        # Create fixture directory if it doesn't exist
        self.fixtures_dir = Path(os.path.dirname(__file__)) / 'fixtures'
        self.fixtures_dir.mkdir(exist_ok=True)
        
        # Create mock data
        self.mock_game_data = {
            'data': [
                {
                    'id': 1,
                    'date': '2023-01-01',
                    'home_team': {'id': 1, 'full_name': 'Boston Celtics', 'abbreviation': 'BOS'},
                    'visitor_team': {'id': 2, 'full_name': 'Los Angeles Lakers', 'abbreviation': 'LAL'},
                    'home_team_score': 105,
                    'visitor_team_score': 98,
                    'season': 2022,
                    'status': 'Final'
                },
                {
                    'id': 2,
                    'date': '2023-01-02',
                    'home_team': {'id': 3, 'full_name': 'Golden State Warriors', 'abbreviation': 'GSW'},
                    'visitor_team': {'id': 4, 'full_name': 'Chicago Bulls', 'abbreviation': 'CHI'},
                    'home_team_score': 110,
                    'visitor_team_score': 112,
                    'season': 2022,
                    'status': 'Final'
                }
            ],
            'meta': {
                'total_pages': 1,
                'current_page': 1,
                'next_page': None,
                'per_page': 25,
                'total_count': 2
            }
        }
    
    def test_init(self):
        """Test initialization of DataCollector"""
        self.assertEqual(self.data_collector.config, self.config)
        self.assertEqual(self.data_collector.metrics['games_collected'], 0)
        self.assertEqual(self.data_collector.metrics['api_errors'], 0)
        
    @patch('src.Model_training_pipeline.data_collector.DataCollector._make_api_request')
    def test_collect_historical_data(self, mock_api_request):
        """Test collecting historical data"""
        # Configure mock to return our fixture data
        mock_api_request.return_value = self.mock_game_data
        
        # Call the method under test
        games, success = self.data_collector.collect_historical_data(seasons=[2022])
        
        # Verify the results
        self.assertTrue(success)
        self.assertEqual(len(games), 2)
        self.assertEqual(games[0]['home_team'], 'Boston Celtics')
        self.assertEqual(games[1]['visitor_team'], 'Chicago Bulls')
        
    @patch('src.Model_training_pipeline.data_collector.DataCollector._make_api_request')
    def test_collect_historical_data_error(self, mock_api_request):
        """Test handling API errors during data collection"""
        # Configure mock to return empty response
        mock_api_request.return_value = {}
        
        # Call the method under test
        games, success = self.data_collector.collect_historical_data(seasons=[2022])
        
        # Verify the results
        self.assertFalse(success)
        self.assertEqual(len(games), 0)
        self.assertTrue(self.data_collector.metrics['api_errors'] > 0)
        
    def test_filter_active_teams(self):
        """Test filtering games to active teams"""
        # Create sample games with both active and inactive teams
        sample_games = [
            {'home_team_id': 1, 'away_team_id': 2},  # Active teams
            {'home_team_id': 99, 'away_team_id': 2}   # Inactive team
        ]
        
        # Mock the active teams function
        with patch('src.Model_training_pipeline.data_collector.get_active_nba_teams') as mock_active:
            mock_active.return_value = [1, 2, 3, 4, 5]  # Active team IDs
            
            # Call the method
            filtered = self.data_collector._filter_active_teams(sample_games)
            
            # Verify only games with active teams are included
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0]['home_team_id'], 1)
    
    def test_get_collection_metrics(self):
        """Test retrieving collection metrics"""
        # Set up some metrics
        self.data_collector.metrics['games_collected'] = 10
        self.data_collector.metrics['teams_filtered'] = 2
        
        # Get metrics
        metrics = self.data_collector.get_collection_metrics()
        
        # Verify metrics are returned correctly
        self.assertEqual(metrics['games_collected'], 10)
        self.assertEqual(metrics['teams_filtered'], 2)


if __name__ == '__main__':
    unittest.main()
