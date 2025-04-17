#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for injury_analysis module

This module contains tests for the injury analysis functionality, including mocks for API responses
to ensure tests can run without actual API access.

Author: Cascade
Date: April 2025
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
from datetime import datetime, timedelta

# Import the module to test
from nba_algorithm.models.injury_analysis import (
    fetch_team_injuries,
    calculate_player_importance,
    get_injury_impact_score,
    get_team_injury_data,
    compare_team_injuries
)


class TestInjuryAnalysis(unittest.TestCase):
    """
    Test cases for the injury analysis module
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        # Sample player injury data for testing
        self.sample_injuries = [
            {
                'player_id': 1,
                'player': {
                    'id': 1,
                    'first_name': 'Test',
                    'last_name': 'Player 1', 
                    'team_id': 1,
                    'position': 'G'
                },
                'team_id': 1,
                'team_name': 'Test Team A',
                'status': 'Out',
                'injury_type': 'Ankle',
                'expected_return': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
                'minutes_per_game': 32.5,
                'points_per_game': 24.3,
                'rebounds_per_game': 6.2,
                'assists_per_game': 5.1,
                'importance_score': 0.85
            },
            {
                'player_id': 2,
                'player': {
                    'id': 2,
                    'first_name': 'Test',
                    'last_name': 'Player 2',
                    'team_id': 1,
                    'position': 'F'
                },
                'team_id': 1,
                'team_name': 'Test Team A',
                'status': 'Day-to-Day',
                'injury_type': 'Illness',
                'expected_return': (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d'),
                'minutes_per_game': 18.2,
                'points_per_game': 8.7,
                'rebounds_per_game': 3.1,
                'assists_per_game': 1.5,
                'importance_score': 0.35
            },
            {
                'player_id': 3,
                'player': {
                    'id': 3,
                    'first_name': 'Test',
                    'last_name': 'Player 3',
                    'team_id': 2,
                    'position': 'C'
                },
                'team_id': 2,
                'team_name': 'Test Team B',
                'status': 'Out',
                'injury_type': 'ACL',
                'expected_return': (datetime.now() + timedelta(days=180)).strftime('%Y-%m-%d'),
                'minutes_per_game': 28.6,
                'points_per_game': 16.8,
                'rebounds_per_game': 8.9,
                'assists_per_game': 1.2,
                'importance_score': 0.75
            }
        ]
    
    @patch('nba_algorithm.models.injury_analysis.BallDontLieClient')
    @patch('nba_algorithm.models.injury_analysis.cache')
    def test_fetch_team_injuries(self, mock_cache, mock_client):
        """
        Test fetching team injuries with mocked API response
        """
        # Configure the mock cache to return None (cache miss)
        mock_cache.get.return_value = None
        
        # Configure the mock client to return our sample data
        mock_client_instance = mock_client.return_value
        mock_client_instance.get_player_injuries.return_value = {
            'data': self.sample_injuries
        }
        
        # Call the function
        result = fetch_team_injuries()
        
        # Assertions
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['player']['first_name'], 'Test')
        self.assertEqual(result[0]['player']['last_name'], 'Player 1')
        mock_client_instance.get_player_injuries.assert_called_once()
        mock_cache.set.assert_called_once()
    
    def test_get_player_importance(self):
        """
        Test calculating player importance
        """
        player_data = {
            'minutes_per_game': 35.0,
            'points_per_game': 28.5,
            'rebounds_per_game': 7.2,
            'assists_per_game': 6.8
        }
        
        # Call the function
        with patch('nba_algorithm.models.injury_analysis.get_player_stats') as mock_get_stats:
            # Mock the player stats return value
            mock_get_stats.return_value = {
                'min': 35.0,
                'pts': 28.5,
                'plus_minus': 8.3,
                'usage_percentage': 29.2
            }
            importance = calculate_player_importance(1)  # Using player ID 1 for test
        
        # Importance should be between 0 and 1
        self.assertTrue(0 <= importance <= 1)
        # A player with these stats should have high importance
        self.assertTrue(importance > 0.7)
    
    @patch('nba_algorithm.models.injury_analysis.calculate_player_importance')
    def test_get_injury_impact_score(self, mock_calc_importance):
        """
        Test calculating team injury impact
        """
        # Configure mock to return importance values
        mock_calc_importance.side_effect = [0.85, 0.35, 0.75]  # Importance values for each player
        
        # Call the function for Team A (has two injuries)
        impact = get_injury_impact_score(self.sample_injuries, 1)
        
        # Assertions
        self.assertTrue(0 <= impact['overall_impact'] <= 1)
        self.assertEqual(impact['num_injuries'], 2)
        self.assertTrue(impact['key_player_injuries'])  # Player 1 is important
        self.assertEqual(len(impact['detail']), 2)
        
        # Call the function for a team with no injuries
        impact = get_injury_impact_score(self.sample_injuries, 3)  # Team ID 3 not in data
        
        # Assertions
        self.assertEqual(impact['overall_impact'], 0.0)
        self.assertEqual(impact['num_injuries'], 0)
        self.assertFalse(impact['key_player_injuries'])
        self.assertEqual(len(impact['detail']), 0)
    
    @patch('nba_algorithm.models.injury_analysis.fetch_team_injuries')
    @patch('nba_algorithm.models.injury_analysis.get_injury_impact_score')
    def test_get_team_injury_data(self, mock_get_impact, mock_fetch):
        """
        Test getting team injury data
        """
        # Configure the mocks
        mock_fetch.return_value = self.sample_injuries
        mock_get_impact.return_value = {
            'team_id': 1,
            'overall_impact': 0.75,
            'num_injuries': 2,
            'key_player_injuries': True,
            'detail': [{'player_name': 'Test Player 1'}, {'player_name': 'Test Player 2'}]
        }
        
        # Call the function
        result = get_team_injury_data(1)  # Team ID 1
        
        # Assertions
        self.assertTrue('overall_impact' in result)
        self.assertTrue('num_injuries' in result)
        self.assertTrue('key_player_injuries' in result)
        self.assertEqual(result['num_injuries'], 2)
        
    @patch('nba_algorithm.models.injury_analysis.fetch_team_injuries')
    @patch('nba_algorithm.models.injury_analysis.get_injury_impact_score')
    def test_compare_team_injuries(self, mock_get_impact, mock_fetch):
        """
        Test comparing injuries between two teams
        """
        # Configure the mocks
        mock_fetch.return_value = self.sample_injuries
        
        # Mock the impact scores to ensure predictable results
        mock_get_impact.side_effect = [
            # Home team (team 1) - higher impact (more injuries)
            {
                'team_id': 1,
                'overall_impact': 0.8,
                'num_injuries': 2,
                'key_player_injuries': True,
                'detail': [{'player_name': 'Test Player 1'}, {'player_name': 'Test Player 2'}]
            },
            # Away team (team 2) - lower impact (fewer injuries)
            {
                'team_id': 2,
                'overall_impact': 0.4,
                'num_injuries': 1,
                'key_player_injuries': True,
                'detail': [{'player_name': 'Test Player 3'}]
            }
        ]
        
        # Call the function
        result = compare_team_injuries(1, 2)  # Team 1 vs Team 2
        
        # Assertions
        self.assertTrue('home_impact' in result)
        self.assertTrue('away_impact' in result)
        self.assertTrue('injury_advantage' in result)
        
        # Team 1 has more injuries including a key player, so advantage should be negative
        # (away team has advantage)
        self.assertTrue(result['injury_advantage'] < 0)


if __name__ == '__main__':
    unittest.main()
