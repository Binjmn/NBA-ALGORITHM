#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for advanced_metrics module

This module contains tests for the advanced metrics functionality, including mocks for API responses
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
from nba_algorithm.models.advanced_metrics import (
    fetch_player_advanced_metrics,
    fetch_team_advanced_metrics,
    calculate_efficiency_score,
    calculate_team_efficiency_score,
    get_top_players_by_efficiency,
    get_player_efficiency_rating,
    get_matchup_efficiency_differential,
    get_team_efficiency_comparison
)


class TestAdvancedMetrics(unittest.TestCase):
    """
    Test cases for the advanced metrics module
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        # Sample player advanced metrics data for testing
        self.sample_player_metrics = [
            {
                'player': {
                    'id': 1,
                    'first_name': 'Test',
                    'last_name': 'Player 1',
                    'position': 'G'
                },
                'game': {
                    'id': 101,
                    'date': '2025-04-01'
                },
                'pie': 0.18,
                'offensive_rating': 116.5,
                'defensive_rating': 104.2,
                'net_rating': 12.3,
                'true_shooting_percentage': 0.58,
                'effective_field_goal_percentage': 0.55,
                'usage_percentage': 28.5,
                'assist_percentage': 32.1
            },
            {
                'player': {
                    'id': 1,
                    'first_name': 'Test',
                    'last_name': 'Player 1',
                    'position': 'G'
                },
                'game': {
                    'id': 102,
                    'date': '2025-04-03'
                },
                'pie': 0.2,
                'offensive_rating': 118.2,
                'defensive_rating': 103.1,
                'net_rating': 15.1,
                'true_shooting_percentage': 0.6,
                'effective_field_goal_percentage': 0.56,
                'usage_percentage': 29.2,
                'assist_percentage': 33.5
            },
            {
                'player': {
                    'id': 2,
                    'first_name': 'Test',
                    'last_name': 'Player 2',
                    'position': 'F'
                },
                'game': {
                    'id': 101,
                    'date': '2025-04-01'
                },
                'pie': 0.14,
                'offensive_rating': 110.8,
                'defensive_rating': 106.5,
                'net_rating': 4.3,
                'true_shooting_percentage': 0.52,
                'effective_field_goal_percentage': 0.48,
                'usage_percentage': 22.1,
                'assist_percentage': 18.7
            }
        ]
        
        # Sample team advanced metrics for testing
        self.sample_team_metrics = {
            'team_id': 1,
            'offensive_rating': 114.5,
            'defensive_rating': 105.8,
            'net_rating': 8.7,
            'pace': 98.6,
            'true_shooting_percentage': 0.56,
            'effective_field_goal_percentage': 0.53,
            'assist_percentage': 64.2,
            'assist_ratio': 19.5,
            'assist_to_turnover': 2.1,
            'turnover_ratio': 12.4,
            'rebound_percentage': 51.2
        }
    
    @patch('nba_algorithm.models.advanced_metrics.BallDontLieClient')
    @patch('nba_algorithm.models.advanced_metrics.cache')
    def test_fetch_player_advanced_metrics(self, mock_cache, mock_client):
        """
        Test fetching player advanced metrics with mocked API response
        """
        # Configure the mock cache to return None (cache miss)
        mock_cache.get.return_value = None
        
        # Configure the mock client to return our sample data
        mock_client_instance = mock_client.return_value
        mock_client_instance.get_advanced_stats.return_value = {
            'data': self.sample_player_metrics[:2]  # Only metrics for player 1
        }
        
        # Call the function
        result = fetch_player_advanced_metrics(1)  # Player ID 1
        
        # Assertions
        self.assertTrue('pie' in result)
        self.assertTrue('offensive_rating' in result)
        self.assertTrue('efficiency_score' in result)
        self.assertTrue(0 <= result['efficiency_score'] <= 1)
        mock_client_instance.get_advanced_stats.assert_called_once()
        mock_cache.set.assert_called_once()
    
    @patch('nba_algorithm.models.advanced_metrics.BallDontLieClient')
    @patch('nba_algorithm.models.advanced_metrics.cache')
    def test_fetch_team_advanced_metrics(self, mock_cache, mock_client):
        """
        Test fetching team advanced metrics with mocked API response
        """
        # Configure the mock cache to return None (cache miss)
        mock_cache.get.return_value = None
        
        # Configure the mock client to return our sample data
        mock_client_instance = mock_client.return_value
        mock_client_instance.get_advanced_stats.return_value = {
            'data': self.sample_player_metrics  # All player metrics
        }
        
        # Call the function
        result = fetch_team_advanced_metrics(1)  # Team ID 1
        
        # Assertions
        self.assertTrue('offensive_rating' in result)
        self.assertTrue('defensive_rating' in result)
        self.assertTrue('team_efficiency_score' in result)
        self.assertTrue(0 <= result['team_efficiency_score'] <= 1)
        self.assertTrue('top_players' in result)
        mock_client_instance.get_advanced_stats.assert_called_once()
        mock_cache.set.assert_called_once()
    
    def test_calculate_efficiency_score(self):
        """
        Test calculating player efficiency score
        """
        # Use first player metrics as test data
        metrics = {
            'pie': 0.18,
            'offensive_rating': 116.5,
            'defensive_rating': 104.2,
            'net_rating': 12.3,
            'true_shooting_percentage': 0.58,
            'effective_field_goal_percentage': 0.55,
            'usage_percentage': 28.5,
            'assist_percentage': 32.1
        }
        
        # Call the function
        score = calculate_efficiency_score(metrics)
        
        # Assertions
        self.assertTrue(0 <= score <= 1)
        # These are good metrics, so score should be high
        self.assertTrue(score > 0.7)
        
        # Test with empty metrics
        self.assertEqual(calculate_efficiency_score({}), 0.0)
    
    def test_calculate_team_efficiency_score(self):
        """
        Test calculating team efficiency score
        """
        # Call the function with sample team metrics
        score = calculate_team_efficiency_score(self.sample_team_metrics)
        
        # Assertions
        self.assertTrue(0 <= score <= 1)
        # These are good team metrics, so score should be high
        self.assertTrue(score > 0.6)
        
        # Test with empty metrics
        self.assertEqual(calculate_team_efficiency_score({}), 0.0)
    
    def test_get_top_players_by_efficiency(self):
        """
        Test extracting top players by efficiency
        """
        # Add extra mock data to ensure we have enough players for the test
        # We need at least 2 players with enough games each
        sample_data = self.sample_player_metrics.copy()
        # Add another game for player 2 to ensure it has enough games
        sample_data.append({
            'player': {
                'id': 2,
                'first_name': 'Test',
                'last_name': 'Player 2',
                'position': 'F'
            },
            'game': {
                'id': 102,
                'date': '2025-04-03'
            },
            'pie': 0.15,
            'offensive_rating': 112.3,
            'defensive_rating': 105.1,
            'net_rating': 7.2,
            'true_shooting_percentage': 0.53,
            'effective_field_goal_percentage': 0.49,
            'usage_percentage': 23.4,
            'assist_percentage': 19.2
        })
        
        # Call the function with our expanded data
        top_players = get_top_players_by_efficiency(sample_data, limit=2)
        
        # Assertions
        self.assertEqual(len(top_players), 2)
        self.assertTrue('player_id' in top_players[0])
        self.assertTrue('player_name' in top_players[0])
        self.assertTrue('efficiency_score' in top_players[0])
        self.assertTrue('games_played' in top_players[0])
        
        # Player 1 should have higher efficiency than Player 2
        self.assertTrue(top_players[0]['efficiency_score'] > top_players[1]['efficiency_score'])
    
    @patch('nba_algorithm.models.advanced_metrics.fetch_player_advanced_metrics')
    def test_get_player_efficiency_rating(self, mock_fetch):
        """
        Test getting player efficiency rating
        """
        # Configure mock to return different values for different calls
        mock_fetch.side_effect = [
            # For season metrics
            {
                'efficiency_score': 0.85,
                'pie': 0.18,
                'offensive_rating': 116.5
            },
            # For recent metrics
            {
                'efficiency_score': 0.88,
                'pie': 0.2,
                'offensive_rating': 118.2
            }
        ]
        
        # Call the function
        result = get_player_efficiency_rating(1)  # Player ID 1
        
        # Assertions
        self.assertTrue('season_efficiency' in result)
        self.assertTrue('recent_efficiency' in result)
        self.assertTrue('trend' in result)
        # Use assertAlmostEqual for floating point comparisons
        self.assertAlmostEqual(result['trend'], 0.03, places=5)  # Recent - Season
    
    @patch('nba_algorithm.models.advanced_metrics.get_player_efficiency_rating')
    def test_get_matchup_efficiency_differential(self, mock_get_rating):
        """
        Test comparing efficiency between two players
        """
        # Configure mock to return different values for different players
        mock_get_rating.side_effect = [
            # Home player
            {
                'season_efficiency': 0.85,
                'recent_efficiency': 0.88
            },
            # Away player
            {
                'season_efficiency': 0.75,
                'recent_efficiency': 0.72
            }
        ]
        
        # Call the function
        result = get_matchup_efficiency_differential(1, 2)  # Home player vs Away player
        
        # Assertions
        self.assertTrue('home_efficiency' in result)
        self.assertTrue('away_efficiency' in result)
        self.assertTrue('differential' in result)
        self.assertTrue('recent_differential' in result)
        # Use assertAlmostEqual for floating point comparisons
        self.assertAlmostEqual(result['differential'], 0.1, places=5)  # Home advantage
        self.assertAlmostEqual(result['recent_differential'], 0.16, places=5)  # Recent home advantage
    
    @patch('nba_algorithm.models.advanced_metrics.fetch_team_advanced_metrics')
    def test_get_team_efficiency_comparison(self, mock_fetch):
        """
        Test comparing efficiency between two teams
        """
        # Configure mock to return different values for different teams
        mock_fetch.side_effect = [
            # Home team
            {
                'team_efficiency_score': 0.75,
                'offensive_rating': 114.5,
                'defensive_rating': 105.8,
                'top_players': [{'player_name': 'Home Star', 'efficiency_score': 0.88}]
            },
            # Away team
            {
                'team_efficiency_score': 0.68,
                'offensive_rating': 112.0,
                'defensive_rating': 108.3,
                'top_players': [{'player_name': 'Away Star', 'efficiency_score': 0.82}]
            }
        ]
        
        # Call the function
        result = get_team_efficiency_comparison(1, 2)  # Home team vs Away team
        
        # Assertions
        self.assertTrue('home_efficiency' in result)
        self.assertTrue('away_efficiency' in result)
        self.assertTrue('offensive_differential' in result)
        self.assertTrue('defensive_differential' in result)
        self.assertTrue('overall_differential' in result)
        self.assertTrue('home_top_players' in result)
        self.assertTrue('away_top_players' in result)
        
        # Home team is better, so differential should be positive
        # Use assertAlmostEqual for floating point comparisons
        self.assertAlmostEqual(result['overall_differential'], 0.07, places=5)
        self.assertAlmostEqual(result['offensive_differential'], 2.5, places=5)
        # For defensive, lower is better, so differential should be positive
        self.assertAlmostEqual(result['defensive_differential'], 2.5, places=5)


if __name__ == '__main__':
    unittest.main()
