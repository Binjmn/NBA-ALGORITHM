#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Metrics Module

This module handles the collection and analysis of advanced NBA metrics
for both players and teams. It provides functions to fetch efficiency ratings
and other advanced statistics to enhance prediction accuracy.

Author: Cascade
Date: April 2025
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

from ..api.balldontlie_client import BallDontLieClient
from ..utils.cache_manager import CacheManager

# Configure logger
logger = logging.getLogger(__name__)

# Initialize cache manager for advanced metrics
cache = CacheManager(cache_name="advanced_metrics", ttl_seconds=86400)  # Cache for 24 hours

# Importance weights for different advanced metrics
METRIC_WEIGHTS = {
    'pie': 0.15,                  # Player Impact Estimate
    'offensive_rating': 0.15,      # Points per 100 possessions
    'defensive_rating': 0.15,      # Points allowed per 100 possessions
    'net_rating': 0.2,             # Point differential per 100 possessions
    'true_shooting_percentage': 0.1,  # Shooting efficiency
    'effective_field_goal_percentage': 0.1,  # Field goal efficiency accounting for 3s
    'usage_percentage': 0.1,       # Percentage of team plays used
    'assist_percentage': 0.05      # Percentage of teammate field goals assisted
}


def fetch_player_advanced_metrics(player_id: int, 
                                  season: Optional[int] = None,
                                  last_n_games: Optional[int] = None) -> Dict[str, Any]:
    """
    Fetch advanced metrics for a specific player
    
    Args:
        player_id: Player ID to fetch advanced metrics for
        season: Optional season to filter by (e.g., 2023 for 2023-24 season)
        last_n_games: Optional number of recent games to consider
        
    Returns:
        Dictionary with player's advanced metrics
    """
    # Construct cache key
    cache_key = f"player_advanced_{player_id}_{season}_{last_n_games}"
    
    # Check cache first
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        logger.info(f"Using cached advanced metrics for player_id={player_id}")
        return cached_data
    
    try:
        client = BallDontLieClient()
        
        # Determine date range if last_n_games is specified
        params = {
            'player_id': player_id,
            'per_page': last_n_games if last_n_games else 100  # Fetch enough to get a good sample
        }
        
        if season:
            params['season'] = season
            
        # Fetch advanced stats
        response = client.get_advanced_stats(**params)
        
        if not response or 'data' not in response or not response['data']:
            logger.warning(f"No advanced metrics data returned for player_id={player_id}")
            return {}
            
        stats = response['data']
        
        # Limit to last_n_games if specified
        if last_n_games and len(stats) > last_n_games:
            # Sort by date to get the most recent games
            stats = sorted(stats, key=lambda x: x.get('game', {}).get('date', ''), reverse=True)[:last_n_games]
        
        # Calculate averages for all metrics
        metrics = {}
        for key in METRIC_WEIGHTS.keys():
            values = [float(stat.get(key, 0)) for stat in stats if key in stat]
            if values:
                metrics[key] = sum(values) / len(values)
            else:
                metrics[key] = 0.0
        
        # Add player info
        if stats and 'player' in stats[0]:
            metrics['player'] = stats[0]['player']
        
        # Calculate composite efficiency score
        efficiency_score = calculate_efficiency_score(metrics)
        metrics['efficiency_score'] = efficiency_score
        
        # Cache the results
        cache.set(cache_key, metrics)
        
        logger.info(f"Fetched advanced metrics for player_id={player_id}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error fetching player advanced metrics: {str(e)}")
        return {}


def fetch_team_advanced_metrics(team_id: int, 
                                season: Optional[int] = None,
                                last_n_games: Optional[int] = None) -> Dict[str, Any]:
    """
    Fetch and aggregate advanced metrics for a team
    
    Args:
        team_id: Team ID to fetch advanced metrics for
        season: Optional season to filter by (e.g., 2023 for 2023-24 season)
        last_n_games: Optional number of recent games to consider
        
    Returns:
        Dictionary with team's aggregated advanced metrics
    """
    # Construct cache key
    cache_key = f"team_advanced_{team_id}_{season}_{last_n_games}"
    
    # Check cache first
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        logger.info(f"Using cached advanced metrics for team_id={team_id}")
        return cached_data
    
    try:
        client = BallDontLieClient()
        
        # Determine parameters for API call
        params = {
            'team_id': team_id,
            'per_page': 100  # Fetch enough to get a good sample
        }
        
        if season:
            params['season'] = season
            
        # Fetch team games with advanced stats
        response = client.get_advanced_stats(**params)
        
        if not response or 'data' not in response or not response['data']:
            logger.warning(f"No advanced metrics data returned for team_id={team_id}")
            return {}
            
        stats = response['data']
        
        # Limit to last_n_games if specified
        if last_n_games and len(stats) > 0:
            # Group by game
            games = {}
            for stat in stats:
                game_id = stat.get('game', {}).get('id')
                if game_id and game_id not in games:
                    games[game_id] = {
                        'date': stat.get('game', {}).get('date', ''),
                        'stats': []
                    }
                if game_id:
                    games[game_id]['stats'].append(stat)
            
            # Sort games by date and take the last n
            sorted_games = sorted(games.values(), key=lambda x: x['date'], reverse=True)[:last_n_games]
            
            # Flatten stats list
            stats = []
            for game in sorted_games:
                stats.extend(game['stats'])
        
        # Calculate team averages for metrics that make sense at team level
        team_metrics = {
            'team_id': team_id,
            'offensive_rating': 0.0,
            'defensive_rating': 0.0,
            'net_rating': 0.0,
            'pace': 0.0,
            'true_shooting_percentage': 0.0,
            'effective_field_goal_percentage': 0.0,
            'assist_percentage': 0.0,
            'assist_ratio': 0.0,
            'assist_to_turnover': 0.0,
            'turnover_ratio': 0.0,
            'rebound_percentage': 0.0
        }
        
        # Count games (unique game IDs in the data)
        game_ids = set()
        for stat in stats:
            game_id = stat.get('game', {}).get('id')
            if game_id:
                game_ids.add(game_id)
        
        # Aggregate metrics across all players
        game_metrics = {}
        for stat in stats:
            game_id = stat.get('game', {}).get('id')
            if not game_id:
                continue
                
            # Initialize game metrics if needed
            if game_id not in game_metrics:
                game_metrics[game_id] = {
                    'offensive_rating': [],
                    'defensive_rating': [],
                    'net_rating': [],
                    'pace': [],
                    'true_shooting_percentage': [],
                    'effective_field_goal_percentage': [],
                    'assist_percentage': [],
                    'assist_ratio': [],
                    'assist_to_turnover': [],
                    'turnover_ratio': [],
                    'rebound_percentage': []
                }
            
            # Add metrics to game
            for key in game_metrics[game_id].keys():
                if key in stat and stat[key] is not None:
                    game_metrics[game_id][key].append(float(stat[key]))
        
        # Calculate team-level averages across games
        for game_id, metrics in game_metrics.items():
            # For each metric, calculate weighted average by minutes played
            for key in metrics.keys():
                if metrics[key]:  # If we have values
                    avg = sum(metrics[key]) / len(metrics[key])
                    
                    # Add to team average
                    if key not in team_metrics:
                        team_metrics[key] = avg / len(game_metrics)
                    else:
                        team_metrics[key] += avg / len(game_metrics)
        
        # Calculate team efficiency score
        efficiency_score = calculate_team_efficiency_score(team_metrics)
        team_metrics['team_efficiency_score'] = efficiency_score
        
        # Calculate top players by efficiency
        top_players = get_top_players_by_efficiency(stats, limit=5)
        team_metrics['top_players'] = top_players
        
        # Cache the results
        cache.set(cache_key, team_metrics)
        
        logger.info(f"Fetched and aggregated advanced metrics for team_id={team_id}")
        return team_metrics
        
    except Exception as e:
        logger.error(f"Error fetching team advanced metrics: {str(e)}")
        return {}


def calculate_efficiency_score(metrics: Dict[str, Any]) -> float:
    """
    Calculate a composite efficiency score for a player based on advanced metrics
    
    Args:
        metrics: Dictionary of player metrics
        
    Returns:
        Composite efficiency score between 0 and 1
    """
    if not metrics:
        return 0.0
        
    try:
        # Start with default score
        score = 0.0
        
        # Calculate weighted sum
        for metric, weight in METRIC_WEIGHTS.items():
            if metric in metrics and metrics[metric] is not None:
                # Handle metrics where lower is better (defensive rating)
                if metric == 'defensive_rating':
                    # Normalize: Good defensive rating is ~100, bad is ~120
                    # Transform so lower values give higher score
                    value = max(0, min(1, (120 - float(metrics[metric])) / 20))
                # Handle metrics with typical percentage ranges
                elif metric in ['true_shooting_percentage', 'effective_field_goal_percentage']:
                    # Normalize: ~40% is poor, ~65% is excellent
                    value = max(0, min(1, (float(metrics[metric]) - 0.4) / 0.25))
                # Handle PIE (Player Impact Estimate)
                elif metric == 'pie':
                    # Normalize: ~0.05 is average, ~0.20 is star level
                    value = max(0, min(1, float(metrics[metric]) / 0.2))
                # Handle usage percentage
                elif metric == 'usage_percentage':
                    # Normalize: ~15% is low, ~35% is very high
                    value = max(0, min(1, (float(metrics[metric]) - 15) / 20))
                # Other metrics where higher is better
                else:
                    # Use a typical range based on metric
                    if metric == 'offensive_rating':
                        # Good offensive rating is ~115, poor is ~95
                        value = max(0, min(1, (float(metrics[metric]) - 95) / 20))
                    elif metric == 'net_rating':
                        # Good net rating is ~+10, poor is ~-10
                        value = max(0, min(1, (float(metrics[metric]) + 10) / 20))
                    elif metric == 'assist_percentage':
                        # Normalize assist percentage to 0-1 scale
                        value = max(0, min(1, float(metrics[metric]) / 50))
                    else:
                        # Default normalization
                        value = float(metrics[metric])
                
                score += value * weight
        
        return score
        
    except Exception as e:
        logger.error(f"Error calculating efficiency score: {str(e)}")
        return 0.0


def calculate_team_efficiency_score(metrics: Dict[str, Any]) -> float:
    """
    Calculate a composite efficiency score for a team based on advanced metrics
    
    Args:
        metrics: Dictionary of team metrics
        
    Returns:
        Composite team efficiency score between 0 and 1
    """
    # Similar to player efficiency but using team-appropriate weights
    team_weights = {
        'offensive_rating': 0.25,
        'defensive_rating': 0.25,
        'net_rating': 0.3,
        'pace': 0.05,
        'assist_ratio': 0.1,
        'rebound_percentage': 0.05
    }
    
    if not metrics:
        return 0.0
        
    try:
        # Start with default score
        score = 0.0
        
        # Calculate weighted sum
        for metric, weight in team_weights.items():
            if metric in metrics and metrics[metric] is not None:
                # Handle metrics where lower is better (defensive rating)
                if metric == 'defensive_rating':
                    # Normalize: Good defensive rating is ~100, bad is ~120
                    value = max(0, min(1, (120 - float(metrics[metric])) / 20))
                # Handle metrics with special ranges
                elif metric == 'offensive_rating':
                    # Good offensive rating is ~115, poor is ~95
                    value = max(0, min(1, (float(metrics[metric]) - 95) / 20))
                elif metric == 'net_rating':
                    # Good net rating is ~+10, poor is ~-10
                    value = max(0, min(1, (float(metrics[metric]) + 10) / 20))
                elif metric == 'pace':
                    # Pace itself isn't good or bad, but deviation from optimal might be
                    # We'll use a neutral value (neither helps nor hurts score)
                    value = 0.5
                elif metric == 'assist_ratio':
                    # Higher assist ratio is generally better (up to a point)
                    value = max(0, min(1, float(metrics[metric]) / 25))
                elif metric == 'rebound_percentage':
                    # Normalize: ~45% is poor, ~55% is excellent
                    value = max(0, min(1, (float(metrics[metric]) - 45) / 10))
                else:
                    # Default normalization
                    value = float(metrics[metric])
                
                score += value * weight
        
        return score
        
    except Exception as e:
        logger.error(f"Error calculating team efficiency score: {str(e)}")
        return 0.0


def get_top_players_by_efficiency(stats: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Extract the top players by efficiency from team stats
    
    Args:
        stats: List of player stat dictionaries
        limit: Maximum number of players to return
        
    Returns:
        List of top players with their efficiency scores
    """
    try:
        # Get unique players
        players = {}
        for stat in stats:
            player = stat.get('player', {})
            player_id = player.get('id')
            
            if not player_id:
                continue
                
            # Initialize player or update
            if player_id not in players:
                players[player_id] = {
                    'player': player,
                    'metrics': {},
                    'games': 0
                }
            
            # Update metrics
            for key in METRIC_WEIGHTS.keys():
                if key in stat and stat[key] is not None:
                    if key not in players[player_id]['metrics']:
                        players[player_id]['metrics'][key] = [float(stat[key])]
                    else:
                        players[player_id]['metrics'][key].append(float(stat[key]))
            
            # Count game
            players[player_id]['games'] += 1
        
        # Calculate average metrics for each player
        player_scores = []
        for player_id, data in players.items():
            if data['games'] < 2:  # Require at least 2 games for meaningful data
                continue
                
            avg_metrics = {}
            for key, values in data['metrics'].items():
                if values:  # If we have values
                    avg_metrics[key] = sum(values) / len(values)
            
            # Calculate efficiency score
            efficiency = calculate_efficiency_score(avg_metrics)
            
            player_info = data['player']
            player_name = f"{player_info.get('first_name', '')} {player_info.get('last_name', '')}".strip()
            
            player_scores.append({
                'player_id': player_id,
                'player_name': player_name,
                'position': player_info.get('position', ''),
                'efficiency_score': efficiency,
                'games_played': data['games']
            })
        
        # Sort by efficiency score and limit
        top_players = sorted(player_scores, key=lambda x: x['efficiency_score'], reverse=True)[:limit]
        
        return top_players
        
    except Exception as e:
        logger.error(f"Error extracting top players: {str(e)}")
        return []


def get_player_efficiency_rating(player_id: int) -> Dict[str, Any]:
    """
    Get a player's efficiency rating and advanced metrics
    
    Args:
        player_id: Player ID to get efficiency rating for
        
    Returns:
        Dictionary with player efficiency rating and metrics
    """
    # Get advanced metrics for current season
    metrics = fetch_player_advanced_metrics(player_id)
    
    # Get metrics for last 10 games for recent form
    recent_metrics = fetch_player_advanced_metrics(player_id, last_n_games=10)
    
    # Combine into a single result
    result = {
        'player_id': player_id,
        'season_metrics': metrics,
        'recent_metrics': recent_metrics,
        'season_efficiency': metrics.get('efficiency_score', 0.0),
        'recent_efficiency': recent_metrics.get('efficiency_score', 0.0)
    }
    
    # Calculate if player is improving or declining
    if result['season_efficiency'] > 0 and result['recent_efficiency'] > 0:
        result['trend'] = result['recent_efficiency'] - result['season_efficiency']
    else:
        result['trend'] = 0.0
    
    return result


def get_matchup_efficiency_differential(home_player_id: int, away_player_id: int) -> Dict[str, Any]:
    """
    Compare efficiency ratings between two players (typically for matchup analysis)
    
    Args:
        home_player_id: Home player ID
        away_player_id: Away player ID
        
    Returns:
        Dictionary with comparative efficiency analysis
    """
    # Get ratings for both players
    home_rating = get_player_efficiency_rating(home_player_id)
    away_rating = get_player_efficiency_rating(away_player_id)
    
    # Calculate differential
    differential = home_rating['season_efficiency'] - away_rating['season_efficiency']
    recent_differential = home_rating['recent_efficiency'] - away_rating['recent_efficiency']
    
    return {
        'home_player_id': home_player_id,
        'away_player_id': away_player_id,
        'home_efficiency': home_rating['season_efficiency'],
        'away_efficiency': away_rating['season_efficiency'],
        'differential': differential,  # Positive = home advantage
        'recent_differential': recent_differential
    }


def get_team_efficiency_comparison(home_team_id: int, away_team_id: int) -> Dict[str, Any]:
    """
    Compare efficiency ratings between two teams
    
    Args:
        home_team_id: Home team ID
        away_team_id: Away team ID
        
    Returns:
        Dictionary with comparative team efficiency analysis
    """
    # Get team metrics
    home_metrics = fetch_team_advanced_metrics(home_team_id)
    away_metrics = fetch_team_advanced_metrics(away_team_id)
    
    # Calculate efficiency differentials
    offensive_diff = home_metrics.get('offensive_rating', 0) - away_metrics.get('offensive_rating', 0)
    defensive_diff = away_metrics.get('defensive_rating', 0) - home_metrics.get('defensive_rating', 0)  # Reversed for defense (lower is better)
    overall_diff = home_metrics.get('team_efficiency_score', 0) - away_metrics.get('team_efficiency_score', 0)
    
    return {
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'home_efficiency': home_metrics.get('team_efficiency_score', 0),
        'away_efficiency': away_metrics.get('team_efficiency_score', 0),
        'offensive_differential': offensive_diff,  # Positive = home advantage
        'defensive_differential': defensive_diff,  # Positive = home advantage
        'overall_differential': overall_diff,      # Positive = home advantage
        'home_top_players': home_metrics.get('top_players', []),
        'away_top_players': away_metrics.get('top_players', [])
    }
