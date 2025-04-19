#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Injury Analysis Module

This module handles the collection and analysis of NBA player injury data.
It provides functions to fetch current injury information, calculate injury impact
scores, and integrate this data into the prediction pipeline.

Author: Cascade
Date: April 2025
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from ..api.balldontlie_client import BallDontLieClient
from ..utils.cache_manager import CacheManager
from ..data.player_data import get_player_stats

# Configure logger
logger = logging.getLogger(__name__)

# Initialize cache manager for injury data
cache = CacheManager(cache_name="injury_data", ttl_seconds=3600)  # Cache for 1 hour

# Injury status severity mapping (higher = more severe)
INJURY_SEVERITY = {
    "Out": 1.0,           # Player definitely out
    "Doubtful": 0.95,    # Very unlikely to play
    "Questionable": 0.7, # 50/50 chance of playing
    "Probable": 0.3,     # Likely to play but may be limited
    "Day-To-Day": 0.2,   # Minor injury, likely to play
    "Game Time Decision": 0.6,  # Unknown until game time
}

# Default importance value for players with no stats
DEFAULT_PLAYER_IMPORTANCE = 0.2


def fetch_team_injuries(team_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch current injury data for a specific team or all teams
    
    Args:
        team_id: Optional team ID to filter injuries
        
    Returns:
        List of injury data dictionaries
    """
    cache_key = f"team_injuries_{team_id if team_id else 'all'}"
    
    # Check cache first
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        logger.info(f"Using cached injury data for team_id={team_id}")
        return cached_data
    
    try:
        client = BallDontLieClient()
        
        # Fetch injury data
        params = {}
        if team_id:
            params['team_ids'] = [team_id]
            
        response = client.get_player_injuries()
        
        if not response or 'data' not in response:
            logger.warning(f"No injury data returned for team_id={team_id}")
            return []
            
        injuries = response['data']
        
        # Filter by team if needed
        if team_id:
            injuries = [injury for injury in injuries 
                       if injury.get('player', {}).get('team_id') == team_id]
        
        # Cache the results
        cache.set(cache_key, injuries)
        
        logger.info(f"Fetched {len(injuries)} injuries for team_id={team_id}")
        return injuries
        
    except Exception as e:
        logger.error(f"Error fetching injury data: {str(e)}")
        return []


def calculate_player_importance(player_id: int, player_stats: Optional[Dict] = None) -> Optional[float]:
    """
    Calculate player importance score based on stats and playing time
    
    Args:
        player_id: Player ID to calculate importance for
        player_stats: Optional player stats dictionary (to avoid refetching)
        
    Returns:
        Player importance score between 0 and 1, or None if insufficient data
    """
    try:
        # If stats not provided, fetch them
        if not player_stats:
            player_stats = get_player_stats(player_id)
            
        if not player_stats:
            logger.warning(f"No stats available for player {player_id}, unable to calculate importance")
            return None  # Return None instead of using default values
            
        # Calculate importance based on multiple factors:
        # 1. Minutes played (more minutes = more important)
        # 2. Points scored (more points = more important)
        # 3. Plus/minus (higher +/- = more important)
        # 4. Usage percentage (higher usage = more important)
        
        minutes = float(player_stats.get('min', 0))
        points = float(player_stats.get('pts', 0))
        plus_minus = float(player_stats.get('plus_minus', 0))
        usage = float(player_stats.get('usage_percentage', 0))
        
        # Check if we have enough meaningful data
        if minutes == 0 and points == 0:
            logger.warning(f"Player {player_id} has no minutes or points data, unable to calculate importance")
            return None  # Return None instead of using default values
        
        # Normalize minutes (typical starter plays 30+ minutes)
        minutes_factor = min(minutes / 36.0, 1.0)  # Cap at 1.0
        
        # Normalize points (20+ points is star level)
        points_factor = min(points / 25.0, 1.0)  # Cap at 1.0
        
        # Normalize plus/minus (scale from -15 to +15)
        plus_minus_norm = (plus_minus + 15) / 30.0
        plus_minus_factor = max(min(plus_minus_norm, 1.0), 0.0)  # Between 0 and 1
        
        # Normalize usage (30%+ is high usage)
        usage_factor = min(usage / 35.0, 1.0)  # Cap at 1.0
        
        # Weighted importance score
        importance = (
            minutes_factor * 0.35 + 
            points_factor * 0.30 + 
            plus_minus_factor * 0.15 +
            usage_factor * 0.20
        )
        
        return importance
        
    except Exception as e:
        logger.error(f"Error calculating player importance: {str(e)}")
        return None  # Return None instead of using default values


def get_injury_impact_score(injuries: List[Dict[str, Any]], team_id: int) -> Dict[str, Any]:
    """
    Calculate the overall injury impact score for a team
    
    Args:
        injuries: List of player injury dictionaries
        team_id: Team ID to calculate impact for
        
    Returns:
        Dictionary with injury impact metrics
    """
    if not injuries:
        return {
            'team_id': team_id,
            'overall_impact': 0.0,
            'num_injuries': 0,
            'key_player_injuries': False,
            'detail': []
        }
        
    try:
        # Filter injuries for this team
        team_injuries = [injury for injury in injuries 
                         if injury.get('player', {}).get('team_id') == team_id]
        
        if not team_injuries:
            return {
                'team_id': team_id,
                'overall_impact': 0.0,
                'num_injuries': 0,
                'key_player_injuries': False,
                'detail': []
            }
        
        total_impact = 0.0
        injury_details = []
        key_player_injured = False
        
        for injury in team_injuries:
            player = injury.get('player', {})
            player_id = player.get('id')
            player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
            status = injury.get('status', 'Unknown')
            position = player.get('position', '')
            
            # Skip if invalid data
            if not player_id or not player_name:
                continue
            
            # Get severity score for this status
            severity = INJURY_SEVERITY.get(status, 0.5)  # Default 0.5 if unknown status
            
            # Calculate player importance
            importance = calculate_player_importance(player_id)
            
            # Check if importance is None
            if importance is None:
                logger.warning(f"Unable to calculate importance for player {player_id}, skipping")
                continue
            
            # Calculate individual impact
            impact = importance * severity
            
            # Check if this is a key player (importance > 0.6)
            is_key_player = importance > 0.6
            if is_key_player and severity > 0.6:  # Key player with serious injury
                key_player_injured = True
            
            injury_details.append({
                'player_id': player_id,
                'player_name': player_name,
                'position': position,
                'status': status,
                'severity': severity,
                'importance': importance,
                'impact': impact,
                'is_key_player': is_key_player
            })
            
            # Add to total impact
            total_impact += impact
        
        # Cap overall impact at 1.0
        overall_impact = min(total_impact, 1.0)
        
        return {
            'team_id': team_id,
            'overall_impact': overall_impact,
            'num_injuries': len(team_injuries),
            'key_player_injuries': key_player_injured,
            'detail': injury_details
        }
        
    except Exception as e:
        logger.error(f"Error calculating injury impact: {str(e)}")
        return {
            'team_id': team_id,
            'overall_impact': 0.0,
            'num_injuries': 0,
            'key_player_injuries': False,
            'detail': []
        }


def get_team_injury_data(team_id: int) -> Dict[str, Any]:
    """
    Get comprehensive injury data for a team
    
    Args:
        team_id: Team ID to get injury data for
        
    Returns:
        Dictionary with team injury metrics and details
    """
    # Fetch all injuries
    injuries = fetch_team_injuries()
    
    # Calculate impact score
    impact_data = get_injury_impact_score(injuries, team_id)
    
    return impact_data


def compare_team_injuries(home_team_id: int, away_team_id: int) -> Dict[str, Any]:
    """
    Compare injury impact between two teams
    
    Args:
        home_team_id: Home team ID
        away_team_id: Away team ID
        
    Returns:
        Dictionary with comparative injury analysis
    """
    # Fetch all injuries (to avoid multiple API calls)
    all_injuries = fetch_team_injuries()
    
    # Calculate impact for both teams
    home_impact = get_injury_impact_score(all_injuries, home_team_id)
    away_impact = get_injury_impact_score(all_injuries, away_team_id)
    
    # Calculate relative advantage
    injury_advantage = away_impact['overall_impact'] - home_impact['overall_impact']
    
    return {
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'home_impact': home_impact,
        'away_impact': away_impact,
        'injury_advantage': injury_advantage,  # Positive = home advantage, negative = away advantage
        'home_key_players_injured': home_impact['key_player_injuries'],
        'away_key_players_injured': away_impact['key_player_injuries']
    }
