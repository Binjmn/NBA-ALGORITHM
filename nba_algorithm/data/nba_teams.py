#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Teams Module

This module provides tools for identifying and filtering current NBA teams
from historical or non-NBA teams that might appear in the API data.
"""

import logging
from typing import Dict, List, Any, Set
from datetime import datetime

logger = logging.getLogger(__name__)


def get_active_nba_teams(teams_data: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    Filter active NBA teams from a list of teams returned by the API
    
    This function identifies current NBA teams by checking for official NBA team IDs
    and validating their current status using organizational data when available.
    
    Args:
        teams_data: List of team dictionaries from the API
        
    Returns:
        Dictionary mapping team IDs to team names for active NBA teams only
    """
    active_teams = {}
    
    # Get the current year for season identification
    current_year = datetime.now().year
    
    # First pass: Identify teams with NBA identifiers
    for team in teams_data:
        # Skip teams missing critical data
        if not team.get('id') or not team.get('full_name'):
            continue
        
        team_id = team.get('id')
        team_name = team.get('full_name')
        
        # Simple check: known current NBA franchises have IDs 1-30
        # This covers the standard 30 NBA teams in most cases
        if 1 <= team_id <= 30:
            active_teams[team_id] = team_name
    
    # If we didn't get a reasonable number of teams, try conference/division filtering as backup
    if len(active_teams) < 25:
        logger.warning("Using backup filtering method based on conference/division membership")
        active_teams.clear()
        
        for team in teams_data:
            if not team.get('id') or not team.get('full_name'):
                continue
                
            team_id = team.get('id')
            team_name = team.get('full_name')
            
            # Check abbreviation existence (all active NBA teams have abbr)
            abbreviation = team.get('abbreviation')
            
            # Check division value if available
            division = team.get('division')
            division_valid = division in ['Atlantic', 'Central', 'Southeast', 'Northwest', 'Pacific', 'Southwest']
            
            # Check conference value if available
            conference = team.get('conference')
            conference_valid = conference in ['East', 'West', 'Eastern', 'Western']
            
            # Check for city/name structure (all NBA teams have this)
            city = team.get('city')
            name = team.get('name')
            
            # Team is likely an active NBA team if it has most of these properties
            # We're being more lenient here as a fallback
            has_enough_properties = sum([
                1 if abbreviation else 0,
                1 if division_valid else 0,
                1 if conference_valid else 0,
                1 if city and name else 0
            ]) >= 2  # At least 2 valid properties
            
            if has_enough_properties:
                active_teams[team_id] = team_name
    
    # Additional validation for teams with season data (extend to 4 years window)
    filtered_teams = {}
    for team in teams_data:
        team_id = team.get('id')
        
        # Only check teams identified as active in previous step
        if team_id in active_teams:
            team_name = active_teams[team_id]
            
            # If season data exists, check recency (within 4 years)
            if 'seasons' in team and team.get('seasons'):
                recent_activity = False
                for season in team.get('seasons', []):
                    season_year = season.get('year')
                    if season_year and (current_year - season_year) <= 4:  # Extended to 4 years
                        recent_activity = True
                        break
                        
                if recent_activity:
                    filtered_teams[team_id] = team_name
            else:
                # If no season data, keep the team (benefit of doubt)
                filtered_teams[team_id] = team_name
    
    # If we found season data, use the filtered list. Otherwise use the broader list
    final_teams = filtered_teams if filtered_teams else active_teams
    
    logger.info(f"Identified {len(final_teams)} active NBA teams")
    
    # Validate that we have a reasonable number of teams
    if len(final_teams) < 20 or len(final_teams) > 32:
        logger.warning(
            f"Unusual number of active teams identified: {len(final_teams)}. "
            f"Expected 30. This may indicate API data issues or changes."
        )
    
    return final_teams


def filter_games_to_active_teams(games: List[Dict[str, Any]], active_teams: Dict[int, str]) -> List[Dict[str, Any]]:
    """
    Filter a list of games to only include those between active NBA teams
    
    Args:
        games: List of game dictionaries from the API
        active_teams: Dictionary of active NBA teams
        
    Returns:
        Filtered list of games between active NBA teams
    """
    # If no active teams identified, default to the standard 30 team IDs (1-30)
    if not active_teams:
        logger.warning("No active teams provided. Falling back to default NBA team IDs (1-30)")
        active_teams = {id: f"Team {id}" for id in range(1, 31)}
    
    filtered_games = []
    
    for game in games:
        home_team = game.get('home_team', {})
        visitor_team = game.get('visitor_team', {})
        
        home_id = home_team.get('id', 0)
        visitor_id = visitor_team.get('id', 0)
        
        # Only keep games where both teams are active NBA teams
        if home_id in active_teams and visitor_id in active_teams:
            filtered_games.append(game)
    
    logger.info(f"Filtered {len(games)} games to {len(filtered_games)} games between active NBA teams")
    return filtered_games
