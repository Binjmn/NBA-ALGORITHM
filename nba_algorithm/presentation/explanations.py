#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Explanations Module

This module generates natural language explanations for NBA game and player prop predictions.

Author: Cascade
Date: April 2025
"""

import logging
import random
from typing import Dict, List, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)


def generate_prediction_explanation(game_data: Dict, win_prob: float, point_spread: float) -> str:
    """
    Generate a natural language explanation of prediction reasoning
    
    This function creates a detailed explanation of why a particular prediction was made,
    referencing team statistics, matchup factors, and other relevant information.
    
    Args:
        game_data: Row of data for the game
        win_prob: Win probability
        point_spread: Predicted point spread
        
    Returns:
        str: Natural language explanation
    """
    try:
        home_team = game_data.get('home_team_name', 'Home Team')
        away_team = game_data.get('away_team_name', 'Away Team')
        
        # Determine favorite and confidence level
        if point_spread > 0:
            favorite = home_team
            underdog = away_team
            location_advantage = "at home"
        else:
            favorite = away_team
            underdog = home_team
            location_advantage = "on the road"
        
        spread_abs = abs(point_spread)
        win_pct = win_prob * 100 if point_spread > 0 else (1 - win_prob) * 100
        
        # Determine confidence level based on win probability
        if win_pct > 70:
            confidence = "strong"
        elif win_pct > 60:
            confidence = "moderate"
        else:
            confidence = "slight"
        
        # Extract relevant statistics
        off_rtg_diff = game_data.get('off_rtg_diff', 0)
        def_rtg_diff = game_data.get('def_rtg_diff', 0)
        net_rtg_diff = game_data.get('net_rtg_diff', 0)
        rest_advantage = game_data.get('rest_advantage', 0)
        
        # Build explanation parts
        explanation_parts = []
        
        # Lead with main prediction
        explanation_parts.append(
            f"Our model gives the {favorite} a {confidence} edge {location_advantage} against the {underdog}, "
            f"projecting a {spread_abs:.1f}-point victory with {win_pct:.1f}% win probability."
        )
        
        # Add contextual factors based on available stats
        factors = []
        
        # Offensive efficiency
        if abs(off_rtg_diff) > 3:
            direction = "superior" if (point_spread > 0 and off_rtg_diff > 0) or (point_spread < 0 and off_rtg_diff < 0) else "inferior"
            factors.append(f"The {favorite}'s {direction} offensive efficiency is a key factor")
        
        # Defensive efficiency
        if abs(def_rtg_diff) > 3:
            direction = "stronger" if (point_spread > 0 and def_rtg_diff > 0) or (point_spread < 0 and def_rtg_diff < 0) else "weaker"
            factors.append(f"The {favorite}'s {direction} defensive rating gives them an edge")
        
        # Net rating
        if abs(net_rtg_diff) > 5:
            factors.append(f"Overall team quality heavily favors the {favorite}")
        
        # Rest advantage
        if abs(rest_advantage) >= 1:
            if rest_advantage > 0 and point_spread > 0:
                factors.append(f"The {home_team} has a rest advantage of {rest_advantage} days")
            elif rest_advantage < 0 and point_spread < 0:
                factors.append(f"The {away_team} has a rest advantage of {abs(rest_advantage)} days")
        
        # Home court
        if point_spread > 0:
            factors.append(f"Home court advantage adds approximately 2-3 points to the {home_team}'s projection")
        
        # Add factors to explanation if available
        if factors:
            explanation_parts.append("Key factors: " + ". ".join(factors) + ".")
        
        # Combine all parts
        explanation = " ".join(explanation_parts)
        
        return explanation
    
    except Exception as e:
        logger.error(f"Error generating prediction explanation: {str(e)}")
        return "Prediction based on team performance metrics and historical matchup data."


def get_prop_recommendation(prediction: float, season_avg: float, prop_type: str) -> Dict[str, Any]:
    """
    Generate over/under recommendation based on prediction vs season average and other factors
    
    Args:
        prediction: Predicted prop value
        season_avg: Season average for this prop
        prop_type: Type of prop
        
    Returns:
        Dictionary with recommendation details
    """
    try:
        # Calculate percentage difference from season average
        if season_avg > 0:
            pct_diff = (prediction - season_avg) / season_avg
        else:
            pct_diff = 0.0
        
        # Determine recommendation strength based on difference
        strength = 'strong' if abs(pct_diff) > 0.15 else 'moderate' if abs(pct_diff) > 0.05 else 'weak'
        
        # Determine recommendation direction (over/under)
        direction = 'over' if prediction > season_avg else 'under'
        
        # Create explanation based on prop type and direction
        prop_name = {'points': 'scoring', 'rebounds': 'rebounding', 'assists': 'assist'}[prop_type]
        
        if direction == 'over':
            explanation = f"Projected {prop_name} is {abs(pct_diff):.1%} higher than season average"
            if strength == 'strong':
                explanation += f". Player's {prop_name} is expected to significantly outperform typical production."
            elif strength == 'moderate':
                explanation += f". Favorable matchup conditions for {prop_name} performance."
        else:  # under
            explanation = f"Projected {prop_name} is {abs(pct_diff):.1%} lower than season average"
            if strength == 'strong':
                explanation += f". Player's {prop_name} is expected to be notably limited in this matchup."
            elif strength == 'moderate':
                explanation += f". Matchup presents challenges for {prop_name} production."
        
        return {
            'direction': direction,
            'strength': strength,
            'explanation': explanation,
            'pct_diff_from_average': pct_diff
        }
    
    except Exception as e:
        logger.error(f"Error generating prop recommendation: {str(e)}")
        return {
            'direction': 'neutral',
            'strength': 'weak',
            'explanation': 'Insufficient data to generate recommendation',
            'pct_diff_from_average': 0.0
        }


def generate_matchup_insight(home_team: Dict[str, Any], away_team: Dict[str, Any]) -> str:
    """
    Generate natural language insights about the matchup dynamics
    
    Args:
        home_team: Home team statistics
        away_team: Away team statistics
        
    Returns:
        Matchup insight text
    """
    try:
        insights = []
        
        # Pace mismatch
        home_pace = home_team.get('pace', 100)
        away_pace = away_team.get('pace', 100)
        pace_diff = home_pace - away_pace
        
        if abs(pace_diff) > 5:
            faster = home_team.get('name', 'Home Team') if home_pace > away_pace else away_team.get('name', 'Away Team')
            slower = away_team.get('name', 'Away Team') if home_pace > away_pace else home_team.get('name', 'Home Team')
            insights.append(f"The {faster} prefer to play at a much faster pace than the {slower}, creating a stylistic mismatch.")
        
        # Offensive vs defensive strength
        home_off = home_team.get('offensive_rating', 110)
        away_def = away_team.get('defensive_rating', 110)
        home_off_v_away_def = home_off - away_def
        
        away_off = away_team.get('offensive_rating', 110)
        home_def = home_team.get('defensive_rating', 110)
        away_off_v_home_def = away_off - home_def
        
        if abs(home_off_v_away_def) > 5:
            advantage = home_team.get('name', 'Home Team') if home_off_v_away_def > 0 else away_team.get('name', 'Away Team')
            insights.append(f"The {advantage}'s offense has a significant advantage in this matchup.")
        
        if abs(away_off_v_home_def) > 5:
            advantage = away_team.get('name', 'Away Team') if away_off_v_home_def > 0 else home_team.get('name', 'Home Team')
            insights.append(f"The {advantage}'s offense has a significant advantage in this matchup.")
        
        # Rest advantage
        home_rest = home_team.get('rest_days', 2)
        away_rest = away_team.get('rest_days', 2)
        rest_diff = home_rest - away_rest
        
        if abs(rest_diff) >= 2:
            rested = home_team.get('name', 'Home Team') if rest_diff > 0 else away_team.get('name', 'Away Team')
            tired = away_team.get('name', 'Away Team') if rest_diff > 0 else home_team.get('name', 'Home Team')
            insights.append(f"The {rested} have a significant rest advantage over the {tired}.")
        
        if not insights:
            # Generate a generic insight if no specific insights were found
            home_name = home_team.get('name', 'Home Team')
            away_name = away_team.get('name', 'Away Team')
            insights.append(f"This matchup between the {home_name} and {away_name} is projected to be closely contested.")
        
        return " ".join(insights)
    
    except Exception as e:
        logger.error(f"Error generating matchup insight: {str(e)}")
        return "Matchup analysis based on team performance metrics and historical data."
