#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Math Utilities Module

This module provides mathematical helper functions for the NBA prediction system.

Author: Cascade
Date: April 2025
"""

import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional
import traceback

# Configure logger
logger = logging.getLogger(__name__)


def spread_to_win_probability(spread: float) -> float:
    """
    Convert point spread to win probability using statistical models
    
    Args:
        spread: Predicted point spread (positive favors home team)
        
    Returns:
        Win probability for home team
    """
    try:
        # Based on historical NBA data, approximately each point is worth ~3.5% win probability
        # centered around 50% for a spread of 0
        # This is a simplified model - more complex models would use logistic regression
        win_prob = 0.5 + (spread * 0.035)
        
        # Ensure probability is between 0.01 and 0.99
        return min(max(win_prob, 0.01), 0.99)
    except Exception as e:
        logger.error(f"Error converting spread to win probability: {str(e)}")
        return 0.5  # Return 50% probability as fallback


def calculate_implied_total(is_home: bool, spread: float, total: float) -> float:
    """
    Calculate implied team total based on Vegas line and over/under
    
    Args:
        is_home: Whether team is home team
        spread: Predicted point spread (positive favors home team)
        total: Over/under total
        
    Returns:
        Implied team total
    """
    try:
        if is_home:
            return (total + spread) / 2
        else:
            return (total - spread) / 2
    except Exception as e:
        logger.error(f"Error calculating implied total: {str(e)}")
        return total / 2  # Split the total evenly as fallback


def calculate_prediction_confidence(model_predictions: Dict[str, Any]) -> float:
    """
    Calculate confidence score based on model agreement and other factors
    
    Args:
        model_predictions: Dictionary of model predictions
        
    Returns:
        Confidence score between 0-1
    """
    try:
        base_confidence = 0.7  # Default moderate confidence
        
        # If we don't have both key predictions, return base confidence
        if 'spread' not in model_predictions and 'win_prob' not in model_predictions:
            return base_confidence
        
        # Calculate confidence based on spread magnitude
        spread_confidence = 0.0
        if 'spread' in model_predictions:
            spread = abs(model_predictions['spread'])
            
            # Larger spread = higher confidence (up to a limit)
            if spread > 12.0:
                spread_confidence = 0.9  # Very high confidence for big spreads
            elif spread > 8.0:
                spread_confidence = 0.8  # High confidence
            elif spread > 4.0:
                spread_confidence = 0.7  # Moderate to high confidence
            elif spread > 2.0:
                spread_confidence = 0.6  # Moderate confidence
            else:
                spread_confidence = 0.5  # Lower confidence for close games
        
        # Calculate confidence based on win probability extremity
        prob_confidence = 0.0
        if 'win_prob' in model_predictions:
            win_prob = model_predictions['win_prob']
            win_prob_distance = abs(win_prob - 0.5)  # Distance from 50/50
            
            # More extreme probability = higher confidence
            prob_confidence = 0.5 + win_prob_distance  # Maps 0.5-1.0 to 0.5-1.0
        
        # Average the confidence metrics, weighing those that are available
        confidences = []
        if spread_confidence > 0:
            confidences.append(spread_confidence)
        if prob_confidence > 0:
            confidences.append(prob_confidence)
        
        if confidences:
            return sum(confidences) / len(confidences)
        else:
            return base_confidence
    
    except Exception as e:
        logger.error(f"Error calculating prediction confidence: {str(e)}")
        return 0.7  # Return moderate confidence as fallback


def calculate_prop_confidence(player: Dict[str, Any], prop_type: str, prediction: float) -> float:
    """
    Calculate confidence level for prop prediction based on player data and context
    
    Args:
        player: Player data including season stats
        prop_type: Prop type (points, assists, rebounds)
        prediction: Predicted value
        
    Returns:
        Confidence score (0-1.0)
    """
    try:
        base_confidence = 0.65  # Default moderate confidence
        
        # Get season stats
        season_stats = player.get('season_stats', {})
        if not season_stats:
            return base_confidence * 0.8  # Reduce confidence if no season stats
        
        # Get games played
        games_played = season_stats.get('games_played', 0)
        if games_played < 10:
            return base_confidence * 0.8  # Reduce confidence for small sample sizes
        
        # Get minutes per game
        minutes = float(season_stats.get('min', 0))
        if minutes < 15:
            return base_confidence * 0.9  # Slightly reduce confidence for bench players
        
        # Get season average for prop type
        season_avg = 0.0
        if prop_type == 'points':
            season_avg = float(season_stats.get('pts', 0))
        elif prop_type == 'rebounds':
            season_avg = float(season_stats.get('reb', 0))
        elif prop_type == 'assists':
            season_avg = float(season_stats.get('ast', 0))
        
        # Calculate confidence boost based on consistent production
        prop_boost = 0.0
        if season_avg > 0:
            # Calculate deviation from average
            deviation_pct = abs(prediction - season_avg) / season_avg
            
            # Less deviation = more confidence
            if deviation_pct < 0.1:
                prop_boost = 0.15  # Very consistent with season average
            elif deviation_pct < 0.2:
                prop_boost = 0.1  # Fairly consistent
            elif deviation_pct < 0.3:
                prop_boost = 0.05  # Somewhat consistent
            elif deviation_pct > 0.5:
                prop_boost = -0.1  # Very inconsistent, reduce confidence
        
        # Final confidence calculation
        confidence = base_confidence + prop_boost
        
        # Ensure confidence is between 0.3 and 0.95
        return min(max(confidence, 0.3), 0.95)
    
    except Exception as e:
        logger.error(f"Error calculating prop confidence: {str(e)}")
        return 0.65  # Return moderate confidence as fallback


def calculate_matchup_difficulty(player: Dict, opponent: Dict) -> float:
    """
    Calculate matchup difficulty based on opponent's defensive metrics
    
    Args:
        player: Player data
        opponent: Opponent team data
        
    Returns:
        Difficulty rating (0-10 scale, 10 being most difficult)
    """
    try:
        # Initialize with neutral difficulty
        difficulty_score = 5.0
        
        # Get player position for position-specific defensive metrics
        player_position = player.get('position', '').upper()
        
        # Get defensive metrics from opponent
        def_rating = opponent.get('defensive_rating')
        if def_rating is None:
            logger.warning(f"No defensive rating available for opponent {opponent.get('name', 'Unknown')}")
            # Try to get alternative defensive metrics
            opp_points_allowed = opponent.get('opp_points_pg')
            if opp_points_allowed:
                # Lower points allowed = better defense = higher difficulty
                if opp_points_allowed < 105:
                    difficulty_score += 2.0  # Excellent defense
                elif opp_points_allowed < 110:
                    difficulty_score += 1.0  # Good defense
                elif opp_points_allowed > 118:
                    difficulty_score -= 2.0  # Poor defense
                elif opp_points_allowed > 114:
                    difficulty_score -= 1.0  # Below average defense
        else:
            # Better defense (lower rating) means higher difficulty
            if def_rating < 105:
                difficulty_score += 2.5  # Excellent defense
            elif def_rating < 110:
                difficulty_score += 1.5  # Good defense
            elif def_rating > 115:
                difficulty_score -= 2.0  # Poor defense
            elif def_rating > 110:
                difficulty_score -= 1.0  # Below average defense
        
        # Adjust for position-specific matchups if available
        if player_position:
            # Guard defenders
            if player_position in ['PG', 'G', 'SG']:
                guard_def = opponent.get('guard_defense_rating')
                if guard_def:
                    difficulty_score += (110 - guard_def) / 10
            # Forward defenders
            elif player_position in ['SF', 'F', 'PF']:
                forward_def = opponent.get('forward_defense_rating')
                if forward_def:
                    difficulty_score += (110 - forward_def) / 10
            # Center defenders
            elif player_position in ['C']:
                center_def = opponent.get('center_defense_rating')
                if center_def:
                    difficulty_score += (110 - center_def) / 10
        
        # Add context: team's overall defensive effectiveness
        steals_pg = opponent.get('steals_pg', 0)
        blocks_pg = opponent.get('blocks_pg', 0)
        
        # Teams with more steals/blocks present greater difficulty
        if steals_pg > 8.5:
            difficulty_score += 0.5
        if blocks_pg > 6.0:
            difficulty_score += 0.5
        
        # Ensure the final score is within the 0-10 range
        return max(0.0, min(10.0, difficulty_score))
    
    except Exception as e:
        logger.error(f"Error calculating matchup difficulty: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't use a default - be explicit about the failure
        raise ValueError(f"Failed to calculate matchup difficulty: {str(e)}")
