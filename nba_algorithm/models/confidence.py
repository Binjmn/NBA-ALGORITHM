# -*- coding: utf-8 -*-
"""
Confidence Scoring Module

This module provides methods for calculating confidence scores and levels
for predictions based on model consensus, data quality, and other factors.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)


def calculate_confidence_level(predictions_df):
    """
    Calculate confidence level for predictions based on model consensus and data quality
    
    This function evaluates the confidence level for each prediction based on:
    1. Model consensus - agreement between different models
    2. Market alignment - agreement between our models and betting markets
    3. Data quality metrics - completeness and recency of features
    
    Args:
        predictions_df: DataFrame of predictions
        
    Returns:
        pandas.DataFrame: Predictions with added confidence scores and levels
    """
    # Check if input DataFrame is valid
    if predictions_df is None or predictions_df.empty:
        logger.error("No predictions data provided. Unable to calculate confidence levels.")
        return predictions_df
    
    df = predictions_df.copy()
    
    # Initialize confidence score and tracking fields
    df['confidence_score'] = np.nan  # No default value initially
    df['confidence_factors_used'] = ''
    df['confidence_calculation_status'] = 'pending'
    
    for i, game in df.iterrows():
        confidence_factors = []
        factors_used = []
        
        # Factor 1: Win probability strength (how far from 50/50)
        if 'win_probability' in game and pd.notna(game['win_probability']):
            win_prob = game['win_probability']
            # Higher confidence when probability is further from 50%
            win_prob_confidence = 2 * abs(win_prob - 0.5)  # Scale from 0 to 1
            confidence_factors.append(win_prob_confidence)
            factors_used.append('win_probability')
        else:
            logger.warning(f"Win probability not available for game {game.get('game_id', 'unknown')}")
        
        # Factor 2: Model consensus (if multiple models available)
        if 'model_agreement' in game and pd.notna(game['model_agreement']):
            model_consensus = game['model_agreement']  # Assumed to be 0-1 scale
            confidence_factors.append(model_consensus)
            factors_used.append('model_agreement')
        elif 'ensemble_confidence' in game and pd.notna(game['ensemble_confidence']):
            # If we have ensemble model confidence, use that
            ensemble_confidence = game['ensemble_confidence']
            confidence_factors.append(ensemble_confidence)
            factors_used.append('ensemble_confidence')
        else:
            logger.warning(f"Model consensus not available for game {game.get('game_id', 'unknown')}")
        
        # Factor 3: Market alignment (if odds available)
        if all(k in game and pd.notna(game[k]) for k in ['win_probability', 'home_odds', 'visitor_odds']):
            win_prob = game['win_probability']
            home_odds = game['home_odds']
            visitor_odds = game['visitor_odds']
            
            # Calculate implied probabilities from odds
            if home_odds > 0:
                home_implied_prob = 100 / (home_odds + 100)
            else:
                home_implied_prob = abs(home_odds) / (abs(home_odds) + 100)
                
            if visitor_odds > 0:
                visitor_implied_prob = 100 / (visitor_odds + 100)
            else:
                visitor_implied_prob = abs(visitor_odds) / (abs(visitor_odds) + 100)
            
            # Normalize implied probabilities to account for vig
            total_implied = home_implied_prob + visitor_implied_prob
            home_implied_prob_normalized = home_implied_prob / total_implied
            
            # Calculate how closely our prediction aligns with the market
            # High agreement = high confidence
            market_alignment = 1 - abs(win_prob - home_implied_prob_normalized)
            confidence_factors.append(market_alignment)
            factors_used.append('market_alignment')
        else:
            logger.warning(f"Market alignment not available for game {game.get('game_id', 'unknown')}")
        
        # Factor 4: Data quality/completeness
        if 'data_quality' in game and pd.notna(game['data_quality']):
            data_quality = game['data_quality']
            confidence_factors.append(data_quality)
            factors_used.append('data_quality')
        else:
            logger.warning(f"Data quality not available for game {game.get('game_id', 'unknown')}")
        
        # Factor 5: Historical matchup data availability
        if 'matchup_sample_size' in game and pd.notna(game['matchup_sample_size']):
            sample_size = game['matchup_sample_size']
            sample_confidence = min(sample_size / 10, 1.0)  # Cap at 1.0 (10+ games is full confidence)
            confidence_factors.append(sample_confidence)
            factors_used.append('matchup_sample_size')
        else:
            logger.warning(f"Historical matchup data not available for game {game.get('game_id', 'unknown')}")
        
        # Calculate overall confidence score - weighted average of factors
        # Win probability strength and model consensus get higher weights
        if confidence_factors:
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            # Truncate weights to match available factors
            weights = weights[:len(confidence_factors)]
            # Normalize weights
            weights = [w/sum(weights) for w in weights]
            
            confidence_score = sum(f * w for f, w in zip(confidence_factors, weights))
            
            # Ensure score is in 0-1 range
            confidence_score = max(0.0, min(1.0, confidence_score))
            df.at[i, 'confidence_score'] = confidence_score
            
            # Map confidence score to descriptive level
            if confidence_score >= 0.8:
                confidence_level = "Very High"
            elif confidence_score >= 0.7:
                confidence_level = "High"
            elif confidence_score >= 0.5:
                confidence_level = "Medium"
            elif confidence_score >= 0.3:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"
                
            df.at[i, 'confidence_level'] = confidence_level
            
            # Add specific confidence scores for spread and total if available
            if 'spread_probability' in game and pd.notna(game['spread_probability']):
                # Spread confidence relates to how far from 50% the probability is
                spread_prob = game['spread_probability']
                spread_confidence = 2 * abs(spread_prob - 0.5)  # Scale from 0 to 1
                # Blend with overall confidence
                df.at[i, 'spread_confidence'] = 0.7 * spread_confidence + 0.3 * confidence_score
            
            if 'over_probability' in game and pd.notna(game['over_probability']):
                # Total confidence relates to how far from 50% the probability is
                over_prob = game['over_probability']
                total_confidence = 2 * abs(over_prob - 0.5)  # Scale from 0 to 1
                # Blend with overall confidence
                df.at[i, 'total_confidence'] = 0.7 * total_confidence + 0.3 * confidence_score
            
            # Update tracking fields
            df.at[i, 'confidence_factors_used'] = ', '.join(factors_used)
            df.at[i, 'confidence_calculation_status'] = 'success'
        else:
            logger.error(f"Unable to calculate confidence for game {game.get('game_id', 'unknown')}")
            df.at[i, 'confidence_calculation_status'] = 'failure'
    
    return df


def calculate_player_confidence(player_predictions_df):
    """
    Calculate confidence level for player predictions based on data quality and model performance
    
    Args:
        player_predictions_df: DataFrame of player predictions
        
    Returns:
        pandas.DataFrame: Player predictions with added confidence scores and levels
    """
    # Check if input DataFrame is valid
    if player_predictions_df is None or player_predictions_df.empty:
        logger.error("No player predictions data provided. Unable to calculate confidence levels.")
        return player_predictions_df
    
    df = player_predictions_df.copy()
    
    # Initialize confidence score and tracking fields
    df['confidence_score'] = np.nan  # No default value initially
    df['confidence_factors_used'] = ''
    df['confidence_calculation_status'] = 'pending'
    
    for i, player in df.iterrows():
        confidence_factors = []
        factors_used = []
        
        # Factor 1: Player sample size/minutes played
        if 'minutes_played' in player and pd.notna(player['minutes_played']):
            minutes = player['minutes_played']
            minutes_confidence = min(minutes / 30, 1.0)  # Cap at 1.0 (30+ mins is full confidence)
            confidence_factors.append(minutes_confidence)
            factors_used.append('minutes_played')
        else:
            logger.warning(f"Minutes played not available for player {player.get('player_id', 'unknown')}")
        
        # Factor 2: Consistency of player performance
        if 'consistency_score' in player and pd.notna(player['consistency_score']):
            consistency = player['consistency_score']
            confidence_factors.append(consistency)
            factors_used.append('consistency_score')
        else:
            logger.warning(f"Consistency score not available for player {player.get('player_id', 'unknown')}")
        
        # Factor 3: Matchup favorability
        if 'matchup_rating' in player and pd.notna(player['matchup_rating']):
            matchup = player['matchup_rating']
            confidence_factors.append(matchup)
            factors_used.append('matchup_rating')
        else:
            logger.warning(f"Matchup rating not available for player {player.get('player_id', 'unknown')}")
        
        # Factor 4: Injury impact
        if 'injury_impact' in player and pd.notna(player['injury_impact']):
            injury_impact = 1 - player['injury_impact']  # Convert impact to confidence
            confidence_factors.append(injury_impact)
            factors_used.append('injury_impact')
        else:
            logger.warning(f"Injury impact not available for player {player.get('player_id', 'unknown')}")
        
        # Factor 5: Overall data quality
        if 'data_quality' in player and pd.notna(player['data_quality']):
            data_quality = player['data_quality']
            confidence_factors.append(data_quality)
            factors_used.append('data_quality')
        else:
            logger.warning(f"Data quality not available for player {player.get('player_id', 'unknown')}")
        
        # Calculate overall confidence score - simple average if no weights specified
        if confidence_factors:
            confidence_score = sum(confidence_factors) / len(confidence_factors)
            
            # Ensure score is in 0-1 range
            confidence_score = max(0.0, min(1.0, confidence_score))
            df.at[i, 'confidence_score'] = confidence_score
            
            # Map confidence score to descriptive level
            if confidence_score >= 0.8:
                confidence_level = "Very High"
            elif confidence_score >= 0.7:
                confidence_level = "High"
            elif confidence_score >= 0.5:
                confidence_level = "Medium"
            elif confidence_score >= 0.3:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"
                
            df.at[i, 'confidence_level'] = confidence_level
            
            # Update tracking fields
            df.at[i, 'confidence_factors_used'] = ', '.join(factors_used)
            df.at[i, 'confidence_calculation_status'] = 'success'
        else:
            logger.error(f"Unable to calculate confidence for player {player.get('player_id', 'unknown')}")
            df.at[i, 'confidence_calculation_status'] = 'failure'
    
    return df
