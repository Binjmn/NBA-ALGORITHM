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
    if predictions_df.empty:
        return predictions_df
    
    # Make a copy to avoid modifying the original
    df = predictions_df.copy()
    
    # Initialize confidence score
    df['confidence_score'] = 0.5  # Default to medium confidence
    
    for i, game in df.iterrows():
        confidence_factors = []
        
        # Factor 1: Win probability strength (how far from 50/50)
        if 'win_probability' in game:
            win_prob = game['win_probability']
            # Higher confidence when probability is further from 50%
            win_prob_confidence = 2 * abs(win_prob - 0.5)  # Scale from 0 to 1
            confidence_factors.append(win_prob_confidence)
        
        # Factor 2: Model consensus (if multiple models available)
        if 'model_agreement' in game:
            model_consensus = game['model_agreement']  # Assumed to be 0-1 scale
            confidence_factors.append(model_consensus)
        elif 'ensemble_confidence' in game:
            # If we have ensemble model confidence, use that
            ensemble_confidence = game['ensemble_confidence']
            confidence_factors.append(ensemble_confidence)
        
        # Factor 3: Market alignment (if odds available)
        if all(k in game for k in ['win_probability', 'home_odds', 'visitor_odds']):
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
        
        # Factor 4: Data quality/completeness
        data_quality = 0.75  # Default to reasonable data quality
        # If we have a data_quality metric, use that instead
        if 'data_quality' in game:
            data_quality = game['data_quality']
        confidence_factors.append(data_quality)
        
        # Factor 5: Historical matchup data availability
        if 'matchup_sample_size' in game:
            sample_size = game['matchup_sample_size']
            sample_confidence = min(sample_size / 10, 1.0)  # Cap at 1.0 (10+ games is full confidence)
            confidence_factors.append(sample_confidence)
        
        # Calculate overall confidence score - weighted average of factors
        # Win probability strength and model consensus get higher weights
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
        if 'spread_probability' in game:
            # Spread confidence relates to how far from 50% the probability is
            spread_prob = game['spread_probability']
            spread_confidence = 2 * abs(spread_prob - 0.5)  # Scale from 0 to 1
            # Blend with overall confidence
            df.at[i, 'spread_confidence'] = 0.7 * spread_confidence + 0.3 * confidence_score
            
        if 'over_probability' in game:
            # Total confidence relates to how far from 50% the probability is
            over_prob = game['over_probability']
            total_confidence = 2 * abs(over_prob - 0.5)  # Scale from 0 to 1
            # Blend with overall confidence
            df.at[i, 'total_confidence'] = 0.7 * total_confidence + 0.3 * confidence_score
    
    return df


def calculate_player_confidence(player_predictions_df):
    """
    Calculate confidence level for player predictions based on data quality and model performance
    
    Args:
        player_predictions_df: DataFrame of player predictions
        
    Returns:
        pandas.DataFrame: Player predictions with added confidence scores and levels
    """
    if player_predictions_df.empty:
        return player_predictions_df
    
    # Make a copy to avoid modifying the original
    df = player_predictions_df.copy()
    
    # Initialize confidence score
    df['confidence_score'] = 0.5  # Default to medium confidence
    
    for i, player in df.iterrows():
        confidence_factors = []
        
        # Factor 1: Player sample size/minutes played
        if 'minutes_played' in player:
            minutes = player['minutes_played']
            minutes_confidence = min(minutes / 30, 1.0)  # Cap at 1.0 (30+ mins is full confidence)
            confidence_factors.append(minutes_confidence)
        
        # Factor 2: Consistency of player performance
        if 'consistency_score' in player:
            consistency = player['consistency_score']
            confidence_factors.append(consistency)
        
        # Factor 3: Matchup favorability
        if 'matchup_rating' in player:
            matchup = player['matchup_rating']
            confidence_factors.append(matchup)
        
        # Factor 4: Injury impact
        if 'injury_impact' in player:
            injury_impact = 1 - player['injury_impact']  # Convert impact to confidence
            confidence_factors.append(injury_impact)
        
        # Factor 5: Overall data quality
        data_quality = 0.7  # Default to reasonable data quality
        if 'data_quality' in player:
            data_quality = player['data_quality']
        confidence_factors.append(data_quality)
        
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
    
    return df
