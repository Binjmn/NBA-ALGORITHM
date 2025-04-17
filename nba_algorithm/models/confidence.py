# -*- coding: utf-8 -*-
"""
Confidence Scoring Module

This module provides methods for calculating confidence scores and levels
for predictions based on model consensus, data quality, and other factors.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union

# Update imports for the new module locations
from ..models.injury_analysis import compare_team_injuries
from ..models.advanced_metrics import get_team_efficiency_comparison

logger = logging.getLogger(__name__)


def calculate_confidence_level(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate confidence level for predictions
    
    Args:
        predictions_df: DataFrame containing prediction results
        
    Returns:
        DataFrame with added confidence level
    """
    if predictions_df.empty:
        return predictions_df
        
    logger.info("Calculating confidence levels for predictions")
    
    # Apply confidence adjustments for each prediction type
    if 'prediction_type' in predictions_df.columns:
        for pred_type, group in predictions_df.groupby('prediction_type'):
            if pred_type == 'moneyline':
                predictions_df.loc[group.index, 'confidence'] = _calculate_moneyline_confidence(group)
            elif pred_type == 'spread':
                predictions_df.loc[group.index, 'confidence'] = _calculate_spread_confidence(group)
            elif pred_type == 'total':
                predictions_df.loc[group.index, 'confidence'] = _calculate_total_confidence(group)
            elif pred_type.startswith('player_'):
                predictions_df.loc[group.index, 'confidence'] = _calculate_player_prop_confidence(group)
    else:
        # If prediction_type not available, apply generic confidence calculation
        predictions_df['confidence'] = _calculate_generic_confidence(predictions_df)
    
    # NEW: Apply injury adjustments to confidence
    try:
        predictions_df = _adjust_confidence_for_injuries(predictions_df)
    except Exception as e:
        logger.warning(f"Error adjusting confidence for injuries: {str(e)}")
    
    # NEW: Apply advanced metrics adjustments to confidence
    try:
        predictions_df = _adjust_confidence_for_advanced_metrics(predictions_df)
    except Exception as e:
        logger.warning(f"Error adjusting confidence for advanced metrics: {str(e)}")
    
    # Ensure confidence is within valid range
    predictions_df['confidence'] = predictions_df['confidence'].clip(0.3, 0.95)
    
    logger.info("Confidence levels calculated successfully")
    return predictions_df


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


def _calculate_moneyline_confidence(group):
    # TO DO: implement moneyline confidence calculation
    return np.nan


def _calculate_spread_confidence(group):
    # TO DO: implement spread confidence calculation
    return np.nan


def _calculate_total_confidence(group):
    # TO DO: implement total confidence calculation
    return np.nan


def _calculate_player_prop_confidence(group):
    # TO DO: implement player prop confidence calculation
    return np.nan


def _calculate_generic_confidence(predictions_df):
    # TO DO: implement generic confidence calculation
    return np.nan


def _adjust_confidence_for_injuries(predictions_df):
    """
    Adjust prediction confidence based on injury data
    
    Args:
        predictions_df: DataFrame of predictions
        
    Returns:
        DataFrame with adjusted confidence values
    """
    if predictions_df.empty:
        return predictions_df
    
    # Make a copy to avoid modifying the original
    df = predictions_df.copy()
    
    # Adjust moneyline and spread confidence based on injuries
    for i, row in df.iterrows():
        try:
            # Skip if needed columns aren't present
            if not all(c in row for c in ['home_team_id', 'away_team_id', 'confidence']):
                continue
                
            # Skip if confidence is missing
            if pd.isna(row['confidence']):
                continue
            
            # Get injury comparison data
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            
            # Get injury data for both teams
            injury_comparison = compare_team_injuries(home_id, away_id)
            
            # Skip if no injury data
            if not injury_comparison:
                continue
            
            # Get baseline confidence
            confidence = row['confidence']
            
            # Adjust confidence downward if any team has key player injuries
            # This represents increased uncertainty in the prediction
            # Specifically, injuries mean our model data (past games) may not reflect current team strength
            if injury_comparison['home_key_players_injured'] or injury_comparison['away_key_players_injured']:
                # Calculate adjustment factor based on injury impact
                home_impact = injury_comparison['home_impact']['overall_impact']
                away_impact = injury_comparison['away_impact']['overall_impact']
                
                # Maximum impact is 1.0, weight by 15% reduction in confidence
                adjustment = (home_impact + away_impact) * 0.15
                
                # Limit adjustment to reasonable range
                adjustment = min(adjustment, 0.3)
                
                # Apply adjustment (reduce confidence)
                new_confidence = confidence * (1 - adjustment)
                
                # Apply the adjusted confidence
                df.at[i, 'confidence'] = new_confidence
                
                # Add adjustment context for display
                df.at[i, 'confidence_injury_note'] = f"Adjusted for injuries: "
                if injury_comparison['home_key_players_injured']:
                    df.at[i, 'confidence_injury_note'] += f"Home ({row['home_team']}) has key player injuries. "
                if injury_comparison['away_key_players_injured']:
                    df.at[i, 'confidence_injury_note'] += f"Away ({row['away_team']}) has key player injuries. "
        except Exception as e:
            logger.warning(f"Error adjusting confidence for injuries in game {i}: {str(e)}")
    
    return df


def _adjust_confidence_for_advanced_metrics(predictions_df):
    """
    Adjust prediction confidence based on advanced metrics
    
    Args:
        predictions_df: DataFrame of predictions
        
    Returns:
        DataFrame with adjusted confidence values
    """
    if predictions_df.empty:
        return predictions_df
    
    # Make a copy to avoid modifying the original
    df = predictions_df.copy()
    
    # Adjust confidence based on advanced metrics
    for i, row in df.iterrows():
        try:
            # Skip if needed columns aren't present
            if not all(c in row for c in ['home_team_id', 'away_team_id', 'confidence']):
                continue
                
            # Skip if confidence is missing
            if pd.isna(row['confidence']):
                continue
            
            # Get efficiency comparison data
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            
            # Get efficiency comparison
            efficiency_comparison = get_team_efficiency_comparison(home_id, away_id)
            
            # Skip if no efficiency data
            if not efficiency_comparison:
                continue
            
            # Get baseline confidence
            confidence = row['confidence']
            
            # Calculate adjustment based on efficiency differential
            # If overall_differential is high (strong advantage for one team), increase confidence
            # If overall_differential is low (teams closely matched), slightly decrease confidence
            overall_diff = abs(efficiency_comparison['overall_differential'])
            
            # Scale adjustment factor: 0.1 increase for strong advantage, up to 0.05 decrease for close matchup
            if overall_diff > 0.2:  # Strong advantage for one team
                # Increase confidence by up to 10% for very strong advantage
                adjustment = min(overall_diff * 0.3, 0.1)
                new_confidence = confidence * (1 + adjustment)
                
                # Add adjustment context
                advantage_team = "home" if efficiency_comparison['overall_differential'] > 0 else "away"
                df.at[i, 'confidence_metrics_note'] = f"Increased confidence: Strong efficiency advantage for {advantage_team} team."
            elif overall_diff < 0.05:  # Very closely matched teams
                # Slightly decrease confidence (by up to 5%) for very close matchups
                adjustment = 0.05 * (1 - (overall_diff / 0.05))
                new_confidence = confidence * (1 - adjustment)
                
                # Add adjustment context
                df.at[i, 'confidence_metrics_note'] = "Slightly decreased confidence: Teams are very closely matched in efficiency metrics."
            else:
                # No significant adjustment needed
                new_confidence = confidence
                df.at[i, 'confidence_metrics_note'] = "No significant efficiency differential impact on confidence."
            
            # Apply the adjusted confidence
            df.at[i, 'confidence'] = new_confidence
        except Exception as e:
            logger.warning(f"Error adjusting confidence for advanced metrics in game {i}: {str(e)}")
    
    return df
