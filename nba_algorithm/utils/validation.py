#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Validation Module

This module provides functions to validate the completeness and quality of data
before using it for model training, eliminating the need for synthetic data.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional, Set

logger = logging.getLogger(__name__)


def validate_data_completeness(data: Dict[str, Any], team_stats: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that the dataset is complete and contains all required fields
    
    Args:
        data: Dictionary containing collected data
        team_stats: Dictionary of team statistics
        
    Returns:
        Tuple containing:
        - Boolean indicating if data is complete
        - Dictionary of missing data elements
    """
    missing_data = {}
    is_complete = True
    
    # Check games data
    games = data.get('games', [])
    if not games:
        is_complete = False
        missing_data['games'] = "No games data available"
    else:
        # Check for missing game data
        incomplete_games = []
        for game in games:
            if not _validate_game_completeness(game):
                incomplete_games.append(game.get('id'))
        
        if incomplete_games:
            is_complete = False
            missing_data['games'] = incomplete_games
    
    # Check team stats
    if not team_stats:
        is_complete = False
        missing_data['team_stats'] = "No team statistics available"
    else:
        # Check for missing team stats
        missing_team_stats = []
        team_ids = _extract_team_ids(games)
        
        for team_id in team_ids:
            if team_id not in team_stats or not team_stats[team_id].get('stats'):
                missing_team_stats.append(team_id)
        
        if missing_team_stats:
            is_complete = False
            missing_data['team_stats'] = missing_team_stats
    
    # Log validation results
    if is_complete:
        logger.info("Data validation passed: All required data is available")
    else:
        logger.warning(f"Data validation failed: Missing {len(missing_data)} data categories")
    
    return is_complete, missing_data


def _validate_game_completeness(game: Dict[str, Any]) -> bool:
    """
    Validate that a game contains all required fields
    
    Args:
        game: Game dictionary
        
    Returns:
        Boolean indicating if game data is complete
    """
    required_fields = ['id', 'date', 'home_team', 'visitor_team', 'home_team_score', 'visitor_team_score']
    
    for field in required_fields:
        if field not in game or game[field] is None:
            return False
    
    if 'home_team' in game and not _validate_team_completeness(game['home_team']):
        return False
    
    if 'visitor_team' in game and not _validate_team_completeness(game['visitor_team']):
        return False
    
    return True


def _validate_team_completeness(team: Dict[str, Any]) -> bool:
    """
    Validate that a team contains all required fields
    
    Args:
        team: Team dictionary
        
    Returns:
        Boolean indicating if team data is complete
    """
    required_fields = ['id', 'full_name', 'abbreviation']
    
    for field in required_fields:
        if field not in team or team[field] is None:
            return False
    
    return True


def _extract_team_ids(games: List[Dict[str, Any]]) -> Set[int]:
    """
    Extract all team IDs from a list of games
    
    Args:
        games: List of game dictionaries
        
    Returns:
        Set of team IDs
    """
    team_ids = set()
    
    for game in games:
        home_team = game.get('home_team', {})
        visitor_team = game.get('visitor_team', {})
        
        if home_team and 'id' in home_team:
            team_ids.add(home_team['id'])
        
        if visitor_team and 'id' in visitor_team:
            team_ids.add(visitor_team['id'])
    
    return team_ids


def validate_feature_quality(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the quality of features in the dataset
    
    Args:
        data: Dictionary containing feature data
        
    Returns:
        Dictionary with quality metrics for each feature
    """
    quality_metrics = {}
    
    # Implement feature quality validation logic here
    # This would check for outliers, missing values, distributions, etc.
    
    return quality_metrics


def validate_training_data(training_df):
    """
    Validate that the training data meets quality requirements for model training
    
    Args:
        training_df: DataFrame containing training data with features
        
    Returns:
        Dict with validation results:
        - is_valid: Boolean indicating if data is valid for training
        - message: Description of validation issues if any
        - metrics: Data quality metrics
    """
    result = {
        'is_valid': True,
        'message': "Training data validation successful",
        'metrics': {}
    }
    
    # Check if DataFrame is empty
    if training_df.empty:
        result['is_valid'] = False
        result['message'] = "Training data is empty"
        return result
    
    # Check minimum number of samples (need at least 100 for reasonable training)
    if len(training_df) < 100:
        result['is_valid'] = False
        result['message'] = f"Insufficient training samples: {len(training_df)} (minimum 100 required)"
        return result
    
    # Check for required target column
    if 'home_team_won' not in training_df.columns:
        result['is_valid'] = False
        result['message'] = "Target column 'home_team_won' not found in training data"
        return result
    
    # Check class balance (prevent highly imbalanced datasets)
    class_counts = training_df['home_team_won'].value_counts()
    if len(class_counts) < 2:
        result['is_valid'] = False
        result['message'] = f"Only one class found in target column: {class_counts.index[0]}"
        return result
    
    min_class_ratio = class_counts.min() / class_counts.sum()
    if min_class_ratio < 0.2:  # Require at least 20% of samples in minority class
        result['is_valid'] = False
        result['message'] = f"Highly imbalanced classes detected. Minority class ratio: {min_class_ratio:.2f}"
        return result
    
    # Check for expected minimum features (prevent training on insufficient features)
    expected_feature_patterns = [
        'home_team_id', 'visitor_team_id',  # Basic identifiers
        'home_', 'visitor_',  # Team stats
        'momentum_', 'matchup_'  # Advanced features
    ]
    
    missing_patterns = []
    for pattern in expected_feature_patterns:
        if not any(col.startswith(pattern) for col in training_df.columns):
            missing_patterns.append(pattern)
    
    if missing_patterns:
        result['is_valid'] = False
        result['message'] = f"Missing required feature types: {', '.join(missing_patterns)}"
        return result
    
    # Check for missing values
    missing_counts = training_df.isnull().sum()
    columns_with_nulls = missing_counts[missing_counts > 0]
    if not columns_with_nulls.empty:
        result['is_valid'] = False
        result['message'] = f"Found {len(columns_with_nulls)} columns with missing values"
        result['metrics']['missing_values'] = columns_with_nulls.to_dict()
        return result
    
    # Add data quality metrics
    result['metrics']['sample_count'] = len(training_df)
    result['metrics']['feature_count'] = len(training_df.columns)
    result['metrics']['class_balance'] = {
        "home_wins": int(class_counts.get(1, 0)),
        "visitor_wins": int(class_counts.get(0, 0)),
        "home_win_ratio": float(class_counts.get(1, 0) / class_counts.sum())
    }
    
    return result
