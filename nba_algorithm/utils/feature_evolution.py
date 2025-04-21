#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Evolution Module for NBA Prediction System

This module provides utilities for feature selection, optimization, and evolution
to improve prediction accuracy over time.

It uses real production data and machine learning techniques to identify the most
important features for different prediction types.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)

class FeatureEvolution:
    """
    Provides feature selection and optimization capabilities for the prediction system
    
    This class helps identify the most important features for different prediction types,
    tracks feature importance over time, and recommends optimal feature sets based on
    historical performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature evolution system
        
        Args:
            config: Configuration dictionary with feature evolution settings
        """
        self.config = config or {}
        
        # Default feature sets based on prediction types
        self.default_features = {
            # Game outcome features
            "moneyline": [
                "home_win_rate", "away_win_rate", "home_points_scored_avg", "away_points_scored_avg",
                "home_points_allowed_avg", "away_points_allowed_avg", "home_recent_form", "away_recent_form",
                "matchup_history_home_win_pct", "home_rest_days", "away_rest_days", "is_home_b2b", "is_away_b2b",
                "home_straight_wins", "away_straight_wins", "home_straight_losses", "away_straight_losses",
                "home_avg_point_diff", "away_avg_point_diff", "home_injury_impact", "away_injury_impact"
            ],
            "spread": [
                "home_ats_record", "away_ats_record", "home_avg_point_diff", "away_avg_point_diff",
                "home_points_scored_avg", "away_points_scored_avg", "home_points_allowed_avg", "away_points_allowed_avg",
                "matchup_history_point_diff", "home_recent_form", "away_recent_form", "home_rest_days", "away_rest_days",
                "is_home_b2b", "is_away_b2b", "home_injury_impact", "away_injury_impact", "home_pace", "away_pace",
                "home_last_10_ats", "away_last_10_ats"
            ],
            "total": [
                "home_points_scored_avg", "away_points_scored_avg", "home_points_allowed_avg", "away_points_allowed_avg",
                "matchup_history_total_points", "home_pace", "away_pace", "home_offensive_rating", "away_offensive_rating",
                "home_defensive_rating", "away_defensive_rating", "league_avg_total", "home_over_rate", "away_over_rate",
                "home_rest_days", "away_rest_days", "is_home_b2b", "is_away_b2b", "home_injury_impact", "away_injury_impact"
            ],
            
            # Player prop features
            "player_points": [
                "player_season_ppg", "player_last_10_ppg", "player_home_away_ppg", "player_minutes_avg",
                "opponent_points_allowed_pg", "opponent_defensive_efficiency", "player_usage_rate",
                "player_injury_status", "team_pace", "opponent_pace", "player_rest_days", "team_injuries"
            ],
            "player_rebounds": [
                "player_season_rpg", "player_last_10_rpg", "player_home_away_rpg", "player_minutes_avg",
                "opponent_rebounds_allowed_pg", "team_rebounding_rate", "opponent_rebounding_rate",
                "player_injury_status", "team_injuries", "player_rest_days"
            ],
            "player_assists": [
                "player_season_apg", "player_last_10_apg", "player_home_away_apg", "player_minutes_avg",
                "opponent_assists_allowed_pg", "team_assist_rate", "player_usage_rate", "team_pace",
                "player_injury_status", "team_injuries", "player_rest_days"
            ],
            "player_threes": [
                "player_season_3pm", "player_last_10_3pm", "player_home_away_3pm", "player_minutes_avg",
                "player_3pt_attempt_rate", "player_3pt_pct", "opponent_3pt_defense", "team_pace", 
                "player_injury_status", "team_injuries", "player_rest_days"
            ],
            "player_steals": [
                "player_season_spg", "player_last_10_spg", "player_home_away_spg", "player_minutes_avg",
                "opponent_turnover_rate", "player_steal_rate", "team_pace", "opponent_pace", 
                "player_injury_status", "team_injuries", "player_rest_days"
            ],
            "player_blocks": [
                "player_season_bpg", "player_last_10_bpg", "player_home_away_bpg", "player_minutes_avg",
                "opponent_blocks_allowed_pg", "player_block_rate", "player_injury_status", "team_injuries",
                "player_rest_days"
            ]
        }
        
        # Track feature importance over time
        self.feature_importance_history = {}
        
    def get_optimized_features(self, prediction_type: str, data: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Get the optimal feature set for a specific prediction type
        
        Args:
            prediction_type: Type of prediction (moneyline, spread, total, player_points, etc.)
            data: Optional DataFrame with training data to analyze
            
        Returns:
            List of feature names to use for prediction
        """
        # Map player prop prediction types to their feature sets
        if prediction_type.startswith("player_"):
            prop_type = prediction_type.replace("player_", "")
            feature_key = f"player_{prop_type}"
        else:
            feature_key = prediction_type
        
        # Check if we have default features for this prediction type
        if feature_key not in self.default_features:
            logger.warning(f"No default features defined for prediction type: {prediction_type}")
            # Return an empty list if no defaults are defined
            return []
        
        # If no data is provided, return the default feature set
        if data is None or data.empty:
            logger.info(f"Using default features for {prediction_type} predictions")
            return self.default_features[feature_key]
        
        # TODO: Implement advanced feature selection based on data analysis
        # For now, return default features
        return self.default_features[feature_key]
    
    def analyze_feature_importance(self, X: pd.DataFrame, y: np.ndarray, 
                                  prediction_type: str, is_classification: bool = True) -> Dict[str, float]:
        """
        Analyze feature importance for a specific prediction task
        
        Args:
            X: Feature DataFrame
            y: Target values
            prediction_type: Type of prediction task
            is_classification: Whether this is a classification task
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            feature_names = X.columns.tolist()
            
            # Use Random Forest for feature importance
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
            # Train model to get feature importance
            model.fit(X, y)
            importances = model.feature_importances_
            
            # Create importance dictionary
            importance_dict = {feature: importance for feature, importance in zip(feature_names, importances)}
            
            # Sort by importance, descending
            sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Track feature importance history
            self.feature_importance_history[prediction_type] = sorted_importances
            
            return dict(sorted_importances)
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            return {}
    
    def select_best_features(self, X: pd.DataFrame, y: np.ndarray, k: int = 10, 
                           is_classification: bool = True) -> List[str]:
        """
        Select k best features based on statistical tests
        
        Args:
            X: Feature DataFrame
            y: Target values
            k: Number of features to select
            is_classification: Whether this is a classification task
            
        Returns:
            List of selected feature names
        """
        try:
            # Choose appropriate statistical test
            if is_classification:
                selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
            else:
                selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
                
            # Apply selection
            selector.fit(X, y)
            
            # Get selected feature names
            feature_mask = selector.get_support()
            selected_features = X.columns[feature_mask].tolist()
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error selecting best features: {str(e)}")
            return []
    
    def create_feature_recommendations(self) -> Dict[str, List[str]]:
        """
        Create feature recommendations based on historical importance
        
        Returns:
            Dictionary mapping prediction types to recommended feature lists
        """
        recommendations = {}
        
        # Use importance history to make recommendations
        for pred_type, importances in self.feature_importance_history.items():
            # Get top 15 features
            top_features = [feature for feature, _ in importances[:15]]
            recommendations[pred_type] = top_features
        
        # Fill in default recommendations for types without history
        for pred_type, features in self.default_features.items():
            if pred_type not in recommendations:
                recommendations[pred_type] = features
        
        return recommendations
