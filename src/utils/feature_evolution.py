#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Evolution System

This module provides tools for automated feature engineering, discovery, and evolution.
It detects new statistical patterns, evaluates new potential features, and adapts to
league-wide changes in playing style over time.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

# Configure logging
logger = logging.getLogger(__name__)

class FeatureEvolution:
    """Manage feature evolution and adaptation for NBA prediction models"""
    
    def __init__(self, storage_dir: str = "data/features/evolution", history_limit: int = 10):
        """
        Initialize the feature evolution system
        
        Args:
            storage_dir: Directory to store feature evolution data
            history_limit: Number of historical feature sets to maintain
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        self.history_limit = history_limit
        self.feature_history = self._load_feature_history()
        
        # Base feature sets for different prediction types
        self.base_features = {
            "moneyline": [
                "home_win_rate", "away_win_rate", "home_points_avg", "away_points_avg",
                "home_defense_rating", "away_defense_rating", "home_offense_rating", "away_offense_rating",
                "rest_days_home", "rest_days_away", "home_streak", "away_streak"
            ],
            "spread": [
                "home_ats_win_rate", "away_ats_win_rate", "home_spread_avg", "away_spread_avg",
                "home_margin_avg", "away_margin_avg", "home_cover_rate", "away_cover_rate"
            ],
            "total": [
                "home_over_rate", "away_over_rate", "home_total_avg", "away_total_avg",
                "home_pace", "away_pace", "home_scoring_avg", "away_scoring_avg"
            ],
            "player_props": [
                "player_pts_avg", "player_reb_avg", "player_ast_avg", "player_min_avg",
                "opponent_def_rating", "opponent_pace", "home_away", "rest_days",
                "recent_performance", "matchup_history"
            ]
        }
        
        # Feature importance thresholds
        self.importance_thresholds = {
            "moneyline": 0.02,
            "spread": 0.02,
            "total": 0.02,
            "player_props": 0.02
        }
    
    def _load_feature_history(self) -> Dict[str, Any]:
        """
        Load feature evolution history
        
        Returns:
            Feature history dictionary
        """
        history_file = self.storage_dir / "feature_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading feature history: {str(e)}")
                return self._initialize_history()
        else:
            return self._initialize_history()
    
    def _initialize_history(self) -> Dict[str, Any]:
        """
        Initialize a new feature history structure
        
        Returns:
            New feature history dictionary
        """
        return {
            "feature_sets": {
                "moneyline": [],
                "spread": [],
                "total": [],
                "player_props": []
            },
            "feature_performance": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_feature_history(self) -> None:
        """
        Save feature history to disk
        """
        history_file = self.storage_dir / "feature_history.json"
        
        try:
            # Update last updated timestamp
            self.feature_history["last_updated"] = datetime.now().isoformat()
            
            with open(history_file, 'w') as f:
                json.dump(self.feature_history, f, indent=2)
            
            logger.info(f"Feature history saved to {history_file}")
        except Exception as e:
            logger.error(f"Error saving feature history: {str(e)}")
    
    def register_feature_set(self, prediction_type: str, features: List[str],
                            performance_metrics: Dict[str, float], model_id: str,
                            description: str = "") -> str:
        """
        Register a new feature set
        
        Args:
            prediction_type: Type of prediction (moneyline, spread, total, player_props)
            features: List of feature names
            performance_metrics: Dictionary with performance metrics
            model_id: ID of the model using these features
            description: Optional description of this feature set
            
        Returns:
            ID of the registered feature set
        """
        if prediction_type not in self.feature_history["feature_sets"]:
            logger.warning(f"Unknown prediction type: {prediction_type}. Adding to history.")
            self.feature_history["feature_sets"][prediction_type] = []
        
        # Generate unique feature set ID
        feature_set_id = f"{prediction_type}_features_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create feature set entry
        feature_set = {
            "id": feature_set_id,
            "prediction_type": prediction_type,
            "features": features,
            "feature_count": len(features),
            "performance_metrics": performance_metrics,
            "model_id": model_id,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        
        # Add to history
        self.feature_history["feature_sets"][prediction_type].append(feature_set)
        
        # Maintain history limit
        if len(self.feature_history["feature_sets"][prediction_type]) > self.history_limit:
            # Remove oldest feature set
            oldest = sorted(self.feature_history["feature_sets"][prediction_type],
                          key=lambda x: x["created_at"])[0]
            self.feature_history["feature_sets"][prediction_type].remove(oldest)
            logger.info(f"Removed oldest feature set: {oldest['id']}")
        
        # Save history
        self._save_feature_history()
        
        logger.info(f"Registered feature set {feature_set_id} for {prediction_type} with {len(features)} features")
        return feature_set_id

    def evaluate_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                                 prediction_type: str, is_classification: bool = False) -> Dict[str, float]:
        """
        Evaluate feature importance for a dataset
        
        Args:
            X: Feature matrix
            y: Target variable
            prediction_type: Type of prediction
            is_classification: Whether this is a classification problem
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if X.empty or y.empty:
            logger.error("Cannot evaluate empty dataset")
            return {}
        
        try:
            importances = {}
            feature_names = list(X.columns)
            
            # Method 1: Random Forest importance
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            rf_importances = model.feature_importances_
            
            for feature, importance in zip(feature_names, rf_importances):
                importances[feature] = float(importance)
            
            # Method 2: Statistical correlation (only for regression)
            if not is_classification:
                try:
                    # Use mutual information for regression
                    mi_importances = mutual_info_regression(X, y, random_state=42)
                    
                    # Normalize to sum to 1
                    mi_sum = sum(mi_importances)
                    if mi_sum > 0:
                        mi_importances = mi_importances / mi_sum
                        
                    # Average with random forest importances
                    for i, feature in enumerate(feature_names):
                        importances[feature] = (importances[feature] + float(mi_importances[i])) / 2
                except Exception as e:
                    logger.warning(f"Error calculating mutual information: {str(e)}")
            
            # Record feature importance in history
            if "feature_importance" not in self.feature_history:
                self.feature_history["feature_importance"] = {}
                
            if prediction_type not in self.feature_history["feature_importance"]:
                self.feature_history["feature_importance"][prediction_type] = {}
            
            # Record with timestamp
            timestamp = datetime.now().isoformat()
            self.feature_history["feature_importance"][prediction_type][timestamp] = importances
            
            # Save history
            self._save_feature_history()
            
            return importances
        except Exception as e:
            logger.error(f"Error evaluating feature importance: {str(e)}")
            return {}
    
    def discover_candidate_features(self, data: pd.DataFrame, existing_features: List[str], 
                                  prediction_type: str) -> List[str]:
        """
        Discover potential new features based on patterns in the data
        
        Args:
            data: DataFrame with all available data columns
            existing_features: List of currently used features
            prediction_type: Type of prediction
            
        Returns:
            List of candidate new feature names
        """
        if data.empty:
            logger.error("Cannot discover features in empty dataset")
            return []
        
        try:
            candidates = []
            
            # 1. Interaction terms between important existing features
            if len(existing_features) >= 2:
                # Get historical importance if available
                importance_dict = {}
                if ("feature_importance" in self.feature_history and 
                    prediction_type in self.feature_history["feature_importance"]):
                    # Get most recent importance
                    timestamps = sorted(self.feature_history["feature_importance"][prediction_type].keys())
                    if timestamps:
                        importance_dict = self.feature_history["feature_importance"][prediction_type][timestamps[-1]]
                
                # Filter existing features by those present in the data
                valid_features = [f for f in existing_features if f in data.columns]
                
                # Sort by importance if available, otherwise use all
                if importance_dict:
                    sorted_features = sorted(
                        [f for f in valid_features if f in importance_dict],
                        key=lambda x: importance_dict.get(x, 0),
                        reverse=True
                    )
                    top_features = sorted_features[:10]  # Use top 10 features
                else:
                    top_features = valid_features[:10] if len(valid_features) > 10 else valid_features
                
                # Generate interaction terms
                for i, feat1 in enumerate(top_features):
                    for feat2 in top_features[i+1:]:
                        interaction_name = f"{feat1}_x_{feat2}"
                        if interaction_name not in existing_features and feat1 in data.columns and feat2 in data.columns:
                            # Check if both features are numeric
                            if pd.api.types.is_numeric_dtype(data[feat1]) and pd.api.types.is_numeric_dtype(data[feat2]):
                                candidates.append(interaction_name)
            
            # 2. Rolling averages and trends for time series features
            time_based_features = []
            for feature in existing_features:
                if feature in data.columns and pd.api.types.is_numeric_dtype(data[feature]):
                    # Rolling averages
                    time_based_features.append(f"{feature}_rolling_avg_3")
                    time_based_features.append(f"{feature}_rolling_avg_5")
                    
                    # Trend features (difference from previous N)
                    time_based_features.append(f"{feature}_trend_3")
                    
                    # Momentum features (rate of change)
                    time_based_features.append(f"{feature}_momentum")
            
            candidates.extend([f for f in time_based_features if f not in existing_features])
            
            # 3. Add sport-specific features based on prediction type
            if prediction_type == "moneyline":
                ml_features = [
                    "win_streak_differential",
                    "home_court_advantage_factor",
                    "back_to_back_game_impact",
                    "travel_distance_impact",
                    "rest_advantage"
                ]
                candidates.extend([f for f in ml_features if f not in existing_features])
                
            elif prediction_type == "spread":
                spread_features = [
                    "home_ats_trend",
                    "away_ats_trend",
                    "favorite_cover_rate",
                    "underdog_cover_rate",
                    "line_movement_direction"
                ]
                candidates.extend([f for f in spread_features if f not in existing_features])
                
            elif prediction_type == "total":
                total_features = [
                    "combined_pace_factor",
                    "defensive_efficiency_combined",
                    "offensive_efficiency_combined",
                    "over_under_trend",
                    "recent_totals_avg"
                ]
                candidates.extend([f for f in total_features if f not in existing_features])
                
            elif prediction_type == "player_props":
                prop_features = [
                    "player_usage_rate",
                    "matchup_defensive_rating",
                    "recent_minutes_trend",
                    "back_to_back_performance",
                    "opponent_position_defense"
                ]
                candidates.extend([f for f in prop_features if f not in existing_features])
            
            logger.info(f"Discovered {len(candidates)} candidate features for {prediction_type}")
            return candidates
        except Exception as e:
            logger.error(f"Error discovering candidate features: {str(e)}")
            return []
    
    def engineer_discovered_features(self, data: pd.DataFrame, candidates: List[str], 
                                    existing_features: List[str]) -> pd.DataFrame:
        """
        Engineer the discovered candidate features
        
        Args:
            data: DataFrame with all available data columns
            candidates: List of candidate feature names
            existing_features: List of currently used features
            
        Returns:
            DataFrame with existing and new engineered features
        """
        if data.empty:
            logger.error("Cannot engineer features for empty dataset")
            return pd.DataFrame()
        
        try:
            # Start with existing features
            feature_cols = [col for col in existing_features if col in data.columns]
            result_df = data[feature_cols].copy()
            
            # Process each candidate feature
            for candidate in candidates:
                # Interaction terms
                if "_x_" in candidate:
                    feat1, feat2 = candidate.split("_x_")
                    if feat1 in data.columns and feat2 in data.columns:
                        if pd.api.types.is_numeric_dtype(data[feat1]) and pd.api.types.is_numeric_dtype(data[feat2]):
                            result_df[candidate] = data[feat1] * data[feat2]
                
                # Rolling average features
                elif "_rolling_avg_" in candidate:
                    base_feat, window_size = candidate.rsplit("_", 1)[0], int(candidate.rsplit("_", 1)[1])
                    base_feat = base_feat.replace("_rolling_avg", "")
                    if base_feat in data.columns and pd.api.types.is_numeric_dtype(data[base_feat]):
                        result_df[candidate] = data[base_feat].rolling(window=window_size, min_periods=1).mean()
                
                # Trend features
                elif "_trend_" in candidate:
                    base_feat, window_size = candidate.rsplit("_", 1)[0], int(candidate.rsplit("_", 1)[1])
                    base_feat = base_feat.replace("_trend", "")
                    if base_feat in data.columns and pd.api.types.is_numeric_dtype(data[base_feat]):
                        result_df[candidate] = data[base_feat].diff(periods=window_size)
                
                # Momentum features
                elif "_momentum" in candidate:
                    base_feat = candidate.replace("_momentum", "")
                    if base_feat in data.columns and pd.api.types.is_numeric_dtype(data[base_feat]):
                        result_df[candidate] = data[base_feat].pct_change()
                
                # Sport-specific features (simplified implementation)
                elif candidate in [
                    "win_streak_differential", "home_court_advantage_factor", "back_to_back_game_impact",
                    "travel_distance_impact", "rest_advantage", "home_ats_trend", "away_ats_trend",
                    "favorite_cover_rate", "underdog_cover_rate", "line_movement_direction",
                    "combined_pace_factor", "defensive_efficiency_combined", "offensive_efficiency_combined",
                    "over_under_trend", "recent_totals_avg", "player_usage_rate", "matchup_defensive_rating",
                    "recent_minutes_trend", "back_to_back_performance", "opponent_position_defense"
                ]:
                    # Placeholder - these would be properly implemented based on available raw data
                    # For demo purposes, we just add placeholder columns
                    result_df[candidate] = np.random.normal(size=len(result_df))
                    logger.warning(f"Added placeholder implementation for advanced feature: {candidate}")
            
            # Fill NaN values
            result_df = result_df.fillna(0)
            
            logger.info(f"Engineered {len(result_df.columns) - len(feature_cols)} new features")
            return result_df
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            return pd.DataFrame(data[existing_features]) if all(f in data.columns for f in existing_features) else pd.DataFrame()
    
    def evaluate_feature_set(self, X: pd.DataFrame, y: pd.Series, 
                           is_classification: bool = False) -> Dict[str, float]:
        """
        Evaluate a feature set by training a model and measuring performance
        
        Args:
            X: Feature matrix
            y: Target variable
            is_classification: Whether this is a classification problem
            
        Returns:
            Dictionary with evaluation metrics
        """
        if X.empty or y.empty:
            logger.error("Cannot evaluate empty dataset")
            return {}
        
        try:
            from sklearn.model_selection import train_test_split
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train a model based on the problem type
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Calculate performance metrics
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                metrics = {
                    "train_accuracy": train_score,
                    "test_accuracy": test_score,
                    "feature_count": X.shape[1]
                }
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Calculate performance metrics
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                
                train_mse = mean_squared_error(y_train, train_preds)
                test_mse = mean_squared_error(y_test, test_preds)
                
                metrics = {
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "feature_count": X.shape[1]
                }
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating feature set: {str(e)}")
            return {}
    
    def compare_feature_sets(self, base_features: List[str], candidate_features: List[str],
                           data: pd.DataFrame, target: pd.Series,
                           prediction_type: str, is_classification: bool = False) -> Dict[str, Any]:
        """
        Compare base feature set with a candidate feature set
        
        Args:
            base_features: List of base feature names
            candidate_features: List of candidate feature names
            data: DataFrame with all available columns
            target: Target variable series
            prediction_type: Type of prediction
            is_classification: Whether this is a classification problem
            
        Returns:
            Comparison results
        """
        if data.empty or target.empty:
            logger.error("Cannot compare feature sets with empty data")
            return {}
        
        # Filter features to ensure they exist in the data
        valid_base = [f for f in base_features if f in data.columns]
        valid_candidate = [f for f in candidate_features if f in data.columns]
        
        if not valid_base or not valid_candidate:
            logger.error("No valid features found in data")
            return {}
        
        try:
            # Prepare feature matrices
            X_base = data[valid_base]
            X_candidate = data[valid_candidate]
            
            # Evaluate each feature set
            base_metrics = self.evaluate_feature_set(X_base, target, is_classification)
            candidate_metrics = self.evaluate_feature_set(X_candidate, target, is_classification)
            
            # Calculate improvements
            improvements = {}
            
            for metric in base_metrics:
                if metric in candidate_metrics:
                    if metric.startswith("test_") or metric.startswith("train_"):
                        # For accuracy, higher is better; for MSE, lower is better
                        if "accuracy" in metric:
                            improvement = candidate_metrics[metric] - base_metrics[metric]
                            improvements[metric] = {
                                "absolute": improvement,
                                "relative": (improvement / base_metrics[metric]) * 100 if base_metrics[metric] > 0 else float('inf'),
                                "better": improvement > 0
                            }
                        elif "mse" in metric:
                            improvement = base_metrics[metric] - candidate_metrics[metric]  # Reduction in error
                            improvements[metric] = {
                                "absolute": improvement,
                                "relative": (improvement / base_metrics[metric]) * 100 if base_metrics[metric] > 0 else float('inf'),
                                "better": improvement > 0
                            }
            
            # Overall assessment
            is_better = False
            for imp in improvements.values():
                if imp["better"]:
                    is_better = True
                    break
            
            # Feature importance if candidate set is better
            feature_importance = {}
            if is_better:
                importance = self.evaluate_feature_importance(X_candidate, target, prediction_type, is_classification)
                feature_importance = importance
            
            comparison = {
                "base_set": {
                    "features": valid_base,
                    "feature_count": len(valid_base),
                    "metrics": base_metrics
                },
                "candidate_set": {
                    "features": valid_candidate,
                    "feature_count": len(valid_candidate),
                    "metrics": candidate_metrics
                },
                "improvements": improvements,
                "is_better": is_better,
                "feature_importance": feature_importance
            }
            
            # Record comparison
            if "feature_comparisons" not in self.feature_history:
                self.feature_history["feature_comparisons"] = {}
                
            if prediction_type not in self.feature_history["feature_comparisons"]:
                self.feature_history["feature_comparisons"][prediction_type] = []
            
            # Add comparison record
            comparison_record = {
                "timestamp": datetime.now().isoformat(),
                "base_feature_count": len(valid_base),
                "candidate_feature_count": len(valid_candidate),
                "improvements": improvements,
                "is_better": is_better
            }
            
            self.feature_history["feature_comparisons"][prediction_type].append(comparison_record)
            
            # Save history
            self._save_feature_history()
            
            return comparison
        except Exception as e:
            logger.error(f"Error comparing feature sets: {str(e)}")
            return {}

    def select_optimal_features(self, data: pd.DataFrame, target: pd.Series, 
                             prediction_type: str, is_classification: bool = False,
                             max_features: int = 20) -> List[str]:
        """
        Select the optimal feature set for a specific prediction task
        
        Args:
            data: DataFrame with all available columns
            target: Target variable
            prediction_type: Type of prediction
            is_classification: Whether this is a classification problem
            max_features: Maximum number of features to include
            
        Returns:
            List of optimal feature names
        """
        if data.empty or target.empty:
            logger.error("Cannot select features from empty dataset")
            return []
        
        try:
            # Get base features for the prediction type
            base_features = self.base_features.get(prediction_type, [])
            
            # Filter to features available in the data
            valid_base = [f for f in base_features if f in data.columns]
            logger.info(f"Starting with {len(valid_base)} valid base features for {prediction_type}")
            
            # Identify correlation matrix to remove highly correlated features
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            valid_numeric = [f for f in valid_base if f in numeric_cols]
            
            if valid_numeric:
                # Compute correlation matrix
                corr_matrix = data[valid_numeric].corr().abs()
                
                # Identify pairs of highly correlated features
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                
                # Find features with correlation > 0.95
                to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
                
                # Remove highly correlated features
                valid_base = [f for f in valid_base if f not in to_drop]
                logger.info(f"Removed {len(to_drop)} highly correlated features")
            
            # Discover candidate features
            candidates = self.discover_candidate_features(data, valid_base, prediction_type)
            logger.info(f"Discovered {len(candidates)} candidate features")
            
            # Engineer features
            feature_df = self.engineer_discovered_features(data, candidates, valid_base)
            
            if feature_df.empty:
                logger.error("Feature engineering failed to produce valid features")
                return valid_base
            
            # Evaluate feature importance
            all_features = list(feature_df.columns)
            
            # Check if we have too many features, if so use feature selection
            if len(all_features) > max_features:
                # Get importance scores
                importance = self.evaluate_feature_importance(feature_df, target, prediction_type, is_classification)
                
                # Sort by importance
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                # Select top features, ensuring we keep at least some base features
                # Reserve slots for at least half of base features
                min_base = min(len(valid_base) // 2, max_features // 2)
                
                # First add important base features
                top_base = [f for f, _ in sorted_features if f in valid_base][:min_base]
                
                # Then add remaining important features up to max_features
                remaining_slots = max_features - len(top_base)
                top_additional = [f for f, _ in sorted_features if f not in top_base][:remaining_slots]
                
                optimal_features = top_base + top_additional
                
                logger.info(f"Selected {len(optimal_features)} optimal features through importance filtering")
            else:
                optimal_features = all_features
            
            # Register the feature set
            feature_set_id = self.register_feature_set(
                prediction_type=prediction_type,
                features=optimal_features,
                performance_metrics={"optimized": True, "feature_count": len(optimal_features)},
                model_id="pending",
                description=f"Optimized feature set for {prediction_type}"
            )
            
            logger.info(f"Registered optimal feature set with ID: {feature_set_id}")
            return optimal_features
        except Exception as e:
            logger.error(f"Error selecting optimal features: {str(e)}")
            return self.base_features.get(prediction_type, [])

    def get_latest_feature_set(self, prediction_type: str) -> List[str]:
        """
        Get the latest registered feature set for a prediction type
        
        Args:
            prediction_type: Type of prediction
            
        Returns:
            List of feature names, or base features if none registered
        """
        if (prediction_type not in self.feature_history["feature_sets"] or
            not self.feature_history["feature_sets"][prediction_type]):
            logger.warning(f"No feature sets registered for {prediction_type}, using base features")
            return self.base_features.get(prediction_type, [])
        
        # Get the latest feature set
        latest = sorted(
            self.feature_history["feature_sets"][prediction_type],
            key=lambda x: x["created_at"],
            reverse=True
        )[0]
        
        logger.info(f"Using latest feature set for {prediction_type} with {len(latest['features'])} features")
        return latest["features"]
    
    def get_production_feature_set(self, prediction_type: str) -> List[str]:
        """
        Get the production feature set for a prediction type
        
        Args:
            prediction_type: Type of prediction
            
        Returns:
            List of feature names marked for production use
        """
        if ("production_features" not in self.feature_history or
            prediction_type not in self.feature_history["production_features"]):
            # Fall back to latest
            logger.warning(f"No production feature set for {prediction_type}, using latest")
            return self.get_latest_feature_set(prediction_type)
        
        prod_features = self.feature_history["production_features"][prediction_type]
        logger.info(f"Using production feature set for {prediction_type} with {len(prod_features)} features")
        return prod_features
    
    def set_production_feature_set(self, prediction_type: str, feature_set_id: str) -> bool:
        """
        Set a feature set as the production set for a prediction type
        
        Args:
            prediction_type: Type of prediction
            feature_set_id: ID of feature set to mark as production
            
        Returns:
            True if successful, False otherwise
        """
        if prediction_type not in self.feature_history["feature_sets"]:
            logger.error(f"Unknown prediction type: {prediction_type}")
            return False
        
        # Find the feature set
        feature_set = None
        for fs in self.feature_history["feature_sets"][prediction_type]:
            if fs["id"] == feature_set_id:
                feature_set = fs
                break
        
        if not feature_set:
            logger.error(f"Feature set with ID {feature_set_id} not found")
            return False
        
        # Initialize production features if not present
        if "production_features" not in self.feature_history:
            self.feature_history["production_features"] = {}
        
        # Set as production
        self.feature_history["production_features"][prediction_type] = feature_set["features"]
        
        # Add metadata
        self.feature_history["production_features_metadata"] = self.feature_history.get("production_features_metadata", {})
        self.feature_history["production_features_metadata"][prediction_type] = {
            "feature_set_id": feature_set_id,
            "set_at": datetime.now().isoformat(),
            "feature_count": len(feature_set["features"])
        }
        
        # Save history
        self._save_feature_history()
        
        logger.info(f"Set feature set {feature_set_id} as production for {prediction_type}")
        return True
    
    def detect_league_changes(self, current_data: pd.DataFrame, historical_data: pd.DataFrame,
                            features: List[str], threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect changes in NBA league patterns by comparing current vs historical distributions
        
        Args:
            current_data: Recent NBA data
            historical_data: Historical NBA data
            features: Features to analyze
            threshold: Threshold for considering a change significant
            
        Returns:
            Dictionary with detected changes
        """
        if current_data.empty or historical_data.empty:
            logger.error("Cannot detect changes with empty dataset")
            return {}
        
        # Filter to common features available in both datasets
        common_features = [f for f in features if f in current_data.columns and f in historical_data.columns]
        
        if not common_features:
            logger.error("No common features found in both datasets")
            return {}
        
        changes = {}
        
        try:
            # Analyze each feature
            for feature in common_features:
                # Check if feature is numeric
                if (pd.api.types.is_numeric_dtype(current_data[feature]) and 
                    pd.api.types.is_numeric_dtype(historical_data[feature])):
                    
                    # Calculate statistics
                    curr_mean = current_data[feature].mean()
                    hist_mean = historical_data[feature].mean()
                    
                    curr_std = current_data[feature].std()
                    hist_std = historical_data[feature].std()
                    
                    # Calculate relative changes
                    mean_change = (curr_mean - hist_mean) / hist_mean if hist_mean != 0 else float('inf')
                    std_change = (curr_std - hist_std) / hist_std if hist_std != 0 else float('inf')
                    
                    # If change exceeds threshold, record it
                    if abs(mean_change) > threshold or abs(std_change) > threshold:
                        changes[feature] = {
                            "mean_change": mean_change,
                            "std_change": std_change,
                            "current_mean": curr_mean,
                            "historical_mean": hist_mean,
                            "current_std": curr_std,
                            "historical_std": hist_std
                        }
            
            # Record detected changes
            if "league_changes" not in self.feature_history:
                self.feature_history["league_changes"] = []
            
            if changes:
                change_record = {
                    "timestamp": datetime.now().isoformat(),
                    "changes": changes,
                    "features_analyzed": len(common_features),
                    "features_changed": len(changes)
                }
                
                self.feature_history["league_changes"].append(change_record)
                self._save_feature_history()
                
                logger.info(f"Detected {len(changes)} significant changes in league patterns")
            
            return changes
        except Exception as e:
            logger.error(f"Error detecting league changes: {str(e)}")
            return {}

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report on feature evolution
        
        Returns:
            Dictionary with report information
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "feature_sets": {
                k: len(v) for k, v in self.feature_history["feature_sets"].items()
            },
            "latest_feature_sets": {},
            "production_feature_sets": {}
        }
        
        # Add latest feature sets
        for pred_type in self.feature_history["feature_sets"]:
            if self.feature_history["feature_sets"][pred_type]:
                latest = sorted(
                    self.feature_history["feature_sets"][pred_type],
                    key=lambda x: x["created_at"],
                    reverse=True
                )[0]
                
                report["latest_feature_sets"][pred_type] = {
                    "id": latest["id"],
                    "feature_count": len(latest["features"]),
                    "created_at": latest["created_at"]
                }
        
        # Add production feature sets
        if "production_features_metadata" in self.feature_history:
            for pred_type, metadata in self.feature_history["production_features_metadata"].items():
                report["production_feature_sets"][pred_type] = metadata
        
        # Add feature importance trends
        report["feature_importance_trends"] = {}
        if "feature_importance" in self.feature_history:
            for pred_type, timestamps in self.feature_history["feature_importance"].items():
                if len(timestamps) >= 2:
                    # Get oldest and newest importance scores
                    sorted_ts = sorted(timestamps.keys())
                    oldest = self.feature_history["feature_importance"][pred_type][sorted_ts[0]]
                    newest = self.feature_history["feature_importance"][pred_type][sorted_ts[-1]]
                    
                    # Find common features
                    common_features = set(oldest.keys()) & set(newest.keys())
                    
                    # Calculate changes
                    importance_changes = {}
                    for feature in common_features:
                        old_score = oldest[feature]
                        new_score = newest[feature]
                        change = new_score - old_score
                        
                        importance_changes[feature] = {
                            "old_score": old_score,
                            "new_score": new_score,
                            "change": change
                        }
                    
                    # Sort by absolute change
                    top_changes = sorted(
                        importance_changes.items(),
                        key=lambda x: abs(x[1]["change"]),
                        reverse=True
                    )[:10]  # Top 10 changes
                    
                    report["feature_importance_trends"][pred_type] = {
                        "period_start": sorted_ts[0],
                        "period_end": sorted_ts[-1],
                        "top_changes": dict(top_changes)
                    }
        
        # Add league changes if available
        if "league_changes" in self.feature_history and self.feature_history["league_changes"]:
            report["league_changes"] = {
                "count": len(self.feature_history["league_changes"]),
                "latest": self.feature_history["league_changes"][-1]
            }
        
        # Add feature comparison summary
        if "feature_comparisons" in self.feature_history:
            report["feature_comparisons"] = {
                pred_type: {
                    "count": len(comparisons),
                    "improvement_rate": sum(1 for comp in comparisons if comp["is_better"]) / len(comparisons) if comparisons else 0
                }
                for pred_type, comparisons in self.feature_history["feature_comparisons"].items()
            }
        
        return report

# Example usage if run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fe = FeatureEvolution()
    print(f"Feature evolution system initialized with {len(fe.base_features)} base feature sets")
    
    # Generate and print a report
    report = fe.generate_report()
    print(json.dumps(report, indent=2))
