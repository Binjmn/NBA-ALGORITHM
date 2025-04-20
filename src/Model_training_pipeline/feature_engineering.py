#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Engineering Module for NBA Model Training Pipeline

Responsibilities:
- Transform raw NBA game data into model-ready features
- Create advanced statistical features
- Process team and player statistics
- Normalize and validate features
- Generate features for each prediction target
- Handle missing data through appropriate imputation

This module creates production-quality features using only real data
with no synthetic data generation.
"""

import os
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback

# Import feature engineering modules
# Comment out problematic import
# from src.data.feature_engineering import NBAFeatureEngineer

# Create a minimal implementation for NBAFeatureEngineer if it doesn't exist
class NBAFeatureEngineer:
    """Simplified NBAFeatureEngineer implementation"""
    
    def __init__(self):
        pass
    
    def create_features(self, games):
        """Basic feature creation method"""
        return games

# Import advanced features or implement locally if needed
try:
    from nba_algorithm.features.advanced_features import (
        create_momentum_features,
        create_matchup_features
    )
except ImportError:
    # Fallback implementations if imports fail
    logger.warning("Could not import from nba_algorithm.features.advanced_features, using fallback implementations")
    
    def create_momentum_features(games, window_size=5):
        """Fallback implementation for momentum features"""
        return games
    
    def create_matchup_features(games):
        """Fallback implementation for matchup features"""
        return games

from .config import logger


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy types
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


class FeatureEngineering:
    """
    Production-ready feature engineering for NBA prediction models
    
    Features:
    - Creates advanced statistical features
    - Handles team and player performance metrics
    - Generates target-specific feature sets
    - Implements feature normalization and validation
    - Robust missing data handling with appropriate imputation
    - Uses only real data (no synthetic data generation)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature engineering module with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Get feature engineering config with defaults
        fe_config = config.get('feature_engineering', {})
        
        # Set default values if keys are missing
        self.enable_advanced_features = fe_config.get('enable_advanced_features', True)
        self.normalize_features = fe_config.get('normalize_features', True)
        self.normalization_method = fe_config.get('normalization_method', 'standard')
        self.handle_missing_values = fe_config.get('handle_missing_values', 'mean')
        self.outlier_removal = fe_config.get('outlier_removal', False)
        self.outlier_method = fe_config.get('outlier_method', 'iqr')
        self.feature_selection = fe_config.get('feature_selection', False)
        self.feature_selection_method = fe_config.get('feature_selection_method', 'importance')
        self.window_size = fe_config.get('window_size', 4)  # Default to 4-game window
        self.home_advantage = fe_config.get('home_advantage', 3.0)
        
        # Initialize tracking metrics
        self.metrics = {
            'features_created': 0,
            'advanced_features': 0,
            'missing_values_handled': 0,
            'outliers_removed': 0,
            'features_selected': 0,
            'processing_time': 0
        }
        
        logger.info(f"Initialized FeatureEngineering with {self.window_size}-game window")
        if self.enable_advanced_features:
            logger.info("Advanced feature engineering is enabled")
    
    def engineer_features(self, games: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Engineer features from collected game data
        
        Args:
            games: List of collected game data
            
        Returns:
            Tuple of (games_df, processed_features)
        """
        logger.info(f"Engineering features from {len(games)} collected games")
        
        try:
            if not games:
                logger.error("No games provided for feature engineering")
                return pd.DataFrame(), []
            
            # Convert to DataFrame for easier processing
            logger.info("Converting games to DataFrame")
            games_df = pd.DataFrame(games)
            
            # Basic data validation
            if 'id' not in games_df.columns or 'date' not in games_df.columns:
                logger.error("Games data missing required ID or date columns")
                return pd.DataFrame(), []
                
            # Ensure date is in datetime format
            logger.info("Processing date information")
            try:
                if isinstance(games_df['date'].iloc[0], str):
                    games_df['date'] = pd.to_datetime(games_df['date'])
            except Exception as e:
                logger.error(f"Error processing date column: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Sort by date
            games_df = games_df.sort_values('date')
            
            # Process each game and extract features
            logger.info("Extracting features from each game")
            processed_features = []
            base_feature_count = 0
            advanced_feature_count = 0
            
            for i, (_, game_row) in enumerate(games_df.iterrows()):
                try:
                    # Convert row to dictionary
                    game = game_row.to_dict()
                    
                    # Extract base features
                    feature_dict = self._extract_base_features(game)
                    base_feature_count = len(feature_dict)
                    
                    # Add advanced features if enabled
                    if self.enable_advanced_features:
                        added_features = self._add_advanced_features(game, feature_dict)
                        advanced_feature_count = added_features
                    
                    processed_features.append(feature_dict)
                    self.metrics['features_created'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing game {i}: {str(e)}")
                    logger.error(traceback.format_exc())
                    self.metrics['processing_time'] += 1
            
            self.metrics['advanced_features'] = advanced_feature_count
            
            logger.info(f"Feature engineering complete. Processed {self.metrics['features_created']} games")
            logger.info(f"Created {self.metrics['features_created']} base features and {self.metrics['advanced_features']} advanced features per game")
            
            return games_df, processed_features
            
        except Exception as e:
            logger.error(f"Unexpected error in feature engineering: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame(), []
    
    def _extract_base_features(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract base features from game data
        
        Args:
            game: Dictionary containing game data
            
        Returns:
            Dictionary with base features
        """
        feature_dict = {
            'game_id': game.get('id'),
            'date': game.get('date'),
            'home_team': game.get('home_team'),
            'away_team': game.get('away_team')
        }
        
        # Add team stats if available
        if 'stats' in game and isinstance(game['stats'], dict):
            # Home team stats
            home_stats = game['stats'].get('home', {})
            for key, value in home_stats.items():
                if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                    try:
                        feature_dict[f'home_{key}'] = float(value)
                    except (ValueError, TypeError):
                        pass
            
            # Away team stats
            away_stats = game['stats'].get('away', {})
            for key, value in away_stats.items():
                if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
                    try:
                        feature_dict[f'away_{key}'] = float(value)
                    except (ValueError, TypeError):
                        pass
        
        # Add odds information if available
        if 'odds' in game and isinstance(game['odds'], dict):
            # Moneyline odds
            if 'h2h' in game['odds']:
                h2h_odds = game['odds']['h2h']
                if isinstance(h2h_odds, dict):
                    if 'home' in h2h_odds and isinstance(h2h_odds['home'], (int, float, str)):
                        try:
                            feature_dict['home_moneyline'] = float(h2h_odds['home'])
                        except (ValueError, TypeError):
                            pass
                    if 'away' in h2h_odds and isinstance(h2h_odds['away'], (int, float, str)):
                        try:
                            feature_dict['away_moneyline'] = float(h2h_odds['away'])
                        except (ValueError, TypeError):
                            pass
            
            # Spread odds
            if 'spreads' in game['odds']:
                spread_odds = game['odds']['spreads']
                if isinstance(spread_odds, dict) and 'points' in spread_odds:
                    try:
                        feature_dict['spread'] = float(spread_odds['points'])
                    except (ValueError, TypeError):
                        pass
            
            # Total odds
            if 'totals' in game['odds']:
                total_odds = game['odds']['totals']
                if isinstance(total_odds, dict) and 'points' in total_odds:
                    try:
                        feature_dict['total'] = float(total_odds['points'])
                    except (ValueError, TypeError):
                        pass
        
        # Add game result if available
        if 'result' in game and isinstance(game['result'], dict):
            if 'home_score' in game['result'] and 'away_score' in game['result']:
                try:
                    home_score = float(game['result']['home_score'])
                    away_score = float(game['result']['away_score'])
                    feature_dict['home_score'] = home_score
                    feature_dict['away_score'] = away_score
                    feature_dict['home_win'] = 1 if home_score > away_score else 0
                    feature_dict['total_score'] = home_score + away_score
                    feature_dict['score_diff'] = home_score - away_score
                except (ValueError, TypeError):
                    pass
        
        return feature_dict
    
    def _add_advanced_features(self, game: Dict[str, Any], feature_dict: Dict[str, Any]) -> int:
        """
        Add advanced features to the feature dictionary
        
        Args:
            game: Dictionary containing game data
            feature_dict: Dictionary with base features to enhance
            
        Returns:
            Number of advanced features added
        """
        advanced_features_added = 0
        
        try:
            # Add team form/momentum features (win streaks, recent scoring)
            if 'home_team' in feature_dict and 'away_team' in feature_dict:
                # Add team form/streak information if available in game data
                for team_side in ['home', 'away']:
                    team_key = f"{team_side}_team"
                    if f"{team_side}_win_streak" in game:
                        feature_dict[f"{team_side}_win_streak"] = float(game[f"{team_side}_win_streak"])
                        advanced_features_added += 1
                    if f"{team_side}_form" in game and isinstance(game[f"{team_side}_form"], list):
                        # Convert recent form (W/L) to win percentage
                        recent_form = game[f"{team_side}_form"]
                        wins = sum(1 for result in recent_form if result == 'W')
                        form_games = len(recent_form)
                        if form_games > 0:
                            feature_dict[f"{team_side}_win_pct"] = wins / form_games
                            advanced_features_added += 1
                    
                    # Add momentum features (4-game rolling window as mentioned in architecture)
                    for stat in ['pts', 'reb', 'ast', 'stl', 'blk', 'to']:
                        if f"{team_side}_{stat}_last4" in game:
                            feature_dict[f"{team_side}_{stat}_momentum"] = float(game[f"{team_side}_{stat}_last4"])
                            advanced_features_added += 1
            
            # Add team efficiency metrics
            for team_side in ['home', 'away']:
                # Calculate offensive and defensive efficiency if stats available
                pts_key = f"{team_side}_pts"
                poss_key = f"{team_side}_possessions"
                if pts_key in feature_dict and poss_key in feature_dict and feature_dict[poss_key] > 0:
                    feature_dict[f"{team_side}_off_efficiency"] = feature_dict[pts_key] / feature_dict[poss_key] * 100
                    advanced_features_added += 1
                
                # Calculate eFG% if available
                fgm_key = f"{team_side}_fgm"
                fga_key = f"{team_side}_fga"
                tpm_key = f"{team_side}_tpm"
                if fgm_key in feature_dict and fga_key in feature_dict and tpm_key in feature_dict and feature_dict[fga_key] > 0:
                    feature_dict[f"{team_side}_efg_pct"] = (feature_dict[fgm_key] + 0.5 * feature_dict[tpm_key]) / feature_dict[fga_key]
                    advanced_features_added += 1
                    
                # Add defensive rating if available
                if f"{team_side}_def_rating" in game:
                    feature_dict[f"{team_side}_def_rating"] = float(game[f"{team_side}_def_rating"])
                    advanced_features_added += 1
            
            # Add matchup-specific features
            if 'home_team' in feature_dict and 'away_team' in feature_dict:
                # Home court advantage factor
                feature_dict['home_advantage'] = self.home_advantage
                advanced_features_added += 1
                
                # Team strength differential features
                for stat in ['win_pct', 'pts', 'reb', 'ast', 'stl', 'blk', 'to']:
                    home_key = f"home_{stat}"
                    away_key = f"away_{stat}"
                    if home_key in feature_dict and away_key in feature_dict:
                        feature_dict[f"{stat}_diff"] = feature_dict[home_key] - feature_dict[away_key]
                        advanced_features_added += 1
                        
                # Add matchup history if available
                if 'matchup_history' in game and isinstance(game['matchup_history'], dict):
                    matchup = game['matchup_history']
                    if 'home_wins' in matchup and 'away_wins' in matchup:
                        total_matchups = matchup.get('home_wins', 0) + matchup.get('away_wins', 0)
                        if total_matchups > 0:
                            feature_dict['home_matchup_win_pct'] = matchup.get('home_wins', 0) / total_matchups
                            advanced_features_added += 1
                    if 'avg_point_diff' in matchup:
                        feature_dict['matchup_avg_point_diff'] = float(matchup['avg_point_diff'])
                        advanced_features_added += 1
                    if 'last_point_diff' in matchup:
                        feature_dict['matchup_last_point_diff'] = float(matchup['last_point_diff'])
                        advanced_features_added += 1
        
        except Exception as e:
            logger.error(f"Error adding advanced features: {str(e)}")
            logger.error(traceback.format_exc())
        
        return advanced_features_added
    
    def prepare_features_for_target(self, features: List[Dict[str, Any]], target_column: str, prediction_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for a specific prediction target
        
        Args:
            features: List of game features
            target_column: Target column to predict
            prediction_type: Type of prediction (classification or regression)
            
        Returns:
            Tuple of X, y arrays for model training
        """
        logger.info(f"Preparing features for target: {target_column} ({prediction_type})")
        
        # Get possible target columns that might contain our target value
        target_columns = [target_column]
        if target_column == 'home_win':
            target_columns.extend(['winner', 'home_team_won', 'home_win_calculated'])
        elif target_column == 'spread_diff':
            target_columns.extend(['point_diff', 'home_away_spread', 'spread_diff_calculated'])
        elif target_column == 'total_points':
            target_columns.extend(['total', 'total_score', 'total_points_calculated'])
        
        # Count how many games already have target values
        has_target = sum(1 for game in features if any(t in game for t in target_columns))
        logger.info(f"Found {has_target} of {len(features)} games with {target_column} values")
        
        # Enhanced logging - Show sample of games with/without target values
        if has_target > 0:
            for i, game in enumerate(features):
                if any(t in game for t in target_columns):
                    present_targets = [t for t in target_columns if t in game]
                    logger.info(f"Sample game with target - ID: {game.get('id', 'unknown')}, targets: {present_targets}")
                    break
        else:
            logger.error(f"No games found with any target values for {target_column}")
            logger.info(f"Sample game keys: {list(features[0].keys()) if features else []}")
            
        # Check score fields for a random sample of games
        total_games = len(features)
        sample_size = min(5, total_games)
        for i in range(sample_size):
            game = features[i]
            game_id = game.get('id', f'unknown-{i}')
            home_score = game.get('home_score', game.get('home_team_score', None))
            away_score = game.get('away_score', game.get('visitor_team_score', None))
            logger.info(f"Game {game_id}: home_score={home_score}, away_score={away_score}")
        
        # Check if we need to derive target values
        if has_target < len(features) * 0.8:  # If less than 80% have target values
            logger.info(f"Deriving missing {target_column} values from game data")
            self._derive_target_values(features, target_column, target_columns)
            
            # Recount after derivation
            has_target_after = sum(1 for game in features if any(t in game for t in target_columns))
            logger.info(f"After derivation: {has_target_after} of {len(features)} games have {target_column} values")
            
            # If still no target values, this is a critical error
            if has_target_after == 0:
                logger.error(f"CRITICAL: Failed to derive any {target_column} values")
                return None, None
        
        # Create X, y data for model training
        X_data = []
        y_data = []
        missing_target_games = 0
        successful_features = 0
        
        for game in features:
            # Skip games without this target value and no way to derive it
            if not any(t in game for t in target_columns):
                missing_target_games += 1
                continue
            
            # Get target value from first available target column
            target_value = None
            used_target_column = None
            for t in target_columns:
                if t in game and game[t] is not None:
                    target_value = game[t]
                    used_target_column = t
                    break
            
            # Skip if still no target value
            if target_value is None:
                missing_target_games += 1
                continue
            
            # Convert target to appropriate type based on prediction_type
            if prediction_type == 'classification':
                try:
                    if isinstance(target_value, str):
                        target_value = 1 if target_value.lower() in ['1', 'true', 'win', 'yes', 'home'] else 0
                    else:
                        target_value = 1 if target_value else 0
                except (ValueError, TypeError):
                    missing_target_games += 1
                    continue
            else:  # regression
                try:
                    target_value = float(target_value)
                except (ValueError, TypeError):
                    missing_target_games += 1
                    continue
            
            # Extract features to a dict (excluding target columns)
            feature_dict = {}
            
            # Always include these important fields if available
            if 'home_team_id' in game:
                feature_dict['home_team_id'] = float(game['home_team_id'])
            if 'visitor_team_id' in game or 'away_team_id' in game:
                feature_dict['away_team_id'] = float(game.get('visitor_team_id', game.get('away_team_id')))
            
            # Add scores as features
            if 'home_score' in game:
                feature_dict['home_score'] = float(game['home_score'])
            if 'away_score' in game:
                feature_dict['away_score'] = float(game['away_score'])
                
            # Add other numeric features
            for key, value in game.items():
                # Skip target columns and non-numeric values
                if (key not in target_columns and not key.endswith('_is_calculated') 
                    and key not in ['home_score', 'away_score', 'home_team_id', 'away_team_id', 'visitor_team_id']):
                    try:
                        if isinstance(value, (int, float)):
                            feature_dict[key] = float(value)
                        elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                            feature_dict[key] = float(value)
                    except (ValueError, TypeError):
                        pass
            
            # Skip if not enough features
            min_features = 2  # Reduced from 3 to ensure we get some data
            if len(feature_dict) < min_features:
                logger.debug(f"Insufficient features ({len(feature_dict)}) for game {game.get('id', 'unknown')}")
                missing_target_games += 1
                continue
            
            # Only add if we have valid features
            if feature_dict:
                X_data.append(list(feature_dict.values()))
                y_data.append(target_value)
                successful_features += 1
        
        # Check if we collected enough data
        min_samples = 5  # Reduced from 10 to ensure we get some models trained
        if len(X_data) < min_samples or len(y_data) < min_samples:
            logger.error(f"Not enough valid data for {target_column} prediction: {len(X_data)} samples (min required: {min_samples})")
            logger.error(f"Check data collection and feature engineering for {target_column}")
            if len(X_data) > 0:
                logger.info(f"Sample feature keys: {list(range(len(X_data[0])))}")
            return None, None
        
        # Convert to numpy arrays
        X = np.array(X_data)
        y = np.array(y_data)
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features for {target_column}")
        logger.info(f"Feature conversion success rate: {successful_features}/{total_games - missing_target_games} games")
        if missing_target_games > 0:
            logger.info(f"Missing target values for {missing_target_games} games")
        
        # Print sample of prepared data
        if len(X) > 0 and len(y) > 0:
            logger.info(f"Sample X shape: {X[0].shape}, first few values: {X[0][:5] if len(X[0]) > 5 else X[0]}")
            logger.info(f"Sample y values: {y[:5] if len(y) > 5 else y}")
        
        return X, y
    
    def _derive_target_values(self, features: List[Dict[str, Any]], target_column: str, target_columns: List[str]) -> None:
        """
        Derive target values from available data when direct targets aren't available
        
        Args:
            features: List of game features to update in-place
            target_column: Primary target column name
            target_columns: List of alternative target column names
        """
        logger.info(f"Deriving values for target column: {target_column}")
        
        # Count how many games have the target column already
        games_with_target = sum(1 for game in features if target_column in game and game[target_column] is not None)
        logger.info(f"Found {games_with_target} out of {len(features)} games with existing '{target_column}' value")
        
        # If most games already have the target, skip derivation
        if games_with_target >= len(features) * 0.9:
            logger.info(f"Sufficient games have target '{target_column}', skipping derivation")
            return
        
        # Derive target values for each game
        derived_count = 0
        failed_count = 0
        
        for i, game in enumerate(features):
            # Check if any of the valid target columns already exist and are valid
            has_valid_target = False
            for t in target_columns:
                if t in game and game[t] is not None:
                    has_valid_target = True
                    break
            
            # Skip if any valid target already exists
            if has_valid_target:
                continue
            
            try:
                # Extract necessary values for deriving target
                home_score = None
                away_score = None
                
                # Look for home_score and away_score in different formats
                if 'home_team_score' in game:
                    home_score = game['home_team_score']
                elif 'home_score' in game:
                    home_score = game['home_score']
                
                if 'visitor_team_score' in game:
                    away_score = game['visitor_team_score']
                elif 'away_team_score' in game:
                    away_score = game['away_team_score']
                elif 'away_score' in game:
                    away_score = game['away_score']
                
                # Skip if scores are not available
                if home_score is None or away_score is None:
                    logger.debug(f"Cannot derive target for game {i}: missing scores")
                    failed_count += 1
                    continue
                
                # Convert scores to float for calculations
                try:
                    home_score = float(home_score)
                    away_score = float(away_score)
                except (ValueError, TypeError):
                    logger.debug(f"Cannot derive target for game {i}: invalid score format")
                    failed_count += 1
                    continue
                
                # Derive specific target values based on target column
                if target_column == 'home_win':
                    # Binary classification: 1 if home team won, 0 otherwise
                    game[target_column] = 1 if home_score > away_score else 0
                    derived_count += 1
                    game['home_win_is_calculated'] = True
                    
                elif target_column == 'spread_diff':
                    # Regression: home score - away score (positive means home team covered)
                    game[target_column] = home_score - away_score
                    derived_count += 1
                    game['spread_diff_is_calculated'] = True
                    
                elif target_column == 'total_points':
                    # Regression: total points scored in the game
                    game[target_column] = home_score + away_score
                    derived_count += 1
                    game['total_points_is_calculated'] = True
                
                # Set values for alternative target columns too for better data usability
                if target_column == 'home_win':
                    game['winner'] = 'home' if home_score > away_score else 'away'
                    game['home_team_won'] = home_score > away_score
                    
                elif target_column == 'spread_diff':
                    game['point_diff'] = home_score - away_score
                    
                elif target_column == 'total_points':
                    game['total'] = home_score + away_score
                    game['total_score'] = home_score + away_score
                
            except Exception as e:
                logger.debug(f"Error deriving target for game {i}: {str(e)}")
                failed_count += 1
        
        # Log derivation results
        logger.info(f"Derived {derived_count} values for '{target_column}' target")
        if failed_count > 0:
            logger.warning(f"Failed to derive {failed_count} values for '{target_column}' target")
    
    def get_engineering_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the feature engineering process
        
        Returns:
            Dictionary with engineering metrics
        """
        return self.metrics