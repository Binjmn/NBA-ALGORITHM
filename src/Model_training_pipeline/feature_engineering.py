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
from src.data.feature_engineering import NBAFeatureEngineer
from nba_algorithm.features.advanced_features import (
    create_momentum_features,
    create_matchup_features,
    calculate_efficiency_metrics
)

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


class FeatureEngineer:
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
        Initialize the feature engineer with configuration
        
        Args:
            config: Configuration dictionary with feature engineering settings
        """
        self.config = config
        self.engineer = NBAFeatureEngineer()
        self.window_size = config['feature_engineering']['window_size']
        self.home_advantage = config['feature_engineering']['home_advantage']
        self.use_advanced_features = config['feature_engineering']['use_advanced_features']
        
        # Initialize metrics
        self.metrics = {
            'games_processed': 0,
            'features_created': 0,
            'advanced_features_created': 0,
            'missing_data_handled': 0,
            'engineering_errors': 0
        }
        
        logger.info(f"Initialized FeatureEngineer with window_size={self.window_size}, home_advantage={self.home_advantage}")
        if self.use_advanced_features:
            logger.info("Advanced feature creation is enabled")
    
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
                    if self.use_advanced_features:
                        added_features = self._add_advanced_features(game, feature_dict)
                        advanced_feature_count = added_features
                    
                    processed_features.append(feature_dict)
                    self.metrics['games_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing game {i}: {str(e)}")
                    logger.error(traceback.format_exc())
                    self.metrics['engineering_errors'] += 1
            
            self.metrics['features_created'] = base_feature_count
            self.metrics['advanced_features_created'] = advanced_feature_count
            
            logger.info(f"Feature engineering complete. Processed {self.metrics['games_processed']} games")
            logger.info(f"Created {self.metrics['features_created']} base features and {self.metrics['advanced_features_created']} advanced features per game")
            
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
        logger.info(f"Preparing features for {target_column} prediction")
        
        X_data = []
        y_data = []
        games_processed = 0
        games_with_target = 0
        missing_target_games = 0
        missing_feature_games = 0
        
        # Check if any features exist
        if not features:
            logger.error(f"No features provided for {target_column} prediction")
            return np.array([]), np.array([])
        
        # Get a sample game to check available columns
        sample_columns = features[0].keys() if features else []
        logger.info(f"Sample columns: {list(sample_columns)[:10]}...")
        
        # Define possible target column names (for flexibility)
        target_columns = [target_column]
        if target_column == 'home_win':
            target_columns.extend(['home_team_won', 'home_winner'])
        elif target_column == 'spread':
            target_columns.extend(['point_spread', 'spread_line'])
        elif target_column == 'total':
            target_columns.extend(['over_under', 'total_points', 'total_score'])
        
        # Log if target exists in sample data
        targets_found = [col for col in target_columns if col in sample_columns]
        if targets_found:
            logger.info(f"Found target columns in data: {targets_found}")
        else:
            logger.warning(f"No direct target columns found for {target_column} in data")
            
            # Try to derive the target from available data
            self._derive_target_values(features, target_column, target_columns)
            
            # Check again if we have targets after derivation
            targets_found = [col for col in target_columns if col in features[0]]
            if targets_found:
                logger.info(f"Successfully derived target columns: {targets_found}")
            else:
                logger.error(f"Failed to derive {target_column} target from available data")
                return np.array([]), np.array([])
        
        # Process each game
        for game in features:
            games_processed += 1
            
            # Skip games without required target
            if not any(target in game for target in target_columns):
                missing_target_games += 1
                continue
            
            games_with_target += 1
            
            # Get target value from any available target column
            target_value = None
            for target in target_columns:
                if target in game:
                    target_value = game[target]
                    break
            
            # Skip if no valid target found
            if target_value is None:
                missing_target_games += 1
                continue
            
            # Convert target to appropriate type
            if prediction_type == 'classification':
                try:
                    target_value = int(target_value)
                except (ValueError, TypeError):
                    # Try to convert boolean-like values
                    if isinstance(target_value, str):
                        target_value = 1 if target_value.lower() in ['true', 'yes', 'win', '1'] else 0
                    else:
                        target_value = 1 if target_value else 0
            else:  # regression
                try:
                    target_value = float(target_value)
                except (ValueError, TypeError):
                    missing_target_games += 1
                    continue
            
            # Create feature dictionary (exclude target columns)
            feature_dict = {}
            for key, value in game.items():
                if key not in target_columns and not key.endswith('_is_calculated'):
                    # Only include numeric features
                    try:
                        if isinstance(value, (int, float)):
                            feature_dict[key] = float(value)
                        elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                            feature_dict[key] = float(value)
                    except (ValueError, TypeError):
                        pass
            
            # Skip if not enough features
            if len(feature_dict) < 3:  # Need at least 3 features for meaningful prediction
                missing_feature_games += 1
                continue
            
            # Add derived features for moneyline prediction
            if target_column == 'home_win':
                try:
                    if 'home_win_pct' in game and 'away_win_pct' in game:
                        feature_dict['win_pct_diff'] = game['home_win_pct'] - game['away_win_pct']
                    if 'home_pts_avg' in game and 'away_pts_avg' in game:
                        feature_dict['pts_avg_diff'] = game['home_pts_avg'] - game['away_pts_avg']
                    if 'home_pts_allowed_avg' in game and 'away_pts_allowed_avg' in game:
                        feature_dict['pts_allowed_avg_diff'] = game['home_pts_allowed_avg'] - game['away_pts_allowed_avg']
                except Exception:
                    logger.debug(f"Error creating derived features", exc_info=True)
            
            # Only add if we have valid features
            if feature_dict:
                X_data.append(feature_dict)
                y_data.append(target_value)
        
        # Provide detailed logging about the data preparation
        logger.info(f"Data preparation summary for {target_column}:")
        logger.info(f"  - Total games processed: {games_processed}")
        logger.info(f"  - Games with target: {games_with_target}")
        logger.info(f"  - Games missing target: {missing_target_games}")
        logger.info(f"  - Games with insufficient features: {missing_feature_games}")
        logger.info(f"  - Final dataset size: {len(X_data)} samples")
        
        # Convert to DataFrame and numpy array
        if not X_data:
            logger.error(f"No valid data found for {target_column} prediction after processing {games_processed} games")
            return np.array([]), np.array([])
        
        # Convert to DataFrame for easier processing
        X_df = pd.DataFrame(X_data)
        
        # Report on data
        logger.info(f"Feature columns for {target_column}: {list(X_df.columns)[:10]}...")
        logger.info(f"X shape: {X_df.shape}, y size: {len(y_data)}")
        
        # Fill any missing values with column means
        X_df = X_df.fillna(X_df.mean())
        # If any columns still have NaNs (all NaN), fill with 0
        X_df = X_df.fillna(0)
        
        # Convert to numpy arrays
        X = X_df.to_numpy()
        y = np.array(y_data)
        
        return X, y
    
    def _derive_target_values(self, features: List[Dict[str, Any]], target_column: str, target_columns: List[str]) -> None:
        """
        Derive target values from available data when direct targets aren't available
        
        Args:
            features: List of game features to update in-place
            target_column: Primary target column name
            target_columns: List of alternative target column names
        """
        derived_count = 0
        
        # For moneyline predictions (home win), derive from game results
        if target_column == 'home_win':
            logger.info("Deriving home_win target from game results")
            for i, game in enumerate(features):
                if 'result' in game and isinstance(game['result'], dict):
                    try:
                        if 'home_score' in game['result'] and 'away_score' in game['result']:
                            features[i]['home_win'] = 1 if game['result']['home_score'] > game['result']['away_score'] else 0
                            derived_count += 1
                    except Exception:
                        logger.debug(f"Error deriving home_win for game {i}", exc_info=True)
                        
                # If we couldn't derive from results, try from stats
                if 'home_win' not in features[i] and 'home_score' in game and 'away_score' in game:
                    try:
                        features[i]['home_win'] = 1 if game['home_score'] > game['away_score'] else 0
                        derived_count += 1
                    except Exception:
                        logger.debug(f"Error deriving home_win from scores for game {i}", exc_info=True)
        
        # For total predictions, derive from game results or odds
        elif target_column == 'total':
            logger.info("Deriving total target from game results or odds")
            for i, game in enumerate(features):
                # First try from game results
                if 'result' in game and isinstance(game['result'], dict):
                    try:
                        if 'home_score' in game['result'] and 'away_score' in game['result']:
                            features[i]['total_score'] = float(game['result']['home_score']) + float(game['result']['away_score'])
                            derived_count += 1
                    except Exception:
                        logger.debug(f"Error deriving total from results for game {i}", exc_info=True)
                
                # If not from results, try from stats
                if 'total_score' not in features[i] and 'home_score' in game and 'away_score' in game:
                    try:
                        features[i]['total_score'] = float(game['home_score']) + float(game['away_score'])
                        derived_count += 1
                    except Exception:
                        logger.debug(f"Error deriving total from scores for game {i}", exc_info=True)
                        
                # If not from scores, try from odds
                if 'total_score' not in features[i] and 'odds' in game and isinstance(game['odds'], dict) and 'totals' in game['odds']:
                    try:
                        totals_data = game['odds']['totals']
                        if isinstance(totals_data, dict) and 'points' in totals_data:
                            features[i]['total'] = float(totals_data['points'])
                            derived_count += 1
                        elif isinstance(totals_data, list) and len(totals_data) > 0:
                            for total in totals_data:
                                if isinstance(total, dict) and 'points' in total:
                                    features[i]['total'] = float(total['points'])
                                    derived_count += 1
                                    break
                    except Exception:
                        logger.debug(f"Error deriving total from odds for game {i}", exc_info=True)
        
        # For spread predictions, derive from game results or odds
        elif target_column == 'spread':
            logger.info("Deriving spread target from game results or odds")
            for i, game in enumerate(features):
                # First try from odds
                if 'odds' in game and isinstance(game['odds'], dict) and 'spreads' in game['odds']:
                    try:
                        spreads_data = game['odds']['spreads']
                        if isinstance(spreads_data, dict) and 'points' in spreads_data:
                            features[i]['spread'] = float(spreads_data['points'])
                            derived_count += 1
                        elif isinstance(spreads_data, list) and len(spreads_data) > 0:
                            for spread in spreads_data:
                                if isinstance(spread, dict) and 'points' in spread:
                                    features[i]['spread'] = float(spread['points'])
                                    derived_count += 1
                                    break
                    except Exception:
                        logger.debug(f"Error deriving spread from odds for game {i}", exc_info=True)
                
                # If not from odds, try to calculate actual spread from results
                if 'spread' not in features[i] and 'result' in game and isinstance(game['result'], dict):
                    try:
                        if 'home_score' in game['result'] and 'away_score' in game['result']:
                            # Convention: positive spread means home team is favored
                            # Actual spread will be away_score - home_score
                            features[i]['spread'] = float(game['result']['away_score']) - float(game['result']['home_score'])
                            derived_count += 1
                    except Exception:
                        logger.debug(f"Error deriving spread from results for game {i}", exc_info=True)
        
        logger.info(f"Derived {derived_count} target values for {target_column}")
    
    def get_engineering_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the feature engineering process
        
        Returns:
            Dictionary with engineering metrics
        """
        return self.metrics