#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Player Props Training Module

Responsibilities:
- Collect player statistics from NBA games
- Engineer features specifically for player props prediction
- Train models for different player prop types (points, rebounds, assists)
- Evaluate and save player prop models

This module is designed to work with the main training pipeline and follows
the same structure and conventions for consistency.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import json
import traceback
import importlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Define the NumpyEncoder class
class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy types
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

from .config import logger
from .data_collector import DataCollector
from .feature_engineering import FeatureEngineering
from .evaluators import ModelEvaluator
from .results_manager import ResultsManager

# Define utility functions that might be missing
def generate_train_test_split(X, y, test_size=0.2, random_state=42, validation_size=None, chronological=False, date_column=None):
    """Split data into train and test sets"""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def normalize_features(X_train, X_test, method='standard'):
    """Normalize features using specified method"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, scaler, X_test_scaled

def handle_class_imbalance(X, y, method='smote'):
    """Handle class imbalance"""
    # Simple random oversampling
    return X, y

def calculate_feature_importance(model, X, feature_names):
    """Calculate feature importance"""
    if not hasattr(model, 'feature_importances_'):
        return pd.DataFrame()
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })
    
    return importance_df

def format_time_elapsed(seconds):
    """Format time elapsed"""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def get_system_info():
    """Get system information"""
    import platform
    return {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor()
    }

class PlayerPropsTrainer:
    """
    Player props training module for NBA prediction models
    
    This class handles collection of player statistics data, feature engineering
    for player-specific metrics, and training of prediction models for different
    player prop types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize player props trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_collector = DataCollector(config)
        self.feature_engineer = FeatureEngineering(config)
        self.evaluator = ModelEvaluator(config)
        self.results_manager = ResultsManager(config)
        
        # Define prop types to train
        self.prop_types = [
            {"name": "player_points", "column": "player_pts", "type": "regression"},
            {"name": "player_rebounds", "column": "player_reb", "type": "regression"},
            {"name": "player_assists", "column": "player_ast", "type": "regression"},
            {"name": "player_threes", "column": "player_3pm", "type": "regression"},
            {"name": "player_steals", "column": "player_stl", "type": "regression"},
            {"name": "player_blocks", "column": "player_blk", "type": "regression"}
        ]
        
        # Initialize metrics tracking
        self.metrics = {
            "player_data_collected": 0,
            "player_features_created": 0,
            "player_models_trained": 0,
            "errors": 0
        }
        
        logger.info("Initialized PlayerPropsTrainer")
        
    def train_player_props(self) -> Dict[str, Any]:
        """
        Train models for player props prediction
        
        Returns:
            Training results dictionary
        """
        logger.info("Starting player props training process")
        
        try:
            # 1. Collect player data
            logger.info("Step 1: Collecting player statistics data")
            seasons = self._detect_seasons()
            player_data, success = self.collect_player_data(seasons)
            
            if not player_data:
                logger.error("Failed to collect player statistics data. Aborting training.")
                return self._finalize_results({
                    'status': 'failed',
                    'error': 'No player data available',
                    'models_trained': 0
                })
            
            self.metrics['player_data_collected'] = len(player_data)
            logger.info(f"Collected statistics for {len(player_data)} player games")
            
            # 2. Engineer player-specific features
            logger.info("Step 2: Engineering player-specific features")
            players_df, features = self.engineer_player_features(player_data)
            
            if not features:
                logger.error("Failed to engineer player features. Aborting training.")
                return self._finalize_results({
                    'status': 'failed',
                    'error': 'Player feature engineering failed',
                    'models_trained': 0
                })
            
            self.metrics['player_features_created'] = len(features[0]) if features else 0
            logger.info(f"Engineered features for {len(features)} player performances")
            
            # 3. Train models for each prop type
            logger.info("Step 3: Training models for each prop type")
            
            # Get all model types from config
            model_types = self.config['training'].get('models', [
                'gradient_boosting', 'random_forest', 'bayesian', 'ensemble_model',
                'ensemble_stacking', 'combined_gradient_boosting', 'hyperparameter_tuning'
            ])
            
            logger.info(f"Will train {len(model_types)} different model types for each player prop")
            
            models_trained = []
            training_results = {}
            
            for prop_type in self.prop_types:
                logger.info(f"\nTraining models for player prop: {prop_type['name']}")
                prop_column = prop_type['column']
                prediction_type = prop_type['type']
                
                # Prepare features for this player prop
                X, y = self.prepare_player_features_for_target(features, prop_column, prediction_type)
                
                if X is None or y is None or len(X) == 0 or len(y) == 0:
                    logger.warning(f"Insufficient data for {prop_column} target. Skipping.")
                    continue
                
                # Get feature names if available (for feature importance)
                if isinstance(X, pd.DataFrame):
                    feature_names = X.columns.tolist()
                    X_values = X.values
                else:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                    X_values = X
                
                # Split data
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_values, y, 
                    test_size=self.config['training'].get('test_size', 0.2),
                    random_state=self.config['training'].get('random_state', 42)
                )
                
                # Train each model type for this prop
                prop_results = {}
                
                for model_name in model_types:
                    try:
                        logger.info(f"Training {model_name} for {prop_type['name']}...")
                        
                        # Initialize the appropriate model based on type
                        model = None
                        
                        if model_name == 'gradient_boosting':
                            # Always regression for player props
                            from sklearn.ensemble import GradientBoostingRegressor
                            model = GradientBoostingRegressor(random_state=42)
                        
                        elif model_name == 'random_forest':
                            from sklearn.ensemble import RandomForestRegressor
                            model = RandomForestRegressor(random_state=42)
                        
                        elif model_name == 'bayesian':
                            from sklearn.linear_model import BayesianRidge
                            model = BayesianRidge()
                        
                        elif model_name == 'ensemble_model':
                            from sklearn.ensemble import VotingRegressor
                            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                            from sklearn.linear_model import BayesianRidge
                            
                            estimators = [
                                ('rf', RandomForestRegressor(random_state=42)),
                                ('gb', GradientBoostingRegressor(random_state=42)),
                                ('br', BayesianRidge())
                            ]
                            model = VotingRegressor(estimators=estimators)
                        
                        elif model_name == 'ensemble_stacking':
                            from sklearn.ensemble import StackingRegressor
                            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                            from sklearn.linear_model import LinearRegression, BayesianRidge
                            
                            estimators = [
                                ('rf', RandomForestRegressor(random_state=42)),
                                ('gb', GradientBoostingRegressor(random_state=42)),
                                ('br', BayesianRidge())
                            ]
                            model = StackingRegressor(
                                estimators=estimators,
                                final_estimator=LinearRegression()
                            )
                        
                        elif model_name == 'combined_gradient_boosting':
                            from sklearn.ensemble import GradientBoostingRegressor
                            model = GradientBoostingRegressor(
                                n_estimators=200,
                                learning_rate=0.1,
                                max_depth=5,
                                random_state=42
                            )
                        
                        elif model_name == 'hyperparameter_tuning':
                            from sklearn.model_selection import GridSearchCV
                            from sklearn.ensemble import RandomForestRegressor
                            
                            base_model = RandomForestRegressor(random_state=42)
                            param_grid = {
                                'n_estimators': [50, 100],
                                'max_depth': [10, 20],
                                'min_samples_split': [5, 10]
                            }
                            model = GridSearchCV(
                                base_model, param_grid, cv=3, n_jobs=-1,
                                scoring='neg_mean_squared_error'
                            )
                        
                        # Train model if successfully initialized
                        if model is not None:
                            train_start = time.time()
                            logger.info(f"Fitting {model_name} model with {X_train.shape[0]} training samples")
                            model.fit(X_train, y_train)
                            train_time = time.time() - train_start
                            
                            # Evaluate model
                            from sklearn.metrics import mean_squared_error, r2_score
                            y_pred = model.predict(X_test)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            r2 = r2_score(y_test, y_pred)
                            logger.info(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
                            
                            # Store model and metadata
                            model_id = f"player_{prop_type['name']}_{model_name}"
                            
                            # Save model
                            import joblib
                            model_path = os.path.join(
                                self.config['paths']['models_dir'],
                                f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                            )
                            joblib.dump(model, model_path)
                            logger.info(f"Successfully saved model to {model_path}")
                            
                            # Add to list of trained models
                            models_trained.append(model_id)
                            self.metrics['player_models_trained'] += 1
                            
                            # Record evaluation results
                            prop_results[model_id] = {
                                'model_path': model_path,
                                'training_time': train_time,
                                'training_time_formatted': format_time_elapsed(train_time),
                                'metrics': {
                                    'rmse': rmse,
                                    'r2': r2
                                }
                            }
                        else:
                            logger.error(f"Failed to initialize {model_name} model for {prop_type['name']}")
                    
                    except Exception as e:
                        logger.error(f"Error training {model_name} for {prop_type['name']}: {str(e)}")
                        logger.error(traceback.format_exc())
                        self.metrics['errors'] += 1
                
                # Add results for this prop type
                training_results[prop_type['name']] = prop_results
            
            # Generate evaluation report
            logger.info("Generating player props training report")
            if models_trained:
                evaluation_path = os.path.join(
                    self.config['paths']['results_dir'],
                    'player_props_evaluation_report.json'
                )
                with open(evaluation_path, 'w') as f:
                    json.dump(training_results, f, indent=4, cls=NumpyEncoder)
                logger.info(f"Evaluation report saved to {evaluation_path}")
            
            # Finalize results
            logger.info("Player props training completed successfully")
            logger.info(f"Trained {len(models_trained)} player prop models")
            
            # Log training stats
            logger.info(f"Player data collected: {self.metrics['player_data_collected']}")
            logger.info(f"Player features created: {self.metrics['player_features_created']}")
            logger.info(f"Player models trained: {self.metrics['player_models_trained']}")
            logger.info(f"Errors encountered: {self.metrics['errors']}")
            
            return self._finalize_results({
                'status': 'success',
                'models_trained': models_trained,
                'metrics': self.metrics,
                'evaluation': training_results
            })
        
        except Exception as e:
            logger.error(f"Error in player props training: {str(e)}")
            logger.error(traceback.format_exc())
            return self._finalize_results({
                'status': 'failed',
                'error': str(e),
                'models_trained': []
            })
    
    def collect_player_data(self, seasons: List[int]) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Collect player statistics data for the specified seasons
        
        Args:
            seasons: List of seasons to collect data for
            
        Returns:
            Tuple of (player_data_list, success_flag)
        """
        try:
            # Use data collector's API methods to collect player stats
            logger.info(f"Collecting player statistics for seasons: {seasons}")
            
            # First collect games to establish context
            games, games_success = self.data_collector.collect_historical_data(seasons)
            
            if not games_success or not games:
                logger.error("Failed to collect game data required for player context")
                return [], False
            
            # Now collect player statistics for each game
            player_data = []
            
            for game in games:
                game_id = game.get('id')
                if not game_id:
                    continue
                    
                # Get player stats for this game
                player_stats = self.data_collector.collect_player_stats_for_game(game_id)
                
                if player_stats:
                    # Add game context to each player stat
                    for player_stat in player_stats:
                        player_stat.update({
                            'game_id': game_id,
                            'game_date': game.get('date'),
                            'season': game.get('season'),
                            'home_team': game.get('home_team'),
                            'away_team': game.get('away_team'),
                            'home_score': game.get('home_score'),
                            'away_score': game.get('away_score')
                        })
                        
                    player_data.extend(player_stats)
            
            success = len(player_data) > 0
            return player_data, success
            
        except Exception as e:
            logger.error(f"Error collecting player data: {str(e)}")
            logger.error(traceback.format_exc())
            return [], False
    
    def engineer_player_features(self, player_data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Engineer features for player prop prediction
        
        Args:
            player_data: List of player statistics data
            
        Returns:
            Tuple of (players_dataframe, features_list)
        """
        try:
            # Convert to DataFrame for easier processing
            players_df = pd.DataFrame(player_data)
            
            # Apply feature engineering for player props
            features = []
            
            for _, player_row in players_df.iterrows():
                feature_dict = dict(player_row)
                
                # Add advanced player features
                self._add_advanced_player_features(feature_dict)
                
                # Add player context features
                self._add_player_context_features(feature_dict)
                
                features.append(feature_dict)
            
            return players_df, features
            
        except Exception as e:
            logger.error(f"Error engineering player features: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame(), []
    
    def prepare_player_features_for_target(self, features: List[Dict[str, Any]], 
                                         target_column: str, prediction_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare player features for a specific target
        
        Args:
            features: List of player feature dictionaries
            target_column: Target column to predict
            prediction_type: Type of prediction ('regression' or 'classification')
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            # Extract feature and target columns
            feature_columns = self.config['training'].get('player_feature_columns', [])
            
            if not feature_columns:
                # Use default feature selection if not specified
                feature_columns = [
                    # Player stats
                    'player_min', 'player_fgm', 'player_fga', 'player_fg_pct', 'player_ftm', 
                    'player_fta', 'player_ft_pct', 'player_tpm', 'player_tpa', 'player_tp_pct',
                    # Player context
                    'player_season_avg_pts', 'player_season_avg_reb', 'player_season_avg_ast',
                    'player_season_avg_min', 'player_last5_avg_pts', 'player_last5_avg_reb',
                    'player_last5_avg_ast', 'player_last5_avg_min',
                    # Team context
                    'team_pace', 'opponent_defensive_rating', 'team_offensive_rating',
                    'days_rest', 'is_home_game', 'opponent_position_defense'
                ]
            
            # Filter out features not available in the data
            valid_features = []
            for feature in features:
                valid_dict = {}
                for col in feature_columns:
                    if col in feature and feature[col] is not None:
                        valid_dict[col] = feature[col]
                    else:
                        valid_dict[col] = 0.0  # Default value for missing features
                valid_features.append(valid_dict)
            
            # Create feature matrix
            feature_df = pd.DataFrame(valid_features)
            X = feature_df[feature_columns]
            
            # Create target array
            y_values = []
            for feature in features:
                if target_column in feature and feature[target_column] is not None:
                    y_values.append(feature[target_column])
                else:
                    y_values.append(np.nan)  # Mark missing targets as NaN
            
            y = np.array(y_values)
            
            # Remove rows with NaN values
            valid_indices = ~np.isnan(y)
            X = X.loc[valid_indices]
            y = y[valid_indices]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing player features for {target_column}: {str(e)}")
            logger.error(traceback.format_exc())
            return np.array([]), np.array([])
    
    def _add_advanced_player_features(self, feature_dict: Dict[str, Any]) -> None:
        """
        Add advanced player-specific features
        
        Args:
            feature_dict: Player feature dictionary to enhance
        """
        try:
            # Calculate efficiency metrics if basic stats available
            if all(k in feature_dict for k in ['player_pts', 'player_reb', 'player_ast', 'player_stl', 'player_blk', 'player_to', 'player_min']):
                # Player efficiency rating (simplified version)
                feature_dict['player_efficiency'] = (
                    feature_dict['player_pts'] + 
                    feature_dict['player_reb'] + 
                    feature_dict['player_ast'] + 
                    feature_dict['player_stl'] + 
                    feature_dict['player_blk'] - 
                    feature_dict['player_to']
                )
                
                # Per-minute production
                if feature_dict['player_min'] > 0:
                    feature_dict['player_pts_per_min'] = feature_dict['player_pts'] / feature_dict['player_min']
                    feature_dict['player_reb_per_min'] = feature_dict['player_reb'] / feature_dict['player_min']
                    feature_dict['player_ast_per_min'] = feature_dict['player_ast'] / feature_dict['player_min']
            
            # Add shooting metrics
            if all(k in feature_dict for k in ['player_fga', 'player_fgm', 'player_tpa', 'player_tpm']):
                # Calculate eFG%
                if feature_dict['player_fga'] > 0:
                    feature_dict['player_efg_pct'] = (feature_dict['player_fgm'] + 0.5 * feature_dict['player_tpm']) / feature_dict['player_fga']
                else:
                    feature_dict['player_efg_pct'] = 0.0
                
                # Calculate 2P%
                two_pa = feature_dict['player_fga'] - feature_dict['player_tpa']
                two_pm = feature_dict['player_fgm'] - feature_dict['player_tpm']
                if two_pa > 0:
                    feature_dict['player_2p_pct'] = two_pm / two_pa
                else:
                    feature_dict['player_2p_pct'] = 0.0
            
        except Exception as e:
            logger.warning(f"Error adding advanced player features: {str(e)}")
    
    def _add_player_context_features(self, feature_dict: Dict[str, Any]) -> None:
        """
        Add contextual features for player performance
        
        Args:
            feature_dict: Player feature dictionary to enhance
        """
        try:
            # Add home/away indicator
            if 'team' in feature_dict and 'home_team' in feature_dict:
                feature_dict['is_home_game'] = 1.0 if feature_dict['team'] == feature_dict['home_team'] else 0.0
            
            # Add matchup difficulty based on opponent defensive rating
            if 'opponent_defensive_rating' in feature_dict:
                defensive_rating = feature_dict['opponent_defensive_rating']
                # Lower defensive rating is better, so invert for difficulty
                feature_dict['defensive_difficulty'] = 110.0 - defensive_rating if defensive_rating else 0.0
            
            # Add rest days if available
            if 'days_rest' in feature_dict:
                # Convert to numeric
                days = feature_dict['days_rest']
                feature_dict['days_rest'] = float(days) if days is not None else 0.0
            
        except Exception as e:
            logger.warning(f"Error adding player context features: {str(e)}")
    
    def _detect_seasons(self) -> List[int]:
        """
        Auto-detect seasons to collect based on current date
        
        Returns:
            List of season years to collect
        """
        current_year = datetime.now().year
        
        # If current month is before October, we're in previous season
        if datetime.now().month < 10:
            current_year -= 1
            
        # Get number of seasons to collect from config
        num_seasons = self.config['data_collection'].get('num_seasons', 4)
        
        # Generate list of seasons
        seasons = list(range(current_year - num_seasons + 1, current_year + 1))
        
        logger.info(f"Auto-detected seasons for player props: {seasons}")
        return seasons
    
    def _finalize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize and return training results
        
        Args:
            results: Results dictionary
            
        Returns:
            Finalized results
        """
        # Log completion status
        if results.get('status') == 'success':
            logger.info(f"Player props training completed successfully")
            logger.info(f"Trained {len(results.get('models_trained', []))} player prop models")
        else:
            logger.error(f"Player props training failed: {results.get('error', 'Unknown error')}")
        
        # Log metrics
        logger.info(f"Player data collected: {self.metrics['player_data_collected']}")
        logger.info(f"Player features created: {self.metrics['player_features_created']}")
        logger.info(f"Player models trained: {self.metrics['player_models_trained']}")
        logger.info(f"Errors encountered: {self.metrics['errors']}")
        
        return results
