#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Random Forest for NBA Moneyline and Player Prop Predictions

This module implements an enhanced Random Forest model optimized for both
moneyline predictions and player statistics (points, assists, rebounds).
It features trend analysis, time-based validation, and ensemble capabilities.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
import os
import joblib
import warnings

from src.models.base_model import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest Classifier and Regressor for predicting NBA game winners (moneyline) and player statistics
    
    Features:
    - Uses team performance metrics to predict home/away win
    - Produces probability outputs for confidence scoring
    - Handles categorical features through encoding
    - Includes hyperparameter optimization
    - Cross-validation for robust performance evaluation
    - Supports player statistics predictions (points, assists, rebounds)
    """
    
    def __init__(self, version: int = 1, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Random Forest model
        
        Args:
            version: Model version number
            params: Optional model parameters for RandomForestClassifier and RandomForestRegressor
        """
        super().__init__(name="RandomForest", model_type="classification", version=version)
        
        # Default parameters (will be tuned during training if not provided)
        self.params = params or {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 4,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }
        
        # Initialize the model
        self.classifier = RandomForestClassifier(**self.params)
        self.regressor = RandomForestRegressor(**self.params)
        
        # Feature preprocessing
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
    
    def _preprocess_features(self, X: pd.DataFrame, fit: bool = False, task: str = 'classification') -> pd.DataFrame:
        """
        Preprocess features with enhanced engineering for player prop predictions
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler (training) or just transform (prediction)
            task: Type of task, either 'classification' or 'regression'
            
        Returns:
            Processed feature matrix
        """
        if X.empty:
            logger.error("Cannot preprocess empty feature matrix")
            raise ValueError("Feature matrix is empty")
            
        # Store original feature names
        feature_names = X.columns.tolist()
        
        # Add trend features if time-related columns exist
        X = self._add_trend_features(X, task)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        if task == 'classification':
            scaler = self.scaler
        else:  # For regression tasks like player stats
            scaler = self.robust_scaler
            
        if fit:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def _add_trend_features(self, X: pd.DataFrame, task: str = 'classification') -> pd.DataFrame:
        """
        Add trend-based features to improve predictive power
        
        Args:
            X: Feature matrix
            task: Type of task, either 'classification' or 'regression'
            
        Returns:
            Enhanced feature matrix with trend features
        """
        # Create a copy to avoid modifying the original
        X_enhanced = X.copy()
        
        try:
            # For classification (moneyline predictions)
            if task == 'classification':
                # Check for suitable columns for trend creation
                form_columns = [col for col in X.columns if any(term in col.lower() for 
                            term in ['rate', 'avg', 'points', 'win', 'streak'])]
                
                if form_columns:
                    # Create ratio features for home vs away metrics
                    for col in form_columns:
                        if 'home_' in col and 'away_' + col[5:] in X.columns:
                            home_col = col
                            away_col = 'away_' + col[5:]
                            ratio_name = f"ratio_{col[5:]}"
                            
                            # Add ratio as a feature (with handling for zero division)
                            home_vals = X[home_col].replace(0, 0.001)  # Avoid division by zero
                            away_vals = X[away_col].replace(0, 0.001)
                            X_enhanced[ratio_name] = home_vals / away_vals
                    
                    # Add trend direction indicators for teams
                    if 'home_win_rate_5g' in X.columns and 'home_win_rate_10g' in X.columns:
                        X_enhanced['home_trend'] = np.where(
                            X['home_win_rate_5g'] > X['home_win_rate_10g'], 1, 
                            np.where(X['home_win_rate_5g'] < X['home_win_rate_10g'], -1, 0)
                        )
                    
                    if 'away_win_rate_5g' in X.columns and 'away_win_rate_10g' in X.columns:
                        X_enhanced['away_trend'] = np.where(
                            X['away_win_rate_5g'] > X['away_win_rate_10g'], 1, 
                            np.where(X['away_win_rate_5g'] < X['away_win_rate_10g'], -1, 0)
                        )
            
            # For regression (player stat predictions)
            elif task == 'regression':
                # Check for player-specific columns
                player_cols = [col for col in X.columns if 'player_' in col]
                stat_cols = [col for col in X.columns if any(stat in col for stat in 
                              ['points', 'assists', 'rebounds', 'blocks', 'steals'])]
                
                if player_cols:
                    # Add interaction terms between player stats and defense ratings
                    if any('defensive' in col for col in X.columns):
                        defense_cols = [col for col in X.columns if 'defensive' in col]
                        
                        for p_col in player_cols:
                            for d_col in defense_cols:
                                if 'player_points' in p_col and 'defensive' in d_col:
                                    # Create interaction feature between player scoring and defense rating
                                    X_enhanced[f"{p_col}_vs_{d_col}"] = X[p_col] / (X[d_col] + 0.001)  # Avoid division by zero
                    
                    # Add trend indicators for player performance
                    for trend_col in [col for col in X.columns if 'trend' in col]:
                        if trend_col in X.columns:
                            # Emphasize trend direction as a feature
                            X_enhanced[f"{trend_col}_direction"] = np.sign(X[trend_col])
                    
                    # Add recency bias features - more weight to recent performance
                    for recent_col in [col for col in X.columns if 'recent_' in col]:
                        if recent_col in X.columns and recent_col.replace('recent_', '') in X.columns:
                            # Calculate ratio of recent to season average
                            season_col = recent_col.replace('recent_', '')
                            if X[season_col].mean() > 0:  # Avoid division by zero
                                X_enhanced[f"{recent_col}_ratio"] = X[recent_col] / (X[season_col] + 0.001)
                
                # Home vs away performance indicators
                if 'is_home_team' in X.columns:
                    home_indicator = X['is_home_team']
                    # Find stats that have home/away versions
                    for col in stat_cols:
                        if 'home_' in col and col.replace('home_', 'away_') in X.columns:
                            home_stat = X[col]
                            away_stat = X[col.replace('home_', 'away_')]
                            # Create home advantage feature
                            X_enhanced[f"home_advantage_{col[5:]}"] = home_stat - away_stat
        
        except Exception as e:
            logger.warning(f"Error adding trend features: {str(e)}")
            # Return original data if feature creation fails
            return X
        
        return X_enhanced

    def train(self, X: pd.DataFrame, y: pd.Series, task: str = 'classification') -> None:
        """
        Train the model with hyperparameter optimization
        
        Args:
            X: Feature matrix
            y: Target variable (1 for home win, 0 for away win) or player statistics
            task: Type of task, either 'classification' or 'regression'
        """
        logger.info(f"Training Random Forest model (version {self.version})")
        
        if X.empty or y.empty:
            logger.error("Cannot train model with empty data")
            raise ValueError("Training data is empty")
        
        try:
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Preprocess features
            X_processed = self._preprocess_features(X, fit=True, task=task)
            
            # Perform hyperparameter tuning if not already tuned
            if not self.is_trained:
                logger.info("Performing hyperparameter optimization")
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 4, 6],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                # Use stratified k-fold to handle potential class imbalance
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                # Grid search with cross-validation
                if task == 'classification':
                    grid_search = GridSearchCV(
                        estimator=RandomForestClassifier(random_state=42),
                        param_grid=param_grid,
                        cv=cv,
                        scoring='accuracy',
                        n_jobs=-1  # Use all available cores
                    )
                elif task == 'regression':
                    grid_search = GridSearchCV(
                        estimator=RandomForestRegressor(random_state=42),
                        param_grid=param_grid,
                        cv=cv,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1  # Use all available cores
                    )
                
                grid_search.fit(X_processed, y)
                
                # Get best parameters
                self.params = grid_search.best_params_
                logger.info(f"Best parameters: {self.params}")
                
                # Update model with best parameters
                if task == 'classification':
                    self.classifier = RandomForestClassifier(**self.params)
                elif task == 'regression':
                    self.regressor = RandomForestRegressor(**self.params)
            
            # Train the model with the best parameters
            if task == 'classification':
                self.classifier.fit(X_processed, y)
            elif task == 'regression':
                self.regressor.fit(X_processed, y)
            
            # Update model metadata
            self.is_trained = True
            self.trained_at = datetime.now(timezone.utc)
            
            logger.info(f"Random Forest model trained successfully with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
            raise ValueError(f"Failed to train model: {str(e)}")
    
    def train_for_player_props(self, X: pd.DataFrame, y: pd.Series, prop_type: str, use_time_series: bool = False) -> None:
        """
        Train the model specifically for player prop predictions
        
        Args:
            X: Feature matrix
            y: Target variable (player statistic values)
            prop_type: Type of prop to predict ('points', 'assists', 'rebounds')
            use_time_series: Whether to use time series cross-validation
        """
        logger.info(f"Training Random Forest model for {prop_type} predictions (version {self.version})")
        
        if X.empty or y.empty:
            logger.error("Cannot train model with empty data")
            raise ValueError("Training data for player props is empty")
        
        try:
            # Set model type to regression for player props
            self.model_type = "regression"
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Preprocess features
            X_processed = self._preprocess_features(X, fit=True, task='regression')
            
            # Hyperparameter grid specific for player prop prediction
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 12, 16, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Choose cross-validation strategy
            if use_time_series:
                cv = TimeSeriesSplit(n_splits=5)
                logger.info("Using TimeSeriesSplit for temporal validation")
            else:
                cv = 5  # Regular k-fold CV
            
            # Grid search with cross-validation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid_search = GridSearchCV(
                    estimator=RandomForestRegressor(random_state=42),
                    param_grid=param_grid,
                    cv=cv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1
                )
                grid_search.fit(X_processed, y)
            
            # Get best parameters
            best_params = grid_search.best_params_
            logger.info(f"Best parameters for {prop_type} prediction: {best_params}")
            
            # Update regressor with best parameters
            self.regressor = RandomForestRegressor(**best_params)
            
            # Train with best parameters
            self.regressor.fit(X_processed, y)
            
            # Calculate training metrics
            train_preds = self.regressor.predict(X_processed)
            mae = mean_absolute_error(y, train_preds)
            logger.info(f"Training MAE for {prop_type}: {mae:.4f}")
            
            # Store metadata
            self.prop_type = prop_type
            self.params = best_params
            self.is_trained = True
            self.trained_at = datetime.now(timezone.utc)
            
            logger.info(f"Random Forest model trained successfully for {prop_type} prediction with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training Random Forest model for {prop_type}: {str(e)}")
            raise ValueError(f"Failed to train model for {prop_type}: {str(e)}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (1 for home win, 0 for away win)
        """
        if not self.is_trained:
            logger.error("Cannot predict with untrained model")
            raise ValueError("Model is not trained")
        
        if X.empty:
            logger.error("Cannot predict with empty feature matrix")
            raise ValueError("Prediction data is empty")
            
        try:
            # Ensure all expected features are present
            missing_features = [f for f in self.feature_names if f not in X.columns]
            if missing_features:
                logger.error(f"Missing features in prediction data: {missing_features}")
                raise ValueError(f"Missing required features for prediction: {missing_features}")
            
            # Preprocess features
            X_processed = self._preprocess_features(X, fit=False)
            
            # Make predictions
            if self.model_type == 'classification':
                return self.classifier.predict(X_processed)
            else:  # regression
                return self.regressor.predict(X_processed)
            
        except Exception as e:
            logger.error(f"Error making predictions with Random Forest model: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input samples
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of class probabilities
        """
        if not self.is_trained:
            logger.error("Cannot predict with untrained model")
            raise ValueError("Model is not trained")
        
        if self.model_type != 'classification':
            logger.error("predict_proba is only available for classification models")
            raise ValueError("Cannot call predict_proba on a regression model")
        
        if X.empty:
            logger.error("Cannot predict with empty feature matrix")
            raise ValueError("Prediction data is empty")
            
        try:
            # Ensure all expected features are present
            missing_features = [f for f in self.feature_names if f not in X.columns]
            if missing_features:
                logger.error(f"Missing features in prediction data: {missing_features}")
                raise ValueError(f"Missing required features for prediction: {missing_features}")
            
            # Preprocess features
            X_processed = self._preprocess_features(X, fit=False)
            
            # Predict probabilities
            return self.classifier.predict_proba(X_processed)
            
        except Exception as e:
            logger.error(f"Error predicting probabilities with Random Forest model: {str(e)}")
            raise RuntimeError(f"Probability prediction failed: {str(e)}")
    
    def predict_player_stat(self, X: pd.DataFrame, prop_type: str = None) -> np.ndarray:
        """
        Make predictions for player statistical performance
        
        Args:
            X: Feature matrix
            prop_type: Type of prop to predict (uses self.prop_type if None)
            
        Returns:
            Array of predicted statistical values
        """
        if not self.is_trained:
            logger.error("Cannot predict with untrained model")
            raise ValueError("Model is not trained")
        
        if X.empty:
            logger.error("Cannot predict with empty feature matrix")
            raise ValueError("Prediction data is empty")
            
        # Check if model is trained for player props
        if self.model_type != 'regression' or not hasattr(self, 'prop_type'):
            logger.error("Model is not trained for player prop predictions")
            raise ValueError("Model is not configured for player prop predictions")
        
        # Use stored prop type if none provided
        if prop_type is None:
            prop_type = self.prop_type
        
        try:
            # Ensure all expected features are present
            missing_features = [f for f in self.feature_names if f not in X.columns]
            if missing_features:
                logger.error(f"Missing features in player prop prediction data: {missing_features}")
                raise ValueError(f"Missing required features for {prop_type} prediction: {missing_features}")
            
            # Preprocess features
            X_processed = self._preprocess_features(X, fit=False, task='regression')
            
            # Make predictions
            predictions = self.regressor.predict(X_processed)
            
            # Ensure predictions are non-negative (player stats can't be negative)
            predictions = np.maximum(predictions, 0)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting {prop_type} with Random Forest model: {str(e)}")
            raise RuntimeError(f"Player stat prediction failed: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance from the trained model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            logger.error("Cannot get feature importance from untrained model")
            return {}
        
        try:
            # Get the trained model
            if self.model_type == 'classification':
                model = self.classifier
            else:  # regression
                model = self.regressor
            
            # Extract feature importance
            importances = model.feature_importances_
            
            # Map to feature names
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                feature_importance[feature] = float(importances[i])
            
            # Sort by importance (highest to lowest)
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def calculate_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate prediction uncertainty using random forest's internal variance
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of uncertainty values for each prediction
        """
        if not self.is_trained:
            logger.error("Cannot calculate uncertainty with untrained model")
            return np.array([])
        
        if X.empty:
            logger.error("Cannot calculate uncertainty with empty feature matrix")
            return np.array([])
            
        try:
            # Preprocess features
            X_processed = self._preprocess_features(X, fit=False, task=self.model_type)
            
            # Get predictions from all trees in the forest
            if self.model_type == 'classification':
                model = self.classifier
                # For classification, get std dev of probability predictions
                predictions = np.array([tree.predict_proba(X_processed) for tree in model.estimators_])
                # Standard deviation across trees (higher means more uncertainty)
                uncertainty = np.std(predictions, axis=0)
                # Take average across classes for a single uncertainty value per sample
                uncertainty = np.mean(uncertainty, axis=1)
            else:  # regression
                model = self.regressor
                # For regression, get std dev of value predictions
                predictions = np.array([tree.predict(X_processed) for tree in model.estimators_])
                # Standard deviation across trees
                uncertainty = np.std(predictions, axis=0)
            
            return uncertainty
            
        except Exception as e:
            logger.error(f"Error calculating prediction uncertainty: {str(e)}")
            return np.array([])
    
    def record_performance(self, metrics: Dict[str, float], prediction_type: str, num_predictions: int, time_window: str = '7d') -> bool:
        """
        Record the performance metrics of the model for a specific prediction type and time window
        
        Args:
            metrics: Dictionary of performance metrics (accuracy, precision, recall, f1, etc.)
            prediction_type: Type of prediction (moneyline, spread, total, etc.)
            num_predictions: Number of predictions made
            time_window: Time window for the metrics (e.g., '7d' for 7 days)
            
        Returns:
            bool: True if performance was successfully recorded
        """
        try:
            # Create the performance data structure
            timestamp = datetime.now(timezone.utc).isoformat()
            performance_data = {
                'model_name': self.name,
                'model_version': self.version,
                'prediction_type': prediction_type,
                'metrics': metrics,
                'num_predictions': num_predictions,
                'timestamp': timestamp,
                'time_window': time_window
            }
            
            # Save the performance data to disk
            performance_dir = os.path.join('data', 'performance')
            os.makedirs(performance_dir, exist_ok=True)
            
            # Create a unique filename based on model, prediction type, and timestamp
            filename = f"{self.name.lower()}_{prediction_type}_{int(datetime.now().timestamp())}.json"
            file_path = os.path.join(performance_dir, filename)
            
            # Save as JSON
            import json
            with open(file_path, 'w') as f:
                json.dump(performance_data, f, indent=2)
                
            logger.info(f"Model performance recorded to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording model performance: {str(e)}")
            return False
    
    def save_model(self, path: str) -> bool:
        """
        Save the model to a file
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_trained:
            logger.error("Cannot save untrained model")
            return False
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Create a dictionary with model data
            model_data = {
                'name': self.name,
                'version': self.version,
                'model_type': self.model_type,
                'params': self.params,
                'feature_names': self.feature_names,
                'trained_at': self.trained_at,
                'classifier': self.classifier if self.model_type == 'classification' else None,
                'regressor': self.regressor if self.model_type == 'regression' else None,
                'scaler': self.scaler,
                'robust_scaler': self.robust_scaler
            }
            
            if hasattr(self, 'prop_type'):
                model_data['prop_type'] = self.prop_type
            
            # Save model data
            joblib.dump(model_data, path)
            
            logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load the model from a file
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False
            
            # Load model data
            model_data = joblib.load(path)
            
            # Update model attributes
            self.name = model_data['name']
            self.version = model_data['version']
            self.model_type = model_data['model_type']
            self.params = model_data['params']
            self.feature_names = model_data['feature_names']
            self.trained_at = model_data['trained_at']
            
            if self.model_type == 'classification':
                self.classifier = model_data['classifier']
            else:  # regression
                self.regressor = model_data['regressor']
            
            self.scaler = model_data['scaler']
            self.robust_scaler = model_data['robust_scaler']
            
            if 'prop_type' in model_data:
                self.prop_type = model_data['prop_type']
            
            self.is_trained = True
            
            logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
