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
            return
        
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
            return
        
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
            self.is_trained = True
            self.trained_at = datetime.now(timezone.utc)
            
            logger.info(f"Random Forest model trained successfully for {prop_type} prediction with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training Random Forest model for {prop_type}: {str(e)}")
    
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
            return np.array([])
        
        try:
            # Ensure all expected features are present
            missing_features = [f for f in self.feature_names if f not in X.columns]
            if missing_features:
                logger.warning(f"Missing features in prediction data: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    X[feature] = 0.0
            
            # Preprocess features
            X_processed = self._preprocess_features(X, fit=False)
            
            # Make predictions
            return self.classifier.predict(X_processed)
            
        except Exception as e:
            logger.error(f"Error making predictions with Random Forest model: {str(e)}")
            return np.array([])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input samples
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for away win (0) and home win (1)
        """
        if not self.is_trained:
            logger.error("Cannot predict probabilities with untrained model")
            return np.array([])
        
        try:
            # Ensure all expected features are present
            missing_features = [f for f in self.feature_names if f not in X.columns]
            if missing_features:
                logger.warning(f"Missing features in prediction data: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    X[feature] = 0.0
            
            # Preprocess features
            X_processed = self._preprocess_features(X, fit=False)
            
            # Predict probabilities
            return self.classifier.predict_proba(X_processed)
            
        except Exception as e:
            logger.error(f"Error predicting probabilities with Random Forest model: {str(e)}")
            return np.array([])
    
    def predict_player_stats(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict player statistics using the trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted player statistics
        """
        if not self.is_trained:
            logger.error("Cannot predict player statistics with untrained model")
            return np.array([])
        
        try:
            # Ensure all expected features are present
            missing_features = [f for f in self.feature_names if f not in X.columns]
            if missing_features:
                logger.warning(f"Missing features in prediction data: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    X[feature] = 0.0
            
            # Preprocess features
            X_processed = self._preprocess_features(X, fit=False, task='regression')
            
            # Predict player statistics
            return self.regressor.predict(X_processed)
            
        except Exception as e:
            logger.error(f"Error predicting player statistics with Random Forest model: {str(e)}")
            return np.array([])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance from the trained model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or not hasattr(self.classifier, 'feature_importances_'):
            logger.error("Cannot get feature importance from untrained model")
            return {}
        
        try:
            # Get feature importances
            importances = self.classifier.feature_importances_
            
            # Create dictionary mapping feature names to importance scores
            importance_dict = dict(zip(self.feature_names, importances))
            
            # Sort by importance (descending)
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def save(self, directory: str = "models") -> str:
        """
        Save the trained model to disk with metadata
        
        Args:
            directory: Directory to save the model in
            
        Returns:
            Path to the saved model file
        """
        if not self.is_trained:
            logger.error("Cannot save untrained model")
            return ""
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Create filename with timestamp and model type info
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type_suffix = self.model_type
            if hasattr(self, 'prop_type'):
                model_type_suffix = f"{self.prop_type}_{model_type_suffix}"
                
            filename = f"{self.name}_{model_type_suffix}_{timestamp}_v{self.version}.pkl"
            filepath = os.path.join(directory, filename)
            
            # Prepare model metadata
            metadata = {
                'classifier': self.classifier,
                'regressor': self.regressor,
                'scaler': self.scaler,
                'robust_scaler': self.robust_scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'prop_type': getattr(self, 'prop_type', None),
                'params': self.params,
                'trained_at': self.trained_at,
                'version': self.version
            }
            
            # Save the model with metadata
            joblib.dump(metadata, filepath)
            logger.info(f"Model saved to {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return ""
    
    @classmethod
    def load(cls, filepath: str) -> 'RandomForestModel':
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Loaded RandomForestModel instance
        """
        try:
            # Load the model metadata
            metadata = joblib.load(filepath)
            
            # Get model configuration
            version = metadata.get('version', 1)
            
            # Create a new model instance
            model = cls(version=version)
            
            # Restore model attributes
            model.classifier = metadata['classifier']
            model.regressor = metadata['regressor']
            model.scaler = metadata['scaler']
            model.robust_scaler = metadata['robust_scaler']
            model.feature_names = metadata['feature_names']
            model.model_type = metadata['model_type']
            if metadata.get('prop_type'):
                model.prop_type = metadata['prop_type']
            model.params = metadata['params']
            model.trained_at = metadata['trained_at']
            model.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {filepath}: {str(e)}")
            return None


# Main function for testing
if __name__ == "__main__":
    # Test model training and prediction
    np.random.seed(42)
    
    # Generate synthetic data for classification (moneyline)
    n_samples = 1000
    n_features = 20
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create dataset
    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)
    
    # Add some basketball-specific features
    X['home_win_rate_5g'] = np.random.uniform(0, 1, n_samples)
    X['home_win_rate_10g'] = X['home_win_rate_5g'] + np.random.normal(0, 0.1, n_samples)
    X['away_win_rate_5g'] = np.random.uniform(0, 1, n_samples)
    X['away_win_rate_10g'] = X['away_win_rate_5g'] + np.random.normal(0, 0.1, n_samples)
    
    # Generate classification target (win/loss)
    y_class = (X['home_win_rate_5g'] > X['away_win_rate_5g']).astype(int)
    
    # Split into train and test
    train_idx = int(0.8 * n_samples)
    X_train, X_test = X[:train_idx], X[train_idx:]
    y_train, y_test = y_class[:train_idx], y_class[train_idx:]
    
    # Test classification model
    model = RandomForestModel(version=2)
    model.train(X_train, y_train, task='classification')
    
    # Make predictions
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    
    # Evaluate classification
    accuracy = accuracy_score(y_test, preds)
    print(f"Classification accuracy: {accuracy:.4f}")
    
    # Save the model
    model_path = model.save()
    print(f"Model saved to: {model_path}")
    
    # Now test regression model for player props
    # Generate synthetic player stat data
    X_player = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)
    
    # Add player-specific features
    X_player['player_points'] = np.random.uniform(10, 30, n_samples)
    X_player['player_assists'] = np.random.uniform(2, 12, n_samples)
    X_player['player_recent_points'] = X_player['player_points'] + np.random.normal(0, 2, n_samples)
    X_player['player_points_trend'] = X_player['player_recent_points'] - X_player['player_points']
    X_player['is_home_team'] = np.random.randint(0, 2, n_samples)
    
    # Target: predict assists
    y_assists = X_player['player_assists'] + X_player['player_points'] * 0.1 + np.random.normal(0, 1, n_samples)
    
    # Split data
    X_player_train, X_player_test = X_player[:train_idx], X_player[train_idx:]
    y_assists_train, y_assists_test = y_assists[:train_idx], y_assists[train_idx:]
    
    # Train for assists prediction
    assists_model = RandomForestModel(version=2)
    assists_model.train_for_player_props(X_player_train, y_assists_train, prop_type='assists')
    
    # Test the assists prediction
    assists_preds = assists_model.predict_player_stats(X_player_test)
    mae = mean_absolute_error(y_assists_test, assists_preds)
    print(f"Assists prediction MAE: {mae:.4f}")
    
    # Save the assists model
    assists_model_path = assists_model.save()
    print(f"Assists model saved to: {assists_model_path}")
