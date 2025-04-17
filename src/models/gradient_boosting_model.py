#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradient Boosting Regressor for NBA Spread and Player Prop Predictions

This module implements an enhanced Gradient Boosting Regressor optimized for 
predicting both point spreads in NBA games and player performance metrics.
It leverages advanced feature engineering, trend analysis, and robust cross-validation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings

from src.models.base_model import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GradientBoostingModel(BaseModel):
    """
    Enhanced Gradient Boosting Regressor for NBA predictions
    
    Features:
    - Optimized for both point spread and player prop predictions
    - Supports advanced time-series based training and evaluation
    - Feature importance analysis with automatic selection
    - Adaptive learning rate and early stopping
    - Performance analysis with detailed metrics
    - Support for both regular GBM and faster HistGradientBoosting
    """
    
    def __init__(self, version: int = 1, params: Optional[Dict[str, Any]] = None,
                use_hist_gradient_boosting: bool = False, prediction_target: str = "spread"):
        """
        Initialize the Gradient Boosting model with enhanced capabilities
        
        Args:
            version: Model version number
            params: Optional model parameters
            use_hist_gradient_boosting: Whether to use the faster HistGradientBoosting variant
            prediction_target: What the model is predicting ("spread", "points", "rebounds", "assists")
        """
        model_name_suffix = "_Hist" if use_hist_gradient_boosting else ""
        super().__init__(name=f"GradientBoosting{model_name_suffix}", model_type="regression", version=version)
        
        self.prediction_target = prediction_target
        self.use_hist_gradient_boosting = use_hist_gradient_boosting
        
        # Enhanced default parameters with better regularization
        self.params = params or {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 3,
            'subsample': 0.8,
            'max_features': 0.8,  # Use 80% of features to reduce overfitting
            'random_state': 42,
            'validation_fraction': 0.2,  # For early stopping
            'n_iter_no_change': 20,      # Early stopping patience
            'tol': 1e-4                  # Tolerance for early stopping
        }
        
        # Choose model type based on configuration
        if use_hist_gradient_boosting:
            # HistGradientBoosting is faster for large datasets
            hist_params = {
                'max_iter': self.params['n_estimators'],
                'learning_rate': self.params['learning_rate'],
                'max_depth': self.params['max_depth'],
                'min_samples_leaf': self.params['min_samples_leaf'],
                'max_bins': 255,  # Higher value = more precise but slower
                'random_state': self.params['random_state'],
                'validation_fraction': self.params['validation_fraction'],
                'n_iter_no_change': self.params['n_iter_no_change'],
                'tol': self.params['tol']
            }
            self.model = HistGradientBoostingRegressor(**hist_params)
        else:
            # Standard GradientBoostingRegressor
            self.model = GradientBoostingRegressor(**self.params)
        
        # Feature preprocessing - Use RobustScaler to handle outliers better
        self.scaler = RobustScaler()
        
        # Performance metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        self.feature_importances = {}
        
        # Set target-specific configurations
        self._configure_for_target()
    
    def _configure_for_target(self):
        """
        Configure model settings based on prediction target
        """
        if self.prediction_target == "spread":
            # Point spread prediction configuration
            self.eval_metric = "neg_mean_absolute_error"
            self.param_grid = {
                'n_estimators': [200, 300, 400],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [5, 10, 15],
                'subsample': [0.7, 0.8, 0.9]
            }
        elif self.prediction_target in ["points", "rebounds", "assists"]:
            # Player prop prediction configuration
            self.eval_metric = "neg_mean_absolute_error"
            # Different grid for player props
            self.param_grid = {
                'n_estimators': [200, 300],
                'learning_rate': [0.01, 0.05],
                'max_depth': [3, 4, 5],
                'min_samples_split': [3, 5, 7],
                'subsample': [0.7, 0.8]
            }
    
    def _preprocess_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Preprocess features with enhanced engineering
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler (training) or just transform (prediction)
            
        Returns:
            Processed feature matrix
        """
        # Store original feature names
        feature_names = X.columns.tolist()
        
        # Add trend features if time-related columns exist
        X = self._add_trend_features(X)
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def _add_trend_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-based features to improve predictive power
        
        Args:
            X: Feature matrix
            
        Returns:
            Enhanced feature matrix with trend features
        """
        # Create a copy to avoid modifying the original
        X_enhanced = X.copy()
        
        try:
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
            
            # For player props, add specific features
            if self.prediction_target in ["points", "rebounds", "assists"]:
                # Check for player-specific columns
                player_cols = [col for col in X.columns if 'player_' in col]
                if player_cols:
                    # Add trend indicator for player performance if available
                    trend_col = f"player_{self.prediction_target}_trend"
                    vs_avg_col = f"player_{self.prediction_target}_vs_average"
                    
                    if trend_col in X.columns:
                        # Emphasize trend direction as a feature
                        X_enhanced[f"{trend_col}_direction"] = np.sign(X[trend_col])
                    
                    if vs_avg_col in X.columns:
                        # Emphasize over/under average performance
                        X_enhanced[f"{vs_avg_col}_direction"] = np.sign(X[vs_avg_col])
        
        except Exception as e:
            logger.warning(f"Error adding trend features: {str(e)}")
            # Return original data if feature creation fails
            return X
        
        return X_enhanced
    
    def train(self, X: pd.DataFrame, y: pd.Series, tune_hyperparams: bool = True,
             use_time_series_cv: bool = False, validation_data: Tuple[pd.DataFrame, pd.Series] = None) -> None:
        """
        Train the model with enhanced validation and feature engineering
        
        Args:
            X: Feature matrix
            y: Target variable (point spread or player stat)
            tune_hyperparams: Whether to perform hyperparameter tuning
            use_time_series_cv: Whether to use time-series cross-validation
            validation_data: Optional separate validation data
        """
        logger.info(f"Training {self.name} model for {self.prediction_target} prediction (version {self.version})")
        
        if X.empty or y.empty:
            logger.error("Cannot train model with empty data")
            return
        
        try:
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Preprocess features
            X_processed = self._preprocess_features(X, fit=True)
            
            # Perform hyperparameter tuning if requested
            if tune_hyperparams and not self.is_trained:
                logger.info("Performing hyperparameter optimization")
                
                # Choose cross-validation strategy
                if use_time_series_cv:
                    # Time series CV for temporally ordered data
                    cv = TimeSeriesSplit(n_splits=5)
                    logger.info("Using TimeSeriesSplit for temporal validation")
                else:
                    # Standard k-fold for non-temporal data
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                
                # Get the correct model class for grid search
                model_class = HistGradientBoostingRegressor if self.use_hist_gradient_boosting else GradientBoostingRegressor
                
                # Adjust param grid based on model type
                if self.use_hist_gradient_boosting:
                    param_grid = {
                        'max_iter': self.param_grid['n_estimators'],
                        'learning_rate': self.param_grid['learning_rate'],
                        'max_depth': self.param_grid['max_depth'],
                        'min_samples_leaf': [1, 3, 5]
                    }
                    base_model = HistGradientBoostingRegressor(random_state=42)
                else:
                    param_grid = self.param_grid
                    base_model = GradientBoostingRegressor(random_state=42)
                
                # Grid search with cross-validation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    grid_search = GridSearchCV(
                        estimator=base_model,
                        param_grid=param_grid,
                        cv=cv,
                        scoring=self.eval_metric,
                        n_jobs=-1,  # Use all available cores
                        verbose=1
                    )
                    
                    grid_search.fit(X_processed, y)
                
                # Get best parameters
                best_params = grid_search.best_params_
                logger.info(f"Best parameters: {best_params}")
                
                # Update model with best parameters
                if self.use_hist_gradient_boosting:
                    # Map parameters for HistGradientBoosting
                    hist_best_params = {
                        'max_iter': best_params.get('max_iter', 300),
                        'learning_rate': best_params.get('learning_rate', 0.05),
                        'max_depth': best_params.get('max_depth', 5),
                        'min_samples_leaf': best_params.get('min_samples_leaf', 3),
                        'max_bins': 255,
                        'random_state': 42,
                        'validation_fraction': 0.2,
                        'n_iter_no_change': 20,
                        'tol': 1e-4
                    }
                    self.model = HistGradientBoostingRegressor(**hist_best_params)
                else:
                    # Update standard GBM parameters
                    for param, value in best_params.items():
                        self.params[param] = value
                    self.model = GradientBoostingRegressor(**self.params)
                
                # Record best CV score
                self.train_metrics['best_cv_score'] = grid_search.best_score_
            
            # Train the model with the best parameters
            self.model.fit(X_processed, y)
            
            # Calculate training metrics
            train_preds = self.model.predict(X_processed)
            self.train_metrics['mae'] = mean_absolute_error(y, train_preds)
            self.train_metrics['rmse'] = np.sqrt(mean_squared_error(y, train_preds))
            self.train_metrics['r2'] = r2_score(y, train_preds)
            
            # Evaluate on validation set if provided
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val_processed = self._preprocess_features(X_val, fit=False)
                val_preds = self.model.predict(X_val_processed)
                
                self.val_metrics['mae'] = mean_absolute_error(y_val, val_preds)
                self.val_metrics['rmse'] = np.sqrt(mean_squared_error(y_val, val_preds))
                self.val_metrics['r2'] = r2_score(y_val, val_preds)
                
                logger.info(f"Validation MAE: {self.val_metrics['mae']:.4f}, RMSE: {self.val_metrics['rmse']:.4f}")
            
            # Calculate and store feature importance
            self._calculate_feature_importance(X_processed)
            
            # Update model metadata
            self.is_trained = True
            self.trained_at = datetime.now(timezone.utc)
            
            logger.info(f"{self.name} model trained successfully with {len(X)} samples")
            logger.info(f"Training MAE: {self.train_metrics['mae']:.4f}, RMSE: {self.train_metrics['rmse']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {self.name} model: {str(e)}")
    
    def _calculate_feature_importance(self, X: pd.DataFrame) -> None:
        """
        Calculate and store feature importance from the trained model
        
        Args:
            X: Processed feature matrix with column names
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model doesn't have feature_importances_ attribute")
            return
        
        try:
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Create dictionary mapping feature names to importance scores
            self.feature_importances = dict(zip(X.columns, importances))
            
            # Sort features by importance
            sorted_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
            
            # Log top features
            logger.info("Top 10 important features:")
            for feature, importance in sorted_features[:10]:
                logger.info(f"{feature}: {importance:.4f}")
                
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model with enhanced preprocessing
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
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
            
            # Apply the same preprocessing as during training
            X_processed = self._preprocess_features(X, fit=False)
            
            # Ensure we use the same features as training in the same order
            common_cols = [col for col in self.feature_names if col in X_processed.columns]
            
            # Make predictions
            predictions = self.model.predict(X_processed[common_cols])
            
            # For player props, ensure non-negative predictions
            if self.prediction_target in ["points", "rebounds", "assists"]:
                predictions = np.maximum(predictions, 0)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with {self.name} model: {str(e)}")
            return np.array([])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        For regression models, this estimates prediction uncertainty
        
        Args:
            X: Feature matrix
            
        Returns:
            Array with prediction confidence estimates
        """
        predictions = self.predict(X)
        
        # Return predictions with a crude confidence estimate
        # We're simulating probabilities for a regression model
        if len(predictions) > 0:
            # Column 0: probability of being below prediction
            # Column 1: probability of being above prediction
            # For regression this is arbitrary, so we use 0.5 for both
            pseudo_proba = np.column_stack((np.ones_like(predictions) * 0.5, np.ones_like(predictions) * 0.5))
            return pseudo_proba
        else:
            return np.array([])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance from the trained model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return self.feature_importances
    
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
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}_v{self.version}.pkl"
            filepath = os.path.join(directory, filename)
            
            # Prepare model metadata
            metadata = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'prediction_target': self.prediction_target,
                'params': self.params,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics,
                'feature_importances': self.feature_importances,
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
    def load(cls, filepath: str) -> 'GradientBoostingModel':
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Loaded GradientBoostingModel instance
        """
        try:
            # Load the model metadata
            metadata = joblib.load(filepath)
            
            # Get model configuration
            version = metadata.get('version', 1)
            prediction_target = metadata.get('prediction_target', 'spread')
            use_hist_gradient_boosting = isinstance(metadata['model'], HistGradientBoostingRegressor)
            
            # Create a new model instance
            model = cls(version=version, prediction_target=prediction_target,
                        use_hist_gradient_boosting=use_hist_gradient_boosting)
            
            # Restore model attributes
            model.model = metadata['model']
            model.scaler = metadata['scaler']
            model.feature_names = metadata['feature_names']
            model.params = metadata['params']
            model.train_metrics = metadata['train_metrics']
            model.val_metrics = metadata['val_metrics']
            model.feature_importances = metadata['feature_importances']
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
    
    # Generate synthetic data for testing
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
    X['home_avg_points_5g'] = np.random.uniform(90, 120, n_samples)
    X['away_avg_points_5g'] = np.random.uniform(90, 120, n_samples)
    
    # Generate target
    y = X['home_avg_points_5g'] - X['away_avg_points_5g'] + np.random.normal(0, 5, n_samples)
    
    # Split into train and test
    train_idx = int(0.8 * n_samples)
    X_train, X_test = X[:train_idx], X[train_idx:]
    y_train, y_test = y[:train_idx], y[train_idx:]
    
    # Create and train model
    model = GradientBoostingModel(version=2, prediction_target="spread")
    model.train(X_train, y_train, tune_hyperparams=True)
    
    # Make predictions
    preds = model.predict(X_test)
    
    # Evaluate
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")
    
    # Save the model
    model.save()
