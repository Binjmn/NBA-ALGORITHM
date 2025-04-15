#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradient Boosting Regressor for NBA Spread Predictions

This module implements a Gradient Boosting Regressor specifically optimized for 
predicting the point spread in NBA games. It forecasts the exact point differential
between teams and is optimized for minimizing regression error.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.base_model import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting Regressor for predicting NBA game point spreads
    
    Features:
    - Forecasts the exact point differential between teams
    - Optimized for minimizing regression error (MAE/RMSE)
    - Includes feature importance analysis
    - Hyperparameter optimization via grid search
    - Cross-validation for robust performance evaluation
    """
    
    def __init__(self, version: int = 1, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Gradient Boosting model
        
        Args:
            version: Model version number
            params: Optional model parameters for GradientBoostingRegressor
        """
        super().__init__(name="GradientBoosting", model_type="regression", version=version)
        
        # Default parameters (will be tuned during training if not provided)
        self.params = params or {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 0.8,
            'random_state': 42
        }
        
        # Initialize the model
        self.model = GradientBoostingRegressor(**self.params)
        
        # Feature preprocessing
        self.scaler = StandardScaler()
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model with hyperparameter optimization
        
        Args:
            X: Feature matrix
            y: Target variable (point spread, positive for home win margin)
        """
        logger.info(f"Training Gradient Boosting model (version {self.version})")
        
        if X.empty or y.empty:
            logger.error("Cannot train model with empty data")
            return
        
        try:
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            # Perform hyperparameter tuning if not already tuned
            if not self.is_trained:
                logger.info("Performing hyperparameter optimization")
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 4, 6],
                    'subsample': [0.7, 0.8, 0.9]
                }
                
                # Use k-fold cross-validation
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    estimator=GradientBoostingRegressor(random_state=42),
                    param_grid=param_grid,
                    cv=cv,
                    scoring='neg_mean_absolute_error',  # Use MAE as primary metric
                    n_jobs=-1  # Use all available cores
                )
                
                grid_search.fit(X_scaled, y)
                
                # Get best parameters
                self.params = grid_search.best_params_
                logger.info(f"Best parameters: {self.params}")
                
                # Update model with best parameters
                self.model = GradientBoostingRegressor(**self.params)
            
            # Train the model with the best parameters
            self.model.fit(X_scaled, y)
            
            # Update model metadata
            self.is_trained = True
            self.trained_at = datetime.now(timezone.utc)
            
            logger.info(f"Gradient Boosting model trained successfully with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training Gradient Boosting model: {str(e)}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted point spreads
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
            
            # Scale features
            X_scaled = self.scaler.transform(X[self.feature_names])
            
            # Make predictions
            return self.model.predict(X_scaled)
            
        except Exception as e:
            logger.error(f"Error making predictions with Gradient Boosting model: {str(e)}")
            return np.array([])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        For regression models, this returns the predictions with a confidence interval
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions (same as predict for regression models)
        """
        # For regression models, predict_proba is not applicable in the same way as classification
        # We could implement quantile regression or other uncertainty estimates here
        # For now, we'll just return the predictions
        return self.predict(X).reshape(-1, 1)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance from the trained model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            logger.error("Cannot get feature importance from untrained model")
            return {}
        
        try:
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Create dictionary mapping feature names to importance scores
            importance_dict = dict(zip(self.feature_names, importances))
            
            # Sort by importance (descending)
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model performance on test data
        
        Args:
            X: Feature matrix
            y: Target variable (point spread)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            logger.error(f"Cannot evaluate untrained model: {self.name}")
            return {}
        
        try:
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate regression metrics
            metrics = {
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred)
            }
            
            # Calculate betting performance if available
            if 'actual_spread' in X.columns and 'vegas_spread' in X.columns:
                # Calculate if our prediction would have beaten Vegas spread
                correct_predictions = ((y_pred > X['vegas_spread']) == (X['actual_spread'] > X['vegas_spread'])).mean()
                metrics['vegas_beat_rate'] = correct_predictions
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model {self.name}: {str(e)}")
            return {}


# Main function for testing
if __name__ == "__main__":
    # Test model training and prediction
    import pandas as pd
    from src.data.feature_engineering import NBAFeatureEngineer
    
    # Load and prepare data
    engineer = NBAFeatureEngineer()
    games_df = engineer.load_games()
    
    if not games_df.empty:
        # Generate features
        features = engineer.generate_features_for_all_games(games_df)
        
        # Prepare training data for spread prediction
        X_train, y_train = engineer.prepare_training_data(features, target='spread')
        
        if not X_train.empty and not y_train.empty:
            # Create and train the model
            model = GradientBoostingModel(version=1)
            model.train(X_train, y_train)
            
            # Evaluate the model
            metrics = model.evaluate(X_train, y_train)
            print(f"Model metrics: {metrics}")
            
            # Get feature importance
            importance = model.get_feature_importance()
            print("Top 10 features by importance:")
            for i, (feature, score) in enumerate(list(importance.items())[:10]):
                print(f"{i+1}. {feature}: {score:.4f}")
            
            # Save the model
            model_path = model.save_to_disk()
            print(f"Model saved to: {model_path}")
        else:
            print("No valid training data available")
    else:
        print("No games data available")
