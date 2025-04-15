#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Random Forest Classifier for NBA Moneyline Predictions

This module implements a Random Forest Classifier specifically optimized for 
predicting the winner (moneyline) in NBA games. It leverages team performance
metrics to predict home/away wins and produces probability outputs for confidence scoring.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest Classifier for predicting NBA game winners (moneyline)
    
    Features:
    - Uses team performance metrics to predict home/away win
    - Produces probability outputs for confidence scoring
    - Handles categorical features through encoding
    - Includes hyperparameter optimization
    - Cross-validation for robust performance evaluation
    """
    
    def __init__(self, version: int = 1, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Random Forest model
        
        Args:
            version: Model version number
            params: Optional model parameters for RandomForestClassifier
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
        self.model = RandomForestClassifier(**self.params)
        
        # Feature preprocessing
        self.scaler = StandardScaler()
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model with hyperparameter optimization
        
        Args:
            X: Feature matrix
            y: Target variable (1 for home win, 0 for away win)
        """
        logger.info(f"Training Random Forest model (version {self.version})")
        
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
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 4, 6],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                # Use stratified k-fold to handle potential class imbalance
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    estimator=RandomForestClassifier(random_state=42),
                    param_grid=param_grid,
                    cv=cv,
                    scoring='accuracy',
                    n_jobs=-1  # Use all available cores
                )
                
                grid_search.fit(X_scaled, y)
                
                # Get best parameters
                self.params = grid_search.best_params_
                logger.info(f"Best parameters: {self.params}")
                
                # Update model with best parameters
                self.model = RandomForestClassifier(**self.params)
            
            # Train the model with the best parameters
            self.model.fit(X_scaled, y)
            
            # Update model metadata
            self.is_trained = True
            self.trained_at = datetime.now(timezone.utc)
            
            logger.info(f"Random Forest model trained successfully with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
    
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
            
            # Scale features
            X_scaled = self.scaler.transform(X[self.feature_names])
            
            # Make predictions
            return self.model.predict(X_scaled)
            
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
            
            # Scale features
            X_scaled = self.scaler.transform(X[self.feature_names])
            
            # Predict probabilities
            return self.model.predict_proba(X_scaled)
            
        except Exception as e:
            logger.error(f"Error predicting probabilities with Random Forest model: {str(e)}")
            return np.array([])
    
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
        
        # Prepare training data
        X_train, y_train = engineer.prepare_training_data(features, target='home_win')
        
        if not X_train.empty and not y_train.empty:
            # Create and train the model
            model = RandomForestModel(version=1)
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
