"""
Random Forest Model for NBA Prediction System

This module implements the Random Forest model for predicting game outcomes and player props.
It leverages the full feature set to capture complex patterns in NBA games, including team
performance metrics, player statistics, and contextual factors.

Features Used:
- Team: Recent performance (5/10/20-game averages), strength of schedule (Elo), home/away splits,
  rest days, travel distance, lineup efficiency, clutch performance, pace/style, recent overtimes,
  fatigue model, venue effects, team chemistry
- Player: Efficiency, injury impact, matchup performance, usage rate, rookie adjustments
- Context: Rivalry/motivation, playoff vs. regular season intensity
- Odds: Game-level odds, player prop odds

The Random Forest model excels at capturing non-linear relationships between these features
and predicting outcomes with high accuracy and resistance to overfitting.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime, timezone

from src.models.base_model import BaseModel

# Configure logging
logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest model for NBA prediction
    
    This model uses Random Forests to predict game outcomes and player performance,
    leveraging the full feature set to capture complex patterns in NBA data.
    
    The model can operate in classification mode (for win/loss predictions) or
    regression mode (for point spreads, totals, and player stats).
    """
    
    def __init__(self, name: str = "RandomForest", prediction_target: str = "moneyline", version: int = 1):
        """
        Initialize the Random Forest model
        
        Args:
            name: Model name
            prediction_target: What the model is predicting ('moneyline', 'spread', 'totals', 'player_points', etc.)
            version: Model version number
        """
        # Determine model type based on prediction target
        if prediction_target in ['moneyline']:
            model_type = 'classification'
        else:  # 'spread', 'totals', 'player_points', etc.
            model_type = 'regression'
            
        super().__init__(name=name, model_type=model_type, version=version)
        self.prediction_target = prediction_target
        
        # Default hyperparameters - these can be tuned
        self.params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the Random Forest model on the provided data
        
        Args:
            X: Feature matrix with all team, player, context, and odds features
            y: Target variable (win/loss for classification, spread/totals/points for regression)
        """
        try:
            # Store feature names for later use
            self.feature_names = list(X.columns)
            
            # Initialize the appropriate Random Forest model based on model type
            if self.model_type == 'classification':
                self.model = RandomForestClassifier(**self.params)
            else:  # 'regression'
                self.model = RandomForestRegressor(**self.params)
            
            # Train the model
            logger.info(f"Training {self.name} model with {len(X)} samples and {len(self.feature_names)} features")
            self.model.fit(X, y)
            
            # Record training time
            self.trained_at = datetime.now(timezone.utc)
            self.is_trained = True
            
            # Log training completion
            logger.info(f"{self.name} model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training {self.name} model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained Random Forest model
        
        Args:
            X: Feature matrix matching the training features
            
        Returns:
            Array of predictions (class labels for classification, continuous values for regression)
        """
        if not self.is_trained:
            logger.error(f"Cannot predict with untrained model: {self.name}")
            return np.array([])
        
        try:
            # Ensure X has the same features as used in training
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing features in prediction data: {missing_features}")
                # Use only the features that are present in both
                common_features = list(set(self.feature_names) & set(X.columns))
                logger.info(f"Proceeding with {len(common_features)} common features")
                X = X[common_features]
            else:
                # Ensure the order of features matches the training data
                X = X[self.feature_names]
            
            # Make predictions
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting with {self.name} model: {str(e)}")
            return np.array([])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for classification models
        
        Args:
            X: Feature matrix matching the training features
            
        Returns:
            Array of class probabilities (only for classification models)
        """
        if not self.is_trained:
            logger.error(f"Cannot predict with untrained model: {self.name}")
            return np.array([])
        
        if self.model_type != 'classification':
            logger.warning(f"predict_proba is only available for classification models, not {self.model_type}")
            return np.array([])
        
        try:
            # Ensure X has the same features as used in training
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing features in prediction data: {missing_features}")
                # Use only the features that are present in both
                common_features = list(set(self.feature_names) & set(X.columns))
                logger.info(f"Proceeding with {len(common_features)} common features")
                X = X[common_features]
            else:
                # Ensure the order of features matches the training data
                X = X[self.feature_names]
            
            # Make probability predictions
            probabilities = self.model.predict_proba(X)
            return probabilities
            
        except Exception as e:
            logger.error(f"Error predicting probabilities with {self.name} model: {str(e)}")
            return np.array([])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance from the trained Random Forest model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            logger.error(f"Cannot get feature importance from untrained model: {self.name}")
            return {}
        
        try:
            # Get feature importances from the model
            importances = self.model.feature_importances_
            
            # Create a dictionary mapping feature names to their importance scores
            feature_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_importance[feature_name] = float(importances[i])
            
            # Sort by importance (highest to lowest)
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {self.name} model: {str(e)}")
            return {}
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model performance on test data
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            logger.error(f"Cannot evaluate untrained model: {self.name}")
            return {}
        
        try:
            # Use the parent class's evaluate method for standard metrics
            metrics = super().evaluate(X, y)
            
            # Add random forest specific metrics if applicable
            if self.model_type == 'classification':
                # For classification, add probability-based metrics
                proba = self.predict_proba(X)
                if len(proba) > 0:
                    # Example: add log loss or AUC if relevant
                    # metrics['log_loss'] = log_loss(y, proba)
                    pass
            else:  # regression
                # For regression, add mean absolute error and mean squared error
                y_pred = self.predict(X)
                if len(y_pred) > 0:
                    metrics['mae'] = np.mean(np.abs(y - y_pred))
                    metrics['mse'] = np.mean((y - y_pred) ** 2)
                    metrics['rmse'] = np.sqrt(metrics['mse'])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {self.name} model: {str(e)}")
            return {}
    
    def save_to_db(self) -> bool:
        """
        Save the model to the database with random forest specific details
        
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if not self.is_trained:
            logger.error(f"Cannot save untrained model: {self.name}")
            return False
        
        try:
            # Get feature importance
            feature_importance = self.get_feature_importance()
            
            # Extract model parameters
            params = self.params.copy()
            
            # Create model weights dictionary with random forest specific details
            model_data = {
                'model_name': self.name,
                'model_type': self.model_type,
                'weights': {
                    'feature_importances': feature_importance,
                    'n_trees': self.model.n_estimators,
                    'oob_score': getattr(self.model, 'oob_score_', None),
                    'prediction_target': self.prediction_target
                },
                'params': params,
                'version': self.version,
                'trained_at': self.trained_at or datetime.now(timezone.utc),
                'active': True
            }
            
            # Save to database
            model_id = ModelWeight.create(model_data)
            
            if model_id:
                logger.info(f"Model {self.name} v{self.version} saved to database with ID {model_id}")
                
                # Deactivate old versions
                ModelWeight.deactivate_old_versions(self.name, self.version)
                
                return True
            else:
                logger.error(f"Failed to save model {self.name} to database")
                return False
                
        except Exception as e:
            logger.error(f"Error saving model {self.name} to database: {str(e)}")
            return False
