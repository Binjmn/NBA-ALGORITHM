"""
Combined Gradient Boosting Model for NBA Prediction System

This module implements a combined model that integrates both XGBoost and LightGBM
for predicting game outcomes and player props with increased accuracy and robustness.
The combination helps leverage the strengths of both algorithms and ensures more
stable predictions across different scenarios.

Features Used:
- Team: Recent performance metrics, strength of schedule, home/away splits, rest days,
  travel distance, lineup efficiency, clutch performance, pace/style, recent overtimes,
  fatigue model, venue effects, team chemistry
- Player: Efficiency metrics, injury impact, matchup performance history, usage rate,
  rookie adjustments
- Context: Rivalry/motivation factors, playoff vs. regular season intensity
- Odds: Game-level odds, player prop odds

The combined approach captures intricate feature interactions like how fatigue and
matchup performance affect spreads, producing more accurate predictions for moneyline,
spread, totals, and player props.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timezone

from src.models.base_model import BaseModel

# Configure logging
logger = logging.getLogger(__name__)


class CombinedGradientBoostingModel(BaseModel):
    """
    Combined Gradient Boosting model using XGBoost and LightGBM
    
    This model blends predictions from both XGBoost and LightGBM to create
    more robust and accurate predictions by leveraging the strengths of
    both algorithms.
    
    The model can operate in classification mode (for win/loss predictions) or
    regression mode (for point spreads, totals, and player stats).
    """
    
    def __init__(self, name: str = "CombinedGBM", prediction_target: str = "moneyline", version: int = 1,
                 xgb_weight: float = 0.5, lgb_weight: float = 0.5):
        """
        Initialize the Combined Gradient Boosting model
        
        Args:
            name: Model name
            prediction_target: What the model is predicting ('moneyline', 'spread', 'totals', 'player_points', etc.)
            version: Model version number
            xgb_weight: Weight given to XGBoost predictions (0.0 to 1.0)
            lgb_weight: Weight given to LightGBM predictions (0.0 to 1.0)
        """
        # Determine model type based on prediction target
        if prediction_target in ['moneyline']:
            model_type = 'classification'
        else:  # 'spread', 'totals', 'player_points', etc.
            model_type = 'regression'
            
        super().__init__(name=name, model_type=model_type, version=version)
        self.prediction_target = prediction_target
        
        # Model mixing weights (should sum to 1.0)
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        
        # Normalize weights to ensure they sum to 1.0
        sum_weights = self.xgb_weight + self.lgb_weight
        if sum_weights != 1.0 and sum_weights > 0:
            self.xgb_weight /= sum_weights
            self.lgb_weight /= sum_weights
        
        # Initialize sub-models
        self.xgb_model = None
        self.lgb_model = None
        
        # Default hyperparameters for XGBoost - these can be tuned
        self.xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        # Default hyperparameters for LightGBM - these can be tuned
        self.lgb_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        # Combined parameters
        self.params = {
            'xgb_params': self.xgb_params,
            'lgb_params': self.lgb_params,
            'xgb_weight': self.xgb_weight,
            'lgb_weight': self.lgb_weight
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train both XGBoost and LightGBM models on the provided data
        
        Args:
            X: Feature matrix with all team, player, context, and odds features
            y: Target variable (win/loss for classification, spread/totals/points for regression)
        """
        try:
            # Store feature names for later use
            self.feature_names = list(X.columns)
            
            # Initialize and train XGBoost model
            if self.model_type == 'classification':
                self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
            else:  # 'regression'
                self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
            
            logger.info(f"Training XGBoost component of {self.name} model with {len(X)} samples")
            self.xgb_model.fit(X, y)
            
            # Initialize and train LightGBM model
            if self.model_type == 'classification':
                self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
            else:  # 'regression'
                self.lgb_model = lgb.LGBMRegressor(**self.lgb_params)
            
            logger.info(f"Training LightGBM component of {self.name} model with {len(X)} samples")
            self.lgb_model.fit(X, y)
            
            # Set combined model (used for feature importance)
            self.model = {
                'xgb': self.xgb_model,
                'lgb': self.lgb_model,
                'xgb_weight': self.xgb_weight,
                'lgb_weight': self.lgb_weight
            }
            
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
        Make combined predictions using both trained models
        
        Args:
            X: Feature matrix matching the training features
            
        Returns:
            Array of weighted predictions from both models
        """
        if not self.is_trained or self.xgb_model is None or self.lgb_model is None:
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
            
            # Get predictions from both models
            xgb_predictions = self.xgb_model.predict(X)
            lgb_predictions = self.lgb_model.predict(X)
            
            # Combine predictions using weights
            if self.model_type == 'classification':
                # For classification, we need to handle classes carefully
                # If the prediction outputs are the same shape, we can apply weighted average
                if xgb_predictions.shape == lgb_predictions.shape:
                    combined_predictions = (
                        xgb_predictions * self.xgb_weight + 
                        lgb_predictions * self.lgb_weight
                    )
                    # Round for class labels (assuming binary classification)
                    if len(combined_predictions.shape) == 1 or combined_predictions.shape[1] == 1:
                        combined_predictions = np.round(combined_predictions).astype(int)
                else:
                    # If shapes differ, just use one model (fallback)
                    logger.warning("Model outputs have different shapes. Using XGBoost predictions only.")
                    combined_predictions = xgb_predictions
            else:  # 'regression'
                # For regression, simple weighted average
                combined_predictions = (
                    xgb_predictions * self.xgb_weight + 
                    lgb_predictions * self.lgb_weight
                )
            
            return combined_predictions
            
        except Exception as e:
            logger.error(f"Error predicting with {self.name} model: {str(e)}")
            return np.array([])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using both trained models (classification only)
        
        Args:
            X: Feature matrix matching the training features
            
        Returns:
            Array of weighted probability predictions from both models
        """
        if not self.is_trained or self.xgb_model is None or self.lgb_model is None:
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
            
            # Get probability predictions from both models
            xgb_proba = self.xgb_model.predict_proba(X)
            lgb_proba = self.lgb_model.predict_proba(X)
            
            # Combine probabilities using weights
            if xgb_proba.shape == lgb_proba.shape:
                combined_proba = (
                    xgb_proba * self.xgb_weight + 
                    lgb_proba * self.lgb_weight
                )
                # Normalize to ensure probabilities sum to 1
                row_sums = combined_proba.sum(axis=1).reshape(-1, 1)
                combined_proba = combined_proba / row_sums
            else:
                # If shapes differ, just use one model (fallback)
                logger.warning("Model probability outputs have different shapes. Using XGBoost probabilities only.")
                combined_proba = xgb_proba
            
            return combined_proba
            
        except Exception as e:
            logger.error(f"Error predicting probabilities with {self.name} model: {str(e)}")
            return np.array([])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the combined feature importance from both models
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.xgb_model is None or self.lgb_model is None:
            logger.error(f"Cannot get feature importance from untrained model: {self.name}")
            return {}
        
        try:
            # Get feature importances from both models
            xgb_importances = self.xgb_model.feature_importances_
            lgb_importances = self.lgb_model.feature_importances_
            
            # Combine feature importances using the same weights as predictions
            combined_importances = (
                xgb_importances * self.xgb_weight + 
                lgb_importances * self.lgb_weight
            )
            
            # Create a dictionary mapping feature names to their importance scores
            feature_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_importance[feature_name] = float(combined_importances[i])
            
            # Sort by importance (highest to lowest)
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {self.name} model: {str(e)}")
            return {}
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the combined model performance on test data
        
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
            
            # Add model specific metrics if applicable
            if self.model_type == 'classification':
                # Add any classification-specific metrics
                pass
            else:  # regression
                # Add any regression-specific metrics
                pass
            
            # Evaluate individual models for comparison
            try:
                # XGBoost individual metrics
                xgb_pred = self.xgb_model.predict(X)
                if self.model_type == 'regression':
                    metrics['xgb_mae'] = np.mean(np.abs(y - xgb_pred))
                    metrics['xgb_mse'] = np.mean((y - xgb_pred) ** 2)
                    metrics['xgb_rmse'] = np.sqrt(metrics['xgb_mse'])
                else:  # classification
                    from sklearn.metrics import accuracy_score
                    metrics['xgb_accuracy'] = accuracy_score(y, xgb_pred)
                
                # LightGBM individual metrics
                lgb_pred = self.lgb_model.predict(X)
                if self.model_type == 'regression':
                    metrics['lgb_mae'] = np.mean(np.abs(y - lgb_pred))
                    metrics['lgb_mse'] = np.mean((y - lgb_pred) ** 2)
                    metrics['lgb_rmse'] = np.sqrt(metrics['lgb_mse'])
                else:  # classification
                    metrics['lgb_accuracy'] = accuracy_score(y, lgb_pred)
            except Exception as inner_e:
                logger.warning(f"Error computing individual model metrics: {str(inner_e)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {self.name} model: {str(e)}")
            return {}
    
    def save_to_db(self) -> bool:
        """
        Save the combined model to the database
        
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if not self.is_trained:
            logger.error(f"Cannot save untrained model: {self.name}")
            return False
        
        try:
            # Get feature importance
            feature_importance = self.get_feature_importance()
            
            # Get individual model feature importances
            xgb_importance = {}
            lgb_importance = {}
            try:
                for i, feature_name in enumerate(self.feature_names):
                    xgb_importance[feature_name] = float(self.xgb_model.feature_importances_[i])
                    lgb_importance[feature_name] = float(self.lgb_model.feature_importances_[i])
                
                # Sort by importance
                xgb_importance = dict(sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True))
                lgb_importance = dict(sorted(lgb_importance.items(), key=lambda x: x[1], reverse=True))
            except Exception as inner_e:
                logger.warning(f"Error extracting individual model importances: {str(inner_e)}")
            
            # Create model weights dictionary with model specific details
            model_data = {
                'model_name': self.name,
                'model_type': self.model_type,
                'weights': {
                    'feature_importances': feature_importance,
                    'xgb_importance': xgb_importance,
                    'lgb_importance': lgb_importance,
                    'xgb_weight': self.xgb_weight,
                    'lgb_weight': self.lgb_weight,
                    'prediction_target': self.prediction_target
                },
                'params': self.params,
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
