#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensemble Stacking Model for NBA Predictions

This module implements a stacking ensemble that combines the outputs from multiple
base models to create more accurate predictions for NBA games. The ensemble uses
a meta-learner to weight the contributions of individual models based on their
performance on validation data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.bayesian_model import BayesianModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Stacking Ensemble Model for NBA predictions
    
    Features:
    - Combines outputs from multiple base models
    - Uses a meta-learner to optimize predictions
    - Weights individual model contributions based on performance
    - Provides final, optimized predictions for games
    - Handles both classification (win/loss) and regression (spread/total) tasks
    """
    
    def __init__(self, name: str = "EnsembleStacking", prediction_target: str = "moneyline", version: int = 1,
                base_models: Optional[List[BaseModel]] = None,
                meta_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Ensemble Stacking model
        
        Args:
            name: Model name
            prediction_target: What the model is predicting ('moneyline', 'spread', 'totals')
            version: Model version number
            base_models: Optional list of pre-trained base models
            meta_params: Optional parameters for the meta-learner
        """
        # Determine model type based on prediction target
        if prediction_target in ['moneyline']:
            model_type = 'classification'
        else:  # 'spread', 'totals', 'player_points', etc.
            model_type = 'regression'
            
        super().__init__(name=name, model_type=model_type, version=version)
        self.prediction_target = prediction_target
        
        # Store prediction type
        self.prediction_type = model_type
        
        # Initialize base models if not provided
        self.base_models = base_models or []
        if not self.base_models:
            if self.prediction_type == "classification":
                # Default classification models for moneyline predictions
                self.base_models = [
                    RandomForestModel(version=1),
                    BayesianModel(prediction_target="moneyline", version=1)
                ]
            else:  # regression
                # Default regression models for spread/total predictions
                self.base_models = [
                    GradientBoostingModel(version=1),
                    BayesianModel(prediction_target="spread", version=1)
                ]
        
        # Initialize meta-learner parameters
        self.meta_params = meta_params or {}
        if self.prediction_type == "classification":
            self.meta_params = self.meta_params or {
                'C': 1.0,
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': 42
            }
            # Use logistic regression as meta-learner for classification
            self.meta_learner = LogisticRegression(**self.meta_params)
        else:  # regression
            self.meta_params = self.meta_params or {
                'alpha': 1.0,
                'fit_intercept': True,
                'normalize': False,
                'random_state': 42
            }
            # Use ridge regression as meta-learner for regression
            self.meta_learner = Ridge(**self.meta_params)
        
        # Feature preprocessing
        self.scaler = StandardScaler()
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the ensemble model
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        logger.info(f"Training Ensemble Stacking model (version {self.version})")
        
        if X.empty or y.empty:
            logger.error("Cannot train model with empty data")
            return
        
        try:
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Ensure base_models is a list, not a dictionary
            if isinstance(self.base_models, dict):
                self.base_models = list(self.base_models.values())
                logger.info(f"Converted base_models dictionary to list with {len(self.base_models)} models")
            
            # Train base models if they aren't already trained
            base_predictions = []
            for i, model in enumerate(self.base_models):
                if not model.is_trained:
                    logger.info(f"Training base model {i+1}: {model.name}")
                    model.train(X, y)
                
                # Generate out-of-fold predictions to avoid overfitting
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                
                # For cross validation, we need to use the raw model instead of our wrapper
                if hasattr(model, 'model') and model.model is not None:
                    # Check if the underlying model has predict_proba (for classification)
                    if self.prediction_type == "classification" and hasattr(model.model, 'predict_proba'):
                        # For classification models with probability predictions
                        pred = cross_val_predict(model.model, X, y, cv=cv, method='predict_proba')
                        # Use probability for positive class
                        base_predictions.append(pred[:, 1].reshape(-1, 1))
                    else:
                        # For regression models or classifiers without predict_proba
                        pred = cross_val_predict(model.model, X, y, cv=cv)
                        base_predictions.append(pred.reshape(-1, 1))
                else:
                    logger.warning(f"Base model {i+1} has no underlying model attribute, skipping")
            
            # If we have no predictions, we can't train the meta model
            if not base_predictions:
                logger.error("No valid base predictions generated, cannot train meta-model")
                return
                
            # Combine base model predictions into a single feature matrix
            meta_features = np.hstack(base_predictions)
            
            # Scale meta features
            meta_features_scaled = self.scaler.fit_transform(meta_features)
            
            # Train the meta-learner
            logger.info(f"Training meta-learner with {meta_features.shape[1]} features")
            self.meta_learner.fit(meta_features_scaled, y)
            
            # Update model metadata
            self.is_trained = True
            self.trained_at = datetime.now(timezone.utc)
            
            logger.info(f"Ensemble Stacking model trained successfully with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training Ensemble Stacking model: {str(e)}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained ensemble model
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            logger.error("Cannot predict with untrained model")
            return np.array([])
        
        try:
            # Get predictions from base models
            base_predictions = []
            for i, model in enumerate(self.base_models):
                if not model.is_trained:
                    logger.error(f"Base model {i+1} is not trained")
                    return np.array([])
                
                # Check if the underlying model has predict_proba (for classification)
                if self.prediction_type == "classification" and hasattr(model.model, 'predict_proba'):
                    pred = model.predict_proba(X)
                    # Use probability for positive class
                    base_predictions.append(pred[:, 1].reshape(-1, 1))
                else:
                    # For regression models or classifiers without predict_proba
                    pred = model.predict(X)
                    base_predictions.append(pred.reshape(-1, 1))
            
            # Combine base model predictions into a single feature matrix
            meta_features = np.hstack(base_predictions)
            
            # Scale meta features
            meta_features_scaled = self.scaler.transform(meta_features)
            
            # Make predictions with the meta-learner
            return self.meta_learner.predict(meta_features_scaled)
            
        except Exception as e:
            logger.error(f"Error making predictions with Ensemble Stacking model: {str(e)}")
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
            logger.error("Cannot predict with untrained model")
            return np.array([])
            
        if self.prediction_type != "classification":
            logger.error("predict_proba is only available for classification models")
            return np.array([])
        
        try:
            # Get predictions from base models
            base_predictions = []
            for i, model in enumerate(self.base_models):
                if hasattr(model, 'predict_proba') and callable(model.predict_proba):
                    try:
                        pred = model.predict_proba(X)
                        # Ensure we get probabilities for positive class (class 1)
                        if pred.shape[1] >= 2:  # Binary classification
                            base_predictions.append(pred[:, 1].reshape(-1, 1))
                        else:
                            base_predictions.append(pred.reshape(-1, 1))
                    except Exception as e:
                        logger.warning(f"Error getting probability predictions from base model {i}: {str(e)}")
                        # Use zeros as fallback
                        base_predictions.append(np.zeros((X.shape[0], 1)))
                else:
                    # For models without predict_proba, use regular predictions converted to 0-1 range
                    try:
                        pred = model.predict(X)
                        # Normalize predictions to [0, 1] range as an approximation of probability
                        pred_norm = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-10)
                        base_predictions.append(pred_norm.reshape(-1, 1))
                    except Exception as e:
                        logger.warning(f"Error getting predictions from base model {i}: {str(e)}")
                        # Use zeros as fallback
                        base_predictions.append(np.zeros((X.shape[0], 1)))
            
            # Combine base model predictions
            if not base_predictions:
                # If no valid predictions, return default probabilities
                return np.array([[0.5, 0.5]] * X.shape[0])
                
            meta_features = np.hstack(base_predictions)
            
            # Scale features
            meta_features_scaled = self.scaler.transform(meta_features)
            
            # Get probability predictions from meta-learner
            if hasattr(self.meta_learner, 'predict_proba'):
                proba = self.meta_learner.predict_proba(meta_features_scaled)
                return proba
            else:
                # If meta-learner doesn't have predict_proba, convert predictions to pseudo-probabilities
                pred = self.meta_learner.predict(meta_features_scaled)
                # Convert to probabilities for binary classification (0/1)
                proba = np.zeros((len(pred), 2))
                proba[:, 1] = (pred > 0.5).astype(float)
                proba[:, 0] = 1 - proba[:, 1]
                return proba
                
        except Exception as e:
            logger.error(f"Error predicting probabilities with Ensemble model: {str(e)}")
            # Return default probabilities as fallback
            return np.array([[0.5, 0.5]] * X.shape[0])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each base model in the ensemble
        
        Returns:
            Dictionary mapping model names to importance scores
        """
        if not self.is_trained:
            logger.error("Cannot get feature importance from untrained model")
            return {}
        
        try:
            # Get coefficients from meta-learner
            if hasattr(self.meta_learner, 'coef_'):
                coeffs = self.meta_learner.coef_
                if len(coeffs.shape) > 1 and coeffs.shape[0] > 1:
                    # For multi-class classification, use mean of coefficients
                    coeffs = np.mean(coeffs, axis=0)
                else:
                    # For binary classification or regression, flatten coefficients
                    coeffs = coeffs.flatten()
                
                # Create dictionary mapping model names to importance scores
                model_names = [f"{model.name} (v{model.version})" for model in self.base_models]
                importance_dict = dict(zip(model_names, np.abs(coeffs)))
                
                # Sort by importance (descending)
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                logger.warning("Meta-learner does not have feature importance information")
                return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {}


# Main function for testing
if __name__ == "__main__":
    # Test ensemble model training and prediction
    import pandas as pd
    from src.data.feature_engineering import NBAFeatureEngineer
    
    # Load and prepare data
    engineer = NBAFeatureEngineer()
    games_df = engineer.load_games()
    
    if not games_df.empty:
        # Generate features
        features = engineer.generate_features_for_all_games(games_df)
        
        # Prepare training data for moneyline prediction
        X_train, y_train = engineer.prepare_training_data(features, target='home_win')
        
        if not X_train.empty and not y_train.empty:
            # Create and train the ensemble model
            ensemble = EnsembleModel(prediction_target="moneyline", version=1)
            ensemble.train(X_train, y_train)
            
            # Evaluate the ensemble model
            metrics = ensemble.evaluate(X_train, y_train)
            print(f"Ensemble model metrics: {metrics}")
            
            # Get model importance
            importance = ensemble.get_feature_importance()
            print("Base model importance:")
            for model_name, score in importance.items():
                print(f"{model_name}: {score:.4f}")
            
            # Save the model
            model_path = ensemble.save_to_disk()
            print(f"Ensemble model saved to: {model_path}")
        else:
            print("No valid training data available")
    else:
        print("No games data available")
