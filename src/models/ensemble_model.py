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
    
    def __init__(self, prediction_type: str = "classification", version: int = 1,
                base_models: Optional[List[BaseModel]] = None,
                meta_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Ensemble Stacking model
        
        Args:
            prediction_type: Type of prediction ('classification' or 'regression')
            version: Model version number
            base_models: Optional list of pre-trained base models
            meta_params: Optional parameters for the meta-learner
        """
        super().__init__(name="EnsembleStacking", model_type=prediction_type, version=version)
        
        # Store prediction type
        self.prediction_type = prediction_type
        
        # Initialize base models if not provided
        self.base_models = base_models or []
        if not self.base_models:
            if prediction_type == "classification":
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
        if prediction_type == "classification":
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
            
            # Train base models if they aren't already trained
            base_predictions = []
            for i, model in enumerate(self.base_models):
                if not model.is_trained:
                    logger.info(f"Training base model {i+1}: {model.name}")
                    model.train(X, y)
                
                # Generate out-of-fold predictions to avoid overfitting
                if self.prediction_type == "classification" and hasattr(model, 'predict_proba'):
                    # For classification, use probability predictions
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    pred = cross_val_predict(model.model, X, y, cv=cv, method='predict_proba')
                    # Use probability for positive class
                    base_predictions.append(pred[:, 1].reshape(-1, 1))
                else:
                    # For regression or non-probabilistic models, use point predictions
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    pred = cross_val_predict(model.model, X, y, cv=cv)
                    base_predictions.append(pred.reshape(-1, 1))
            
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
                
                # Use probability predictions for classification models
                if self.prediction_type == "classification" and hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                    # Use probability for positive class
                    base_predictions.append(pred[:, 1].reshape(-1, 1))
                else:
                    # Use point predictions for regression models
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
        Predict class probabilities for the input samples
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of class probabilities (only for classification models)
        """
        if not self.is_trained:
            logger.error("Cannot predict probabilities with untrained model")
            return np.array([])
        
        if self.prediction_type != "classification":
            logger.warning("predict_proba is not applicable for regression models")
            return self.predict(X).reshape(-1, 1)
        
        try:
            # Get predictions from base models
            base_predictions = []
            for i, model in enumerate(self.base_models):
                if not model.is_trained:
                    logger.error(f"Base model {i+1} is not trained")
                    return np.array([])
                
                # Use probability predictions for classification models
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                    # Use probability for positive class
                    base_predictions.append(pred[:, 1].reshape(-1, 1))
                else:
                    # Use point predictions for models without predict_proba
                    pred = model.predict(X)
                    base_predictions.append(pred.reshape(-1, 1))
            
            # Combine base model predictions into a single feature matrix
            meta_features = np.hstack(base_predictions)
            
            # Scale meta features
            meta_features_scaled = self.scaler.transform(meta_features)
            
            # Make probability predictions with the meta-learner
            return self.meta_learner.predict_proba(meta_features_scaled)
            
        except Exception as e:
            logger.error(f"Error predicting probabilities with Ensemble Stacking model: {str(e)}")
            return np.array([])
    
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
            ensemble = EnsembleModel(prediction_type="classification", version=1)
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
