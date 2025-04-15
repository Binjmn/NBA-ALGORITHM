"""
Ensemble Stacking Model for NBA Prediction System

This module implements ensemble stacking, a technique where a meta-model is
trained to combine the predictions of several base models. Unlike simple
model mixing, stacking trains a model to learn optimal combinations based on
the data, potentially capturing complex relationships between model outputs.

Features Used:
- Outputs from base models: Random Forests, Combined Gradient Boosting, Bayesian
- Original features: All team, player, context, and odds features

Ensemble stacking optimizes prediction accuracy by learning how to best combine
the outputs of different models based on the input features, leveraging the
strengths of each model while compensating for their weaknesses.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from datetime import datetime, timezone

from src.models.base_model import BaseModel

# Configure logging
logger = logging.getLogger(__name__)


class EnsembleStackingModel(BaseModel):
    """
    Ensemble Stacking model for NBA prediction
    
    This model uses a meta-learner to combine predictions from multiple base models,
    potentially along with original features, to make more accurate predictions.
    
    The model supports both classification (for win/loss predictions) and
    regression (for spread/totals/player stats predictions).
    """
    
    def __init__(self, name: str = "EnsembleStacking", prediction_target: str = "moneyline", version: int = 1):
        """
        Initialize the Ensemble Stacking model
        
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
        
        # Storage for base models
        self.base_models = {}
        self.base_model_names = []
        
        # Default hyperparameters for the meta-learner
        if self.model_type == 'classification':
            self.params = {
                'meta_learner': 'logistic_regression',
                'cv_folds': 5,          # Number of cross-validation folds
                'use_proba': True,       # Use probability outputs from base models
                'include_original': True,  # Include original features in meta-model
                'regularization': 1.0,   # Regularization strength
                'max_iter': 1000,        # Maximum iterations
                'random_state': 42
            }
        else:  # regression
            self.params = {
                'meta_learner': 'ridge',
                'cv_folds': 5,           # Number of cross-validation folds
                'include_original': True,  # Include original features in meta-model
                'alpha': 1.0,            # Regularization strength
                'max_iter': 1000,        # Maximum iterations
                'random_state': 42
            }
        
        # Initialize containers for cross-validation
        self.cv_base_models = {}  # Models trained on CV folds
        self.meta_features = None  # For storing meta-features during prediction
    
    def set_base_models(self, models: Dict[str, BaseModel]) -> None:
        """
        Set the base models to use for stacking
        
        Args:
            models: Dictionary mapping model names to model instances
        """
        self.base_models = models
        self.base_model_names = list(models.keys())
        logger.info(f"Set {len(models)} base models for {self.name}: {self.base_model_names}")
    
    def _generate_meta_features(self, X: pd.DataFrame, base_models: Dict[str, BaseModel]) -> pd.DataFrame:
        """
        Generate meta-features from base model predictions
        
        Args:
            X: Original feature matrix
            base_models: Dictionary of base models to use
            
        Returns:
            DataFrame with meta-features
        """
        meta_features_list = []
        
        # Add predictions from each base model
        for model_name, model in base_models.items():
            try:
                # For classification models, we can use probabilities
                if self.model_type == 'classification' and self.params['use_proba'] and hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    if proba.shape[1] == 2:  # Binary classification
                        # Just use the probability of class 1
                        meta_features_list.append(pd.DataFrame(
                            proba[:, 1].reshape(-1, 1),
                            columns=[f"{model_name}_proba_1"]
                        ))
                    else:  # Multi-class
                        # Use probabilities for all classes
                        meta_features_list.append(pd.DataFrame(
                            proba,
                            columns=[f"{model_name}_proba_{i}" for i in range(proba.shape[1])]
                        ))
                else:  # For regression or when use_proba is False
                    # Use the raw predictions
                    preds = model.predict(X).reshape(-1, 1)
                    meta_features_list.append(pd.DataFrame(
                        preds,
                        columns=[f"{model_name}_pred"]
                    ))
            except Exception as e:
                logger.warning(f"Error generating meta-features for {model_name}: {str(e)}")
                # If a model fails, we'll use a column of zeros
                meta_features_list.append(pd.DataFrame(
                    np.zeros((X.shape[0], 1)),
                    columns=[f"{model_name}_pred"]
                ))
        
        # Combine all meta-features
        meta_features = pd.concat(meta_features_list, axis=1)
        
        # Add original features if configured to do so
        if self.params['include_original']:
            # Only include top features to avoid dimensionality issues
            # This can be tuned based on feature importance or domain knowledge
            top_feature_count = min(20, X.shape[1])  # Limit to top 20 features or fewer if X has fewer
            
            # If we have feature importances, use them to select top features
            if hasattr(self, 'feature_importances_') and self.feature_importances_ is not None:
                top_features = list(self.feature_importances_.keys())[:top_feature_count]
            else:  # Otherwise just take the first N features
                top_features = list(X.columns)[:top_feature_count]
            
            # Add the original features
            meta_features = pd.concat([meta_features, X[top_features]], axis=1)
        
        return meta_features
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the ensemble stacking model on the provided data
        
        Args:
            X: Feature matrix with all team, player, context, and odds features
            y: Target variable (win/loss for classification, values for regression)
        """
        if not self.base_models:
            raise ValueError("No base models provided. Call set_base_models() first.")
            
        try:
            # Store feature names for later use
            self.feature_names = list(X.columns)
            
            # First, train all base models on the full dataset
            for model_name, model in self.base_models.items():
                logger.info(f"Training base model {model_name} for {self.name}")
                try:
                    model.train(X, y)
                except Exception as inner_e:
                    logger.error(f"Error training base model {model_name}: {str(inner_e)}")
                    logger.warning(f"Continuing without base model {model_name}")
            
            # Generate meta-features using cross-validation
            logger.info(f"Generating meta-features using {self.params['cv_folds']}-fold cross-validation")
            
            # Initialize cross-validation
            kf = KFold(n_splits=self.params['cv_folds'], shuffle=True, random_state=self.params['random_state'])
            
            # Initialize containers for meta-features and CV models
            meta_features = np.zeros((X.shape[0], 0))  # Empty array to hold meta-features
            self.cv_base_models = {i: {} for i in range(self.params['cv_folds'])}  # CV models by fold
            
            # Generate meta-features using cross-validation
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train base models on this fold
                fold_base_models = {}
                for model_name, model in self.base_models.items():
                    # Create a fresh copy of the model for this fold
                    fold_model = type(model)(
                        name=f"{model_name}_fold{fold_idx}",
                        prediction_target=self.prediction_target,
                        version=1
                    )
                    # Train on fold training data
                    try:
                        fold_model.train(X_train, y_train)
                        fold_base_models[model_name] = fold_model
                    except Exception as inner_e:
                        logger.warning(f"Error training {model_name} on fold {fold_idx}: {str(inner_e)}")
                
                # Store the fold models
                self.cv_base_models[fold_idx] = fold_base_models
                
                # Generate meta-features for validation set
                fold_meta_features = self._generate_meta_features(X_val, fold_base_models)
                
                # Store these meta-features in the correct positions
                if fold_idx == 0:  # First fold, initialize meta_features with correct shape
                    meta_features = np.zeros((X.shape[0], fold_meta_features.shape[1]))
                
                meta_features[val_idx, :] = fold_meta_features.values
            
            # Convert meta-features to DataFrame
            meta_features_df = pd.DataFrame(meta_features, columns=fold_meta_features.columns)
            
            # Train the meta-learner on the meta-features
            logger.info(f"Training meta-learner for {self.name}")
            
            # Initialize the meta-learner based on model type
            if self.model_type == 'classification':
                if self.params['meta_learner'] == 'logistic_regression':
                    self.model = LogisticRegression(
                        C=self.params['regularization'],
                        max_iter=self.params['max_iter'],
                        random_state=self.params['random_state']
                    )
                else:  # Default to logistic regression
                    logger.warning(f"Unknown meta-learner '{self.params['meta_learner']}'. Using logistic_regression.")
                    self.model = LogisticRegression(
                        C=self.params['regularization'],
                        max_iter=self.params['max_iter'],
                        random_state=self.params['random_state']
                    )
            else:  # regression
                if self.params['meta_learner'] == 'ridge':
                    self.model = Ridge(
                        alpha=self.params['alpha'],
                        max_iter=self.params['max_iter'],
                        random_state=self.params['random_state']
                    )
                else:  # Default to ridge regression
                    logger.warning(f"Unknown meta-learner '{self.params['meta_learner']}'. Using ridge.")
                    self.model = Ridge(
                        alpha=self.params['alpha'],
                        max_iter=self.params['max_iter'],
                        random_state=self.params['random_state']
                    )
            
            # Train meta-learner on the meta-features
            self.model.fit(meta_features_df, y)
            
            # Generate feature importances for the meta-learner
            if hasattr(self.model, 'coef_'):
                # Get the coefficients as importances
                coef = self.model.coef_
                if len(coef.shape) > 1:  # For multi-class classification
                    coef = np.abs(coef).mean(axis=0)  # Average absolute coefficients across classes
                
                # Create feature importance dictionary
                self.feature_importances_ = {}
                for i, feature_name in enumerate(meta_features_df.columns):
                    self.feature_importances_[feature_name] = abs(float(coef[i]))
                
                # Sort by importance
                self.feature_importances_ = dict(sorted(
                    self.feature_importances_.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
            
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
        Make predictions using the trained ensemble stacking model
        
        Args:
            X: Feature matrix matching the training features
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            logger.error(f"Cannot predict with untrained model: {self.name}")
            return np.array([])
        
        try:
            # Generate meta-features using the fully trained base models
            meta_features = self._generate_meta_features(X, self.base_models)
            self.meta_features = meta_features  # Store for potential inspection
            
            # Make predictions using the meta-learner
            predictions = self.model.predict(meta_features)
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
        
        if not hasattr(self.model, 'predict_proba'):
            logger.warning(f"Meta-learner {self.params['meta_learner']} does not support predict_proba")
            return np.array([])
        
        try:
            # Generate meta-features using the fully trained base models
            meta_features = self._generate_meta_features(X, self.base_models)
            self.meta_features = meta_features  # Store for potential inspection
            
            # Make probability predictions using the meta-learner
            probabilities = self.model.predict_proba(meta_features)
            return probabilities
            
        except Exception as e:
            logger.error(f"Error predicting probabilities with {self.name} model: {str(e)}")
            return np.array([])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance for meta-features
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            logger.error(f"Cannot get feature importance from untrained model: {self.name}")
            return {}
        
        try:
            # Return the pre-computed feature importances if available
            if hasattr(self, 'feature_importances_') and self.feature_importances_ is not None:
                return self.feature_importances_
            
            # Otherwise, if we have coefficients, compute them now
            if hasattr(self.model, 'coef_'):
                # Get the coefficients as importances
                coef = self.model.coef_
                if len(coef.shape) > 1:  # For multi-class classification
                    coef = np.abs(coef).mean(axis=0)  # Average absolute coefficients across classes
                
                # We need meta-features to get column names
                if self.meta_features is None:
                    logger.warning("No meta-features available for feature importance calculation")
                    return {}
                
                # Create feature importance dictionary
                feature_importance = {}
                for i, feature_name in enumerate(self.meta_features.columns):
                    feature_importance[feature_name] = abs(float(coef[i]))
                
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                return feature_importance
            
            # If we don't have coefficients or pre-computed importances, return empty dict
            logger.warning("Meta-learner does not provide feature importances")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {self.name} model: {str(e)}")
            return {}
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the ensemble stacking model performance on test data
        
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
            
            # Add comparison to base models
            for model_name, model in self.base_models.items():
                try:
                    base_metrics = model.evaluate(X, y)
                    for metric_name, value in base_metrics.items():
                        metrics[f"base_{model_name}_{metric_name}"] = value
                except Exception as inner_e:
                    logger.warning(f"Error evaluating base model {model_name}: {str(inner_e)}")
            
            # Calculate improvement over base models
            if 'accuracy' in metrics:
                base_accuracies = [v for k, v in metrics.items() if k.startswith('base_') and k.endswith('_accuracy')]
                if base_accuracies:
                    metrics['improvement_over_avg_base'] = metrics['accuracy'] - sum(base_accuracies) / len(base_accuracies)
                    metrics['improvement_over_best_base'] = metrics['accuracy'] - max(base_accuracies)
            
            # Calculate meta-feature importance summary
            importances = self.get_feature_importance()
            if importances:
                # Group importances by base model
                model_importance = {}
                for feature, importance in importances.items():
                    for model_name in self.base_model_names:
                        if feature.startswith(model_name):
                            model_importance[model_name] = model_importance.get(model_name, 0) + importance
                            break
                
                # Include in metrics
                for model_name, importance in model_importance.items():
                    metrics[f"meta_importance_{model_name}"] = importance
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {self.name} model: {str(e)}")
            return {}
    
    def save_to_db(self) -> bool:
        """
        Save the ensemble stacking model to the database
        
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if not self.is_trained:
            logger.error(f"Cannot save untrained model: {self.name}")
            return False
        
        try:
            # Get feature importance
            feature_importance = self.get_feature_importance()
            
            # Get meta-learner coefficients if available
            meta_coefficients = None
            if hasattr(self.model, 'coef_'):
                meta_coefficients = self.model.coef_.tolist()
            
            # Create model weights dictionary
            model_data = {
                'model_name': self.name,
                'model_type': self.model_type,
                'weights': {
                    'feature_importances': feature_importance,
                    'meta_coefficients': meta_coefficients,
                    'base_models': self.base_model_names,
                    'meta_learner': self.params['meta_learner'],
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
