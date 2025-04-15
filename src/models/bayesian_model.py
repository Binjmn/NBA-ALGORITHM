"""
Bayesian Model for NBA Prediction System

This module implements a Bayesian model for generating probabilistic predictions
about game outcomes and player performance. It excels at updating probabilities
based on new information and provides uncertainty estimates with each prediction.

Features Used:
- Team: Recent performance metrics, strength of schedule, home/away splits, rest days,
  travel distance, lineup efficiency, clutch performance, pace/style, recent overtimes,
  fatigue model, venue effects, team chemistry
- Player: Efficiency metrics, injury impact, matchup performance history, usage rate,
  rookie adjustments
- Context: Rivalry/motivation factors, playoff vs. regular season intensity
- Odds: Game-level odds, player prop odds

The Bayesian approach handles uncertainty naturally, producing probabilities rather
than point estimates, which is particularly valuable for models that inform betting
decisions. It's especially responsive to dynamic inputs like injury status and odds shifts.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import BayesianRidge
from datetime import datetime, timezone

from src.models.base_model import BaseModel

# Configure logging
logger = logging.getLogger(__name__)


class BayesianModel(BaseModel):
    """
    Bayesian model for NBA prediction
    
    This model uses Bayesian methods to generate probabilistic predictions,
    providing not just point estimates but also confidence intervals and
    uncertainty measures.
    
    It can use Gaussian Naive Bayes for classification tasks, Bayesian Ridge
    for regression tasks, or more advanced Bayesian methods depending on
    the prediction requirements.
    """
    
    def __init__(self, name: str = "Bayesian", prediction_target: str = "moneyline", version: int = 1):
        """
        Initialize the Bayesian model
        
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
        
        # Default hyperparameters
        if self.model_type == 'classification':
            self.params = {
                'var_smoothing': 1e-9,  # For GaussianNB
                'use_bagging': True,    # Whether to use bagging for stability
                'n_estimators': 10,     # Number of bagged estimators
                'max_samples': 0.8,     # Fraction of samples for bagging
                'random_state': 42
            }
        else:  # regression
            self.params = {
                'alpha_1': 1e-6,          # Precision of noise
                'alpha_2': 1e-6,          # Precision of weights
                'lambda_1': 1e-6,         # Precision of noise
                'lambda_2': 1e-6,         # Precision of weights
                'n_iter': 300,            # Max iterations
                'compute_score': True,    # Compute objective function
                'fit_intercept': True,    # Fit intercept term
                'normalize': False,       # Normalize features
                'random_state': 42
            }
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the Bayesian model on the provided data
        
        Args:
            X: Feature matrix with all relevant basketball and betting features
            y: Target variable (win/loss for classification, values for regression)
        """
        try:
            # Store feature names for later use
            self.feature_names = list(X.columns)
            
            # Initialize the appropriate Bayesian model based on model type
            if self.model_type == 'classification':
                base_model = GaussianNB(var_smoothing=self.params['var_smoothing'])
                
                # Use bagging with the Naive Bayes classifier for more stability
                if self.params.get('use_bagging', True):
                    self.model = BaggingClassifier(
                        base_estimator=base_model,
                        n_estimators=self.params['n_estimators'],
                        max_samples=self.params['max_samples'],
                        random_state=self.params['random_state']
                    )
                else:
                    self.model = base_model
            else:  # 'regression'
                # Use Bayesian Ridge Regression
                self.model = BayesianRidge(
                    alpha_1=self.params['alpha_1'],
                    alpha_2=self.params['alpha_2'],
                    lambda_1=self.params['lambda_1'],
                    lambda_2=self.params['lambda_2'],
                    n_iter=self.params['n_iter'],
                    compute_score=self.params['compute_score'],
                    fit_intercept=self.params['fit_intercept'],
                    normalize=self.params['normalize'],
                    random_state=self.params['random_state']
                )
            
            # Handle categorical features
            # Bayesian models often need preprocessing for categorical features
            # For this implementation, we assume the features are already properly encoded
            
            # Train the model
            logger.info(f"Training {self.name} model with {len(X)} samples")
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
        Make predictions using the trained Bayesian model
        
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
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
                return probabilities
            else:
                logger.warning(f"{self.name} model does not support predict_proba")
                return np.array([])
            
        except Exception as e:
            logger.error(f"Error predicting probabilities with {self.name} model: {str(e)}")
            return np.array([])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance from the trained Bayesian model
        
        This is more challenging for Bayesian models, especially Naive Bayes,
        but we can approximate it using coefficient magnitudes for regression
        or class conditional variance for classification.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            logger.error(f"Cannot get feature importance from untrained model: {self.name}")
            return {}
        
        try:
            feature_importance = {}
            
            if self.model_type == 'classification':
                # For Gaussian Naive Bayes, we can use the variance of each feature as
                # a rough proxy for importance (higher variance → more information)
                if isinstance(self.model, GaussianNB):
                    for i, feature_name in enumerate(self.feature_names):
                        # Combine variances across classes (weighted by class prior)
                        weighted_var = np.sum(self.model.var_ * self.model.class_prior_[:, np.newaxis], axis=0)[i]
                        feature_importance[feature_name] = float(weighted_var)
                elif hasattr(self.model, 'estimators_'):
                    # For bagged models, we can get feature importances from estimators if they're available
                    importances = np.zeros(len(self.feature_names))
                    for estimator in self.model.estimators_:
                        if hasattr(estimator, 'var_'):
                            # Similar approach as above for each base estimator
                            weighted_var = np.sum(estimator.var_ * estimator.class_prior_[:, np.newaxis], axis=0)
                            importances += weighted_var
                    importances /= len(self.model.estimators_)
                    for i, feature_name in enumerate(self.feature_names):
                        feature_importance[feature_name] = float(importances[i])
                else:
                    # Fallback: equal importance
                    for feature_name in self.feature_names:
                        feature_importance[feature_name] = 1.0 / len(self.feature_names)
            else:  # 'regression'
                # For Bayesian Ridge, we can use the coefficient magnitudes
                if hasattr(self.model, 'coef_'):
                    for i, feature_name in enumerate(self.feature_names):
                        feature_importance[feature_name] = abs(float(self.model.coef_[i]))
                else:
                    # Fallback: equal importance
                    for feature_name in self.feature_names:
                        feature_importance[feature_name] = 1.0 / len(self.feature_names)
            
            # Normalize to sum to 1
            total = sum(feature_importance.values())
            if total > 0:
                for feature in feature_importance:
                    feature_importance[feature] /= total
            
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
            
            # Add Bayesian specific metrics if applicable
            if self.model_type == 'regression' and hasattr(self.model, 'scores_'):
                # For Bayesian Ridge, we can include the model score progression
                metrics['final_score'] = float(self.model.scores_[-1])
                
                # Add standard error of predictions if available
                if hasattr(self.model, 'predict') and hasattr(self.model, 'predict_std'):
                    _, std = self.model.predict(X, return_std=True)
                    metrics['avg_prediction_std'] = float(np.mean(std))
            
            # Add calibration metrics for classification (if probabilities available)
            if self.model_type == 'classification' and hasattr(self.model, 'predict_proba'):
                try:
                    from sklearn.calibration import calibration_curve
                    prob_pos = self.model.predict_proba(X)[:, 1]
                    fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos, n_bins=10)
                    # Calculate the calibration error (mean absolute difference)
                    calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                    metrics['calibration_error'] = float(calibration_error)
                except Exception as cal_err:
                    logger.warning(f"Error calculating calibration metrics: {str(cal_err)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {self.name} model: {str(e)}")
            return {}
    
    def save_to_db(self) -> bool:
        """
        Save the model to the database with Bayesian specific details
        
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
            
            # Create model weights dictionary with additional information
            model_data = {
                'model_name': self.name,
                'model_type': self.model_type,
                'weights': {
                    'feature_importances': feature_importance,
                    'prediction_target': self.prediction_target
                },
                'params': params,
                'version': self.version,
                'trained_at': self.trained_at or datetime.now(timezone.utc),
                'active': True
            }
            
            # Add model-specific additional data
            if self.model_type == 'classification':
                if hasattr(self.model, 'class_prior_'):
                    model_data['weights']['class_prior'] = self.model.class_prior_.tolist()
                    
                # If using BaggingClassifier, include number of estimators
                if hasattr(self.model, 'estimators_'):
                    model_data['weights']['n_estimators'] = len(self.model.estimators_)
            else:  # regression
                if hasattr(self.model, 'alpha_') and hasattr(self.model, 'lambda_'):
                    model_data['weights']['alpha'] = float(self.model.alpha_)
                    model_data['weights']['lambda'] = float(self.model.lambda_)
                    
                if hasattr(self.model, 'coef_'):
                    model_data['weights']['coefficients'] = self.model.coef_.tolist()
                
                if hasattr(self.model, 'sigma_'):
                    model_data['weights']['sigma'] = float(self.model.sigma_)
            
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
