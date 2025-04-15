"""
Model Mixing for NBA Prediction System

This module implements a model mixing approach that combines outputs from multiple
prediction models based on their recent performance. Instead of relying on a single
model, this approach creates a weighted ensemble that leverages the strengths of
each model to produce more robust predictions.

Features Used:
- Outputs from other models: Random Forests, Combined Gradient Boosting, Bayesian
- Performance Metrics: 7-day accuracy for each model on different prediction types
- Context: Season phase (regular season vs playoffs) for adaptive weighting

The model mixing approach helps manage model uncertainty by giving more weight to
models that have performed well recently on similar prediction tasks, resulting in
more reliable predictions overall.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from src.models.base_model import BaseModel
from src.database.models import ModelPerformance

# Configure logging
logger = logging.getLogger(__name__)


class ModelMixing(BaseModel):
    """
    Model Mixing class that combines predictions from multiple models
    
    This class doesn't train a model directly but rather combines the outputs
    of other trained models based on their recent performance metrics.
    """
    
    def __init__(self, name: str = "ModelMixing", prediction_target: str = "moneyline", version: int = 1):
        """
        Initialize the Model Mixing component
        
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
        
        # Model configuration
        self.params = {
            'base_models': ['RandomForest', 'CombinedGBM', 'Bayesian'],
            'min_weight': 0.05,  # Minimum weight for any model
            'performance_window': '7d',  # Performance window to consider
            'weight_method': 'performance_based',  # How to determine weights
            'use_playoffs_weighting': True,  # Whether to adjust weights for playoffs
            'default_weights': {  # Default weights if no performance data available
                'RandomForest': 0.4,
                'CombinedGBM': 0.4,
                'Bayesian': 0.2,
                'AnomalyDetection': 0.0,  # Anomaly detection not used by default in mixing
            }
        }
        
        # Attributes for model weights
        self.model_weights = self.params['default_weights'].copy()
        self.is_playoffs = False  # Flag for playoff vs regular season
        self.base_models = {}  # Will hold the actual model instances
    
    def set_base_models(self, models: Dict[str, BaseModel]) -> None:
        """
        Set the base models to use for mixing
        
        Args:
            models: Dictionary mapping model names to model instances
        """
        self.base_models = models
        # Update the list of base model names
        self.params['base_models'] = list(models.keys())
        # Initialize weights
        self.update_weights()
    
    def set_playoffs_mode(self, is_playoffs: bool) -> None:
        """
        Set whether current predictions are for playoff games
        
        Args:
            is_playoffs: True if predictions are for playoff games, False otherwise
        """
        self.is_playoffs = is_playoffs
        # Update weights when changing season phase
        self.update_weights()
    
    def update_weights(self) -> None:
        """
        Update model weights based on recent performance
        
        This method retrieves recent performance metrics from the database
        and adjusts weights accordingly.
        """
        try:
            # Start with default weights
            model_weights = self.params['default_weights'].copy()
            
            # If using performance-based weighting, get recent model performance
            if self.params['weight_method'] == 'performance_based':
                weights = {}
                total_weight = 0.0
                min_weight = self.params['min_weight']
                
                # Get performance for each base model
                for model_name in self.params['base_models']:
                    # Skip models that aren't in the system
                    if model_name not in self.model_weights:
                        continue
                        
                    try:
                        # Get latest performance from database
                        performance = ModelPerformance.get_latest_performance(
                            model_name=model_name,
                            prediction_type=self.prediction_target,
                            window=self.params['performance_window']
                        )
                        
                        if performance and 'metrics' in performance:
                            # Use accuracy or relevant metric based on model type
                            if self.model_type == 'classification':
                                metric_value = performance['metrics'].get('accuracy', 0.5)
                            else:  # regression
                                # For regression, we want lower error (convert to higher = better)
                                metric_value = 1.0 / (1.0 + performance['metrics'].get('rmse', 1.0))
                            
                            # Ensure metric is within reasonable bounds
                            metric_value = max(0.01, min(0.99, metric_value))
                            
                            # Set weight based on metric value
                            weights[model_name] = metric_value
                            total_weight += metric_value
                        else:
                            # No performance data, use default weight
                            weights[model_name] = self.model_weights[model_name]
                            total_weight += weights[model_name]
                            
                    except Exception as inner_e:
                        logger.warning(f"Error getting performance for {model_name}: {str(inner_e)}")
                        # Fall back to default weight
                        weights[model_name] = self.model_weights[model_name]
                        total_weight += weights[model_name]
                
                # Normalize weights if we have a positive total
                if total_weight > 0:
                    for model_name in weights:
                        # Normalize and ensure minimum weight
                        weights[model_name] = max(min_weight, weights[model_name] / total_weight)
                    
                    # Re-normalize after enforcing minimum weights
                    total_weight = sum(weights.values())
                    for model_name in weights:
                        weights[model_name] /= total_weight
                    
                    # Update model weights
                    model_weights = weights
            
            # Adjust weights for playoffs if needed
            if self.params['use_playoffs_weighting'] and self.is_playoffs:
                # During playoffs, we might want to adjust weights differently
                # For example, increase weight for models that handle playoff intensity well
                # This is just an example adjustment - customize based on historical performance
                if 'Bayesian' in model_weights:
                    # Example: Increase Bayesian model weight for playoffs (more uncertainty)
                    bayesian_weight = model_weights['Bayesian'] * 1.2  # 20% increase
                    model_weights['Bayesian'] = bayesian_weight
                    
                    # Reduce other weights proportionally
                    total_adjustment = bayesian_weight - self.model_weights['Bayesian']
                    for model_name in model_weights:
                        if model_name != 'Bayesian':
                            weight_reduction = total_adjustment * (model_weights[model_name] / sum(
                                model_weights[m] for m in model_weights if m != 'Bayesian'
                            ))
                            model_weights[model_name] -= weight_reduction
            
            # Update the model weights
            self.model_weights = model_weights
            logger.info(f"Updated model weights for {self.name}: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Error updating model weights: {str(e)}")
            # Keep existing weights if update fails
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Model Mixing doesn't require traditional training
        
        Instead, it updates weights based on recent performance metrics
        
        Args:
            X: Feature matrix (not used directly)
            y: Target variable (not used directly)
        """
        try:
            # Store feature names for consistency with other models
            self.feature_names = list(X.columns)
            
            # Update model weights
            self.update_weights()
            
            # Set the model attribute to the weights dictionary for serialization
            self.model = self.model_weights
            
            # Record training time
            self.trained_at = datetime.now(timezone.utc)
            self.is_trained = True
            
            logger.info(f"{self.name} model initialized with weights: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Error initializing {self.name} model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions by combining outputs from base models
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of mixed predictions
        """
        if not self.is_trained:
            logger.error(f"Cannot predict with untrained model: {self.name}")
            return np.array([])
        
        if not self.base_models:
            logger.error(f"No base models provided for {self.name}")
            return np.array([])
        
        try:
            # Get predictions from each base model
            model_predictions = {}
            for model_name, model in self.base_models.items():
                if model_name in self.model_weights and self.model_weights[model_name] > 0:
                    model_predictions[model_name] = model.predict(X)
            
            if not model_predictions:
                logger.error("No valid predictions from base models")
                return np.array([])
            
            # Check that predictions have compatible shapes
            shapes = [pred.shape for pred in model_predictions.values()]
            if not all(shape[0] == shapes[0][0] for shape in shapes):
                logger.error("Incompatible prediction shapes from base models")
                return np.array([])
            
            # Combine predictions using weights
            if self.model_type == 'classification':
                # For classification, we need special handling
                # First, get probabilities if available
                try:
                    return self.predict_proba(X).argmax(axis=1) if X.shape[0] > 0 else np.array([])
                except:
                    # If predict_proba fails, try to combine class labels directly
                    # This is less ideal but can work for binary classification
                    weighted_sum = np.zeros(shapes[0][0])
                    for model_name, predictions in model_predictions.items():
                        weighted_sum += self.model_weights[model_name] * predictions
                    
                    # Round to nearest class label
                    return np.round(weighted_sum).astype(int)
            else:  # regression
                # For regression, simple weighted average
                weighted_sum = np.zeros(shapes[0][0])
                for model_name, predictions in model_predictions.items():
                    weighted_sum += self.model_weights[model_name] * predictions
                
                return weighted_sum
            
        except Exception as e:
            logger.error(f"Error predicting with {self.name} model: {str(e)}")
            return np.array([])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities by combining probability outputs from base models
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of mixed class probabilities
        """
        if not self.is_trained:
            logger.error(f"Cannot predict with untrained model: {self.name}")
            return np.array([])
        
        if self.model_type != 'classification':
            logger.warning(f"predict_proba is only available for classification models, not {self.model_type}")
            return np.array([])
        
        if not self.base_models:
            logger.error(f"No base models provided for {self.name}")
            return np.array([])
        
        try:
            # Get probability predictions from each base model
            model_probas = {}
            for model_name, model in self.base_models.items():
                if model_name in self.model_weights and self.model_weights[model_name] > 0:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        if proba.shape[0] > 0:  # Check for non-empty array
                            model_probas[model_name] = proba
            
            if not model_probas:
                logger.error("No valid probability predictions from base models")
                return np.array([])
            
            # Check that probability outputs have compatible shapes
            shapes = [proba.shape for proba in model_probas.values()]
            if not all(shape == shapes[0] for shape in shapes):
                logger.error("Incompatible probability shapes from base models")
                return np.array([])
            
            # Combine probabilities using weights
            weighted_proba = np.zeros(shapes[0])
            total_weight = 0.0
            
            for model_name, proba in model_probas.items():
                model_weight = self.model_weights.get(model_name, 0.0)
                weighted_proba += model_weight * proba
                total_weight += model_weight
            
            # Normalize by total weight if not zero
            if total_weight > 0:
                weighted_proba /= total_weight
                
            # Ensure probabilities sum to 1 across classes
            row_sums = weighted_proba.sum(axis=1).reshape(-1, 1)
            weighted_proba = weighted_proba / row_sums
            
            return weighted_proba
            
        except Exception as e:
            logger.error(f"Error predicting probabilities with {self.name} model: {str(e)}")
            return np.array([])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance by combining importances from base models
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            logger.error(f"Cannot get feature importance from untrained model: {self.name}")
            return {}
        
        if not self.base_models:
            logger.error(f"No base models provided for {self.name}")
            return {}
        
        try:
            # Get feature importances from each base model
            all_importances = {}
            
            for model_name, model in self.base_models.items():
                if model_name in self.model_weights and self.model_weights[model_name] > 0:
                    model_importance = model.get_feature_importance()
                    if model_importance:
                        all_importances[model_name] = model_importance
            
            if not all_importances:
                logger.warning("No feature importances available from base models")
                return {}
            
            # Combine feature importances using weights
            combined_importance = {}
            features = set()
            
            # Collect all unique features
            for importances in all_importances.values():
                features.update(importances.keys())
            
            # Initialize combined importance for all features
            for feature in features:
                combined_importance[feature] = 0.0
            
            # Add weighted importances
            total_weight = 0.0
            for model_name, importances in all_importances.items():
                model_weight = self.model_weights.get(model_name, 0.0)
                if model_weight > 0:
                    for feature, importance in importances.items():
                        if feature in combined_importance:
                            combined_importance[feature] += model_weight * importance
                    total_weight += model_weight
            
            # Normalize by total weight if not zero
            if total_weight > 0:
                for feature in combined_importance:
                    combined_importance[feature] /= total_weight
            
            # Sort by importance (highest to lowest)
            combined_importance = dict(sorted(combined_importance.items(), key=lambda x: x[1], reverse=True))
            
            return combined_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {self.name} model: {str(e)}")
            return {}
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model mixing performance on test data
        
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
            
            # Add model weights to metrics
            for model_name, weight in self.model_weights.items():
                metrics[f"{model_name}_weight"] = weight
            
            # Evaluate individual models for comparison
            if self.base_models:
                for model_name, model in self.base_models.items():
                    if model_name in self.model_weights and self.model_weights[model_name] > 0:
                        model_metrics = model.evaluate(X, y)
                        for metric_name, value in model_metrics.items():
                            metrics[f"{model_name}_{metric_name}"] = value
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {self.name} model: {str(e)}")
            return {}
    
    def save_to_db(self) -> bool:
        """
        Save the model mixing weights to the database
        
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if not self.is_trained:
            logger.error(f"Cannot save untrained model: {self.name}")
            return False
        
        try:
            # Get feature importance
            feature_importance = self.get_feature_importance()
            
            # Create model weights dictionary
            model_data = {
                'model_name': self.name,
                'model_type': self.model_type,
                'weights': {
                    'feature_importances': feature_importance,
                    'model_weights': self.model_weights,
                    'prediction_target': self.prediction_target,
                    'is_playoffs': self.is_playoffs
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

# Create an alias for the class that matches what's imported in auto_train_manager.py
# This ensures backward compatibility with existing code
ModelMixingModel = ModelMixing
