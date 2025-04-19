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
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.base_model import BaseModel

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
        self.weights_source = "default"  # Track the source of weights for logging
        self.using_default_weights = True  # Flag to track if we're using default weights
    
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
        
        This method retrieves recent performance metrics from disk
        and adjusts weights accordingly.
        """
        try:
            # Start with default weights
            model_weights = self.params['default_weights'].copy()
            using_default_weights = True  # Flag to track if we're using default weights
            weights_source = "default"  # Track the source of weights for logging
            
            # If using performance-based weighting, get recent model performance
            if self.params['weight_method'] == 'performance_based':
                weights = {}
                total_weight = 0.0
                min_weight = self.params['min_weight']
                performance_data_found = False  # Flag to track if any performance data was found
                
                # Get performance for each base model
                for model_name in self.params['base_models']:
                    # Skip models that aren't in the system
                    if model_name not in self.model_weights:
                        continue
                        
                    try:
                        # Get latest performance from files
                        perf_dir = os.path.join('data', 'performance')
                        if not os.path.exists(perf_dir):
                            logger.warning(f"Performance directory not found: {perf_dir}")
                            # Use default weight
                            weights[model_name] = self.model_weights[model_name]
                            total_weight += weights[model_name]
                            continue
                            
                        # Find all performance files for this model and prediction type
                        window = self.params['performance_window']
                        prefix = f"{model_name}_{self.prediction_target}_{window}_"
                        performance_files = [f for f in os.listdir(perf_dir) 
                                       if f.startswith(prefix) and f.endswith('.json')]
                        
                        if not performance_files:
                            logger.info(f"No performance data found for {model_name}, using default weight")
                            weights[model_name] = self.model_weights[model_name]
                            total_weight += weights[model_name]
                            continue
                            
                        # Sort by timestamp and get the latest file
                        performance_files.sort(reverse=True)
                        latest_file = os.path.join(perf_dir, performance_files[0])
                        
                        # Load the performance data
                        with open(latest_file, 'r') as f:
                            performance = json.load(f)
                        
                        if performance and 'metrics' in performance:
                            performance_data_found = True  # Found valid performance data
                            
                            # Use accuracy or relevant metric based on model type
                            if self.model_type == 'classification':
                                metric_value = performance['metrics'].get('accuracy', 0.5)
                                logger.info(f"Found {model_name} classification accuracy: {metric_value:.4f}")
                            else:  # regression
                                # For regression, we want lower error (convert to higher = better)
                                rmse = performance['metrics'].get('rmse', 1.0)
                                metric_value = 1.0 / (1.0 + rmse)
                                logger.info(f"Found {model_name} regression RMSE: {rmse:.4f} (converted to {metric_value:.4f})")
                            
                            # Ensure metric is within reasonable bounds
                            metric_value = max(0.01, min(0.99, metric_value))
                            
                            # Set weight based on metric value
                            weights[model_name] = metric_value
                            total_weight += metric_value
                        else:
                            # Invalid performance data, use default weight
                            logger.warning(f"Invalid performance data for {model_name}, using default weight")
                            weights[model_name] = self.model_weights[model_name]
                            total_weight += weights[model_name]
                            
                    except Exception as inner_e:
                        logger.warning(f"Error getting performance for {model_name}: {str(inner_e)}")
                        # Use default weight
                        weights[model_name] = self.model_weights[model_name]
                        total_weight += weights[model_name]
                
                # Normalize weights if we have a positive total and found performance data
                if total_weight > 0:
                    if performance_data_found:
                        using_default_weights = False
                        weights_source = "performance-based"
                        
                        for model_name in weights:
                            # Normalize and ensure minimum weight
                            weights[model_name] = max(min_weight, weights[model_name] / total_weight)
                        
                        # Re-normalize after enforcing minimum weights
                        total_weight = sum(weights.values())
                        for model_name in weights:
                            weights[model_name] /= total_weight
                        
                        # Update model weights
                        model_weights = weights
                        
                        # Compare with default weights to see the difference
                        diff_str = []
                        for model_name in model_weights:
                            if model_name in self.params['default_weights']:
                                default = self.params['default_weights'][model_name]
                                current = model_weights[model_name]
                                diff = current - default
                                diff_str.append(f"{model_name}: {current:.3f} ({diff:+.3f})")
                        
                        logger.info(f"Using performance-based weights: {', '.join(diff_str)}")
                    else:
                        logger.warning("No valid performance data found for any model, using default weights")
            
            # Adjust weights for playoffs if needed
            if self.params['use_playoffs_weighting'] and self.is_playoffs:
                weights_source += "+playoffs-adjusted"
                logger.info("Applying playoff adjustments to model weights")
                
                # During playoffs, we might want to adjust weights differently
                # For example, increase weight for models that handle playoff intensity well
                # This is just an example adjustment - customize based on historical performance
                if 'Bayesian' in model_weights:
                    # Example: Increase Bayesian model weight for playoffs (more uncertainty)
                    original_weight = model_weights['Bayesian']
                    bayesian_weight = original_weight * 1.2  # 20% increase
                    model_weights['Bayesian'] = bayesian_weight
                    
                    # Reduce other weights proportionally
                    total_adjustment = bayesian_weight - original_weight
                    for model_name in model_weights:
                        if model_name != 'Bayesian':
                            weight_reduction = total_adjustment * (model_weights[model_name] / sum(
                                model_weights[m] for m in model_weights if m != 'Bayesian'
                            ))
                            model_weights[model_name] -= weight_reduction
                    
                    # Log the playoff adjustments
                    logger.info(f"Playoffs adjustment: Increased Bayesian model weight from {original_weight:.3f} to {bayesian_weight:.3f}")
            
            # Track if weights changed from previous weights
            weights_changed = any(abs(self.model_weights.get(model, 0) - weight) > 0.01 
                                for model, weight in model_weights.items())
            
            # Update the model weights
            self.model_weights = model_weights
            
            # Store weight source for reference
            self.weights_source = weights_source
            self.using_default_weights = using_default_weights
            
            # Log weight update details
            if weights_changed:
                logger.info(f"Updated model weights for {self.name} ({weights_source}): {self.model_weights}")
            else:
                logger.info(f"Model weights unchanged for {self.name} (source: {weights_source})")
            
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
                except Exception as e:
                    logger.error(f"Error in predict_proba for classification: {str(e)}")
                    # If predict_proba fails, we should not proceed with a fallback
                    return np.array([])
            else:  # regression
                # For regression, simple weighted average
                weighted_sum = np.zeros(shapes[0][0])
                for model_name, predictions in model_predictions.items():
                    weighted_sum += self.model_weights[model_name] * predictions
                
                return weighted_sum
            
        except Exception as e:
            logger.error(f"Error predicting with {self.name} model: {str(e)}")
            return np.array([])
    
    def record_prediction_performance(self, X: pd.DataFrame, y_true: np.ndarray) -> bool:
        """
        Record performance metrics for all base models after making predictions
        
        This method should be called after predictions are made and actual outcomes are known.
        It ensures that performance metrics are regularly updated to keep model weights accurate.
        
        Args:
            X: Feature matrix used for predictions
            y_true: Actual outcome values
            
        Returns:
            bool: True if recording was successful for at least one model, False otherwise
        """
        if not self.is_trained or not self.base_models:
            logger.error("Cannot record performance without trained base models")
            return False
            
        success = False
        logger.info(f"Recording performance metrics for {len(self.base_models)} base models")
        
        try:
            for model_name, model in self.base_models.items():
                if model_name not in self.model_weights:
                    continue
                    
                try:
                    # Get predictions from this model
                    if self.model_type == 'classification':
                        y_pred = model.predict(X)
                        
                        # Calculate metrics
                        metrics = {
                            'accuracy': float(accuracy_score(y_true, y_pred)),
                            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                        }
                    else:  # regression
                        y_pred = model.predict(X)
                        
                        # Calculate metrics
                        metrics = {
                            'rmse': float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                            'mae': float(np.mean(np.abs(y_true - y_pred))),
                            'r2': float(max(0, 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)))
                        }
                    
                    # Record performance
                    perf_success = model.record_performance(
                        metrics=metrics,
                        prediction_type=self.prediction_target,
                        num_predictions=len(y_true),
                        time_window=self.params['performance_window']
                    )
                    
                    if perf_success:
                        logger.info(f"Recorded performance metrics for {model_name}: {metrics}")
                        success = True
                    else:
                        logger.warning(f"Failed to record performance metrics for {model_name}")
                        
                except Exception as model_error:
                    logger.error(f"Error recording performance for {model_name}: {str(model_error)}")
            
            # Update weights based on new performance data if recording was successful
            if success:
                logger.info("Updating model weights based on new performance data")
                self.update_weights()
                
                # Check if weights changed from defaults
                if not self.using_default_weights:
                    logger.info("Successfully using performance-based weights")
                else:
                    logger.warning("Still using default weights even after recording performance")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in recording prediction performance: {str(e)}")
            return False
    
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
            # Avoid division by zero
            valid_indices = row_sums > 0
            if valid_indices.any():
                weighted_proba[valid_indices.flatten()] = weighted_proba[valid_indices.flatten()] / row_sums[valid_indices]
            
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
                total_weight += model_weight
                
                for feature, importance in importances.items():
                    combined_importance[feature] += model_weight * importance
            
            # Normalize by total weight if not zero
            if total_weight > 0:
                for feature in combined_importance:
                    combined_importance[feature] /= total_weight
            
            return combined_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {self.name} model: {str(e)}")
            return {}
    
    def save_to_disk(self, models_dir: str = 'data/models') -> str:
        """
        Save the model mixing weights and configuration to disk
        
        Args:
            models_dir: Directory to save the model file
            
        Returns:
            Path to the saved model file or empty string if saving failed
        """
        # Create a special version of the model for saving
        save_data = {
            'model_weights': self.model_weights,
            'is_playoffs': self.is_playoffs,
            'params': self.params
        }
        
        # Set the model to this data for base class to save
        self.model = save_data
        
        # Use the base class implementation
        return super().save_to_disk(models_dir)
    
    def load_from_disk(self, models_dir: str = 'data/models', version: Optional[int] = None) -> bool:
        """
        Load the model mixing weights and configuration from disk
        
        Args:
            models_dir: Directory containing the model files
            version: Specific version to load, or latest if None
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        # Use the base class implementation to load the file
        success = super().load_from_disk(models_dir, version)
        
        if success and isinstance(self.model, dict):
            # Extract model weights and configuration
            self.model_weights = self.model.get('model_weights', self.params['default_weights'].copy())
            self.is_playoffs = self.model.get('is_playoffs', False)
            
            # Update params if present in the loaded model
            if 'params' in self.model:
                self.params.update(self.model['params'])
                
            logger.info(f"Loaded model mixing configuration with weights: {self.model_weights}")
            return True
        
        return success

# Create an alias for the class that matches what's imported in auto_train_manager.py
# This ensures backward compatibility with existing code
ModelMixingModel = ModelMixing
