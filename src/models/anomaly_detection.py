"""
Anomaly Detection Model for NBA Prediction System

This module implements anomaly detection for identifying outliers in NBA game
and player data. It focuses on detecting unusual patterns that might indicate
prediction opportunities or data quality issues.

Features Used:
- Team: Recent performance, clutch performance, fatigue model, team chemistry
- Player: Efficiency, injury impact, matchup performance, usage rate
- Context: Rivalry/motivation (for unexpected rivalry effects)
- Odds: Game-level odds, player prop odds (for betting anomalies)
- Live Data: Live scores and odds (for in-game anomalies)

The anomaly detection system is valuable for identifying betting opportunities
where the market may have mispriced odds due to overlooked factors or unusual
situations that statistical models might miss.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from datetime import datetime, timezone

from src.models.base_model import BaseModel

# Configure logging
logger = logging.getLogger(__name__)


class AnomalyDetectionModel(BaseModel):
    """
    Anomaly Detection model for identifying outliers in NBA data
    
    This model uses unsupervised learning techniques to detect unusual patterns
    in the data that might indicate betting opportunities or data quality issues.
    It can focus on different types of anomalies depending on the detection_target.
    """
    
    VALID_ALGORITHMS = ['isolation_forest', 'local_outlier_factor', 'one_class_svm']
    
    def __init__(self, name: str = "AnomalyDetection", detection_target: str = "game_outliers", 
                 algorithm: str = "isolation_forest", version: int = 1):
        """
        Initialize the Anomaly Detection model
        
        Args:
            name: Model name
            detection_target: What anomalies to detect ('game_outliers', 'player_outliers', 'odds_outliers')
            algorithm: Algorithm to use for anomaly detection
            version: Model version number
        """
        super().__init__(name=name, model_type='anomaly_detection', version=version)
        
        self.detection_target = detection_target
        
        # Validate and set the algorithm
        if algorithm not in self.VALID_ALGORITHMS:
            logger.warning(f"Invalid algorithm '{algorithm}'. Using isolation_forest instead.")
            self.algorithm = 'isolation_forest'
        else:
            self.algorithm = algorithm
        
        # Default hyperparameters - these can be tuned
        if self.algorithm == 'isolation_forest':
            self.params = {
                'n_estimators': 100,
                'max_samples': 'auto',
                'contamination': 'auto',  # Proportion of outliers in the data
                'max_features': 1.0,      # Features to draw from X
                'bootstrap': False,
                'random_state': 42
            }
        elif self.algorithm == 'local_outlier_factor':
            self.params = {
                'n_neighbors': 20,
                'algorithm': 'auto',
                'leaf_size': 30,
                'metric': 'minkowski',
                'contamination': 'auto'
            }
        elif self.algorithm == 'one_class_svm':
            self.params = {
                'kernel': 'rbf',
                'gamma': 'scale',
                'nu': 0.1,          # Upper bound on the fraction of outliers
                'shrinking': True
            }
    
    def train(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Train the anomaly detection model on the provided data
        
        Args:
            X: Feature matrix with selected features for anomaly detection
            y: Optional target variable (not used for unsupervised methods)
        """
        try:
            # Store feature names for later use
            self.feature_names = list(X.columns)
            
            # Initialize the appropriate anomaly detection model
            if self.algorithm == 'isolation_forest':
                self.model = IsolationForest(**self.params)
            elif self.algorithm == 'local_outlier_factor':
                self.model = LocalOutlierFactor(novelty=True, **self.params)
            elif self.algorithm == 'one_class_svm':
                self.model = OneClassSVM(**self.params)
            
            # Train the model
            logger.info(f"Training {self.name} model with {len(X)} samples and {len(self.feature_names)} features")
            self.model.fit(X)
            
            # Record training time
            self.trained_at = datetime.now(timezone.utc)
            self.is_trained = True
            
            # Log training completion
            logger.info(f"{self.name} model trained successfully using {self.algorithm} algorithm")
            
        except Exception as e:
            logger.error(f"Error training {self.name} model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly scores or labels for the input samples
        
        Args:
            X: Feature matrix matching the training features
            
        Returns:
            Array of anomaly scores or labels (-1 for outliers, 1 for inliers)
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
            
            # Get anomaly predictions
            if self.algorithm == 'local_outlier_factor':
                predictions = self.model.predict(X)
            else:  # isolation_forest or one_class_svm
                predictions = self.model.predict(X)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting with {self.name} model: {str(e)}")
            return np.array([])
    
    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get the anomaly scores for the input samples
        
        Higher score = more normal
        Lower score = more anomalous
        
        Args:
            X: Feature matrix matching the training features
            
        Returns:
            Array of anomaly scores (lower = more anomalous)
        """
        if not self.is_trained:
            logger.error(f"Cannot score with untrained model: {self.name}")
            return np.array([])
        
        if not hasattr(self.model, 'decision_function'):
            logger.warning(f"{self.algorithm} does not have decision_function method")
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
            
            # Get anomaly scores
            scores = self.model.decision_function(X)
            return scores
            
        except Exception as e:
            logger.error(f"Error scoring with {self.name} model: {str(e)}")
            return np.array([])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        This method is not implemented for anomaly detection models
        
        Args:
            X: Feature matrix
            
        Returns:
            Empty array
        """
        logger.warning(f"predict_proba is not implemented for {self.model_type} models")
        return np.array([])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance from the trained model
        
        This is challenging for anomaly detection models, but for Isolation Forest
        we can estimate it based on the average path length decrease caused by features.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            logger.error(f"Cannot get feature importance from untrained model: {self.name}")
            return {}
        
        try:
            feature_importance = {}
            
            # For Isolation Forest, we can get feature importances
            if self.algorithm == 'isolation_forest' and hasattr(self.model, 'estimators_'):
                importances = np.zeros(len(self.feature_names))
                
                # Calculate mean feature importances across all trees
                for tree in self.model.estimators_:
                    if hasattr(tree, 'feature_importances_'):
                        importances += tree.feature_importances_
                
                importances /= len(self.model.estimators_)
                
                # Create a dictionary mapping feature names to their importance scores
                for i, feature_name in enumerate(self.feature_names):
                    feature_importance[feature_name] = float(importances[i])
                    
            else:
                # For other algorithms, we could use permutation importance,
                # but for simplicity we'll just set equal weights
                for feature_name in self.feature_names:
                    feature_importance[feature_name] = 1.0 / len(self.feature_names)
            
            # Sort by importance (highest to lowest)
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {self.name} model: {str(e)}")
            return {}
    
    def evaluate(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Evaluate the model performance
        
        For anomaly detection, standard classification metrics might not apply,
        but we can still compute some useful statistics.
        
        Args:
            X: Feature matrix
            y: Optional ground truth labels (1 for normal, -1 for anomalous)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            logger.error(f"Cannot evaluate untrained model: {self.name}")
            return {}
        
        try:
            metrics = {}
            
            # Make predictions
            predictions = self.predict(X)
            
            # Calculate the proportion of anomalies detected
            anomaly_ratio = np.mean(predictions == -1)
            metrics['anomaly_ratio'] = float(anomaly_ratio)
            
            # If ground truth labels are provided, calculate accuracy metrics
            if y is not None and len(y) == len(predictions):
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                # Convert y to -1/1 format if it's in 0/1 format
                if set(np.unique(y)).issubset({0, 1}):
                    y_converted = np.where(y == 0, -1, 1)
                else:
                    y_converted = y
                
                metrics['accuracy'] = accuracy_score(y_converted, predictions)
                metrics['precision'] = precision_score(y_converted, predictions, pos_label=-1, zero_division=0)
                metrics['recall'] = recall_score(y_converted, predictions, pos_label=-1, zero_division=0)
                metrics['f1_score'] = f1_score(y_converted, predictions, pos_label=-1, zero_division=0)
            
            # For Isolation Forest, add average path length as a metric
            if self.algorithm == 'isolation_forest' and hasattr(self.model, 'estimators_'):
                avg_path_length = np.mean([tree.max_depth for tree in self.model.estimators_])
                metrics['avg_path_length'] = float(avg_path_length)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {self.name} model: {str(e)}")
            return {}
    
    def get_anomalies(self, X: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        """
        Identify anomalies in the data based on anomaly scores
        
        Args:
            X: Feature matrix
            threshold: Optional threshold for anomaly scores (if None, use model default)
            
        Returns:
            DataFrame with original data and anomaly scores, sorted by anomaly score
        """
        if not self.is_trained:
            logger.error(f"Cannot detect anomalies with untrained model: {self.name}")
            return pd.DataFrame()
        
        try:
            # Get anomaly predictions (-1 for anomalies, 1 for normal)
            predictions = self.predict(X)
            
            # Get anomaly scores if available
            if hasattr(self.model, 'decision_function'):
                scores = self.decision_function(X)
                
                # Create a copy of X with anomaly scores
                result = X.copy()
                result['anomaly_score'] = scores
                result['is_anomaly'] = predictions == -1
                
                # If threshold is provided, override model predictions
                if threshold is not None:
                    result['is_anomaly'] = result['anomaly_score'] < threshold
                
                # Sort by anomaly score (ascending, so most anomalous first)
                result = result.sort_values('anomaly_score')
                
            else:
                # Without scores, just use the predictions
                result = X.copy()
                result['is_anomaly'] = predictions == -1
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting anomalies with {self.name} model: {str(e)}")
            return pd.DataFrame()
    
    def save_to_db(self) -> bool:
        """
        Save the model to the database with anomaly detection specific details
        
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if not self.is_trained:
            logger.error(f"Cannot save untrained model: {self.name}")
            return False
        
        try:
            # Get feature importance
            feature_importance = self.get_feature_importance()
            
            # Create model weights dictionary with anomaly detection specific details
            model_data = {
                'model_name': self.name,
                'model_type': self.model_type,
                'weights': {
                    'feature_importances': feature_importance,
                    'algorithm': self.algorithm,
                    'detection_target': self.detection_target
                },
                'params': self.params,
                'version': self.version,
                'trained_at': self.trained_at or datetime.now(timezone.utc),
                'active': True
            }
            
            # Add algorithm-specific data
            if self.algorithm == 'isolation_forest':
                if hasattr(self.model, 'max_samples_'):
                    model_data['weights']['max_samples'] = int(self.model.max_samples_)
                if hasattr(self.model, 'estimators_'):
                    model_data['weights']['n_estimators'] = len(self.model.estimators_)
            
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
