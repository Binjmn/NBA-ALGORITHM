"""
Base Model for NBA Prediction System

This module provides the BaseModel class that serves as a foundation for all prediction
models in the system. It defines common interfaces and functionality that all models
should implement.

The BaseModel handles model persistence, serialization, loading from the database,
and shared evaluation metrics.
"""

import logging
import os
import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.database.models import ModelWeight, ModelPerformance

# Configure logging
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all prediction models
    
    This class defines the interface that all prediction models must implement
    and provides common functionality for model management.
    """
    
    def __init__(self, name: str, model_type: str, version: int = 1):
        """
        Initialize the base model
        
        Args:
            name: Name of the model (e.g., 'RandomForest', 'Bayesian')
            model_type: Type of model (e.g., 'classification', 'regression')
            version: Model version number
        """
        self.name = name
        self.model_type = model_type
        self.version = version
        self.model = None
        self.trained_at = None
        self.feature_names = []
        self.params = {}
        self.is_trained = False
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model on the provided data
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input samples
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of class probabilities
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance from the trained model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
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
            # Make predictions
            y_pred = self.predict(X)
            
            # For probability predictions, get class predictions
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
            
            # Calculate profit metrics if profit data available
            if hasattr(X, 'profit_factor'):
                metrics['profit_factor'] = X.profit_factor.mean()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model {self.name}: {str(e)}")
            return {}
    
    def save_to_disk(self, models_dir: str = 'data/models') -> str:
        """
        Save the model to disk
        
        Args:
            models_dir: Directory to save the model file
            
        Returns:
            Path to the saved model file or empty string if saving failed
        """
        if not self.is_trained:
            logger.error(f"Cannot save untrained model: {self.name}")
            return ""
        
        try:
            # Create the models directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)
            
            # Create model metadata
            metadata = {
                'name': self.name,
                'model_type': self.model_type,
                'version': self.version,
                'trained_at': self.trained_at.isoformat() if self.trained_at else None,
                'feature_names': self.feature_names,
                'params': self.params
            }
            
            # Save the metadata to a JSON file
            filename_base = f"{self.name.lower()}_v{self.version}"
            metadata_path = os.path.join(models_dir, f"{filename_base}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save the model to a pickle file
            model_path = os.path.join(models_dir, f"{filename_base}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info(f"Model {self.name} v{self.version} saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model {self.name} to disk: {str(e)}")
            return ""
    
    def load_from_disk(self, models_dir: str = 'data/models', version: Optional[int] = None) -> bool:
        """
        Load the model from disk
        
        Args:
            models_dir: Directory containing the model files
            version: Specific version to load, or latest if None
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # If version is not specified, find the latest version
            if version is None:
                # Find all metadata files for this model
                model_prefix = f"{self.name.lower()}_v"
                metadata_files = [f for f in os.listdir(models_dir) if f.startswith(model_prefix) and f.endswith('_metadata.json')]
                
                if not metadata_files:
                    logger.error(f"No model files found for {self.name} in {models_dir}")
                    return False
                
                # Find the highest version number
                versions = [int(f.split('_v')[1].split('_')[0]) for f in metadata_files]
                version = max(versions)
            
            # Load the metadata file
            filename_base = f"{self.name.lower()}_v{version}"
            metadata_path = os.path.join(models_dir, f"{filename_base}_metadata.json")
            
            if not os.path.exists(metadata_path):
                logger.error(f"Metadata file not found: {metadata_path}")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Update model attributes from metadata
            self.version = metadata['version']
            self.model_type = metadata['model_type']
            self.feature_names = metadata['feature_names']
            self.params = metadata['params']
            self.trained_at = datetime.fromisoformat(metadata['trained_at']) if metadata['trained_at'] else None
            
            # Load the model file
            model_path = os.path.join(models_dir, f"{filename_base}.pkl")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.is_trained = True
            logger.info(f"Model {self.name} v{version} loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {self.name} from disk: {str(e)}")
            return False
    
    def save_to_db(self) -> bool:
        """
        Save the model to the database
        
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
                    # Additional model-specific weights can be added by subclasses
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
    
    def load_from_db(self, version: Optional[int] = None) -> bool:
        """
        Load the model from the database
        
        Args:
            version: Specific version to load, or latest active if None
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            # Find the model in the database
            if version is None:
                # Get the latest active version
                model_data = ModelWeight.find_latest_active(self.name)
            else:
                # Find the specific version
                # This is a simplified example - in a real implementation,
                # you would need to add a method to find a specific version
                model_data = None
                
            if not model_data:
                logger.error(f"No model data found for {self.name} in the database")
                return False
            
            # Update model attributes from database
            self.version = model_data['version']
            self.model_type = model_data['model_type']
            self.params = model_data['params']
            self.trained_at = model_data['trained_at']
            
            # Here you would need to deserialize the model
            # This is just a placeholder - actual implementation would depend on how models are stored
            # For example, models might be stored as pickle files with paths in the database
            # or serialized directly into the database
            
            # For this example, we'll assume the model needs to be loaded from disk
            # using the path stored in the database
            model_path = model_data['weights'].get('model_path', '')
            if model_path and os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                    
                self.is_trained = True
                logger.info(f"Model {self.name} v{self.version} loaded from database")
                return True
            else:
                logger.error(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {self.name} from database: {str(e)}")
            return False
    
    def record_performance(self, metrics: Dict[str, float], prediction_type: str, num_predictions: int, time_window: str = '7d') -> bool:
        """
        Record model performance metrics in the database
        
        Args:
            metrics: Dictionary of performance metrics
            prediction_type: Type of prediction (e.g., 'moneyline', 'spread', 'player_points')
            num_predictions: Number of predictions evaluated
            time_window: Time window for the metrics (e.g., '7d', '30d', 'season')
            
        Returns:
            bool: True if recording was successful, False otherwise
        """
        try:
            performance_data = {
                'model_name': self.name,
                'date': datetime.now(timezone.utc),
                'metrics': metrics,
                'prediction_type': prediction_type,
                'num_predictions': num_predictions,
                'time_window': time_window
            }
            
            perf_id = ModelPerformance.create(performance_data)
            
            if perf_id:
                logger.info(f"Performance metrics for {self.name} recorded with ID {perf_id}")
                return True
            else:
                logger.error(f"Failed to record performance metrics for {self.name}")
                return False
                
        except Exception as e:
            logger.error(f"Error recording performance metrics for {self.name}: {str(e)}")
            return False
            
    def get_latest_performance(self, prediction_type: str, time_window: str = '7d') -> Optional[Dict[str, Any]]:
        """
        Get the latest performance metrics for this model
        
        Args:
            prediction_type: Type of prediction (e.g., 'moneyline', 'spread', 'player_points')
            time_window: Time window for the metrics (e.g., '7d', '30d', 'season')
            
        Returns:
            Dictionary containing performance data or None if not found
        """
        try:
            return ModelPerformance.get_latest_performance(self.name, prediction_type, time_window, time_window_column='time_window')
        except Exception as e:
            logger.error(f"Error getting latest performance for {self.name}: {str(e)}")
            return None
    
    def get_performance_trend(self, prediction_type: str, time_window: str = '7d', days: int = 30) -> List[Dict[str, Any]]:
        """
        Get the performance trend for this model over time
        
        Args:
            prediction_type: Type of prediction
            time_window: Time window for the metrics
            days: Number of days of history to retrieve
            
        Returns:
            List of dictionaries containing performance data ordered by date
        """
        try:
            return ModelPerformance.get_performance_trend(self.name, prediction_type, time_window, days, time_window_column='time_window')
        except Exception as e:
            logger.error(f"Error getting performance trend for {self.name}: {str(e)}")
            return []
