"""
Base Model for NBA Prediction System

This module provides the BaseModel class that serves as a foundation for all prediction
models in the system. It defines common interfaces and functionality that all models
should implement.

The BaseModel handles model persistence, serialization, and shared evaluation metrics.
"""

import logging
import os
import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
                
                if not os.path.exists(models_dir):
                    logger.error(f"Models directory not found: {models_dir}")
                    return False
                    
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
    
    def record_performance(self, metrics: Dict[str, float], prediction_type: str, num_predictions: int, time_window: str = '7d') -> bool:
        """
        Record the performance metrics of the model for a specific prediction type and time window
        
        Args:
            metrics: Dictionary of performance metrics (accuracy, precision, recall, f1, etc.)
            prediction_type: Type of prediction (moneyline, spread, total, etc.)
            num_predictions: Number of predictions made
            time_window: Time window for the metrics (e.g., '7d' for 7 days)
            
        Returns:
            bool: True if performance was successfully recorded
        """
        try:
            # Create the performance data structure
            timestamp = datetime.now(timezone.utc).isoformat()
            performance_data = {
                'model_name': self.name,
                'model_version': self.version,
                'prediction_type': prediction_type,
                'metrics': metrics,
                'num_predictions': num_predictions,
                'timestamp': timestamp,
                'time_window': time_window
            }
            
            # Save the performance data to disk
            performance_dir = os.path.join('data', 'performance')
            os.makedirs(performance_dir, exist_ok=True)
            
            # Create a unique filename based on model, prediction type, and timestamp
            filename = f"{self.name.lower()}_{prediction_type}_{int(datetime.now().timestamp())}.json"
            file_path = os.path.join(performance_dir, filename)
            
            # Save as JSON
            with open(file_path, 'w') as f:
                json.dump(performance_data, f, indent=2)
                
            logger.info(f"Model performance recorded to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording model performance: {str(e)}")
            return False
    
    def get_latest_performance(self, prediction_type: str, time_window: str = '7d') -> Optional[Dict[str, Any]]:
        """
        Get the latest performance metrics for this model from disk
        
        Args:
            prediction_type: Type of prediction (e.g., 'moneyline', 'spread', 'player_points')
            time_window: Time window for the metrics (e.g., '7d', '30d', 'season')
            
        Returns:
            Dictionary containing performance data or None if not found
        """
        try:
            perf_dir = os.path.join('data', 'performance')
            if not os.path.exists(perf_dir):
                logger.warning(f"Performance directory not found: {perf_dir}")
                return None
                
            # Find all performance files for this model, prediction type, and time window
            prefix = f"{self.name}_{prediction_type}_{time_window}_"
            performance_files = [f for f in os.listdir(perf_dir) if f.startswith(prefix) and f.endswith('.json')]
            
            if not performance_files:
                logger.warning(f"No performance data found for {self.name}, {prediction_type}, {time_window}")
                return None
                
            # Sort by timestamp (which is embedded in the filename)
            performance_files.sort(reverse=True)
            
            # Load the latest performance file
            latest_file = os.path.join(perf_dir, performance_files[0])
            with open(latest_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error getting latest performance for {self.name}: {str(e)}")
            return None
    
    def get_performance_trend(self, prediction_type: str, time_window: str = '7d', days: int = 30) -> List[Dict[str, Any]]:
        """
        Get the performance trend for this model over time from disk
        
        Args:
            prediction_type: Type of prediction
            time_window: Time window for the metrics
            days: Number of days of history to retrieve
            
        Returns:
            List of dictionaries containing performance data ordered by date
        """
        try:
            perf_dir = os.path.join('data', 'performance')
            if not os.path.exists(perf_dir):
                logger.warning(f"Performance directory not found: {perf_dir}")
                return []
                
            # Find all performance files for this model, prediction type, and time window
            prefix = f"{self.name}_{prediction_type}_{time_window}_"
            performance_files = [f for f in os.listdir(perf_dir) if f.startswith(prefix) and f.endswith('.json')]
            
            if not performance_files:
                logger.warning(f"No performance data found for {self.name}, {prediction_type}, {time_window}")
                return []
                
            # Load all performance files
            performances = []
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            for filename in performance_files:
                file_path = os.path.join(perf_dir, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Check if the performance data is within the specified time range
                    perf_date = datetime.fromisoformat(data['date'])
                    if perf_date >= cutoff_date:
                        performances.append(data)
            
            # Sort by date
            performances.sort(key=lambda x: x['date'])
            return performances
                
        except Exception as e:
            logger.error(f"Error getting performance trend for {self.name}: {str(e)}")
            return []

    def save(self, path: Optional[str] = None) -> bool:
        """
        Save the model to a file
        
        Args:
            path: Optional path to save the model to. If None, a default path will be used.
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            import pickle
            from pathlib import Path
            
            # Create models directory if it doesn't exist
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            # Use default path if none provided
            if path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = models_dir / f"{self.name}_{timestamp}_v{self.version}.pkl"
            else:
                path = Path(path)
                
            # Ensure parent directories exist
            path.parent.mkdir(exist_ok=True, parents=True)
            
            # Save the model
            with open(path, 'wb') as f:
                pickle.dump(self, f)
                
            logger.info(f"Saved {self.name} to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving {self.name}: {str(e)}")
            return False
