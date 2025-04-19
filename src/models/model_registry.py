#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Registry System

This module implements a comprehensive model registry for tracking, versioning,
and managing all prediction models. It supports:
- Version tracking and history
- Model metadata and performance metrics
- A/B testing capabilities
- Production model selection
"""

import os
import json
import logging
import shutil
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Centralized registry for model versioning and management"""
    
    def __init__(self, registry_dir: str = "models/registry"):
        """
        Initialize the model registry
        
        Args:
            registry_dir: Directory for storing the registry and models
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True, parents=True)
        
        # Load or initialize registry
        self.registry = self._load_registry()
        
        # Current production models by type
        self.production_models = self.registry.get("production_models", {})
    
    def _load_registry(self) -> Dict[str, Any]:
        """
        Load the registry from disk or initialize a new one
        
        Returns:
            Registry dictionary
        """
        registry_file = self.registry_dir / "registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading model registry: {str(e)}")
                return self._initialize_registry()
        else:
            return self._initialize_registry()
    
    def _initialize_registry(self) -> Dict[str, Any]:
        """
        Initialize a new model registry structure
        
        Returns:
            New registry dictionary
        """
        return {
            "models": {},
            "production_models": {},
            "model_types": [
                "moneyline",
                "spread",
                "total",
                "player_props_points",
                "player_props_rebounds",
                "player_props_assists"
            ],
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_registry(self) -> None:
        """
        Save the registry to disk
        """
        registry_file = self.registry_dir / "registry.json"
        
        try:
            # Update last updated timestamp
            self.registry["last_updated"] = datetime.now().isoformat()
            
            with open(registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2)
            
            logger.info(f"Model registry saved to {registry_file}")
        except Exception as e:
            logger.error(f"Error saving model registry: {str(e)}")
    
    def register_model(self, model_path: str, model_info: Dict[str, Any]) -> str:
        """
        Register a model in the registry
        
        Args:
            model_path: Path to the model file
            model_info: Dictionary containing model metadata
                Required keys:
                - name: Model name
                - version: Model version
                - type: Model type (e.g., 'moneyline', 'spread')
                - metrics: Dictionary of performance metrics
                
        Returns:
            Model ID (name_version) if registration successful, empty string otherwise
        """
        try:
            model_name = model_info.get("name")
            model_version = model_info.get("version")
            model_type = model_info.get("type")
            
            if not all([model_name, model_version, model_type, model_path]):
                logger.error("Missing required model information for registration")
                return ""
                
            # Ensure model file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                return ""
                
            # Create unique model ID
            model_id = f"{model_name}_{model_version}"
            
            # Copy model file to registry
            destination = self.registry_dir / f"{model_id}.pkl"
            shutil.copy2(model_path, destination)
            
            # Save metadata
            model_info["registered_at"] = datetime.now().isoformat()
            model_info["path"] = str(destination)
            
            # Add to registry
            if model_name not in self.registry["models"]:
                self.registry["models"][model_name] = {}
                
            self.registry["models"][model_name][str(model_version)] = model_info
            
            # Update registry file
            self._save_registry()
            
            logger.info(f"Registered model {model_id} in registry")
            return model_id
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            return ""
    
    def set_production_model(self, model_name: str, model_version: int, model_type: str) -> bool:
        """
        Set a specific model version as the production model for a given type
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            model_type: Type of prediction (e.g., 'moneyline', 'spread')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            model_id = f"{model_name}_{model_version}"
            
            # Verify model exists in registry
            if (model_name not in self.registry["models"] or 
                str(model_version) not in self.registry["models"][model_name]):
                logger.error(f"Model {model_id} not found in registry")
                return False
                
            # Set as production model for this type
            self.registry["production_models"][model_type] = {
                "name": model_name,
                "version": model_version,
                "set_at": datetime.now().isoformat()
            }
            
            # Update instance cache
            self.production_models = self.registry["production_models"]
            
            # Save registry
            self._save_registry()
            
            logger.info(f"Set {model_id} as production model for {model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting production model: {str(e)}")
            return False
    
    def get_production_model_path(self, model_type: str) -> str:
        """
        Get the path to the current production model for a given type
        
        Args:
            model_type: Type of prediction (e.g., 'moneyline', 'spread')
            
        Returns:
            Path to the model file or empty string if not found
        """
        try:
            # Check if we have a production model for this type
            if model_type not in self.production_models:
                logger.warning(f"No production model set for {model_type}")
                return ""
                
            # Get model details
            prod_model = self.production_models[model_type]
            model_name = prod_model["name"]
            model_version = prod_model["version"]
            
            # Look up model in registry
            if (model_name not in self.registry["models"] or 
                str(model_version) not in self.registry["models"][model_name]):
                logger.error(f"Production model {model_name}_{model_version} not found in registry")
                return ""
                
            # Get model path
            model_info = self.registry["models"][model_name][str(model_version)]
            model_path = model_info.get("path", "")
            
            if not model_path or not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                return ""
                
            return model_path
            
        except Exception as e:
            logger.error(f"Error getting production model path: {str(e)}")
            return ""
    
    def load_production_model(self, model_type: str) -> Any:
        """
        Load the current production model for a given type
        
        Args:
            model_type: Type of prediction (e.g., 'moneyline', 'spread')
            
        Returns:
            Loaded model or None if not found
        """
        try:
            model_path = self.get_production_model_path(model_type)
            
            if not model_path:
                logger.error(f"No valid production model path found for {model_type}")
                return None
                
            # Load the model from pickle file
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            logger.info(f"Loaded production model for {model_type}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading production model: {str(e)}")
            return None
    
    def get_model_info(self, model_name: str, model_version: Optional[int] = None) -> Dict[str, Any]:
        """
        Get information about a specific model or its latest version
        
        Args:
            model_name: Name of the model
            model_version: Version of the model, or None for latest
            
        Returns:
            Dictionary with model information or empty dict if not found
        """
        try:
            # Check if model exists
            if model_name not in self.registry["models"]:
                logger.warning(f"Model {model_name} not found in registry")
                return {}
                
            model_versions = self.registry["models"][model_name]
            
            # If version not specified, find the latest
            if model_version is None:
                # Convert version strings to integers and find max
                versions = [int(v) for v in model_versions.keys()]
                if not versions:
                    return {}
                model_version = max(versions)
                
            # Get model info
            version_str = str(model_version)
            if version_str not in model_versions:
                logger.warning(f"Version {model_version} of {model_name} not found in registry")
                return {}
                
            return model_versions[version_str]
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
    
    def get_model_metrics(self, model_name: str, model_version: Optional[int] = None) -> Dict[str, float]:
        """
        Get performance metrics for a specific model
        
        Args:
            model_name: Name of the model
            model_version: Version of the model, or None for latest
            
        Returns:
            Dictionary of metrics or empty dict if not found
        """
        try:
            model_info = self.get_model_info(model_name, model_version)
            return model_info.get("metrics", {})
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            return {}
    
    def compare_models(self, model_type: str, metric: str = "accuracy") -> List[Dict[str, Any]]:
        """
        Compare all models of a given type using a specific metric
        
        Args:
            model_type: Type of prediction (e.g., 'moneyline', 'spread')
            metric: Metric to use for comparison (e.g., 'accuracy', 'f1_score')
            
        Returns:
            List of model info dictionaries sorted by metric value (descending)
        """
        try:
            # Find all models for this type
            models_of_type = []
            
            for model_name, versions in self.registry["models"].items():
                for version_str, model_info in versions.items():
                    if model_info.get("type") == model_type:
                        # Add the metric value if available
                        metric_value = model_info.get("metrics", {}).get(metric, 0.0)
                        
                        models_of_type.append({
                            "name": model_name,
                            "version": int(version_str),
                            "registered_at": model_info.get("registered_at", ""),
                            metric: metric_value,
                            "is_production": False  # Will update below
                        })
            
            # Mark production model
            if model_type in self.production_models:
                prod_name = self.production_models[model_type]["name"]
                prod_version = self.production_models[model_type]["version"]
                
                for model in models_of_type:
                    if model["name"] == prod_name and model["version"] == prod_version:
                        model["is_production"] = True
                        break
            
            # Sort by metric value (descending)
            return sorted(models_of_type, key=lambda x: x.get(metric, 0.0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            return []
    
    def delete_model(self, model_name: str, model_version: int) -> bool:
        """
        Delete a model from the registry
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Check if model exists
            if (model_name not in self.registry["models"] or 
                str(model_version) not in self.registry["models"][model_name]):
                logger.warning(f"Model {model_name}_{model_version} not found in registry")
                return False
                
            # Check if model is in production
            for model_type, prod_model in self.production_models.items():
                if prod_model["name"] == model_name and prod_model["version"] == model_version:
                    logger.error(f"Cannot delete production model {model_name}_{model_version}")
                    return False
            
            # Get model path
            model_info = self.registry["models"][model_name][str(model_version)]
            model_path = model_info.get("path", "")
            
            # Delete model file if it exists
            if model_path and os.path.exists(model_path):
                os.remove(model_path)
                
            # Remove from registry
            del self.registry["models"][model_name][str(model_version)]
            
            # If no more versions, remove the model entry
            if not self.registry["models"][model_name]:
                del self.registry["models"][model_name]
                
            # Save registry
            self._save_registry()
            
            logger.info(f"Deleted model {model_name}_{model_version} from registry")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize registry
    registry = ModelRegistry()
