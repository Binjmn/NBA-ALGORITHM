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
    
    def register_model(self, model_name: str, model_type: str, version: str, 
                      model_path: str, metrics: Dict[str, Any],
                      metadata: Optional[Dict[str, Any]] = None,
                      register_as_production: bool = False) -> str:
        """
        Register a model in the registry
        
        Args:
            model_name: Name of the model
            model_type: Type of model (moneyline, spread, total, player_props_*)
            version: Version string (e.g., "1.0.0")
            model_path: Path to the saved model file
            metrics: Performance metrics for the model
            metadata: Optional additional metadata
            register_as_production: Whether to register this model as the production model
            
        Returns:
            Registry ID for the registered model
        """
        # Check if model type is valid
        if model_type not in self.registry["model_types"]:
            logger.warning(f"Unknown model type: {model_type}. Adding to registry anyway.")
            self.registry["model_types"].append(model_type)
        
        # Generate unique registry ID
        registry_id = f"{model_name}_{model_type}_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Ensure model path exists
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model file not found at: {model_path}")
            return ""
        
        # Get file size
        file_size = model_path.stat().st_size
        
        # Copy model to registry
        registry_model_path = self.registry_dir / "models" / f"{registry_id}.pkl"
        registry_model_path.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            shutil.copy(model_path, registry_model_path)
        except Exception as e:
            logger.error(f"Error copying model to registry: {str(e)}")
            return ""
        
        # Create model entry
        model_entry = {
            "id": registry_id,
            "name": model_name,
            "type": model_type,
            "version": version,
            "path": str(registry_model_path),
            "original_path": str(model_path),
            "metrics": metrics,
            "metadata": metadata or {},
            "file_size": file_size,
            "registered_at": datetime.now().isoformat(),
            "is_production": register_as_production
        }
        
        # Update registry
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {
                "versions": {}
            }
        
        self.registry["models"][model_name]["versions"][version] = model_entry
        
        # Set as production model if requested
        if register_as_production:
            self.set_production_model(model_name, model_type, version)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Model {model_name} (v{version}) registered with ID: {registry_id}")
        return registry_id
    
    def load_model(self, model_name: str, model_type: str, version: Optional[str] = None) -> Any:
        """
        Load a model from the registry
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            version: Optional version string (loads latest if not specified)
            
        Returns:
            Loaded model object or None if not found
        """
        # Get model entry
        model_entry = self.get_model_entry(model_name, model_type, version)
        
        if not model_entry:
            return None
        
        # Load model from path
        model_path = model_entry["path"]
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Loaded model {model_name} (v{model_entry['version']}) from registry")
            return model
        except Exception as e:
            logger.error(f"Error loading model from registry: {str(e)}")
            return None
    
    def get_model_entry(self, model_name: str, model_type: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a model entry from the registry
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            version: Optional version string (gets latest if not specified)
            
        Returns:
            Model entry dictionary or empty dict if not found
        """
        if model_name not in self.registry["models"]:
            logger.warning(f"Model {model_name} not found in registry")
            return {}
        
        versions = self.registry["models"][model_name]["versions"]
        
        if not versions:
            logger.warning(f"No versions found for model {model_name}")
            return {}
        
        if version:
            # Get specific version
            if version in versions:
                model_entry = versions[version]
                
                # Check if model type matches
                if model_entry["type"] != model_type:
                    logger.warning(f"Model type mismatch: requested {model_type}, found {model_entry['type']}")
                    return {}
                
                return model_entry
            else:
                logger.warning(f"Version {version} not found for model {model_name}")
                return {}
        else:
            # Get latest version with matching type
            matching_versions = [v for v in versions.values() if v["type"] == model_type]
            
            if not matching_versions:
                logger.warning(f"No versions with type {model_type} found for model {model_name}")
                return {}
            
            # Sort by registered date (latest first)
            matching_versions.sort(key=lambda x: x["registered_at"], reverse=True)
            return matching_versions[0]
    
    def set_production_model(self, model_name: str, model_type: str, version: str) -> bool:
        """
        Set a model as the production model for its type
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            version: Version string
            
        Returns:
            True if successful, False otherwise
        """
        # Get model entry
        if model_name not in self.registry["models"]:
            logger.error(f"Model {model_name} not found in registry")
            return False
        
        versions = self.registry["models"][model_name]["versions"]
        
        if version not in versions:
            logger.error(f"Version {version} not found for model {model_name}")
            return False
        
        model_entry = versions[version]
        
        if model_entry["type"] != model_type:
            logger.error(f"Model type mismatch: requested {model_type}, found {model_entry['type']}")
            return False
        
        # Update production status
        # First, unset any existing production model of this type
        for m_name, m_data in self.registry["models"].items():
            for v_id, v_data in m_data["versions"].items():
                if v_data["type"] == model_type and v_data.get("is_production", False):
                    v_data["is_production"] = False
                    logger.info(f"Unset production status for {m_name} (v{v_id})")
        
        # Set this model as production
        model_entry["is_production"] = True
        model_entry["production_set_at"] = datetime.now().isoformat()
        
        # Update production models lookup
        self.production_models[model_type] = {
            "name": model_name,
            "version": version,
            "id": model_entry["id"],
            "path": model_entry["path"]
        }
        
        self.registry["production_models"] = self.production_models
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Set {model_name} (v{version}) as production model for {model_type}")
        return True
    
    def get_production_model(self, model_type: str) -> Any:
        """
        Get the current production model for a specific type
        
        Args:
            model_type: Type of model
            
        Returns:
            Loaded production model or None if not found
        """
        if model_type not in self.production_models:
            logger.warning(f"No production model set for type: {model_type}")
            return None
        
        prod_info = self.production_models[model_type]
        model_path = prod_info["path"]
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Loaded production model {prod_info['name']} (v{prod_info['version']}) for {model_type}")
            return model
        except Exception as e:
            logger.error(f"Error loading production model: {str(e)}")
            return None
    
    def list_models(self, model_type: Optional[str] = None, production_only: bool = False) -> List[Dict[str, Any]]:
        """
        List models in the registry
        
        Args:
            model_type: Optional type to filter by
            production_only: Whether to list only production models
            
        Returns:
            List of model entries
        """
        models = []
        
        for model_name, model_data in self.registry["models"].items():
            for version, version_data in model_data["versions"].items():
                # Apply filters
                if model_type and version_data["type"] != model_type:
                    continue
                    
                if production_only and not version_data.get("is_production", False):
                    continue
                
                models.append(version_data)
        
        return models
    
    def compare_models(self, model1_id: str, model2_id: str) -> Dict[str, Any]:
        """
        Compare two models in the registry
        
        Args:
            model1_id: Registry ID of first model
            model2_id: Registry ID of second model
            
        Returns:
            Comparison results
        """
        # Find model entries
        model1 = None
        model2 = None
        
        for model_name, model_data in self.registry["models"].items():
            for version, version_data in model_data["versions"].items():
                if version_data["id"] == model1_id:
                    model1 = version_data
                if version_data["id"] == model2_id:
                    model2 = version_data
        
        if not model1 or not model2:
            logger.error(f"Models not found: {model1_id}, {model2_id}")
            return {}
        
        # Ensure models are comparable (same type)
        if model1["type"] != model2["type"]:
            logger.warning(f"Comparing different model types: {model1['type']} vs {model2['type']}")
        
        # Compare metrics
        comparison = {
            "model1": {
                "id": model1["id"],
                "name": model1["name"],
                "version": model1["version"],
                "metrics": model1["metrics"]
            },
            "model2": {
                "id": model2["id"],
                "name": model2["name"],
                "version": model2["version"],
                "metrics": model2["metrics"]
            },
            "metric_differences": {}
        }
        
        # Calculate differences for common metrics
        common_metrics = set(model1["metrics"].keys()) & set(model2["metrics"].keys())
        
        for metric in common_metrics:
            value1 = model1["metrics"][metric]
            value2 = model2["metrics"][metric]
            
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                difference = value1 - value2
                percent_change = (difference / value2) * 100 if value2 != 0 else float('inf')
                
                comparison["metric_differences"][metric] = {
                    "absolute": difference,
                    "percent": percent_change,
                    "better": difference > 0 if metric != "error" else difference < 0
                }
        
        return comparison
    
    def delete_model(self, model_name: str, version: str) -> bool:
        """
        Delete a model from the registry
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.registry["models"]:
            logger.error(f"Model {model_name} not found in registry")
            return False
        
        versions = self.registry["models"][model_name]["versions"]
        
        if version not in versions:
            logger.error(f"Version {version} not found for model {model_name}")
            return False
        
        model_entry = versions[version]
        
        # Check if this is a production model
        if model_entry.get("is_production", False):
            logger.error(f"Cannot delete production model: {model_name} (v{version})")
            return False
        
        # Remove model file
        model_path = Path(model_entry["path"])
        if model_path.exists():
            try:
                model_path.unlink()
                logger.info(f"Deleted model file: {model_path}")
            except Exception as e:
                logger.error(f"Error deleting model file: {str(e)}")
        
        # Remove from registry
        del versions[version]
        
        # If this was the last version, remove the model entirely
        if not versions:
            del self.registry["models"][model_name]
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Deleted model {model_name} (v{version}) from registry")
        return True
    
    def setup_ab_test(self, model_type: str, model_a_id: str, model_b_id: str, test_name: str, 
                     allocation: float = 0.5, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Setup an A/B test between two models
        
        Args:
            model_type: Type of model
            model_a_id: Registry ID of model A
            model_b_id: Registry ID of model B
            test_name: Name for the A/B test
            allocation: Proportion of traffic to route to model B (0-1)
            metadata: Optional additional metadata
            
        Returns:
            ID of the A/B test
        """
        # Find model entries
        model_a = None
        model_b = None
        
        for model_name, model_data in self.registry["models"].items():
            for version, version_data in model_data["versions"].items():
                if version_data["id"] == model_a_id:
                    model_a = version_data
                if version_data["id"] == model_b_id:
                    model_b = version_data
        
        if not model_a or not model_b:
            logger.error(f"Models not found: {model_a_id}, {model_b_id}")
            return ""
        
        # Ensure models are comparable (same type)
        if model_a["type"] != model_type or model_b["type"] != model_type:
            logger.error(f"Model type mismatch for A/B test")
            return ""
        
        # Generate test ID
        test_id = f"ab_test_{test_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Initialize AB tests in registry if not present
        if "ab_tests" not in self.registry:
            self.registry["ab_tests"] = {}
        
        # Create test entry
        test_entry = {
            "id": test_id,
            "name": test_name,
            "model_type": model_type,
            "model_a": {
                "id": model_a_id,
                "name": model_a["name"],
                "version": model_a["version"]
            },
            "model_b": {
                "id": model_b_id,
                "name": model_b["name"],
                "version": model_b["version"]
            },
            "allocation": allocation,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "results": {}
        }
        
        # Add to registry
        self.registry["ab_tests"][test_id] = test_entry
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Created A/B test {test_name} ({test_id}) between {model_a['name']} and {model_b['name']}")
        return test_id
    
    def update_ab_test_results(self, test_id: str, results: Dict[str, Any]) -> bool:
        """
        Update results for an A/B test
        
        Args:
            test_id: ID of the A/B test
            results: Dictionary with test results
            
        Returns:
            True if successful, False otherwise
        """
        if "ab_tests" not in self.registry or test_id not in self.registry["ab_tests"]:
            logger.error(f"A/B test {test_id} not found")
            return False
        
        test_entry = self.registry["ab_tests"][test_id]
        
        # Update results
        test_entry["results"] = results
        test_entry["updated_at"] = datetime.now().isoformat()
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Updated results for A/B test {test_id}")
        return True
    
    def promote_ab_test_winner(self, test_id: str, winner: str) -> bool:
        """
        Promote the winner of an A/B test to production
        
        Args:
            test_id: ID of the A/B test
            winner: 'a' or 'b' to indicate which model won
            
        Returns:
            True if successful, False otherwise
        """
        if "ab_tests" not in self.registry or test_id not in self.registry["ab_tests"]:
            logger.error(f"A/B test {test_id} not found")
            return False
        
        test_entry = self.registry["ab_tests"][test_id]
        
        # Determine winner model
        if winner.lower() == 'a':
            winner_model = test_entry["model_a"]
        elif winner.lower() == 'b':
            winner_model = test_entry["model_b"]
        else:
            logger.error(f"Invalid winner value: {winner}. Must be 'a' or 'b'.")
            return False
        
        # Get model details
        model_name = winner_model["name"]
        model_version = winner_model["version"]
        model_type = test_entry["model_type"]
        
        # Set as production model
        success = self.set_production_model(model_name, model_type, model_version)
        
        if success:
            # Update test status
            test_entry["status"] = "completed"
            test_entry["winner"] = winner.lower()
            test_entry["completed_at"] = datetime.now().isoformat()
            
            # Save registry
            self._save_registry()
            
            logger.info(f"Promoted {model_name} (v{model_version}) as winner of A/B test {test_id}")
            return True
        else:
            logger.error(f"Failed to promote winner of A/B test {test_id}")
            return False
    
    def get_model_by_id(self, model_id: str) -> Dict[str, Any]:
        """
        Get a model entry by its registry ID
        
        Args:
            model_id: Registry ID
            
        Returns:
            Model entry or empty dict if not found
        """
        for model_name, model_data in self.registry["models"].items():
            for version, version_data in model_data["versions"].items():
                if version_data["id"] == model_id:
                    return version_data
        
        logger.warning(f"Model with ID {model_id} not found in registry")
        return {}
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the registry
        
        Returns:
            Dictionary with registry summary
        """
        summary = {
            "models_count": len(self.registry["models"]),
            "model_types": self.registry["model_types"],
            "production_models": {},
            "ab_tests": {
                "active": 0,
                "completed": 0
            },
            "last_updated": self.registry["last_updated"]
        }
        
        # Count versions
        versions_count = 0
        for model_name, model_data in self.registry["models"].items():
            versions_count += len(model_data["versions"])
        
        summary["versions_count"] = versions_count
        
        # Summarize production models
        for model_type, model_info in self.production_models.items():
            summary["production_models"][model_type] = {
                "name": model_info["name"],
                "version": model_info["version"]
            }
        
        # Count A/B tests
        if "ab_tests" in self.registry:
            for test_id, test_data in self.registry["ab_tests"].items():
                if test_data["status"] == "active":
                    summary["ab_tests"]["active"] += 1
                elif test_data["status"] == "completed":
                    summary["ab_tests"]["completed"] += 1
        
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize registry
    registry = ModelRegistry()
    
    # Print registry summary
    summary = registry.get_registry_summary()
    print(json.dumps(summary, indent=2))
