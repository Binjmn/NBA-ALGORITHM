#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Registry and Management System

This module provides tools for managing machine learning models, including:
- Model versioning and tracking
- Performance metrics tracking
- Model promotion/demotion
- Automated cleanup of old models
"""

import os
import json
import shutil
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import glob
import re
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for tracking and managing machine learning models"""
    
    def __init__(self, registry_path: str = "models/registry.json", models_dir: str = "models",
                 max_models_per_type: int = 5):
        """
        Initialize the model registry
        
        Args:
            registry_path: Path to the registry JSON file
            models_dir: Directory containing model files
            max_models_per_type: Maximum number of models to keep per type
        """
        self.registry_path = registry_path
        self.models_dir = models_dir
        self.max_models_per_type = max_models_per_type
        self.registry = self._load_registry()
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load the registry from disk, or create if it doesn't exist"""
        registry_dir = os.path.dirname(self.registry_path)
        if registry_dir and not os.path.exists(registry_dir):
            os.makedirs(registry_dir, exist_ok=True)
            
        if not os.path.exists(self.registry_path):
            # Create default registry structure
            registry = {
                "models": {},
                "production": {},
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
            self._save_registry(registry)
            return registry
        
        try:
            with open(self.registry_path, 'r') as f:
                registry = json.load(f)
            return registry
        except Exception as e:
            logger.error(f"Error loading registry: {str(e)}")
            # Return a new registry if loading fails
            return {
                "models": {},
                "production": {},
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
    
    def _save_registry(self, registry: Dict[str, Any] = None) -> None:
        """Save the registry to disk"""
        if registry is None:
            registry = self.registry
            
        registry["last_updated"] = datetime.now().isoformat()
        
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")
    
    def scan_models(self) -> None:
        """Scan the models directory and update registry with found models"""
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory {self.models_dir} does not exist")
            return
        
        # Get all model files
        model_files = glob.glob(os.path.join(self.models_dir, "*.pkl"))
        
        # Parse model info from filenames
        for model_path in model_files:
            try:
                model_file = os.path.basename(model_path)
                # Parse model type and timestamp from filename
                # Expected format: ModelType_YYYYMMDD_HHMMSS_vX.pkl
                # or ModelType_task_YYYYMMDD_HHMMSS_vX.pkl
                
                # Match pattern with or without task
                match = re.match(r"([A-Za-z]+)(?:_([a-z_]+))?_([0-9]{8}_[0-9]{6})_v([0-9]+)\.pkl", model_file)
                if not match:
                    logger.warning(f"Couldn't parse model filename: {model_file}")
                    continue
                    
                if len(match.groups()) == 4:
                    model_type, task, timestamp, version = match.groups()
                    if task is None:
                        task = "default"
                else:
                    logger.warning(f"Unexpected match groups in filename: {model_file}")
                    continue
                
                # Convert timestamp string to datetime
                try:
                    dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    created_at = dt.isoformat()
                except ValueError:
                    logger.warning(f"Invalid timestamp format in filename: {model_file}")
                    created_at = datetime.now().isoformat()
                
                # Create model entry
                model_id = f"{model_type}_{task}_{timestamp}_v{version}"
                
                # Add to registry if not already present
                if model_id not in self.registry["models"]:
                    self.registry["models"][model_id] = {
                        "id": model_id,
                        "type": model_type,
                        "task": task,
                        "version": int(version),
                        "file_path": model_path,
                        "created_at": created_at,
                        "metrics": {},
                        "is_production": False,
                        "status": "active"
                    }
            except Exception as e:
                logger.error(f"Error processing model file {model_file}: {str(e)}")
        
        # Save updated registry
        self._save_registry()
    
    def register_model(self, model_path: str, metrics: Dict[str, float] = None) -> Optional[str]:
        """Register a new model in the registry"""
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} does not exist")
            return None
        
        try:
            model_file = os.path.basename(model_path)
            
            # Parse model info from filename
            match = re.match(r"([A-Za-z]+)(?:_([a-z_]+))?_([0-9]{8}_[0-9]{6})_v([0-9]+)\.pkl", model_file)
            if not match:
                logger.warning(f"Couldn't parse model filename: {model_file}")
                return None
                
            if len(match.groups()) == 4:
                model_type, task, timestamp, version = match.groups()
                if task is None:
                    task = "default"
            else:
                logger.warning(f"Unexpected match groups in filename: {model_file}")
                return None
            
            # Convert timestamp string to datetime
            try:
                dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                created_at = dt.isoformat()
            except ValueError:
                logger.warning(f"Invalid timestamp format in filename: {model_file}")
                created_at = datetime.now().isoformat()
            
            # Create model entry
            model_id = f"{model_type}_{task}_{timestamp}_v{version}"
            
            # Add to registry
            self.registry["models"][model_id] = {
                "id": model_id,
                "type": model_type,
                "task": task,
                "version": int(version),
                "file_path": model_path,
                "created_at": created_at,
                "metrics": metrics or {},
                "is_production": False,
                "status": "active"
            }
            
            # Save updated registry
            self._save_registry()
            
            return model_id
        except Exception as e:
            logger.error(f"Error registering model {model_path}: {str(e)}")
            return None
    
    def update_metrics(self, model_id: str, metrics: Dict[str, float]) -> bool:
        """Update metrics for a model"""
        if model_id not in self.registry["models"]:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        try:
            self.registry["models"][model_id]["metrics"].update(metrics)
            self.registry["models"][model_id]["last_updated"] = datetime.now().isoformat()
            
            # Save updated registry
            self._save_registry()
            return True
        except Exception as e:
            logger.error(f"Error updating metrics for model {model_id}: {str(e)}")
            return False
    
    def promote_to_production(self, model_id: str) -> bool:
        """Promote a model to production status"""
        if model_id not in self.registry["models"]:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        try:
            model = self.registry["models"][model_id]
            model_type = model["type"]
            task = model["task"]
            
            # Check if there's an existing production model of this type
            prod_key = f"{model_type}_{task}"
            if prod_key in self.registry["production"]:
                # Demote the current production model
                current_prod_id = self.registry["production"][prod_key]
                if current_prod_id in self.registry["models"]:
                    self.registry["models"][current_prod_id]["is_production"] = False
            
            # Promote the new model
            self.registry["models"][model_id]["is_production"] = True
            self.registry["production"][prod_key] = model_id
            
            # Copy model file to a production alias for easy access
            src_path = model["file_path"]
            dest_name = f"{model_type}_{task}_production.pkl"
            dest_path = os.path.join(self.models_dir, dest_name)
            
            # Create symlink or copy depending on platform
            if os.path.exists(dest_path):
                os.remove(dest_path)
                
            shutil.copy2(src_path, dest_path)
            logger.info(f"Promoted model {model_id} to production for {prod_key}")
            
            # Save updated registry
            self._save_registry()
            return True
        except Exception as e:
            logger.error(f"Error promoting model {model_id} to production: {str(e)}")
            return False
    
    def get_production_model(self, model_type: str, task: str = "default") -> Optional[str]:
        """Get the current production model for a given type and task"""
        prod_key = f"{model_type}_{task}"
        return self.registry["production"].get(prod_key)
    
    def get_best_model(self, model_type: str, task: str = "default", metric: str = "accuracy",
                      higher_is_better: bool = True) -> Optional[str]:
        """Get the best model for a given type based on a metric"""
        # Filter models by type and task
        matching_models = [
            (model_id, model) for model_id, model in self.registry["models"].items()
            if model["type"] == model_type and model["task"] == task and metric in model.get("metrics", {})
        ]
        
        if not matching_models:
            logger.warning(f"No models found for type={model_type}, task={task} with metric={metric}")
            return None
        
        # Sort by metric value
        sorted_models = sorted(
            matching_models,
            key=lambda x: x[1]["metrics"][metric],
            reverse=higher_is_better
        )
        
        # Return the best model ID
        return sorted_models[0][0] if sorted_models else None
    
    def cleanup_old_models(self) -> int:
        """Remove excess models to maintain only max_models_per_type for each type"""
        # Group models by type and task
        model_groups = {}
        for model_id, model in self.registry["models"].items():
            key = f"{model['type']}_{model['task']}"
            if key not in model_groups:
                model_groups[key] = []
            model_groups[key].append((model_id, model))
        
        removed_count = 0
        
        # For each group, keep only the max_models_per_type newest models
        for key, models in model_groups.items():
            # Sort by creation date, newest first
            sorted_models = sorted(
                models,
                key=lambda x: x[1]["created_at"],
                reverse=True
            )
            
            # Keep only the newest and production models
            if len(sorted_models) > self.max_models_per_type:
                # Always keep production models
                models_to_keep = []
                models_to_remove = []
                
                for model_id, model in sorted_models:
                    if model["is_production"] or len(models_to_keep) < self.max_models_per_type:
                        models_to_keep.append(model_id)
                    else:
                        models_to_remove.append(model_id)
                
                # Remove excess models
                for model_id in models_to_remove:
                    try:
                        model_path = self.registry["models"][model_id]["file_path"]
                        if os.path.exists(model_path):
                            os.remove(model_path)
                            logger.info(f"Removed old model file: {model_path}")
                        
                        del self.registry["models"][model_id]
                        removed_count += 1
                    except Exception as e:
                        logger.error(f"Error removing model {model_id}: {str(e)}")
        
        # Save updated registry
        if removed_count > 0:
            self._save_registry()
        
        return removed_count


def promote_best_models_to_production(registry_path: str = "models/registry.json", 
                                     models_dir: str = "models",
                                     metric: str = "accuracy",
                                     higher_is_better: bool = True) -> Dict[str, Any]:
    """
    Automatically promote the best models to production based on metrics
    
    Args:
        registry_path: Path to the registry JSON file
        models_dir: Directory containing model files
        metric: Metric to use for determining best model
        higher_is_better: Whether higher metric values are better
        
    Returns:
        Dictionary with results of promotion
    """
    registry = ModelRegistry(registry_path=registry_path, models_dir=models_dir)
    registry.scan_models()
    
    # Get all unique model type and task combinations
    model_types = set()
    for model_id, model in registry.registry["models"].items():
        model_types.add((model["type"], model["task"]))
    
    results = {
        "promoted": [],
        "no_change": [],
        "failed": []
    }
    
    # For each type, promote the best model
    for model_type, task in model_types:
        try:
            best_model_id = registry.get_best_model(
                model_type=model_type,
                task=task,
                metric=metric,
                higher_is_better=higher_is_better
            )
            
            if not best_model_id:
                results["no_change"].append(f"{model_type}_{task}")
                continue
            
            # Check if already production
            current_prod_id = registry.get_production_model(model_type, task)
            if current_prod_id == best_model_id:
                results["no_change"].append(f"{model_type}_{task}")
                continue
            
            # Promote to production
            success = registry.promote_to_production(best_model_id)
            if success:
                results["promoted"].append(best_model_id)
            else:
                results["failed"].append(best_model_id)
        except Exception as e:
            logger.error(f"Error promoting best model for {model_type}_{task}: {str(e)}")
            results["failed"].append(f"{model_type}_{task}")
    
    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run a registry scan and cleanup
    registry = ModelRegistry()
    registry.scan_models()
    
    removed_count = registry.cleanup_old_models()
    logger.info(f"Cleaned up {removed_count} old models")
    
    # Promote best models to production
    results = promote_best_models_to_production()
    logger.info(f"Promotion results: {json.dumps(results, indent=2)}")
