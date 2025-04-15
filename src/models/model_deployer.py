#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Model Deployment System

This module handles the deployment of trained models to production:
1. Loads trained models from storage
2. Registers them with the prediction API
3. Ensures correct versioning and rollback capabilities
4. Provides health checks and monitoring

The deployment system connects trained models to API endpoints for real-time predictions.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone

# Import model classes
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.bayesian_model import BayesianModel
from src.models.ensemble_model import EnsembleModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Models directory
MODELS_DIR = Path('data/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Production directory for deployed models
PRODUCTION_DIR = Path('data/production_models')
PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

# Model registry file to track active models
MODEL_REGISTRY = PRODUCTION_DIR / 'model_registry.json'


class ModelDeployer:
    """
    Handles the deployment of trained models to production
    """
    
    def __init__(self):
        """Initialize the model deployer"""
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the model registry"""
        if MODEL_REGISTRY.exists():
            try:
                with MODEL_REGISTRY.open('r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading model registry: {str(e)}")
                return self._create_default_registry()
        else:
            return self._create_default_registry()
    
    def _create_default_registry(self) -> Dict[str, Any]:
        """Create a default model registry"""
        registry = {
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'active_models': {
                'moneyline': None,
                'spread': None,
                'total': None
            },
            'model_history': {
                'moneyline': [],
                'spread': [],
                'total': []
            }
        }
        return registry
    
    def _save_registry(self) -> None:
        """Save the model registry"""
        try:
            self.registry['last_updated'] = datetime.now(timezone.utc).isoformat()
            with MODEL_REGISTRY.open('w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model registry: {str(e)}")
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models in the models directory"""
        available_models = {}
        
        # Check for model files
        for model_file in MODELS_DIR.glob('*.pkl'):
            model_name = model_file.stem.split('_v')[0].lower()
            version = model_file.stem.split('_v')[1] if '_v' in model_file.stem else '1'
            
            # Group by model type
            if model_name not in available_models:
                available_models[model_name] = []
            
            available_models[model_name].append(version)
        
        return available_models
    
    def get_latest_model_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model"""
        available = self.list_available_models()
        
        if model_name not in available or not available[model_name]:
            return None
        
        # Sort versions numerically (assuming versions are integers)
        versions = sorted(available[model_name], key=lambda x: int(x))
        return versions[-1]  # Return the highest version number
    
    def deploy_model(self, model_name: str, model_version: Optional[str] = None,
                    prediction_target: str = 'moneyline') -> bool:
        """Deploy a model to production
        
        Args:
            model_name: Name of the model (e.g., 'randomforest', 'ensemble')
            model_version: Version of the model to deploy, or None for latest
            prediction_target: What the model predicts ('moneyline', 'spread', 'total')
            
        Returns:
            bool: True if deployment was successful
        """
        model_name = model_name.lower()
        
        # Get the latest version if not specified
        if not model_version:
            model_version = self.get_latest_model_version(model_name)
            if not model_version:
                logger.error(f"No versions found for model {model_name}")
                return False
        
        # Source model file
        source_path = MODELS_DIR / f"{model_name}_v{model_version}.pkl"
        metadata_path = MODELS_DIR / f"{model_name}_v{model_version}_metadata.json"
        
        if not source_path.exists() or not metadata_path.exists():
            logger.error(f"Model files not found: {source_path} or {metadata_path}")
            return False
        
        try:
            # Create target directories
            target_dir = PRODUCTION_DIR / prediction_target
            target_dir.mkdir(exist_ok=True)
            
            # Copy model files to production directory
            target_path = target_dir / f"{model_name}.pkl"
            target_metadata = target_dir / f"{model_name}_metadata.json"
            
            shutil.copy2(str(source_path), str(target_path))
            shutil.copy2(str(metadata_path), str(target_metadata))
            
            # Load metadata
            with metadata_path.open('r') as f:
                metadata = json.load(f)
            
            # Update registry
            deployment_info = {
                'model_name': model_name,
                'version': model_version,
                'deployed_at': datetime.now(timezone.utc).isoformat(),
                'file_path': str(target_path),
                'metadata': metadata
            }
            
            # Archive previous model if exists
            if self.registry['active_models'][prediction_target]:
                self.registry['model_history'][prediction_target].append(
                    self.registry['active_models'][prediction_target]
                )
            
            # Set as active model
            self.registry['active_models'][prediction_target] = deployment_info
            
            # Save updated registry
            self._save_registry()
            
            logger.info(f"Model {model_name} v{model_version} successfully deployed for {prediction_target} prediction")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model {model_name} v{model_version}: {str(e)}")
            return False
    
    def load_production_model(self, prediction_target: str) -> Optional[Any]:
        """Load a production model for use in predictions
        
        Args:
            prediction_target: Target prediction type ('moneyline', 'spread', 'total')
            
        Returns:
            Model instance or None if not available
        """
        try:
            # Check if model is registered
            if not self.registry['active_models'][prediction_target]:
                logger.warning(f"No active model registered for {prediction_target} prediction")
                return None
            
            model_info = self.registry['active_models'][prediction_target]
            model_name = model_info['model_name']
            version = model_info['version']
            
            # Determine model class based on name
            if 'randomforest' in model_name.lower():
                model = RandomForestModel(version=int(version))
            elif 'gradient' in model_name.lower() or 'boost' in model_name.lower():
                model = GradientBoostingModel(version=int(version))
            elif 'bayes' in model_name.lower():
                model = BayesianModel(prediction_target=prediction_target, version=int(version))
            elif 'ensemble' in model_name.lower():
                # For ensemble, determine prediction type
                if prediction_target == 'moneyline':
                    model = EnsembleModel(prediction_type='classification', version=int(version))
                else:  # 'spread' or 'total'
                    model = EnsembleModel(prediction_type='regression', version=int(version))
            else:
                logger.error(f"Unknown model type: {model_name}")
                return None
            
            # Load model from production directory
            model_path = PRODUCTION_DIR / prediction_target
            loaded = model.load_from_disk(models_dir=str(model_path), version=None)
            
            if loaded:
                logger.info(f"Loaded production model for {prediction_target}: {model_name} v{version}")
                return model
            else:
                logger.error(f"Failed to load production model for {prediction_target}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading production model for {prediction_target}: {str(e)}")
            return None
    
    def rollback_deployment(self, prediction_target: str) -> bool:
        """Rollback to the previous model version
        
        Args:
            prediction_target: Target prediction type to rollback
            
        Returns:
            bool: True if rollback was successful
        """
        try:
            # Check if there's a previous version to roll back to
            if not self.registry['model_history'][prediction_target]:
                logger.warning(f"No previous model found for {prediction_target} to roll back to")
                return False
            
            # Get last model from history
            previous_model = self.registry['model_history'][prediction_target].pop()
            current_model = self.registry['active_models'][prediction_target]
            
            # Swap models
            self.registry['active_models'][prediction_target] = previous_model
            if current_model:  # Add current model to history
                self.registry['model_history'][prediction_target].append(current_model)
            
            # Save updated registry
            self._save_registry()
            
            logger.info(f"Rolled back {prediction_target} model to previous version: {previous_model['model_name']} v{previous_model['version']}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back {prediction_target} model: {str(e)}")
            return False
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get the current deployment status
        
        Returns:
            Dictionary with deployment status information
        """
        status = {
            'last_updated': self.registry['last_updated'],
            'active_models': {}
        }
        
        # Format active models information
        for target, model_info in self.registry['active_models'].items():
            if model_info:
                status['active_models'][target] = {
                    'name': model_info['model_name'],
                    'version': model_info['version'],
                    'deployed_at': model_info['deployed_at']
                }
            else:
                status['active_models'][target] = None
        
        # Add model history count
        status['history_count'] = {
            target: len(history) for target, history in self.registry['model_history'].items()
        }
        
        return status


# Main function for testing
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NBA Model Deployment System')
    parser.add_argument('--deploy', action='store_true', help='Deploy models to production')
    parser.add_argument('--model', type=str, help='Model name to deploy')
    parser.add_argument('--version', type=str, help='Model version to deploy')
    parser.add_argument('--target', type=str, choices=['moneyline', 'spread', 'total'],
                      help='Prediction target')
    parser.add_argument('--status', action='store_true', help='Show deployment status')
    parser.add_argument('--rollback', type=str, choices=['moneyline', 'spread', 'total'],
                      help='Rollback specified prediction target')
    args = parser.parse_args()
    
    # Create deployer
    deployer = ModelDeployer()
    
    # Handle commands
    if args.deploy and args.model and args.target:
        # Deploy a model
        result = deployer.deploy_model(
            model_name=args.model,
            model_version=args.version,  # None will use latest
            prediction_target=args.target
        )
        print(f"Deployment {'successful' if result else 'failed'}")
    
    elif args.rollback:
        # Rollback a model
        result = deployer.rollback_deployment(args.rollback)
        print(f"Rollback {'successful' if result else 'failed'}")
    
    elif args.status or not (args.deploy or args.rollback):
        # Show status (default action)
        status = deployer.get_deployment_status()
        print("\nCurrent Deployment Status:")
        print(f"Last Updated: {status['last_updated']}\n")
        
        print("Active Models:")
        for target, model in status['active_models'].items():
            if model:
                print(f"  {target}: {model['name']} v{model['version']} (deployed: {model['deployed_at']})")
            else:
                print(f"  {target}: None")
        
        print("\nAvailable Models:")
        available = deployer.list_available_models()
        for model_name, versions in available.items():
            print(f"  {model_name}: {', '.join(versions)}")
