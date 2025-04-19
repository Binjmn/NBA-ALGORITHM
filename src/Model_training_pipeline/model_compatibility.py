#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Compatibility Module

Ensures backward compatibility with existing prediction systems.
Provides tools for model versioning, validation, and format conversion.
"""

import os
import json
import pickle
import hashlib
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from .config import logger


class ModelCompatibilityManager:
    """
    Manages model compatibility with existing prediction systems
    
    Responsibilities:
    - Ensure trained models are compatible with production formats
    - Validate model outputs against expected schemas
    - Convert between model formats if needed
    - Track model versions and maintain backward compatibility
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the compatibility manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.production_model_schema = {
            'random_forest': {
                'required_methods': ['predict', 'predict_proba'],
                'required_attributes': ['feature_importances_', 'n_estimators']
            },
            'gradient_boosting': {
                'required_methods': ['predict', 'predict_proba'],
                'required_attributes': ['feature_importances_', 'n_estimators']
            },
            'ensemble': {
                'required_methods': ['predict'],
                'required_attributes': ['estimators_']
            }
        }
        
        self.metrics = {
            'models_validated': 0,
            'models_converted': 0,
            'compatibility_issues': 0
        }
        
        logger.info("Initialized ModelCompatibilityManager")
    
    def validate_model_compatibility(self, model: Any, model_type: str) -> Tuple[bool, List[str]]:
        """
        Validate that a model is compatible with the production prediction system
        
        Args:
            model: Trained model object
            model_type: Type of model ('random_forest', 'gradient_boosting', etc.)
            
        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        logger.info(f"Validating compatibility for {model_type} model")
        issues = []
        
        # Check if model type is supported
        if model_type not in self.production_model_schema:
            issues.append(f"Model type '{model_type}' is not supported in production")
            return False, issues
        
        # Check required methods
        for method_name in self.production_model_schema[model_type]['required_methods']:
            if not hasattr(model, method_name) or not callable(getattr(model, method_name)):
                issues.append(f"Missing required method: {method_name}")
        
        # Check required attributes
        for attr_name in self.production_model_schema[model_type]['required_attributes']:
            if not hasattr(model, attr_name):
                issues.append(f"Missing required attribute: {attr_name}")
        
        # Check if the model can be serialized properly
        try:
            pickle.dumps(model)
        except Exception as e:
            issues.append(f"Model serialization failed: {str(e)}")
        
        self.metrics['models_validated'] += 1
        is_compatible = len(issues) == 0
        
        if is_compatible:
            logger.info("Model is compatible with production system")
        else:
            logger.warning(f"Model compatibility issues found: {len(issues)}")
            logger.warning("\n".join(issues))
            self.metrics['compatibility_issues'] += 1
        
        return is_compatible, issues
    
    def prepare_model_for_production(self, model_id: str, model: Any, metadata: Dict[str, Any]) -> str:
        """
        Prepare a model for production use by validating and converting if needed
        
        Args:
            model_id: Unique identifier for the model
            model: Trained model object
            metadata: Model metadata
            
        Returns:
            Path to production-ready model file
        """
        model_type = metadata.get('model_type', 'unknown')
        logger.info(f"Preparing model {model_id} for production deployment")
        
        # Validate compatibility
        is_compatible, issues = self.validate_model_compatibility(model, model_type)
        
        if not is_compatible:
            # Try to fix compatibility issues
            model = self._convert_model_format(model, model_type, issues)
            self.metrics['models_converted'] += 1
            
            # Re-validate after conversion
            is_compatible, issues = self.validate_model_compatibility(model, model_type)
            if not is_compatible:
                logger.error(f"Failed to make model {model_id} compatible with production")
                logger.error("\n".join(issues))
                return ""
        
        # Add production metadata
        production_metadata = metadata.copy()
        production_metadata.update({
            'production_ready': True,
            'compatibility_checked': True,
            'production_timestamp': datetime.now().isoformat(),
            'model_format_version': '2.0',
            'prediction_api_version': '1.0'
        })
        
        # Create production model file
        production_dir = os.path.join(self.config['paths']['models_dir'], 'production')
        os.makedirs(production_dir, exist_ok=True)
        production_path = os.path.join(production_dir, f"production_{model_id}.pkl")
        
        # Save model with production metadata
        try:
            with open(production_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'metadata': production_metadata
                }, f)
            
            # Save metadata separately as JSON for easy access
            meta_path = os.path.join(production_dir, f"production_{model_id}_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(production_metadata, f, indent=2)
                
            logger.info(f"Model prepared for production and saved to {production_path}")
            return production_path
            
        except Exception as e:
            logger.error(f"Error saving production model: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
    
    def _convert_model_format(self, model: Any, model_type: str, issues: List[str]) -> Any:
        """
        Attempt to convert a model to be compatible with production
        
        Args:
            model: Model to convert
            model_type: Type of model
            issues: List of compatibility issues
            
        Returns:
            Converted model or original if conversion not possible
        """
        logger.info(f"Attempting to convert {model_type} model format")
        
        # Implement specific conversions based on issues and model type
        try:
            # Check for specific issues that can be addressed
            missing_methods = [issue.split(': ')[1] for issue in issues 
                              if issue.startswith('Missing required method:')]
            
            missing_attrs = [issue.split(': ')[1] for issue in issues 
                            if issue.startswith('Missing required attribute:')]
            
            # Example: Adding wrapper methods to ensure API compatibility
            if model_type == 'random_forest' and 'predict_proba' in missing_methods:
                # Add predict_proba wrapper if only predict exists
                if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
                    original_predict = model.predict
                    
                    def predict_proba_wrapper(X):
                        preds = original_predict(X)
                        # Convert to probability-like format
                        if len(preds.shape) == 1:
                            # For binary classification
                            probs = np.zeros((len(preds), 2))
                            probs[:, 1] = preds
                            probs[:, 0] = 1 - preds
                            return probs
                        return preds
                    
                    setattr(model, 'predict_proba', predict_proba_wrapper)
                    logger.info("Added predict_proba wrapper method")
            
            # Include additional conversion logic as needed
            
            return model
            
        except Exception as e:
            logger.error(f"Error converting model format: {str(e)}")
            logger.error(traceback.format_exc())
            return model
    
    def get_compatibility_metrics(self) -> Dict[str, Any]:
        """
        Get metrics related to model compatibility
        
        Returns:
            Dictionary of compatibility metrics
        """
        return self.metrics.copy()


def check_backward_compatibility(model_path: str, reference_model_path: str) -> Tuple[bool, List[str]]:
    """
    Check if a new model is backward compatible with a reference model
    
    Args:
        model_path: Path to new model file
        reference_model_path: Path to reference model file
        
    Returns:
        Tuple of (is_compatible, list_of_issues)
    """
    issues = []
    
    try:
        # Load both models
        with open(model_path, 'rb') as f:
            new_model_data = pickle.load(f)
            
        with open(reference_model_path, 'rb') as f:
            reference_model_data = pickle.load(f)
        
        new_model = new_model_data.get('model', new_model_data)
        reference_model = reference_model_data.get('model', reference_model_data)
        
        # Check for method compatibility
        for method_name in ['predict', 'predict_proba']:
            if hasattr(reference_model, method_name) and callable(getattr(reference_model, method_name)):
                if not hasattr(new_model, method_name) or not callable(getattr(new_model, method_name)):
                    issues.append(f"New model is missing method {method_name} from reference model")
        
        # Check input requirements if available in metadata
        new_metadata = new_model_data.get('metadata', {})
        ref_metadata = reference_model_data.get('metadata', {})
        
        if 'feature_count' in ref_metadata and 'feature_count' in new_metadata:
            if new_metadata['feature_count'] != ref_metadata['feature_count']:
                issues.append(f"Feature count mismatch: reference={ref_metadata['feature_count']}, new={new_metadata['feature_count']}")
        
        # Create test data and check that prediction shapes match
        if hasattr(reference_model, 'predict'):
            # Generate dummy input based on model type
            if hasattr(reference_model, 'n_features_in_'):
                n_features = reference_model.n_features_in_
            elif 'feature_count' in ref_metadata:
                n_features = ref_metadata['feature_count']
            else:
                n_features = 10  # Default fallback
                
            X_test = np.random.rand(5, n_features)
            
            try:
                ref_preds = reference_model.predict(X_test)
                new_preds = new_model.predict(X_test)
                
                if ref_preds.shape != new_preds.shape:
                    issues.append(f"Prediction shape mismatch: reference={ref_preds.shape}, new={new_preds.shape}")
            except Exception as e:
                issues.append(f"Prediction compatibility test failed: {str(e)}")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        issues.append(f"Compatibility check failed: {str(e)}")
        logger.error(f"Error checking backward compatibility: {str(e)}")
        logger.error(traceback.format_exc())
        return False, issues
