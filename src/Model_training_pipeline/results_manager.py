#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Results Manager Module for NBA Model Training Pipeline

Responsibilities:
- Save trained models to disk in standardized formats
- Store and organize training results including metrics
- Generate summary reports for model training runs
- Track model versioning and history
- Export model metadata for deployment
- Manage model artifacts directory structure
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
import pickle
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback

from .config import logger


class ResultsManager:
    """
    Production-ready results manager for NBA prediction models
    
    Features:
    - Standardized model artifact management
    - Versioned model storage
    - Comprehensive results reporting
    - Model metadata generation
    - Training history tracking
    - Exportable model artifacts for deployment
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the results manager with configuration
        
        Args:
            config: Configuration dictionary with model storage settings
        """
        self.config = config
        self.models_dir = config['paths']['models_dir']
        self.results_dir = config['paths']['results_dir']
        self.production_dir = config['paths']['production_dir']
        self.backup_enabled = config['results'].get('backup_enabled', True)
        self.backup_dir = config['paths'].get('backup_dir', os.path.join(self.results_dir, 'backups'))
        
        # Initialize metrics and results tracking
        self.metrics = {
            'models_saved': 0,
            'results_saved': 0,
            'deployments_prepared': 0,
            'errors': 0
        }
        
        self.current_results = {
            'models_trained': [],
            'start_time': datetime.now().isoformat(),
            'metrics': {},
            'config': {},
            'feature_importance': {}
        }
        
        # Ensure required directories exist
        self._ensure_directories()
        
        logger.info(f"Initialized ResultsManager with models_dir={self.models_dir}, results_dir={self.results_dir}")
    
    def _ensure_directories(self) -> None:
        """
        Ensure all required directories exist
        """
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.production_dir, exist_ok=True)
        if self.backup_enabled:
            os.makedirs(self.backup_dir, exist_ok=True)
    
    def save_model(self, model_name: str, model: Any, metadata: Dict[str, Any] = None, 
                  is_production: bool = False) -> str:
        """
        Save trained model to disk with metadata
        
        Args:
            model_name: Name identifier for the model
            model: Trained model object
            metadata: Additional model metadata
            is_production: Whether to mark as production model
            
        Returns:
            Path to saved model file
        """
        logger.info(f"Saving model {model_name}")
        
        try:
            # Generate timestamp and version
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version = metadata.get('version', '1.0.0') if metadata else '1.0.0'
            
            # Create safe filename
            safe_name = model_name.lower().replace(' ', '_')
            filename = f"{safe_name}_{timestamp}.pkl"
            
            # Set appropriate directory based on production flag
            target_dir = self.production_dir if is_production else self.models_dir
            model_path = os.path.join(target_dir, filename)
            
            # Prepare complete metadata
            full_metadata = {
                'model_name': model_name,
                'timestamp': timestamp,
                'version': version,
                'is_production': is_production,
                'file_path': model_path
            }
            
            # Add user-provided metadata if any
            if metadata:
                full_metadata.update(metadata)
            
            # Save model with joblib (more efficient than pickle for scikit-learn models)
            joblib.dump(model, model_path)
            
            # Save metadata alongside model
            metadata_path = os.path.join(target_dir, f"{safe_name}_{timestamp}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)
            
            # If production model, create a symlink or copy to latest version
            if is_production:
                latest_path = os.path.join(self.production_dir, f"{safe_name}_latest.pkl")
                latest_metadata_path = os.path.join(self.production_dir, f"{safe_name}_latest_metadata.json")
                
                # Remove existing latest if exists
                if os.path.exists(latest_path):
                    os.remove(latest_path)
                if os.path.exists(latest_metadata_path):
                    os.remove(latest_metadata_path)
                
                # Create copies for latest
                shutil.copy2(model_path, latest_path)
                shutil.copy2(metadata_path, latest_metadata_path)
                
                logger.info(f"Updated production model at {latest_path}")
            
            # Track model in current results
            self.current_results['models_trained'].append({
                'name': model_name,
                'path': model_path,
                'metadata': full_metadata
            })
            
            self.metrics['models_saved'] += 1
            logger.info(f"Successfully saved model to {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics['errors'] += 1
            return ""
    
    def save_training_results(self, results: Dict[str, Any], training_id: Optional[str] = None) -> str:
        """
        Save training results and metrics to disk
        
        Args:
            results: Dictionary containing training results
            training_id: Optional identifier for the training run
            
        Returns:
            Path to saved results file
        """
        logger.info("Saving training results")
        
        try:
            # Generate timestamp and ID if not provided
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_id = training_id if training_id else f"training_run_{timestamp}"
            
            # Merge with tracked results
            full_results = self.current_results.copy()
            full_results.update(results)
            full_results['end_time'] = datetime.now().isoformat()
            full_results['run_id'] = run_id
            
            # Calculate training duration if possible
            try:
                start_time = datetime.fromisoformat(full_results['start_time'])
                end_time = datetime.fromisoformat(full_results['end_time'])
                duration_seconds = (end_time - start_time).total_seconds()
                full_results['duration_seconds'] = duration_seconds
                full_results['duration_formatted'] = str(end_time - start_time)
            except (ValueError, KeyError):
                pass
            
            # Save results to JSON file
            results_path = os.path.join(self.results_dir, f"{run_id}_results.json")
            with open(results_path, 'w') as f:
                json.dump(full_results, f, indent=2, default=str)
            
            # Save a summary CSV for easy comparison
            self._update_results_summary(full_results)
            
            self.metrics['results_saved'] += 1
            logger.info(f"Successfully saved training results to {results_path}")
            
            # Create backup if enabled
            if self.backup_enabled:
                backup_path = os.path.join(self.backup_dir, f"{run_id}_results_backup.json")
                shutil.copy2(results_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
            
            return results_path
            
        except Exception as e:
            logger.error(f"Error saving training results: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics['errors'] += 1
            return ""
    
    def _update_results_summary(self, results: Dict[str, Any]) -> None:
        """
        Update the summary CSV file with latest training results
        
        Args:
            results: Dictionary containing training results
        """
        try:
            summary_path = os.path.join(self.results_dir, "training_summary.csv")
            
            # Extract key metrics for summary
            summary_row = {
                'run_id': results.get('run_id', ''),
                'timestamp': results.get('end_time', ''),
                'models_trained': len(results.get('models_trained', [])),
                'duration_seconds': results.get('duration_seconds', 0)
            }
            
            # Add top-level metrics
            if 'metrics' in results and isinstance(results['metrics'], dict):
                for key, value in results['metrics'].items():
                    if isinstance(value, (int, float, str, bool)):
                        summary_row[f"metric_{key}"] = value
            
            # Create DataFrame for this row
            df_row = pd.DataFrame([summary_row])
            
            # Append to existing CSV or create new one
            if os.path.exists(summary_path):
                df_existing = pd.read_csv(summary_path)
                df_updated = pd.concat([df_existing, df_row], ignore_index=True)
                df_updated.to_csv(summary_path, index=False)
            else:
                df_row.to_csv(summary_path, index=False)
                
        except Exception as e:
            logger.warning(f"Error updating results summary: {str(e)}")
    
    def prepare_model_for_deployment(self, model_name: str, source_path: str, metadata: Dict[str, Any] = None) -> str:
        """
        Prepare a model for deployment by copying to production directory with appropriate metadata
        
        Args:
            model_name: Name identifier for the model
            source_path: Path to source model file
            metadata: Additional deployment metadata
            
        Returns:
            Path to deployment-ready model file
        """
        logger.info(f"Preparing {model_name} for deployment")
        
        try:
            # Ensure source model exists
            if not os.path.exists(source_path):
                logger.error(f"Source model not found at {source_path}")
                return ""
            
            # Generate timestamp and version
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version = metadata.get('version', '1.0.0') if metadata else '1.0.0'
            
            # Create safe filename
            safe_name = model_name.lower().replace(' ', '_')
            filename = f"production_{safe_name}_{timestamp}.pkl"
            
            # Define deployment path
            deployment_path = os.path.join(self.production_dir, filename)
            
            # Copy model to production directory
            shutil.copy2(source_path, deployment_path)
            
            # Prepare deployment metadata
            deploy_metadata = {
                'model_name': model_name,
                'timestamp': timestamp,
                'version': version,
                'source_path': source_path,
                'deployment_path': deployment_path,
                'is_production': True
            }
            
            # Add user-provided metadata if any
            if metadata:
                deploy_metadata.update(metadata)
            
            # Save metadata alongside model
            metadata_path = os.path.join(self.production_dir, f"production_{safe_name}_{timestamp}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(deploy_metadata, f, indent=2, default=str)
            
            # Create alias to latest production model
            latest_path = os.path.join(self.production_dir, f"production_{safe_name}.pkl")
            latest_metadata_path = os.path.join(self.production_dir, f"production_{safe_name}_metadata.json")
            
            # Remove existing latest if exists
            if os.path.exists(latest_path):
                os.remove(latest_path)
            if os.path.exists(latest_metadata_path):
                os.remove(latest_metadata_path)
            
            # Create copies for latest
            shutil.copy2(deployment_path, latest_path)
            shutil.copy2(metadata_path, latest_metadata_path)
            
            self.metrics['deployments_prepared'] += 1
            logger.info(f"Successfully prepared model for deployment at {deployment_path}")
            logger.info(f"Production alias created at {latest_path}")
            
            return deployment_path
            
        except Exception as e:
            logger.error(f"Error preparing model for deployment: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics['errors'] += 1
            return ""
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a model from disk
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model object or None if error
        """
        logger.info(f"Loading model from {model_path}")
        
        try:
            # Ensure model exists
            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}")
                return None
            
            # Load model using joblib
            model = joblib.load(model_path)
            logger.info(f"Successfully loaded model from {model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def get_available_models(self, production_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of available models with metadata
        
        Args:
            production_only: Whether to only include production models
            
        Returns:
            List of dictionaries with model information
        """
        models_info = []
        
        try:
            # Determine which directory to scan
            target_dir = self.production_dir if production_only else self.models_dir
            
            # List all pickle files in directory
            pickle_files = [f for f in os.listdir(target_dir) if f.endswith('.pkl') and not f.endswith('_latest.pkl')]
            
            for pkl_file in pickle_files:
                # Try to find corresponding metadata file
                base_name = os.path.splitext(pkl_file)[0]
                metadata_file = f"{base_name}_metadata.json"
                metadata_path = os.path.join(target_dir, metadata_file)
                
                model_info = {
                    'file_name': pkl_file,
                    'path': os.path.join(target_dir, pkl_file),
                    'metadata': {}
                }
                
                # Load metadata if available
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            model_info['metadata'] = metadata
                            # Extract model name from metadata if available
                            if 'model_name' in metadata:
                                model_info['model_name'] = metadata['model_name']
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse metadata file {metadata_path}")
                
                # Extract model name from filename if not in metadata
                if 'model_name' not in model_info:
                    parts = base_name.split('_')
                    if len(parts) > 1:
                        model_info['model_name'] = '_'.join(parts[:-1])  # Exclude timestamp
                    else:
                        model_info['model_name'] = base_name
                
                models_info.append(model_info)
            
            # Sort by timestamp if available
            models_info.sort(key=lambda x: x['metadata'].get('timestamp', ''), reverse=True)
            
            logger.info(f"Found {len(models_info)} available models")
            return models_info
            
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def get_latest_training_results(self, count: int = 1) -> List[Dict[str, Any]]:
        """
        Get the most recent training results
        
        Args:
            count: Number of recent training results to retrieve
            
        Returns:
            List of training result dictionaries
        """
        results = []
        
        try:
            # List all JSON result files in results directory
            result_files = [f for f in os.listdir(self.results_dir) if f.endswith('_results.json')]
            
            # Sort by modification time (most recent first)
            result_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.results_dir, x)), reverse=True)
            
            # Load the most recent results
            for i, result_file in enumerate(result_files[:count]):
                result_path = os.path.join(self.results_dir, result_file)
                try:
                    with open(result_path, 'r') as f:
                        result_data = json.load(f)
                        results.append(result_data)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse result file {result_path}")
            
            logger.info(f"Retrieved {len(results)} recent training results")
            return results
            
        except Exception as e:
            logger.error(f"Error getting latest training results: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def get_results_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the results management process
        
        Returns:
            Dictionary with results metrics
        """
        return self.metrics