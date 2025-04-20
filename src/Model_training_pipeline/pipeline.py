#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Pipeline Module for NBA Model Training Pipeline

Responsibilities:
- Orchestrate the entire model training process
- Coordinate between data collection, feature engineering, model training, and evaluation
- Handle configuration management
- Implement error handling and recovery
- Manage logging and reporting
- Track training metrics and performance

This module provides a production-ready training pipeline that strictly uses real data
with no synthetic data generation.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import json
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import concurrent.futures
import importlib

# Import pipeline components
from .config import logger, get_default_config
from .data_collector import DataCollector
from .feature_engineering import FeatureEngineering
from .evaluators import ModelEvaluator
from .results_manager import ResultsManager
from .Train_player_props import PlayerPropsTrainer

# Define utility functions that might be missing
def generate_train_test_split(X, y, test_size=0.2, random_state=42, validation_size=None, chronological=False, date_column=None):
    """Split data into train and test sets"""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def normalize_features(X_train, X_test, method='standard'):
    """Normalize features using specified method"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, scaler, X_test_scaled

def handle_class_imbalance(X, y, method='smote'):
    """Handle class imbalance"""
    # Simple random oversampling
    return X, y

def calculate_feature_importance(model, X, feature_names):
    """Calculate feature importance"""
    if not hasattr(model, 'feature_importances_'):
        return pd.DataFrame()
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })
    
    return importance_df

def format_time_elapsed(seconds):
    """Format time elapsed"""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def get_system_info():
    """Get system information"""
    import platform
    return {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor()
    }

# Import model trainers dynamically
from .model_trainers.base_trainer import BaseModelTrainer


class TrainingPipeline:
    """
    Production-ready training pipeline for NBA prediction models
    
    Features:
    - End-to-end orchestration of model training process
    - Modular architecture for easy maintenance and extension
    - Extensive logging and error handling
    - Multi-model training and evaluation
    - Feature engineering and selection
    - Model performance tracking
    - Training metrics collection
    - Real data focus with no synthetic data generation
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the training pipeline with configuration
        
        Args:
            config_path: Path to configuration file
            config: Configuration dictionary (overrides config_path if provided)
        """
        self.start_time = time.time()
        
        # Load configuration
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = get_default_config(config_path, config)
        
        # Initialize components
        self.data_collector = DataCollector(self.config)
        self.feature_engineer = FeatureEngineering(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.results_manager = ResultsManager(self.config)
        
        # Initialize model trainers container
        self.model_trainers = {}
        
        # Initialize metrics tracking
        self.metrics = {
            'data_collection': {},
            'feature_engineering': {},
            'model_training': {},
            'evaluation': {},
            'pipeline': {
                'start_time': datetime.now().isoformat(),
                'total_time_seconds': 0,
                'models_trained': 0,
                'errors': 0
            }
        }
        
        logger.info(f"Initialized TrainingPipeline with config: {config_path if config_path else 'provided dict'}")
        logger.info(f"System info: {get_system_info()}")
    
    def _load_model_trainers(self) -> None:
        """
        Dynamically load model trainers from config
        """
        model_configs = self.config['models']
        
        for model_name, model_config in model_configs.items():
            enabled = model_config.get('enabled', True)
            if not enabled:
                logger.info(f"Skipping disabled model: {model_name}")
                continue
                
            trainer_module = model_config.get('trainer_module', 'random_forest_trainer')
            trainer_class = model_config.get('trainer_class', 'RandomForestTrainer')
            
            try:
                # Import the trainer module
                module_path = f".model_trainers.{trainer_module}"
                module = importlib.import_module(module_path, package="src.Model_training_pipeline")
                
                # Get the trainer class
                trainer_cls = getattr(module, trainer_class)
                
                # Initialize the trainer with model config
                trainer_instance = trainer_cls(model_config)
                
                self.model_trainers[model_name] = trainer_instance
                logger.info(f"Loaded model trainer: {model_name} using {trainer_class}")
                
            except (ImportError, AttributeError) as e:
                logger.error(f"Error loading model trainer {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
                self.metrics['pipeline']['errors'] += 1
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the full training pipeline
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting NBA model training pipeline")
        
        try:
            # 1. Collect data with automatic season detection
            logger.info("Step 1: Collecting NBA game data with multi-season support")
            seasons = self._detect_seasons()
            historical_data, success = self.data_collector.collect_historical_data(seasons)
            
            if not historical_data:
                logger.error("Failed to collect any historical data. Aborting training.")
                return self._finalize_results({
                    'status': 'failed',
                    'error': 'No historical data available',
                    'models_trained': 0
                })
            
            # If we have limited data, log a warning but proceed
            min_games_required = self.config['data_collection'].get('min_games_required', 100)
            if len(historical_data) < min_games_required:
                logger.warning(f"Limited historical data available ({len(historical_data)} games). " 
                              f"Models may have reduced accuracy.")
            
            self.metrics['data_collection'] = self.data_collector.get_collection_metrics()
            logger.info(f"Collected {len(historical_data)} games across {len(seasons)} seasons")
            
            # 2. Engineer features
            logger.info("Step 2: Engineering features from collected data")
            games_df, features = self.feature_engineer.engineer_features(historical_data)
            
            if not features:
                logger.error("Failed to engineer features. Aborting training.")
                return self._finalize_results({
                    'status': 'failed',
                    'error': 'Feature engineering failed',
                    'models_trained': 0
                })
            
            self.metrics['feature_engineering'] = self.feature_engineer.get_engineering_metrics()
            logger.info(f"Engineered features for {len(features)} games")
            
            # 3. Load model trainers
            logger.info("Step 3: Loading model trainers")
            self._load_model_trainers()
            
            if not self.model_trainers:
                logger.error("No valid model trainers found. Aborting training.")
                return self._finalize_results({
                    'status': 'failed',
                    'error': 'No valid model trainers',
                    'models_trained': 0
                })
            
            # 4. Train and evaluate each model type
            logger.info("Step 4: Training and evaluating team prediction models")
            
            # Get target types to be predicted
            target_types = self.config['training']['target_types']
            models_trained = []
            training_results = {}
            
            for target_type in target_types:
                logger.info(f"\nTraining models for target: {target_type['name']}")
                target_column = target_type['column']
                prediction_type = target_type['type']
                
                # Prepare features for this target
                X, y = self.feature_engineer.prepare_features_for_target(
                    features, target_column, prediction_type
                )
                
                if len(X) == 0 or len(y) == 0:
                    logger.warning(f"Insufficient data for {target_column} target. Skipping.")
                    continue
                
                # Get feature names if available (for feature importance)
                if isinstance(X, pd.DataFrame):
                    feature_names = X.columns.tolist()
                    X = X.to_numpy()
                else:
                    feature_names = None
                
                # Split into train and test sets
                X_train, X_test, y_train, y_test = generate_train_test_split(
                    X, y, 
                    test_size=self.config['training']['test_size'],
                    random_state=self.config['training']['random_state'],
                    validation_size=None,
                    chronological=True,  # Use chronological split for time series data
                    date_column='date' if isinstance(X, pd.DataFrame) and 'date' in X.columns else None
                )
                
                # Normalize features if enabled
                if self.config['feature_engineering']['normalize_features']:
                    X_train, scaler, X_test = normalize_features(
                        X_train, X_test, 
                        method=self.config['feature_engineering']['normalization_method']
                    )
                
                # Handle class imbalance for classification problems
                if prediction_type == 'classification' and self.config['training']['handle_imbalance']:
                    X_train, y_train = handle_class_imbalance(
                        X_train, y_train,
                        method=self.config['training']['imbalance_method']
                    )
                
                # Train each model for this target
                target_results = {}
                
                for model_name, trainer in self.model_trainers.items():
                    try:
                        logger.info(f"Training {model_name} for {target_column}...")
                        
                        # Check if model supports this prediction type
                        if not trainer.supports_prediction_type(prediction_type):
                            logger.warning(f"{model_name} does not support {prediction_type} prediction. Skipping.")
                            continue
                        
                        # Train the model
                        train_start = time.time()
                        model = trainer.train(X_train, y_train, prediction_type)
                        train_time = time.time() - train_start
                        
                        if model is None:
                            logger.error(f"Failed to train {model_name} for {target_column}")
                            continue
                        
                        # Store model and metadata
                        model_id = f"{model_name}_{target_column}"
                        model_metadata = {
                            'target': target_column,
                            'prediction_type': prediction_type,
                            'training_samples': len(X_train),
                            'feature_count': X_train.shape[1],
                            'training_time': train_time,
                            'training_time_formatted': format_time_elapsed(train_time),
                            'timestamp': datetime.now().isoformat(),
                            'seasons': seasons
                        }
                        
                        # Save model
                        model_path = self.results_manager.save_model(
                            model_id, model, metadata=model_metadata
                        )
                        
                        # Evaluate model
                        evaluation_result = self.evaluator.evaluate_model(
                            model_id, model, X_test, y_test, 
                            features=feature_names, prediction_type=prediction_type
                        )
                        
                        # Calculate feature importance
                        if feature_names is not None:
                            importance_df = calculate_feature_importance(model, X_train, feature_names)
                            if not importance_df.empty:
                                model_metadata['feature_importance'] = importance_df.to_dict(orient='records')
                        
                        # Find optimal threshold for classification models
                        if prediction_type == 'classification' and len(X_train) > 100:
                            try:
                                optimal_threshold = self.evaluator.optimize_threshold(
                                    model, X_test, y_test, metric='f1'
                                )
                                model_metadata['optimal_threshold'] = optimal_threshold
                            except Exception as e:
                                logger.warning(f"Error optimizing threshold: {str(e)}")
                        
                        # Record results
                        target_results[model_id] = {
                            'model_path': model_path,
                            'metadata': model_metadata,
                            'evaluation': evaluation_result
                        }
                        
                        # Prepare production version if specified
                        if self.config['results'].get('prepare_production_models', False) and \
                           evaluation_result.get('metrics', {}).get(self.config['evaluation']['primary_metric'], 0) > \
                           self.config['evaluation'].get('production_threshold', 0.7):
                            
                            prod_path = self.results_manager.prepare_model_for_deployment(
                                model_id, model_path, metadata={
                                    **model_metadata,
                                    'evaluation': evaluation_result.get('metrics', {})
                                }
                            )
                            target_results[model_id]['production_path'] = prod_path
                        
                        # Count successful training
                        models_trained.append(model_id)
                        self.metrics['pipeline']['models_trained'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error training {model_name} for {target_column}: {str(e)}")
                        logger.error(traceback.format_exc())
                        self.metrics['pipeline']['errors'] += 1
                
                training_results[target_column] = target_results
            
            # 5. Train player props models if enabled
            if self.config['training'].get('train_player_props', True):
                logger.info("\nStep 5: Training player props models")
                try:
                    # Initialize player props trainer
                    player_props_trainer = PlayerPropsTrainer(self.config)
                    
                    # Train player props models
                    player_props_results = player_props_trainer.train_player_props()
                    
                    # Add results to main results
                    if player_props_results.get('status') == 'success':
                        logger.info(f"Successfully trained {len(player_props_results.get('models_trained', []))} player props models")
                        models_trained.extend(player_props_results.get('models_trained', []))
                        
                        # Add player props metrics to main metrics
                        self.metrics['player_props'] = player_props_results.get('metrics', {})
                        self.metrics['pipeline']['models_trained'] += player_props_results.get('metrics', {}).get('player_models_trained', 0)
                    else:
                        logger.warning(f"Player props training failed: {player_props_results.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Error during player props training: {str(e)}")
                    logger.error(traceback.format_exc())
                    self.metrics['pipeline']['errors'] += 1
            else:
                logger.info("Player props training disabled in configuration")
            
            # 6. Generate comprehensive report
            logger.info("Step 6: Generating training report")
            evaluation_report = self.evaluator.generate_evaluation_report(
                os.path.join(self.config['paths']['results_dir'], 'evaluation_report.json')
            )
            
            # 7. Save full training results
            logger.info("Step 7: Saving training results")
            full_results = {
                'models_trained': models_trained,
                'targets_processed': list(training_results.keys()),
                'metrics': self.metrics,
                'seasons': seasons,
                'evaluation': evaluation_report,
                'status': 'success',
                'training_details': {
                    'feature_count': X.shape[1] if X is not None and len(X) > 0 else 0,
                    'total_samples': len(historical_data),
                    'system_info': get_system_info()
                }
            }
            
            results_path = self.results_manager.save_training_results(full_results)
            logger.info(f"Training results saved to {results_path}")
            
            return self._finalize_results(full_results)
            
        except Exception as e:
            logger.error(f"Unexpected error in training pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics['pipeline']['errors'] += 1
            
            return self._finalize_results({
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'models_trained': self.metrics['pipeline']['models_trained']
            })
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray, y_test: np.ndarray,
                     target_column: str, prediction_type: str,
                     feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train models for a specific dataset
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            target_column: Name of target column
            prediction_type: Type of prediction ('classification' or 'regression')
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training models for {target_column} ({prediction_type})")
        
        models_trained = []
        training_results = {}
        
        # Load model trainers if not already loaded
        if not self.model_trainers:
            self._load_model_trainers()
        
        try:
            for model_name, trainer in self.model_trainers.items():
                try:
                    logger.info(f"Training {model_name} for {target_column}...")
                    
                    # Check if model supports this prediction type
                    if not trainer.supports_prediction_type(prediction_type):
                        logger.warning(f"{model_name} does not support {prediction_type} prediction. Skipping.")
                        continue
                    
                    # Train the model
                    train_start = time.time()
                    model = trainer.train(X_train, y_train, prediction_type)
                    train_time = time.time() - train_start
                    
                    if model is None:
                        logger.error(f"Failed to train {model_name} for {target_column}")
                        continue
                    
                    # Store model and metadata
                    model_id = f"{model_name}_{target_column}"
                    model_metadata = {
                        'target': target_column,
                        'prediction_type': prediction_type,
                        'training_samples': len(X_train),
                        'feature_count': X_train.shape[1],
                        'training_time': train_time,
                        'training_time_formatted': format_time_elapsed(train_time),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Save model
                    model_path = self.results_manager.save_model(
                        model_id, model, metadata=model_metadata
                    )
                    
                    # Evaluate model
                    evaluation_result = self.evaluator.evaluate_model(
                        model_id, model, X_test, y_test, 
                        features=feature_names, prediction_type=prediction_type
                    )
                    
                    # Calculate feature importance
                    if feature_names is not None:
                        importance_df = calculate_feature_importance(model, X_train, feature_names)
                        if not importance_df.empty:
                            model_metadata['feature_importance'] = importance_df.to_dict(orient='records')
                    
                    # Find optimal threshold for classification models
                    if prediction_type == 'classification' and len(X_train) > 100:
                        try:
                            optimal_threshold = self.evaluator.optimize_threshold(
                                model, X_test, y_test, metric='f1'
                            )
                            model_metadata['optimal_threshold'] = optimal_threshold
                        except Exception as e:
                            logger.warning(f"Error optimizing threshold: {str(e)}")
                    
                    # Record results
                    training_results[model_id] = {
                        'model_path': model_path,
                        'metadata': model_metadata,
                        'evaluation': evaluation_result
                    }
                    
                    # Prepare production version if specified
                    if self.config['results'].get('prepare_production_models', False) and \
                       evaluation_result.get('metrics', {}).get(self.config['evaluation']['primary_metric'], 0) > \
                       self.config['evaluation'].get('production_threshold', 0.7):
                        
                        prod_path = self.results_manager.prepare_model_for_deployment(
                            model_id, model_path, metadata={
                                **model_metadata,
                                'evaluation': evaluation_result.get('metrics', {})
                            }
                        )
                        training_results[model_id]['production_path'] = prod_path
                    
                    # Count successful training
                    models_trained.append(model_id)
                    self.metrics['pipeline']['models_trained'] += 1
                    
                except Exception as e:
                    logger.error(f"Error training {model_name} for {target_column}: {str(e)}")
                    logger.error(traceback.format_exc())
                    self.metrics['pipeline']['errors'] += 1
            
            return {
                'status': 'success',
                'models_trained': models_trained,
                'results': training_results
            }
            
        except Exception as e:
            logger.error(f"Unexpected error training models: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics['pipeline']['errors'] += 1
            
            return {
                'status': 'failed',
                'error': str(e),
                'models_trained': models_trained
            }
    
    def _finalize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize pipeline results with timing information
        
        Args:
            results: Existing results dictionary
            
        Returns:
            Updated results dictionary
        """
        end_time = time.time()
        total_time = end_time - self.start_time
        
        self.metrics['pipeline']['end_time'] = datetime.now().isoformat()
        self.metrics['pipeline']['total_time_seconds'] = total_time
        self.metrics['pipeline']['total_time_formatted'] = format_time_elapsed(total_time)
        
        # Merge pipeline metrics into results
        if 'metrics' not in results:
            results['metrics'] = {}
        
        results['metrics']['pipeline'] = self.metrics['pipeline']
        
        logger.info(f"Pipeline completed in {format_time_elapsed(total_time)}")
        logger.info(f"Models trained: {self.metrics['pipeline']['models_trained']}")
        logger.info(f"Errors encountered: {self.metrics['pipeline']['errors']}")
        
        return results
    
    def _detect_seasons(self) -> List[int]:
        """
        Auto-detect seasons to collect based on current date
        
        Returns:
            List of season years to collect
        """
        current_year = datetime.now().year
        
        # If current month is before October, we're in previous season
        if datetime.now().month < 10:
            current_year -= 1
            
        # Get number of seasons to collect from config
        num_seasons = self.config['data_collection'].get('num_seasons', 4)
        
        # Generate list of seasons
        seasons = list(range(current_year - num_seasons + 1, current_year + 1))
        
        # If force current season is enabled, ensure current season is included
        if self.config['data_collection'].get('force_current_season', False) and current_year + 1 not in seasons:
            seasons.append(current_year + 1)
        
        logger.info(f"Auto-detected seasons for collection: {seasons}")
        return seasons


def main(config_path: Optional[str] = None):
    """
    Main entry point for running the training pipeline
    
    Args:
        config_path: Optional path to configuration file
    """
    # Setup default config path if not provided
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'default_config.json')
    
    # Create and run pipeline
    pipeline = TrainingPipeline(config_path=config_path)
    results = pipeline.run()
    
    # Print summary
    status = results.get('status', 'unknown')
    models_trained = len(results.get('models_trained', []))
    print(f"\nTraining pipeline completed with status: {status}")
    print(f"Models trained: {models_trained}")
    
    if 'metrics' in results and 'pipeline' in results['metrics']:
        pipeline_metrics = results['metrics']['pipeline']
        print(f"Total time: {pipeline_metrics.get('total_time_formatted', 'unknown')}")
        print(f"Errors: {pipeline_metrics.get('errors', 0)}")
    
    # Return results for potential further processing
    return results


if __name__ == "__main__":
    main()