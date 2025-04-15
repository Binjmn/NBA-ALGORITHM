#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Model Training Pipeline

This module coordinates the entire training process for NBA prediction models:
1. Fetches historical data from the data collection module
2. Processes features using the engineering pipeline
3. Trains all models with appropriate validation
4. Evaluates performance against betting markets
5. Deploys the best models to production

The pipeline can be run manually or scheduled for regular retraining.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import time

# Import data collection and feature engineering modules
from src.data.historical_collector import HistoricalDataCollector
from src.data.feature_engineering import NBAFeatureEngineer

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
        logging.FileHandler('logs/training_pipeline.log'),
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

# Results directory for storing training outputs
RESULTS_DIR = Path('data/training_results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ModelTrainingPipeline:
    """
    Orchestrates the entire model training pipeline from data collection to deployment
    """
    
    def __init__(self, season: str = "2024"):
        """
        Initialize the training pipeline
        
        Args:
            season: NBA season to train models for (e.g., "2024" for 2023-2024 season)
        """
        self.season = season
        self.collector = HistoricalDataCollector()
        self.engineer = NBAFeatureEngineer()
        
        # Training configuration
        self.config = {
            'season': season,
            'data_collection': {
                'include_stats': True,
                'include_odds': True
            },
            'feature_engineering': {
                'window_size': 10,  # Number of games to use for form metrics
                'home_advantage': 3.0  # Home court advantage in points
            },
            'training': {
                'test_split': 0.2,  # Fraction of data to use for testing
                'random_state': 42,
                'optimize_hyperparams': True
            },
            'models': {
                'random_forest': True,
                'gradient_boosting': True,
                'bayesian': True,
                'ensemble': True
            },
            'prediction_targets': {
                'moneyline': True,  # Win/loss prediction
                'spread': True,     # Point spread prediction
                'total': True       # Over/under prediction
            },
            'logging': {
                'verbose': True,
                'save_metrics': True
            }
        }
        
        # Metadata about the training run
        self.training_metadata = {
            'start_time': None,
            'end_time': None,
            'data_collected': False,
            'features_engineered': False,
            'models_trained': [],
            'performance_metrics': {}
        }
    
    def collect_historical_data(self) -> bool:
        """
        Collect historical NBA data for the specified season
        
        Returns:
            bool: True if data collection was successful
        """
        logger.info(f"Collecting historical data for {self.season} season")
        
        try:
            # Collect data for the entire season
            result = self.collector.collect_data_for_season(
                season=self.season,
                include_stats=self.config['data_collection']['include_stats'],
                include_odds=self.config['data_collection']['include_odds']
            )
            
            # Update metadata
            self.training_metadata['data_collected'] = True
            logger.info(f"Data collection completed: {result}")
            
            return True
        except Exception as e:
            logger.error(f"Error collecting historical data: {str(e)}")
            return False
    
    def engineer_features(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process raw data into engineered features for model training
        
        Returns:
            Tuple containing:
            - DataFrame of games data
            - Dictionary of all game features
        """
        logger.info("Engineering features from historical data")
        
        try:
            # Load games data
            games_df = self.engineer.load_games()
            
            if games_df.empty:
                logger.error("No games data available for feature engineering")
                return pd.DataFrame(), []
            
            # Generate features for all games
            features = self.engineer.generate_features_for_all_games(games_df)
            
            # Update metadata
            self.training_metadata['features_engineered'] = True
            logger.info(f"Feature engineering completed: {len(features)} games processed")
            
            return games_df, features
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            return pd.DataFrame(), []
    
    def train_models(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train all specified models
        
        Args:
            features: List of game features to use for training
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training prediction models")
        
        if not features:
            logger.error("No features available for model training")
            return {}
        
        try:
            results = {}
            
            # Train models for each prediction target
            targets = [
                ('moneyline', 'home_win', 'classification'),
                ('spread', 'spread', 'regression'),
                ('total', 'total', 'regression')
            ]
            
            for target_name, target_column, prediction_type in targets:
                if not self.config['prediction_targets'][target_name]:
                    continue
                
                logger.info(f"Training models for {target_name} prediction")
                
                # Prepare training data
                X, y = self.engineer.prepare_training_data(features, target=target_column)
                
                if X.empty or y.empty:
                    logger.warning(f"No valid training data for {target_name} prediction")
                    continue
                
                target_results = {}
                
                # Train Random Forest model (for classification targets)
                if self.config['models']['random_forest'] and prediction_type == 'classification':
                    try:
                        rf_model = RandomForestModel(version=int(datetime.now().timestamp()))
                        rf_model.train(X, y)
                        metrics = rf_model.evaluate(X, y)
                        model_path = rf_model.save_to_disk()
                        
                        target_results['random_forest'] = {
                            'metrics': metrics,
                            'path': model_path,
                            'importance': rf_model.get_feature_importance()
                        }
                        
                        self.training_metadata['models_trained'].append(f"random_forest_{target_name}")
                        logger.info(f"Random Forest model for {target_name} trained: {metrics}")
                    except Exception as e:
                        logger.error(f"Error training Random Forest model for {target_name}: {str(e)}")
                
                # Train Gradient Boosting model (for regression targets)
                if self.config['models']['gradient_boosting'] and prediction_type == 'regression':
                    try:
                        gb_model = GradientBoostingModel(version=int(datetime.now().timestamp()))
                        gb_model.train(X, y)
                        metrics = gb_model.evaluate(X, y)
                        model_path = gb_model.save_to_disk()
                        
                        target_results['gradient_boosting'] = {
                            'metrics': metrics,
                            'path': model_path,
                            'importance': gb_model.get_feature_importance()
                        }
                        
                        self.training_metadata['models_trained'].append(f"gradient_boosting_{target_name}")
                        logger.info(f"Gradient Boosting model for {target_name} trained: {metrics}")
                    except Exception as e:
                        logger.error(f"Error training Gradient Boosting model for {target_name}: {str(e)}")
                
                # Train Bayesian model
                if self.config['models']['bayesian']:
                    try:
                        bayes_model = BayesianModel(
                            prediction_target=target_name,
                            version=int(datetime.now().timestamp())
                        )
                        bayes_model.train(X, y)
                        metrics = bayes_model.evaluate(X, y)
                        model_path = bayes_model.save_to_disk()
                        
                        target_results['bayesian'] = {
                            'metrics': metrics,
                            'path': model_path
                        }
                        
                        self.training_metadata['models_trained'].append(f"bayesian_{target_name}")
                        logger.info(f"Bayesian model for {target_name} trained: {metrics}")
                    except Exception as e:
                        logger.error(f"Error training Bayesian model for {target_name}: {str(e)}")
                
                # Train Ensemble model if at least two base models were trained
                if self.config['models']['ensemble'] and len(target_results) >= 2:
                    try:
                        # Create ensemble with appropriate base models
                        base_models = []
                        if 'random_forest' in target_results:
                            rf_model = RandomForestModel()
                            rf_model.load_from_disk()  # Load the latest version
                            base_models.append(rf_model)
                        
                        if 'gradient_boosting' in target_results:
                            gb_model = GradientBoostingModel()
                            gb_model.load_from_disk()  # Load the latest version
                            base_models.append(gb_model)
                        
                        if 'bayesian' in target_results:
                            bayes_model = BayesianModel(prediction_target=target_name)
                            bayes_model.load_from_disk()  # Load the latest version
                            base_models.append(bayes_model)
                        
                        ensemble = EnsembleModel(
                            prediction_type=prediction_type,
                            version=int(datetime.now().timestamp()),
                            base_models=base_models
                        )
                        
                        ensemble.train(X, y)
                        metrics = ensemble.evaluate(X, y)
                        model_path = ensemble.save_to_disk()
                        
                        target_results['ensemble'] = {
                            'metrics': metrics,
                            'path': model_path,
                            'model_importance': ensemble.get_feature_importance()
                        }
                        
                        self.training_metadata['models_trained'].append(f"ensemble_{target_name}")
                        logger.info(f"Ensemble model for {target_name} trained: {metrics}")
                    except Exception as e:
                        logger.error(f"Error training Ensemble model for {target_name}: {str(e)}")
                
                # Add results for this target
                results[target_name] = target_results
            
            # Update performance metrics
            self.training_metadata['performance_metrics'] = results
            
            return results
        
        except Exception as e:
            logger.error(f"Error in model training process: {str(e)}")
            return {}
    
    def save_training_results(self, training_id: Optional[str] = None) -> str:
        """
        Save the training results to disk
        
        Args:
            training_id: Optional ID for the training run
            
        Returns:
            Path to the saved results file
        """
        try:
            # Create a unique training ID if not provided
            if not training_id:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                training_id = f"training_{self.season}_{timestamp}"
            
            # Update metadata
            self.training_metadata['end_time'] = datetime.now(timezone.utc).isoformat()
            
            # Save results to file
            results_file = RESULTS_DIR / f"{training_id}.json"
            with results_file.open('w') as f:
                json.dump(self.training_metadata, f, indent=2)
            
            logger.info(f"Training results saved to {results_file}")
            return str(results_file)
        
        except Exception as e:
            logger.error(f"Error saving training results: {str(e)}")
            return ""
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline
        
        Returns:
            Dictionary with pipeline results
        """
        try:
            # Set start time
            start_time = datetime.now(timezone.utc)
            self.training_metadata['start_time'] = start_time.isoformat()
            
            logger.info(f"Starting training pipeline for {self.season} season at {start_time}")
            
            # Step 1: Collect historical data
            data_success = self.collect_historical_data()
            if not data_success:
                logger.error("Data collection failed, aborting pipeline")
                self.save_training_results()
                return {'status': 'failed', 'stage': 'data_collection'}
            
            # Step 2: Engineer features
            games_df, features = self.engineer_features()
            if games_df.empty or not features:
                logger.error("Feature engineering failed, aborting pipeline")
                self.save_training_results()
                return {'status': 'failed', 'stage': 'feature_engineering'}
            
            # Step 3: Train models
            training_results = self.train_models(features)
            if not training_results:
                logger.warning("Model training yielded no results")
            
            # Step 4: Save results
            results_path = self.save_training_results()
            
            # Calculate elapsed time
            end_time = datetime.now(timezone.utc)
            elapsed = (end_time - start_time).total_seconds() / 60.0  # minutes
            
            logger.info(f"Training pipeline completed in {elapsed:.2f} minutes")
            
            return {
                'status': 'success',
                'models_trained': self.training_metadata['models_trained'],
                'results_path': results_path,
                'elapsed_minutes': elapsed
            }
        
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            self.save_training_results()
            return {'status': 'failed', 'error': str(e)}


# Main function for running the pipeline
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run NBA prediction model training pipeline')
    parser.add_argument('--season', type=str, default="2024", help='NBA season to train for (e.g., 2024)')
    parser.add_argument('--skip-data-collection', action='store_true', help='Skip data collection step')
    args = parser.parse_args()
    
    # Create and run the pipeline
    pipeline = ModelTrainingPipeline(season=args.season)
    
    # Customize pipeline if needed
    if args.skip_data_collection:
        pipeline.config['skip_data_collection'] = True
    
    # Run the pipeline
    result = pipeline.run_pipeline()
    
    print(f"Pipeline result: {result}")
