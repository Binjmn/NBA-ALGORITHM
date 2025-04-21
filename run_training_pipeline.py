#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Algorithm Training Pipeline Runner

This script runs the complete NBA model training pipeline with proper error handling,
logging, and metrics tracking. It ensures all modules are properly configured and
trained, including both team prediction models and player props models.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Configure logging
os.makedirs('logs', exist_ok=True)
log_file = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("nba_training")

# Add src directory to path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Main function
def main():
    """
    Main entry point for running the NBA training pipeline
    """
    try:
        # Import dependencies here to ensure paths are set up
        from src.Model_training_pipeline.config import get_default_config
        from src.Model_training_pipeline.pipeline import TrainingPipeline
        
        logger.info("Starting NBA Model Training Pipeline")
        
        # Set up required directories
        required_dirs = [
            'results',
            'results/models',
            'logs',
            'data'
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
            else:
                logger.info(f"Directory already exists: {directory}")
        
        # Create a default configuration with required values
        config = {
            'season': datetime.now().year,
            'data_collection': {
                'api': {
                    'base_url': 'https://api.balldontlie.io/v1',
                    'timeout': 10,
                    'key': os.environ.get('BALLDONTLIE_API_KEY', ''),
                    'odds_key': os.environ.get('THE_ODDS_API_KEY', '')
                },
                'days_back': 365,
                'include_stats': True,
                'include_odds': True,
                'filter_active_teams': True,
                'use_real_data_only': True,
                'min_games_required': 100,
                'num_seasons': 4,
                'retry_count': 3,
                'retry_delay': 2.0
            },
            'feature_engineering': {
                'enable_advanced_features': True,
                'normalize_features': True,
                'normalization_method': 'standard'
            },
            'training': {
                'train_player_props': True,
                'train_game_outcomes': True,  # Explicitly enable game outcome models
                'test_size': 0.2,
                'random_state': 42,
                'handle_imbalance': True,
                'imbalance_method': 'smote',
                'models': [
                    'gradient_boosting',
                    'random_forest',
                    'bayesian',
                    'ensemble_model',
                    'ensemble_stacking',
                    'combined_gradient_boosting',
                    'hyperparameter_tuning'
                ],
                'target_types': [
                    {'name': 'moneyline', 'column': 'home_win', 'type': 'classification'},
                    {'name': 'spread', 'column': 'spread_diff', 'type': 'regression'},
                    {'name': 'totals', 'column': 'total_points', 'type': 'regression'}
                ],
                'player_targets': [
                    {'name': 'points', 'column': 'player_pts', 'type': 'regression'},
                    {'name': 'rebounds', 'column': 'player_reb', 'type': 'regression'},
                    {'name': 'assists', 'column': 'player_ast', 'type': 'regression'},
                    {'name': 'threes', 'column': 'player_fg3m', 'type': 'regression'},
                    {'name': 'blocks', 'column': 'player_blk', 'type': 'regression'},
                    {'name': 'steals', 'column': 'player_stl', 'type': 'regression'}
                ]
            },
            'models': {
                'random_forest': {
                    'enabled': True,
                    'trainer_module': 'random_forest_trainer',
                    'trainer_class': 'RandomForestTrainer',
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 20,
                        'min_samples_split': 10,
                        'random_state': 42
                    }
                },
                'gradient_boosting': {
                    'enabled': True,
                    'trainer_module': 'gradient_boosting_trainer',
                    'trainer_class': 'GradientBoostingTrainer',
                    'params': {
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 3,
                        'subsample': 1.0,
                        'random_state': 42
                    }
                },
                'ensemble': {
                    'enabled': True,
                    'trainer_module': 'ensemble_trainer',
                    'trainer_class': 'EnsembleTrainer',
                    'method': 'stacking',
                    'base_models': ['random_forest', 'gradient_boosting']
                }
            },
            'player_props': {
                'enabled': True,
                'models': {
                    'points_model': {
                        'trainer_module': 'gradient_boosting_trainer',
                        'trainer_class': 'GradientBoostingTrainer',
                        'params': {
                            'n_estimators': 150,
                            'learning_rate': 0.05,
                            'max_depth': 4,
                            'subsample': 0.9,
                            'random_state': 42
                        }
                    },
                    'rebounds_model': {
                        'trainer_module': 'gradient_boosting_trainer',
                        'trainer_class': 'GradientBoostingTrainer',
                        'params': {
                            'n_estimators': 150,
                            'learning_rate': 0.05,
                            'max_depth': 4,
                            'subsample': 0.9,
                            'random_state': 42
                        }
                    },
                    'assists_model': {
                        'trainer_module': 'gradient_boosting_trainer',
                        'trainer_class': 'GradientBoostingTrainer',
                        'params': {
                            'n_estimators': 150,
                            'learning_rate': 0.05,
                            'max_depth': 4,
                            'subsample': 0.9,
                            'random_state': 42
                        }
                    },
                    'threes_model': {
                        'trainer_module': 'gradient_boosting_trainer',
                        'trainer_class': 'GradientBoostingTrainer',
                        'params': {
                            'n_estimators': 150,
                            'learning_rate': 0.05,
                            'max_depth': 4,
                            'subsample': 0.9,
                            'random_state': 42
                        }
                    }
                }
            },
            'evaluation': {
                'metrics': {
                    'classification': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                    'regression': ['mse', 'rmse', 'mae', 'r2', 'mape']
                },
                'primary_metric': 'f1',  # For classification
                'primary_metric_regression': 'rmse',  # For regression
                'threshold': 0.5,  # For binary classification
                'production_threshold': 0.75  # Minimum primary metric value for production models
            },
            'results': {
                'save_models': True,
                'save_evaluation': True,
                'save_predictions': True,
                'save_feature_importance': True,
                'model_naming_format': 'production_{model_type}_{prediction_type}.pkl'
            },
            'paths': {
                'data_dir': os.path.join(current_dir, 'data'),
                'results_dir': os.path.join(current_dir, 'results'),
                'models_dir': os.path.join(current_dir, 'results', 'models'),
                'production_dir': os.path.join(current_dir, 'results', 'production')
            },
            'logging': {
                'level': 'INFO',
                'save_to_file': True,
                'console_output': True,
                'file_path': 'logs/training_pipeline.log'
            }
        }
        
        # Ensure model directories exist and are writable
        model_dirs = [
            config['paths']['models_dir'],
            config['paths']['production_dir']
        ]
        
        for directory in model_dirs:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    logger.info(f"Created model directory: {directory}")
                # Test directory is writable by creating a temporary file
                test_file = os.path.join(directory, '_test_write.txt')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                logger.info(f"Verified model directory is writable: {directory}")
            except Exception as e:
                logger.error(f"Error creating or validating model directory {directory}: {str(e)}")
                logger.error(traceback.format_exc())
                return 1
        
        # Initialize and run pipeline
        pipeline = TrainingPipeline(config=config)
        pipeline_results = pipeline.run()
        
        # Log results
        if pipeline_results.get('status') == 'success':
            logger.info("Training pipeline completed successfully")
            models_trained = pipeline_results.get('metrics', {}).get('pipeline', {}).get('models_trained', 0)
            game_models = pipeline_results.get('metrics', {}).get('pipeline', {}).get('game_outcome_models_trained', 0)
            player_models = pipeline_results.get('metrics', {}).get('pipeline', {}).get('player_props_models_trained', 0)
            
            logger.info(f"Trained {models_trained} models total")
            
            # Check for specific model types
            if game_models == 0 and config['training'].get('train_game_outcomes', True):
                logger.warning("No game outcome models were trained despite being enabled in config")
            
            # Only warn about player props if they're enabled but none were trained
            if player_models == 0 and config['training'].get('train_player_props', True):
                logger.warning("No player prop models were trained despite being enabled in config")
            
            # Save results to file
            with open('results/training_results.json', 'w') as f:
                json.dump(pipeline_results, f, indent=2)
                
            logger.info(f"Results saved to results/training_results.json")
        else:
            logger.error(f"Training pipeline failed: {pipeline_results.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Unhandled exception in training pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
