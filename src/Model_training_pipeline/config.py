#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training Pipeline Configuration

This module defines configuration settings and constants for the model training pipeline:
- Base paths and directories
- Default training parameters
- Environment setup
- Logging configuration

Used by all pipeline components to ensure consistent settings.
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

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

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Models directory
MODELS_DIR = BASE_DIR / 'data' / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Results directory for storing training outputs
RESULTS_DIR = BASE_DIR / 'data' / 'training_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_current_season() -> str:
    """
    Auto-detect the current NBA season based on the date
    
    Returns:
        str: Current NBA season (e.g., "2025" for 2024-2025 season)
    """
    try:
        from nba_algorithm.data.season_utils import get_current_season as get_season
        return str(get_season())
    except Exception as e:
        logger.warning(f"Could not auto-detect season from utility: {str(e)}")
        
        # Calculate based on current date
        now = datetime.now()
        # NBA season starts in October and ends in June
        # The season year is the ending year (e.g., 2024-2025 season is 2025)
        if now.month >= 10:  # October or later (new season starts)
            return str(now.year + 1)
        else:
            return str(now.year)


def get_default_config(season: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default configuration for the training pipeline
    
    Args:
        season: NBA season to train models for (e.g., "2025" for 2024-2025 season)
               If None, automatically detects the current season
    
    Returns:
        Dictionary with default configuration settings
    """
    if season is None:
        season = get_current_season()
    
    DEFAULT_CONFIG = {
        'data_collection': {
            'api': {
                'base_url': 'https://www.balldontlie.io/api/v1',
                'key': None,  # API key if required
                'timeout': 10,  # Request timeout in seconds
                'max_retries': 3,  # Max retry attempts for failed requests
                'retry_delay': 2.0  # Delay between retries in seconds
            },
            'num_seasons': 4,              # Number of seasons to collect
            'min_games_required': 100,     # Minimum games required for training
            'team_filter': 'active_nba',   # Filter type ('all', 'active_nba', 'custom')
            'pagination_size': 100,        # API pagination size
            'max_pages_per_season': 25,    # Maximum pages to request per season
            'request_delay': 0.5,          # Delay between API requests in seconds
            'retry_count': 3,              # Number of retries for failed requests
            'filter_active_teams': True,   # Whether to filter for active teams only
            'force_current_season': False  # Whether to force inclusion of current season
        },
        'feature_engineering': {
            'window_size': 10,  # Window size for rolling averages
            'use_advanced_features': True,  # Whether to use advanced statistical features
            'home_advantage': 3.0,  # Home court advantage factor
            'normalize_features': True,  # Whether to normalize features
            'normalization_method': 'standard',  # Normalization method ('standard', 'minmax', 'robust')
            'handle_missing_values': True,  # Whether to handle missing values
            'missing_value_strategy': 'mean',  # Strategy for missing values ('mean', 'median', 'zero')
            'drop_missing_threshold': 0.5  # Drop columns with more than this fraction of missing values
        },
        'training': {
            'test_size': 0.2,  # Fraction of data to use for testing
            'validation_size': 0.1,  # Fraction of data to use for validation
            'random_state': 42,  # Random seed for reproducibility
            'handle_imbalance': True,  # Whether to handle class imbalance
            'imbalance_method': 'smote',  # Method for handling imbalance
            'use_cross_validation': True,  # Whether to use cross-validation
            'cv_folds': 5,  # Number of cross-validation folds
            'target_types': [
                {
                    'name': 'Moneyline',
                    'column': 'home_win',
                    'type': 'classification'
                },
                {
                    'name': 'Spread',
                    'column': 'spread',
                    'type': 'regression'
                },
                {
                    'name': 'Total',
                    'column': 'total',
                    'type': 'regression'
                }
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
                    'max_depth': 5,
                    'random_state': 42
                }
            },
            'stacking_ensemble': {
                'enabled': True,
                'trainer_module': 'ensemble_trainer',
                'trainer_class': 'StackingEnsembleTrainer',
                'base_models': ['random_forest', 'gradient_boosting'],
                'meta_model': 'logistic_regression',
                'params': {
                    'cv': 5,
                    'random_state': 42
                }
            },
            'bayesian': {
                'enabled': True,
                'trainer_module': 'bayesian_trainer',
                'trainer_class': 'BayesianTrainer',
                'params': {
                    'alpha': 1.0, 
                    'random_state': 42
                }
            }
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'rmse', 'mae', 'r2'],
            'baseline_strategy': 'majority',  # Strategy for baseline model ('majority', 'stratified')
            'threshold': 0.5,  # Classification threshold
            'primary_metric': 'f1',  # Primary metric for model comparison
            'production_threshold': 0.7  # Threshold for promoting to production
        },
        'results': {
            'save_models': True,  # Whether to save trained models
            'save_predictions': True,  # Whether to save predictions
            'save_feature_importance': True,  # Whether to save feature importance
            'backup_enabled': True,  # Whether to create backups
            'prepare_production_models': True  # Whether to prepare production models
        },
        'logging': {
            'level': 'INFO',  # Logging level
            'to_console': True,  # Whether to log to console
            'to_file': True,  # Whether to log to file
            'file_path': 'logs/training_pipeline.log',  # Path to log file
            'rotation': '1 day',  # Log rotation interval
            'retention': '30 days'  # Log retention period
        },
        'paths': {
            'models_dir': 'models',  # Directory for trained models
            'results_dir': 'results',  # Directory for training results
            'production_dir': 'models/production',  # Directory for production models
            'data_dir': 'data',  # Directory for collected data
            'output_dir': 'output',  # Directory for outputs
            'backup_dir': 'results/backups'  # Directory for backups
        }
    }
    
    return {
        'season': season,
        'data_collection': DEFAULT_CONFIG['data_collection'],
        'feature_engineering': DEFAULT_CONFIG['feature_engineering'],
        'training': DEFAULT_CONFIG['training'],
        'models': DEFAULT_CONFIG['models'],
        'evaluation': DEFAULT_CONFIG['evaluation'],
        'results': DEFAULT_CONFIG['results'],
        'logging': DEFAULT_CONFIG['logging'],
        'paths': DEFAULT_CONFIG['paths']
    }


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize configuration settings
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated and normalized configuration
    """
    # Get default config to compare against
    default_config = get_default_config(config.get('season'))
    
    # Ensure all required sections exist
    for section in default_config.keys():
        if section not in config:
            config[section] = default_config[section]
            logger.warning(f"Missing configuration section '{section}', using defaults")
        elif isinstance(default_config[section], dict):
            # For nested sections, ensure all fields exist
            for key in default_config[section].keys():
                if key not in config[section]:
                    config[section][key] = default_config[section][key]
                    logger.warning(f"Missing configuration key '{section}.{key}', using default")
    
    # Validate specific settings
    if config['data_collection']['days_back'] < 30:
        logger.warning(f"days_back={config['data_collection']['days_back']} is too low for reliable training, minimum recommended is 30")
    
    # Ensure use_real_data_only is True for production
    if not config['data_collection']['use_real_data_only']:
        logger.warning("Setting use_real_data_only to True for production reliability")
        config['data_collection']['use_real_data_only'] = True
    
    # Validate test split ratio
    if not (0.1 <= config['training']['test_split'] <= 0.3):
        logger.warning(f"test_split={config['training']['test_split']} outside recommended range [0.1, 0.3], adjusting")
        config['training']['test_split'] = max(0.1, min(0.3, config['training']['test_split']))
    
    return config