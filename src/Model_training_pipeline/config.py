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
import json
import traceback

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


def get_default_config(config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None, season: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default configuration for the training pipeline
    
    Args:
        config_path: Path to a JSON configuration file
        config: Configuration dictionary (overrides config_path if provided)
        season: NBA season to train models for (e.g., "2025" for 2024-2025 season)
               If None, automatically detects the current season
    
    Returns:
        Dictionary with default configuration settings
    """
    if season is None:
        season = get_current_season()
    
    # Start with base config
    default_config = {
        'paths': {
            'data_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data'),
            'results_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results'),
            'models_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results', 'models'),
            'logs_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        },
        'data_collection': {
            'num_seasons': 4,
            'min_games_required': 100,
            'api': {
                'base_url': 'https://api.balldontlie.io/v1',
                'timeout': 10,
                'key': os.environ.get('BALLDONTLIE_API_KEY', '')
            },
            'retry_count': 3,
            'retry_delay': 2.0,
            'force_current_season': True,
            'enable_logging': True,
            'filter_active_teams': True
        },
        'feature_engineering': {
            'enable_advanced_features': True,
            'normalize_features': True,
            'normalization_method': 'standard',  # 'standard', 'minmax', 'robust'
            'handle_missing_values': 'mean',  # 'mean', 'median', 'mode', 'zero'
            'outlier_removal': False,
            'outlier_method': 'iqr',  # 'iqr', 'zscore'
            'feature_selection': False,
            'feature_selection_method': 'importance',  # 'importance', 'correlation'
            'home_advantage': 3.0  # Average home advantage in points
        },
        'training': {
            'train_player_props': True,  # Enable player props training
            'target_types': [
                {'name': 'moneyline', 'column': 'home_win', 'type': 'classification'},
                {'name': 'spread', 'column': 'spread_diff', 'type': 'regression'},
                {'name': 'totals', 'column': 'total_points', 'type': 'regression'}
            ],
            'models': ['random_forest', 'gradient_boosting', 'ensemble'],
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42,
            'handle_imbalance': True,
            'imbalance_method': 'smote',  # 'smote', 'adasyn', 'random_over', 'random_under'
            'player_feature_columns': [
                # Player stats
                'player_min', 'player_fgm', 'player_fga', 'player_fg_pct', 'player_ftm', 
                'player_fta', 'player_ft_pct', 'player_tpm', 'player_tpa', 'player_tp_pct',
                # Player context
                'player_season_avg_pts', 'player_season_avg_reb', 'player_season_avg_ast',
                'player_season_avg_min', 'player_last5_avg_pts', 'player_last5_avg_reb',
                'player_last5_avg_ast', 'player_last5_avg_min',
                # Team context
                'team_pace', 'opponent_defensive_rating', 'team_offensive_rating',
                'days_rest', 'is_home_game', 'opponent_position_defense'
            ]
        },
        'models': {
            'random_forest': {
                'params': {
                    'n_estimators': 100,
                    'max_depth': 20,
                    'min_samples_split': 10,
                    'random_state': 42
                },
                'optimize_hyperparams': False,
                'tuning_method': 'random',  # 'grid', 'random'
                'cv_folds': 5,
                'n_iter': 10  # For random search
            },
            'gradient_boosting': {
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'subsample': 1.0,
                    'random_state': 42
                },
                'optimize_hyperparams': False,
                'tuning_method': 'random',  # 'grid', 'random'
                'cv_folds': 5,
                'n_iter': 10  # For random search
            },
            'ensemble': {
                'ensemble_method': 'stacking',  # 'stacking', 'voting'
                'weights': None,  # For voting ensemble
                'rf_params': {
                    'n_estimators': 100,
                    'max_depth': 20,
                    'random_state': 42
                },
                'gb_params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': 42
                },
                'final_estimator_params': {},  # For stacking ensemble
                'cv_folds': 5
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
            'prepare_production_models': True,
            'check_backward_compatibility': True,
            'model_format': 'pickle'  # 'pickle', 'joblib', 'onnx'
        },
        'logging': {
            'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'file': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs', 'training.log'),
            'console': True
        }
    }
    
    # Load config from file if provided
    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from file: {str(e)}")
            config = None
    
    # Override default config with provided config
    if config:
        default_config.update(config)
    
    return {
        'season': season,
        'data_collection': default_config['data_collection'],
        'feature_engineering': default_config['feature_engineering'],
        'training': default_config['training'],
        'models': default_config['models'],
        'evaluation': default_config['evaluation'],
        'results': default_config['results'],
        'logging': default_config['logging'],
        'paths': default_config['paths']
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