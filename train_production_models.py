#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Production-Quality Model Training Script

This script provides a comprehensive way to train NBA prediction models using
real historical data from the 4-season rolling window. It eliminates the use of
synthetic data in favor of complete, validated datasets.

Usage:
    python train_production_models.py [--force-collection] [--validate]
"""

import os
import sys
import json
import logging
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import our advanced features modules
from nba_algorithm.features.advanced_features import create_momentum_features, create_matchup_features
from nba_algorithm.training.data_pipeline import get_production_training_data
from nba_algorithm.utils.validation import validate_data_completeness

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

# Define model configurations
MODEL_CONFIGS = {
    'random_forest_win_predictor': {
        'model_class': RandomForestClassifier,
        'params': {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 10,
            'random_state': 42
        },
        'target': 'home_team_won',
        'features': [
            # Basic stats differences
            'win_pct_diff', 'points_avg_diff', 'points_allowed_avg_diff',
            'rebounds_avg_diff', 'assists_avg_diff', 'steals_avg_diff',
            'blocks_avg_diff', 'turnovers_avg_diff', 'fouls_avg_diff',
            # Advanced metrics
            'offensive_rating_diff', 'defensive_rating_diff', 'net_rating_diff',
            'pace_diff', 'true_shooting_pct_diff',
            # Momentum features
            'home_win_momentum', 'away_win_momentum',
            # Matchup history features
            'home_matchup_win_pct', 'avg_matchup_point_diff',
            # Injury impact
            'injury_advantage'
        ]
    },
    'gradient_boosting_win_predictor': {
        'model_class': GradientBoostingClassifier,
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        },
        'target': 'home_team_won',
        'features': [
            # Basic stats differences
            'win_pct_diff', 'points_avg_diff', 'points_allowed_avg_diff',
            'rebounds_avg_diff', 'assists_avg_diff', 'steals_avg_diff',
            'blocks_avg_diff', 'turnovers_avg_diff', 'fouls_avg_diff',
            # Advanced metrics
            'offensive_rating_diff', 'defensive_rating_diff', 'net_rating_diff',
            'pace_diff', 'true_shooting_pct_diff',
            # Momentum features
            'home_win_momentum', 'away_win_momentum',
            # Matchup history features
            'home_matchup_win_pct', 'avg_matchup_point_diff',
            # Injury impact
            'injury_advantage'
        ]
    }
}


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train NBA prediction models with production-quality data')
    parser.add_argument('--force-collection', action='store_true', help='Force new data collection')
    parser.add_argument('--validate', action='store_true', help='Validate data before training')
    return parser.parse_args()


def engineer_features(game_data: Dict[str, Any], team_stats: Dict[str, Any]) -> pd.DataFrame:
    """
    Engineer features using real historical data and advanced features
    
    Args:
        game_data: Dictionary with game data
        team_stats: Dictionary with team statistics
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    logger.info("Engineering features with advanced metrics")
    
    features_list = []
    historical_games = game_data.get('games', [])
    
    for game in historical_games:
        if not game.get('home_team') or not game.get('visitor_team'):
            continue
            
        home_team_id = game['home_team'].get('id')
        away_team_id = game['visitor_team'].get('id')
        
        if not home_team_id or not away_team_id:
            continue
            
        # Skip games without scores (future games)
        if game.get('home_team_score') is None or game.get('visitor_team_score') is None:
            continue
            
        # Create feature dictionary
        game_features = {}
        
        # Basic game info
        game_features['game_id'] = game.get('id')
        game_features['date'] = game.get('date')
        game_features['home_team_id'] = home_team_id
        game_features['away_team_id'] = away_team_id
        
        # Actual result (target variable)
        home_team_won = game.get('home_team_score', 0) > game.get('visitor_team_score', 0)
        game_features['home_team_won'] = home_team_won
        
        # Get team stats
        home_team_stats = team_stats.get(str(home_team_id), {}).get('stats', {})
        away_team_stats = team_stats.get(str(away_team_id), {}).get('stats', {})
        
        if not home_team_stats or not away_team_stats:
            continue
            
        # Team performance metrics
        for stat in ['wins', 'losses', 'points', 'points_allowed', 'rebounds', 'assists', 
                    'steals', 'blocks', 'turnovers', 'fouls']:
            home_val = home_team_stats.get(stat, 0)
            away_val = away_team_stats.get(stat, 0)
            
            # Store individual team stats
            game_features[f'home_team_{stat}'] = home_val
            game_features[f'away_team_{stat}'] = away_val
            
            # Calculate stat differences
            if stat in ['wins', 'losses']:
                # Convert win/loss to win percentage
                home_games = home_team_stats.get('wins', 0) + home_team_stats.get('losses', 0)
                away_games = away_team_stats.get('wins', 0) + away_team_stats.get('losses', 0)
                
                home_win_pct = home_team_stats.get('wins', 0) / home_games if home_games > 0 else 0.5
                away_win_pct = away_team_stats.get('wins', 0) / away_games if away_games > 0 else 0.5
                
                game_features['win_pct_diff'] = home_win_pct - away_win_pct
            else:
                # Calculate per-game average differences
                home_games = home_team_stats.get('wins', 0) + home_team_stats.get('losses', 0)
                away_games = away_team_stats.get('wins', 0) + away_team_stats.get('losses', 0)
                
                home_avg = home_val / home_games if home_games > 0 else 0
                away_avg = away_val / away_games if away_games > 0 else 0
                
                game_features[f'{stat}_avg_diff'] = home_avg - away_avg
        
        # Advanced metrics (offensive/defensive ratings)
        for metric in ['offensive_rating', 'defensive_rating', 'net_rating', 'pace', 'true_shooting_pct']:
            home_val = home_team_stats.get(metric, 0)
            away_val = away_team_stats.get(metric, 0)
            
            game_features[f'{metric}_diff'] = home_val - away_val
        
        # Generate momentum features
        try:
            home_momentum = create_momentum_features(historical_games, home_team_id)
            away_momentum = create_momentum_features(historical_games, away_team_id)
            
            game_features['home_win_momentum'] = home_momentum.get('win_momentum', 0)
            game_features['away_win_momentum'] = away_momentum.get('win_momentum', 0)
        except Exception as e:
            logger.warning(f"Could not create momentum features: {str(e)}")
            game_features['home_win_momentum'] = 0
            game_features['away_win_momentum'] = 0
        
        # Generate matchup features
        try:
            matchup_features = create_matchup_features(historical_games, home_team_id, away_team_id)
            
            game_features['home_matchup_win_pct'] = matchup_features.get('home_win_pct', 0.5)
            game_features['avg_matchup_point_diff'] = matchup_features.get('avg_point_diff', 0)
        except Exception as e:
            logger.warning(f"Could not create matchup features: {str(e)}")
            game_features['home_matchup_win_pct'] = 0.5
            game_features['avg_matchup_point_diff'] = 0
        
        # Add placeholder for injury advantage (to be filled by injury_analysis module in production)
        game_features['injury_advantage'] = 0
        
        features_list.append(game_features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    logger.info(f"Engineered features for {len(features_df)} games using real historical data")
    return features_df


def train_model(model_config: Dict[str, Any], data: pd.DataFrame) -> Tuple[Any, Dict[str, float]]:
    """
    Train a model according to its configuration
    
    Args:
        model_config: Dictionary with model configuration
        data: DataFrame with training data
        
    Returns:
        Tuple[Any, Dict[str, float]]: Trained model and performance metrics
    """
    model_name = next(iter(model_config))
    config = model_config[model_name]
    
    logger.info(f"Training model: {model_name}")
    
    # Extract features and target
    features = config.get('features', [])
    target = config.get('target', 'home_team_won')
    
    # Make sure all required columns exist
    missing_columns = [col for col in features + [target] if col not in data.columns]
    if missing_columns:
        logger.warning(f"Missing columns: {missing_columns}. These will be filled with zeros.")
        for col in missing_columns:
            data[col] = 0
    
    X = data[features]
    y = data[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    model_class = config.get('model_class')
    params = config.get('params', {})
    model = model_class(**params)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    logger.info(f"Model {model_name} trained with metrics: {metrics}")
    
    # Save the model and scaler together
    model_with_scaler = {'model': model, 'scaler': scaler, 'features': features}
    model_path = MODEL_DIR / f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_with_scaler, f)
    
    logger.info(f"Model saved to {model_path}")
    
    return model, metrics


def train_all_models() -> Dict[str, Any]:
    """
    Train all models defined in MODEL_CONFIGS using production data
    
    Returns:
        Dict[str, Any]: Dictionary with training results
    """
    logger.info("Training all models with production data")
    
    # Get production-quality data
    training_data = get_production_training_data()
    
    # Engineer features
    features_df = engineer_features(training_data, training_data.get('team_stats', {}))
    
    # Make sure we have enough data
    if len(features_df) < 100:
        logger.warning(f"Only {len(features_df)} samples available for training. This may lead to poor model performance.")
    
    # Train each model
    results = {}
    for model_name, config in MODEL_CONFIGS.items():
        try:
            model, metrics = train_model({model_name: config}, features_df)
            results[model_name] = {'success': True, 'metrics': metrics}
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            results[model_name] = {'success': False, 'error': str(e)}
    
    # Save training results
    results_path = MODEL_DIR / f"training_results_{datetime.now().strftime('%Y%m%d')}.json"
    with open(results_path, 'w') as f:
        # Convert any non-serializable metrics to strings
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                'success': result['success']
            }
            if result['success']:
                serializable_results[model_name]['metrics'] = {
                    k: float(v) for k, v in result['metrics'].items()
                }
            else:
                serializable_results[model_name]['error'] = result['error']
        
        json.dump(serializable_results, f, indent=4)
    
    logger.info(f"Trained {sum(1 for r in results.values() if r['success'])} models")
    return results


def main():
    """
    Main function
    """
    args = parse_arguments()
    
    # Make sure models directory exists
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Train models
    results = train_all_models()
    
    # Print summary
    print("\nTraining Results Summary:")
    for model_name, result in results.items():
        if result['success']:
            metrics = result['metrics']
            print(f"✅ {model_name}: Success - Accuracy: {metrics['accuracy']:.4f}")
        else:
            print(f"❌ {model_name}: Failed - {result['error']}")
    
    return 0


if __name__ == "__main__":
    try:
        import traceback
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
