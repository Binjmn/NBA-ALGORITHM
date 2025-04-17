#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Player Props Model Training Script

This script specifically trains models for player prop predictions (points, rebounds, assists).
It ensures proper data collection, feature engineering, and model training specifically
for player performance predictions.

Usage:
    python scripts/train_player_props.py [--force-collection] [--days_back 60]
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import json
import pickle

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, parent_dir)

# Import project modules with proper relative imports
from nba_algorithm.data.player_data import fetch_player_data
from nba_algorithm.features.player_features import extract_player_features
# Import the correct model loading/saving functions
from nba_algorithm.models.loader import load_model_from_file
import pickle
from nba_algorithm.utils.production_readiness import setup_production_logging

# Configure logging
logger = logging.getLogger("player_props_training")
setup_production_logging(app_name="player_props_training")

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train Player Props Models")
    parser.add_argument(
        "--force-collection",
        action="store_true",
        help="Force collection of player data even if it already exists"
    )
    parser.add_argument(
        "--days_back",
        type=int,
        default=60,
        help="Number of days back to collect player data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/player_props",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/player_stats",
        help="Directory to save/load player data"
    )
    return parser.parse_args()

def collect_player_data(args):
    """
    Collect player data for training
    
    Args:
        args: Command line arguments
        
    Returns:
        pandas.DataFrame: Player data for training
    """
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    season_averages_path = data_dir / "season_averages.csv"
    
    # Check if we need to collect data
    if args.force_collection or not season_averages_path.exists():
        logger.info("Collecting player data for training...")
        try:
            # Use nba_algorithm module to fetch player data
            games = fetch_games_for_player_stats()
            player_data = fetch_player_data(games)
            
            # Process the player data
            processed_data = process_player_data(player_data)
            
            # Save to CSV
            processed_data.to_csv(season_averages_path, index=False)
            logger.info(f"Saved player data to {season_averages_path}")
            
            return processed_data
        except Exception as e:
            logger.error(f"Error collecting player data: {str(e)}")
            if not season_averages_path.exists():
                logger.error("No existing player data found. Cannot proceed with training.")
                return None
            logger.warning("Using existing player data...")
    
    # Load existing data
    try:
        logger.info(f"Loading player data from {season_averages_path}")
        return pd.read_csv(season_averages_path)
    except Exception as e:
        logger.error(f"Error loading player data: {str(e)}")
        return None

def fetch_games_for_player_stats():
    """
    Fetch recent games for player stats
    
    Returns:
        List of games for fetching player stats
    """
    try:
        from nba_algorithm.api.balldontlie_client import BallDontLieClient
        client = BallDontLieClient()
        
        # Get games from the past 60 days
        today = datetime.now()
        start_date = (today - timedelta(days=60)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        
        games = client.get_games(start_date=start_date, end_date=end_date)
        if isinstance(games, dict) and 'data' in games:
            games = games['data']
        
        logger.info(f"Fetched {len(games)} games for player stats")
        return games
    except Exception as e:
        logger.error(f"Error fetching games for player stats: {str(e)}")
        return []

def process_player_data(player_data):
    """
    Process raw player data into training format
    
    Args:
        player_data: Raw player data
        
    Returns:
        pandas.DataFrame: Processed player data
    """
    # Implementation depends on the format of player_data
    # This is a placeholder for the actual implementation
    try:
        if isinstance(player_data, list):
            # Convert list to DataFrame
            df = pd.DataFrame(player_data)
        elif isinstance(player_data, pd.DataFrame):
            df = player_data
        else:
            logger.error(f"Unexpected player data type: {type(player_data)}")
            return pd.DataFrame()
        
        # Ensure we have the necessary columns
        required_columns = ['player_id', 'player_name', 'team_id', 'points', 'rebounds', 'assists']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
        
        # Fill missing values
        df = df.fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Error processing player data: {str(e)}")
        return pd.DataFrame()

def engineer_features(player_data):
    """
    Engineer features for player prop predictions
    
    Args:
        player_data: DataFrame with player data
        
    Returns:
        Dict with feature sets for different prop types
    """
    if player_data is None or player_data.empty:
        logger.error("No player data available for feature engineering")
        return {}
    
    try:
        logger.info("Engineering features for player prop predictions")
        
        # Define the target columns for each prop type
        prop_targets = {
            "points": "points",
            "rebounds": "rebounds",
            "assists": "assists"
        }
        
        result = {}
        
        for prop_name, target_col in prop_targets.items():
            if target_col in player_data.columns:
                # Get features (everything except the target columns we're not predicting)
                features = player_data.drop(
                    [c for c in prop_targets.values() if c != target_col],
                    axis=1, 
                    errors='ignore'
                )
                
                # Get target
                target = player_data[target_col]
                
                result[prop_name] = (features, target)
                logger.info(f"Prepared {len(features)} samples for {prop_name} prediction")
            else:
                logger.warning(f"Target column '{target_col}' not found for {prop_name} prediction")
        
        return result
    except Exception as e:
        logger.error(f"Error engineering features: {str(e)}")
        return {}

def train_prop_models(feature_sets, args):
    """
    Train models for each prop type
    
    Args:
        feature_sets: Dict with feature sets for different prop types
        args: Command line arguments
        
    Returns:
        Dict with trained models
    """
    if not feature_sets:
        logger.error("No feature sets available for training")
        return {}
    
    try:
        logger.info("Training player prop models")
        
        # Initialize models to train - updated paths to point to src/models
        from src.models.gradient_boosting_model import GradientBoostingModel
        from src.models.random_forest_model import RandomForestModel
        
        # Use common model classes for all prop types
        model_classes = {
            "GradientBoosting": GradientBoostingModel,
            "RandomForest": RandomForestModel
        }
        
        trained_models = {}
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir) / "player_props"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Train models for each prop type
        for prop_name, (X, y) in feature_sets.items():
            logger.info(f"Training models for {prop_name} prediction")
            prop_models = {}
            
            # Preprocess data - remove non-numeric columns
            X_numeric = X.select_dtypes(include=['number'])
            if len(X_numeric.columns) < len(X.columns):
                logger.info(f"Removed {len(X.columns) - len(X_numeric.columns)} non-numeric columns for training")
            
            for model_name, model_class in model_classes.items():
                try:
                    logger.info(f"Training {model_name} for {prop_name} prediction")
                    
                    # Initialize model
                    model = model_class()
                    
                    # Set prediction target if the model supports it
                    if hasattr(model, 'prediction_target'):
                        model.prediction_target = prop_name
                    
                    # For RandomForest, we need to ensure the scaler is properly fit
                    if model_name == "RandomForest":
                        # Explicitly call preprocess with fit=True to ensure the scaler is fit
                        if hasattr(model, '_preprocess_features'):
                            try:
                                # This will fit the scaler internally
                                model._preprocess_features(X_numeric, fit=True, task='regression')
                            except Exception as e:
                                logger.warning(f"Error preprocessing features: {str(e)}")
                    
                    # Train model with appropriate method based on model type
                    if hasattr(model, 'train_for_player_props'):
                        model.train_for_player_props(X_numeric, y, prop_type=prop_name)
                    elif model_name == "GradientBoosting":
                        # GradientBoostingModel doesn't accept 'task' parameter
                        model.train(X_numeric, y)
                    elif hasattr(model, 'train'):
                        model.train(X_numeric, y)
                    else:
                        model.fit(X_numeric, y)
                    
                    # Save model
                    model_filename = f"{model_name}_{prop_name}_{datetime.now().strftime('%Y%m%d')}.pkl"
                    model_path = output_dir / model_filename
                    
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    
                    logger.info(f"Saved {model_name} for {prop_name} to {model_path}")
                    
                    # Add to trained models
                    prop_models[model_name] = model
                    
                except Exception as e:
                    logger.error(f"Error training {model_name} for {prop_name}: {str(e)}")
            
            trained_models[prop_name] = prop_models
        
        return trained_models
    except Exception as e:
        logger.error(f"Error training prop models: {str(e)}")
        return {}

def validate_prop_models(trained_models, feature_sets):
    """
    Validate trained prop models
    
    Args:
        trained_models: Dict with trained models
        feature_sets: Dict with feature sets for different prop types
        
    Returns:
        Dict with validation results
    """
    if not trained_models or not feature_sets:
        logger.error("No models or feature sets to validate")
        return {}
    
    try:
        logger.info("Validating player prop models")
        
        validation_results = {}
        
        for prop_name, models in trained_models.items():
            if prop_name not in feature_sets:
                logger.warning(f"No feature set available for {prop_name} validation")
                continue
                
            X, y = feature_sets[prop_name]
            # Use only numeric columns
            X_numeric = X.select_dtypes(include=['number'])
            
            prop_results = {}
            
            for model_name, model in models.items():
                try:
                    logger.info(f"Validating {model_name} for {prop_name} prediction")
                    
                    # Make predictions
                    if hasattr(model, 'predict'):
                        predictions = model.predict(X_numeric)
                    else:
                        logger.warning(f"Model {model_name} has no predict method")
                        continue
                    
                    # Calculate metrics
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    
                    metrics = {
                        "mse": float(mean_squared_error(y, predictions)),
                        "mae": float(mean_absolute_error(y, predictions)),
                        "r2": float(r2_score(y, predictions))
                    }
                    
                    logger.info(f"{model_name} for {prop_name}: MSE={metrics['mse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.2f}")
                    
                    prop_results[model_name] = metrics
                    
                except Exception as e:
                    logger.error(f"Error validating {model_name} for {prop_name}: {str(e)}")
            
            validation_results[prop_name] = prop_results
        
        # Save validation results
        import json
        from datetime import datetime
        
        output_dir = Path("models") / "player_props"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results_path = output_dir / f"validation_results_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(results_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Saved validation results to {results_path}")
        
        return validation_results
    except Exception as e:
        logger.error(f"Error validating prop models: {str(e)}")
        return {}

def main():
    """
    Main function
    """
    try:
        args = parse_arguments()
        
        logger.info("Starting player props model training")
        
        # Collect player data
        player_data = collect_player_data(args)
        if player_data is None or player_data.empty:
            logger.error("No player data available. Exiting.")
            return 1
        
        # Engineer features
        feature_sets = engineer_features(player_data)
        if not feature_sets:
            logger.error("No feature sets created. Exiting.")
            return 1
        
        # Train models
        trained_models = train_prop_models(feature_sets, args)
        if not trained_models:
            logger.error("No models trained. Exiting.")
            return 1
        
        # Validate models
        validation_results = validate_prop_models(trained_models, feature_sets)
        
        logger.info("Player props model training completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
