#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive NBA Prediction Training Pipeline

This script orchestrates the complete training pipeline for the NBA prediction system:
1. Collects historical data if needed
2. Engineers robust features from the raw data
3. Trains ALL available prediction models
4. Evaluates model performance
5. Deploys trained models to production

Usage:
    python -m src.training_pipeline [--season YEAR] [--force-collection] [--skip-collection]
"""

import os
import sys
import time
import json
import logging
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Import models
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.bayesian_model import BayesianModel
from src.models.combined_gradient_boosting import CombinedGradientBoostingModel
from src.models.ensemble_model import EnsembleModel
from src.models.ensemble_stacking import EnsembleStackingModel

# Import data collectors
from src.data.historical_collector import HistoricalDataCollector

# Import robustness utilities
from src.utils.robustness import (
    validate_training_data,
    train_with_checkpointing,
    analyze_feature_importance,
    prune_low_importance_features,
    train_models_parallel,
    should_tune_hyperparameters,
    detect_and_handle_outliers,
    check_feature_drift
)

# Import project modules
from src.features.advanced_features import FeatureEngineer
from src.features.advanced_features_plus import EnhancedFeatureEngineer
from src.models.base_model import BaseModel
# Import SeasonManager for automatic season detection
from src.utils.season_manager import SeasonManager
from src.utils.model_registry import ModelRegistry, promote_best_models_to_production

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / f"training_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Suppress sklearn future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ... (rest of the code remains the same)

def setup_database():
    """
    Initialize the database connection and create necessary tables
    
    Returns:
        bool: True if database setup was successful, False otherwise
    """
    try:
        from src.database.connection import init_db
        success = init_db()
        if success:
            logger.info("Database setup completed successfully")
        else:
            logger.error("Database setup failed")
        return success
    except ImportError:
        logger.warning("Database module not found, continuing without database connection")
        return False
    except Exception as e:
        logger.error(f"Error setting up database: {str(e)}")
        return False


def collect_historical_data(season: str, force: bool = False, skip: bool = False) -> bool:
    """
    Collect historical NBA data
    
    Args:
        season: Current season to collect data for (e.g., "2025")
        force: Force collection even if data exists
        skip: Skip collection entirely
    
    Returns:
        bool: True if collection was successful or skipped, False otherwise
    """
    if skip:
        logger.info("Skipping data collection as requested")
        return True
    
    logger.info(f"Starting historical data collection for season {season} and previous seasons")
    
    try:
        # Check if we already have data for this season
        data_path = Path("data/historical")
        has_data = False
        
        if not force and data_path.exists():
            # Check if we have enough game data
            games_dir = data_path / "games"
            if games_dir.exists() and len(list(games_dir.glob("*.json"))) > 0:
                logger.info(f"Found existing game data, skipping collection")
                has_data = True
        
        if force or not has_data:
            # Initialize collector
            collector = HistoricalDataCollector()
            
            # Define seasons to collect (current season and 3 previous seasons)
            current_season = int(season)
            seasons_to_collect = [
                str(current_season),      # Current season
                str(current_season - 1),  # Previous season
                str(current_season - 2),  # Two seasons ago
                str(current_season - 3)   # Three seasons ago
            ]
            
            logger.info(f"Collecting data for multiple seasons: {seasons_to_collect}")
            
            # Collect data for multiple seasons
            result = collector.collect_data_for_multiple_seasons(
                seasons=seasons_to_collect,
                include_stats=True,
                include_odds=False  # Skip odds collection as it was causing issues
            )
            
            logger.info(f"Multi-season data collection result: {result}")
            
            # Check if data was collected
            if not result or 'total_games' not in result or result['total_games'] == 0:
                logger.error("No game data was collected")
                return False
            
            logger.info(f"Successfully collected {result['total_games']} games across {result['seasons_collected']} seasons")
            return True
        
        return True
    except Exception as e:
        logger.error(f"Error collecting historical data: {str(e)}")
        return False


def engineer_training_features() -> bool:
    """
    Run the feature engineering pipeline
    
    Returns:
        bool: True if feature engineering was successful, False otherwise
    """
    logger.info("Starting feature engineering process")
    
    try:
        # Run feature engineering
        features_df = run_feature_engineering()
        
        if features_df.empty:
            logger.error("Feature engineering produced no data")
            return False
        
        logger.info(f"Feature engineering completed successfully with {len(features_df)} records")
        return True
    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        return False


def train_all_models() -> Dict[str, Any]:
    """
    Train all available prediction models including enhanced player prop prediction models
    
    Returns:
        Dict[str, Any]: Dictionary with training results for each model
    """
    logger.info("Starting comprehensive model training")
    
    results = {}
    
    # Initialize models with enhanced versions
    models = {
        "RandomForestModel": RandomForestModel(version=1),
        "GradientBoostingModel": GradientBoostingModel(version=1),
        "BayesianModel": BayesianModel(prediction_target="moneyline", version=1),
        "CombinedGradientBoostingModel": CombinedGradientBoostingModel(prediction_target="moneyline", version=1)
    }
    
    # Player prop specific models
    player_prop_models = {
        "PointsPredictor": GradientBoostingModel(version=1, prediction_target="points"),
        "AssistsPredictor": RandomForestModel(version=1),  # Will be configured for assists
        "ReboundsPredictor": GradientBoostingModel(version=1, prediction_target="rebounds")
    }
    
    # Initialize and train ensemble models after base models
    # We'll train them separately after the base models are trained
    ensemble_models = {
        "EnsembleModel": EnsembleModel(prediction_target="moneyline", version=1),
        "EnsembleStackingModel": EnsembleStackingModel(prediction_target="moneyline", version=1)
    }
    
    feature_path = Path("data/features/engineered_features.csv")
    if not feature_path.exists():
        logger.error(f"Feature file not found: {feature_path}")
        return {"status": "error", "message": "Feature file not found"}
    
    # Load the feature data
    try:
        logger.info(f"Loading feature data from {feature_path}")
        import pandas as pd
        features_df = pd.read_csv(feature_path)
        
        # Ensure we have data
        if features_df.empty:
            logger.error("Feature file is empty")
            return {"status": "error", "message": "Feature file is empty"}
            
        logger.info(f"Loaded {len(features_df)} samples with {len(features_df.columns)} features")
        
        # Calculate point differential if not present
        if 'point_diff' not in features_df.columns and 'home_score' in features_df.columns and 'away_score' in features_df.columns:
            features_df['point_diff'] = features_df['home_score'] - features_df['away_score']
            logger.info("Created point_diff column from score difference")

        # Identify target columns we need to preserve
        target_cols = ['home_won', 'point_diff', 'game_id', 'date', 'home_team_id', 'away_team_id', 'home_score', 'away_score']
        preserved_targets = {col: features_df[col] for col in target_cols if col in features_df.columns}
        
        # Ensure all metrics are numeric for feature columns
        numeric_cols = features_df.select_dtypes(include=['number']).columns
        features_df = features_df[numeric_cols]
        
        # Add back the target columns that might have been removed
        for col, data in preserved_targets.items():
            if col not in features_df.columns:
                features_df[col] = data
                logger.info(f"Preserved target column: {col}")
        
        # Identify feature columns (exclude targets and metadata)
        meta_cols = ['game_id', 'date', 'home_team_id', 'away_team_id', 'home_score', 'away_score', 'home_won', 'point_diff']
        feature_cols = [col for col in features_df.columns if col not in meta_cols]
        
        # Create data splits for each prediction target
        prediction_targets = {
            "moneyline": "home_won",    # Binary classification target
            "spread": "point_diff",      # Regression target
            "combined": "home_won"      # Default to moneyline for combined models
        }
        
        # Check for feature drift and handle outliers
        drift_detected, _ = check_feature_drift(features_df)
        if drift_detected:
            logger.warning("Feature drift detected, proceeding with caution in model training")
        
        # Handle outliers in the feature data
        features_df = detect_and_handle_outliers(features_df, columns=feature_cols)
        
        data_splits = {}
        for target_name, target_col in prediction_targets.items():
            if target_col in features_df.columns:
                # Get features and target
                X = features_df[feature_cols]
                y = features_df[target_col]
                
                # Handle missing values
                X = X.fillna(0)
                
                data_splits[target_name] = (X, y)
                logger.info(f"Prepared data for {target_name} prediction with {len(X)} samples")
            else:
                logger.warning(f"Target column '{target_col}' not found in features")
        
        # Check if we have player stats data for training prop models
        player_stats_path = Path("data/player_stats/season_averages.csv")
        player_prop_training_ready = False
        
        if player_stats_path.exists():
            try:
                player_stats_df = pd.read_csv(player_stats_path)
                if not player_stats_df.empty:
                    logger.info(f"Found player stats data with {len(player_stats_df)} entries")
                    player_prop_training_ready = True
                    
                    # Prepare feature datasets for each prop type
                    # These targets are the actual stat values we want to predict
                    prop_targets = {
                        "points": "pts", 
                        "assists": "ast", 
                        "rebounds": "reb"
                    }
                    
                    for prop_name, target_col in prop_targets.items():
                        if target_col in player_stats_df.columns:
                            # Extract relevant features for this prop
                            prop_features = player_stats_df.drop([c for c in prop_targets.values() if c != target_col], axis=1, errors='ignore')
                            
                            # Get target values
                            prop_target = player_stats_df[target_col]
                            
                            # Handle missing values
                            prop_features = prop_features.fillna(0)
                            
                            data_splits[prop_name] = (prop_features, prop_target)
                            logger.info(f"Prepared data for {prop_name} prediction with {len(prop_features)} samples")
            except Exception as e:
                logger.warning(f"Error loading player stats for prop models: {str(e)}")
                player_prop_training_ready = False
        else:
            logger.warning(f"Player stats file not found: {player_stats_path}")
        
        if not data_splits:
            logger.error("No valid target columns found in feature data")
            return {"status": "error", "message": "No valid targets found"}
            
    except Exception as e:
        logger.error(f"Error loading feature data: {str(e)}")
        return {"status": "error", "message": f"Error loading data: {str(e)}"}

    # Determine if we should perform hyperparameter tuning
    perform_tuning = should_tune_hyperparameters()
    if perform_tuning:
        logger.info("Will perform hyperparameter tuning during model training")
        # Update model parameters to enable tuning
        for model in list(models.values()) + list(player_prop_models.values()):
            if hasattr(model, 'params') and isinstance(model.params, dict):
                model.params['tune_hyperparameters'] = True
    else:
        logger.info("Skipping hyperparameter tuning (use cached parameters)")
        for model in list(models.values()) + list(player_prop_models.values()):
            if hasattr(model, 'params') and isinstance(model.params, dict):
                model.params['tune_hyperparameters'] = False

    # Get a count of how many models we're expecting to train
    total_models = len(models) + len(ensemble_models)
    if player_prop_training_ready:
        total_models += len(player_prop_models)
    logger.info(f"Attempting to train {total_models} models ({len(models)} base models, "
                f"{len(player_prop_models)} player prop models, and {len(ensemble_models)} ensemble models)")

    # Train base models in parallel
    logger.info("Training base game prediction models in parallel")
    X_moneyline, y_moneyline = data_splits["moneyline"]
    base_results = train_models_parallel(models, X_moneyline, y_moneyline)
    
    # Process base model results
    trained_models = {}
    for model_name, res in base_results.items():
        model = res["model"]
        success = res["success"]
        duration = res.get("duration", 0)
        
        if success:
            trained_models[model_name] = model
            results[model_name] = {
                "status": "success",
                "training_time": duration
            }
            
            # Evaluate the model if it has an evaluate method
            if hasattr(model, "evaluate"):
                try:
                    metrics = model.evaluate(X_moneyline, y_moneyline)
                    results[model_name]["metrics"] = metrics
                    logger.info(f"Metrics for {model_name}: {metrics}")
                except Exception as e:
                    logger.warning(f"Error evaluating {model_name}: {str(e)}")
            
            # Save model if it has a save method
            if hasattr(model, "save") and callable(model.save):
                try:
                    model.save()
                    logger.info(f"Saved {model_name} to disk")
                except Exception as e:
                    logger.warning(f"Error saving {model_name}: {str(e)}")
            else:
                logger.warning(f"{model_name} does not have a save method, skipping persistence")
                
            logger.info(f"Successfully trained {model_name} in {duration:.2f} seconds")
        else:
            results[model_name] = {
                "status": "error",
                "error": res.get("error", "Unknown error during training")
            }
            logger.error(f"Error training {model_name}: {res.get('error', 'Unknown error')}")
    
    # Train player prop models if we have the data
    if player_prop_training_ready:
        logger.info("Training player prop prediction models")
        prop_trained_models = {}
        
        # Configure and train each prop model
        for model_name, model in player_prop_models.items():
            prop_type = ""
            
            if "Points" in model_name:
                prop_type = "points"
            elif "Assists" in model_name:
                prop_type = "assists"
                # Configure RandomForest for assists prediction
                if hasattr(model, 'train_for_player_props'):
                    X_prop, y_prop = data_splits.get(prop_type, (None, None))
                    if X_prop is not None and y_prop is not None:
                        try:
                            logger.info(f"Training {model_name} for {prop_type} prediction")
                            start_time = time.time()
                            model.train_for_player_props(X_prop, y_prop, prop_type=prop_type)
                            duration = time.time() - start_time
                            
                            prop_trained_models[model_name] = model
                            results[model_name] = {
                                "status": "success",
                                "training_time": duration,
                                "prop_type": prop_type
                            }
                            
                            # Save the model
                            if hasattr(model, "save") and callable(model.save):
                                model_path = model.save()
                                results[model_name]["model_path"] = model_path
                                logger.info(f"Saved {model_name} to {model_path}")
                                
                            logger.info(f"Successfully trained {model_name} for {prop_type} in {duration:.2f} seconds")
                        except Exception as e:
                            logger.error(f"Error training {model_name} for {prop_type}: {str(e)}")
                            results[model_name] = {
                                "status": "error",
                                "error": str(e),
                                "prop_type": prop_type
                            }
                else:
                    logger.warning(f"{model_name} does not support player prop training method")
            elif "Rebounds" in model_name:
                prop_type = "rebounds"
            
            # If we haven't trained with the specialized method, use standard training
            if model_name not in prop_trained_models and prop_type:
                X_prop, y_prop = data_splits.get(prop_type, (None, None))
                if X_prop is not None and y_prop is not None:
                    try:
                        logger.info(f"Training {model_name} for {prop_type} prediction")
                        start_time = time.time()
                        
                        # For GradientBoostingModel, set prediction_target
                        if hasattr(model, 'prediction_target'):
                            model.prediction_target = prop_type
                        
                        # Use regression training for props
                        if hasattr(model, 'train'):
                            model.train(X_prop, y_prop, task='regression')
                        else:
                            model.fit(X_prop, y_prop)
                            
                        duration = time.time() - start_time
                        
                        prop_trained_models[model_name] = model
                        results[model_name] = {
                            "status": "success",
                            "training_time": duration,
                            "prop_type": prop_type
                        }
                        
                        # Save the model
                        if hasattr(model, "save") and callable(model.save):
                            model_path = model.save()
                            results[model_name]["model_path"] = model_path
                            logger.info(f"Saved {model_name} to {model_path}")
                            
                        logger.info(f"Successfully trained {model_name} for {prop_type} in {duration:.2f} seconds")
                    except Exception as e:
                        logger.error(f"Error training {model_name} for {prop_type}: {str(e)}")
                        results[model_name] = {
                            "status": "error",
                            "error": str(e),
                            "prop_type": prop_type
                        }
        
        # Add prop models to the trained models dictionary
        trained_models.update(prop_trained_models)
    else:
        logger.warning("Skipping player prop model training due to missing player stats data")
    
    # Analyze feature importance from trained models
    if trained_models:
        importance_df = analyze_feature_importance(trained_models, feature_cols)
        
        if not importance_df.empty and 'avg_importance_normalized' in importance_df.columns:
            # Save feature importance analysis
            importance_file = Path("data/analysis/feature_importance.csv")
            importance_file.parent.mkdir(exist_ok=True)
            importance_df.to_csv(importance_file)
            logger.info(f"Saved feature importance analysis to {importance_file}")
            
            # Log top and bottom features
            top_features = importance_df.head(10).index.tolist()
            bottom_features = importance_df.tail(10).index.tolist()
            logger.info(f"Top 10 important features: {', '.join(top_features)}")
            logger.info(f"Least important features: {', '.join(bottom_features)}")
    
    # Initialize and train ensemble models with the base models
    for model_name, model in ensemble_models.items():
        logger.info(f"Training ensemble model: {model_name}")
        
        try:
            # For ensemble models that need base models
            if hasattr(model, "set_base_models"):
                model.set_base_models(trained_models)
            
            # Get the appropriate data split for this model's prediction target
            prediction_target = getattr(model, "prediction_target", "moneyline")
            
            if prediction_target not in data_splits:
                logger.warning(f"No data split available for {prediction_target}, using moneyline")
                prediction_target = "moneyline"
            
            X_target, y_target = data_splits[prediction_target]
            
            # Start timing
            start_time = time.time()
            
            # Train the ensemble model with the base models
            if hasattr(model, "train"):
                model.train(X_target, y_target)
            elif hasattr(model, "fit"):
                model.fit(X_target, y_target)
            else:
                logger.error(f"{model_name} does not have train or fit method")
                continue
                
            duration = time.time() - start_time
            
            # Evaluate the model if it has an evaluate method
            metrics = {}
            if hasattr(model, "evaluate"):
                try:
                    metrics = model.evaluate(X_target, y_target)
                except Exception as e:
                    logger.warning(f"Error evaluating {model_name}: {str(e)}")
            
            # Save the model if it has a save method
            if hasattr(model, "save") and callable(model.save):
                try:
                    model.save()
                    logger.info(f"Saved {model_name} to disk")
                except Exception as e:
                    logger.warning(f"Error saving {model_name}: {str(e)}")
            
            results[model_name] = {
                "status": "success",
                "training_time": duration,
                "metrics": metrics
            }
            
            logger.info(f"Successfully trained {model_name} in {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            results[model_name] = {
                "status": "error",
                "error": str(e)
            }
    
    # Calculate overall statistics
    success_count = sum(1 for res in results.values() if res.get("status") == "success")
    total_time = sum(res.get("training_time", 0) for res in results.values())
    
    logger.info(f"Training pipeline completed: {success_count}/{total_models} models trained successfully")
    logger.info(f"Total training time: {total_time:.2f} seconds")
    
    results["summary"] = {
        "total_models": total_models,
        "success_count": success_count,
        "total_time": total_time
    }
    
    return results


def load_feature_data(use_enhanced_features=True, historical_days=30):
    """
    Load feature data from CSV file or generate new features if file doesn't exist
    
    Args:
        use_enhanced_features: Whether to use the enhanced feature engineering module
        historical_days: Number of days of historical data to use
        
    Returns:
        DataFrame with features data
    """
    feature_path = Path("data/features/engineered_features.csv")
    
    # Check if feature file exists and is recent enough
    if feature_path.exists():
        file_age_days = (datetime.now() - datetime.fromtimestamp(feature_path.stat().st_mtime)).days
        
        # Only use cached features if less than 1 day old
        if file_age_days < 1:
            try:
                logger.info(f"Loading existing feature data from {feature_path}")
                features_df = pd.read_csv(feature_path)
                
                # Validate the loaded feature data
                validation_results = validate_training_data(features_df)
                if validation_results["status"] == "error":
                    logger.warning("Feature data failed validation, regenerating features")
                else:
                    # Check for feature drift against historical stats
                    drift_detected, drift_report = check_feature_drift(features_df)
                    if drift_detected:
                        logger.warning("Feature drift detected, proceeding with caution")
                    
                    # Handle outliers in the feature data
                    features_df = detect_and_handle_outliers(features_df)
                    
                    return features_df
            except Exception as e:
                logger.error(f"Error loading feature data: {str(e)}")
    
    # Generate new features
    logger.info("Generating new feature data")
    
    try:
        # Collect historical game data
        historical_collector = HistoricalDataCollector()
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=historical_days)).strftime("%Y-%m-%d")
        
        logger.info(f"Collecting game data from {start_date} to {end_date}")
        game_df = historical_collector.get_games_by_date_range(start_date, end_date)
        
        if game_df.empty:
            logger.error("No game data found for the specified date range")
            return pd.DataFrame()
        
        logger.info(f"Collected {len(game_df)} games")
        
        # Apply feature engineering
        if use_enhanced_features:
            logger.info("Using enhanced feature engineering")
            feature_engineer = EnhancedFeatureEngineer(lookback_days=historical_days)
        else:
            logger.info("Using standard feature engineering")
            feature_engineer = FeatureEngineer(lookback_days=historical_days)
            
        features_df = feature_engineer.engineer_features(game_df)
        
        # Validate the newly generated feature data
        validation_results = validate_training_data(features_df)
        if validation_results["status"] == "error":
            logger.error("Newly generated feature data failed validation")
            if not features_df.empty:
                logger.warning("Proceeding with caution despite validation errors")
            else:
                return pd.DataFrame()
        
        # Handle outliers in the feature data
        features_df = detect_and_handle_outliers(features_df)
        
        # Update feature drift baseline with the new data
        _, _ = check_feature_drift(features_df)
        
        # Save features to CSV
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        features_df.to_csv(feature_path, index=False)
        logger.info(f"Saved {len(features_df)} feature records to {feature_path}")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error generating feature data: {str(e)}")
        return pd.DataFrame()


def deploy_models(results: Dict[str, Any]) -> bool:
    """
    Deploy trained models to production
    
    Args:
        results: Dictionary with training results
        
    Returns:
        bool: True if deployment was successful, False otherwise
    """
    logger.info("Deploying trained models to production")
    
    try:
        # Create a model registry instance
        registry = ModelRegistry()
        
        # Scan for existing models and register them
        registry.scan_models()
        
        # Register and update metrics for newly trained models
        promotion_candidates = []
        
        for model_name, result in results.items():
            # Skip the summary result
            if model_name == "summary":
                continue
                
            if result.get("status") == "success":
                # Get the model path from the results
                model_path = result.get("model_path")
                
                # Skip if no model path found
                if not model_path:
                    logger.warning(f"No model path found for {model_name}, skipping registration")
                    continue
                    
                # Try to extract model type and task from the model name
                model_type = model_name
                task = "default"
                
                # Handle player prop models specially
                if "prop_type" in result:
                    task = result["prop_type"]  # e.g., "points", "assists", "rebounds"
                    
                # Register the model
                model_id = registry.register_model(model_path, metrics=result.get("metrics", {}))
                
                if model_id:
                    logger.info(f"Registered model {model_id} in registry")
                    promotion_candidates.append(model_id)
                else:
                    logger.warning(f"Failed to register model {model_name} in registry")
        
        # Clean up old models to save disk space
        removed_count = registry.cleanup_old_models()
        logger.info(f"Cleaned up {removed_count} old models to save disk space")
        
        # Promote the best models to production based on metrics
        promotion_results = promote_best_models_to_production()
        
        # Log the results
        if promotion_results["promoted"]:
            logger.info(f"Promoted models to production: {', '.join(promotion_results['promoted'])}")
        
        if promotion_results["no_change"]:
            logger.info(f"Models already in production: {', '.join(promotion_results['no_change'])}")
            
        if promotion_results["failed"]:
            logger.warning(f"Failed to promote models: {', '.join(promotion_results['failed'])}")
        
        return len(promotion_results["failed"]) == 0
    except Exception as e:
        logger.error(f"Error deploying models: {str(e)}")
        return False


def run_training_pipeline(args: argparse.Namespace) -> bool:
    """
    Run the complete training pipeline
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Success/failure status
    """
    logger.info("Starting NBA prediction training pipeline")
    logger.info(f"Arguments: {args}")
    
    # Auto-detect current NBA season if not explicitly specified
    if args.auto_detect_season:
        # Initialize SeasonManager to auto-detect the current season
        season_manager = SeasonManager()
        current_season = str(season_manager.get_current_season_year())
        if current_season != args.season:
            logger.info(f"Auto-detected current NBA season: {current_season} (was set to {args.season})")
            args.season = current_season
    else:
        logger.info(f"Using specified NBA season: {args.season}")
    
    # Initialize the database
    if not args.skip_database:
        db_success = setup_database()
        if not db_success and not args.ignore_database_errors:
            logger.error("Database setup failed, stopping pipeline")
            return False
    
    # If forcing collection, delete existing features to trigger regeneration
    if args.force_collection:
        feature_path = Path("data/features/engineered_features.csv")
        if feature_path.exists():
            logger.info(f"Forcing collection: Removing existing feature file {feature_path}")
            feature_path.unlink()
    
    # Collect historical data for the detected/specified season
    if not args.skip_collection:
        collection_success = collect_historical_data(
            season=args.season,
            force=args.force_collection
        )
        if not collection_success:
            logger.error("Historical data collection failed, stopping pipeline")
            return False
    
    # Load features data with enhanced engineering
    feature_data = load_feature_data(use_enhanced_features=args.use_enhanced_features)
    
    if feature_data.empty:
        logger.error("Failed to load or generate feature data")
        return False
    
    # Train all models
    results = train_all_models()
    
    if results.get("status") == "error":
        logger.error(f"All model training failed: {results.get('message', 'Unknown error')}")
        return False
    
    # Deploy models
    if not args.skip_deployment:
        deploy_success = deploy_models(results)
        if deploy_success:
            logger.info("Successfully deployed models to production")
        else:
            logger.warning("Model deployment failed or partially succeeded")
    
    logger.info("Training pipeline completed successfully")
    return True


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description="NBA Prediction Training Pipeline")
    parser.add_argument("--season", type=str, default="2025", help="Season to train on (e.g., '2025')")
    parser.add_argument("--auto-detect-season", action="store_true", help="Automatically detect current NBA season (overrides --season)")
    parser.add_argument("--force-collection", action="store_true", help="Force data collection even if data exists")
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection entirely")
    parser.add_argument("--skip-database", action="store_true", help="Skip database setup")
    parser.add_argument("--skip-deployment", action="store_true", help="Skip model deployment")
    parser.add_argument("--ignore-database-errors", action="store_true", help="Continue pipeline even if database setup fails")
    parser.add_argument("--use-enhanced-features", action="store_true", help="Use enhanced feature engineering")
    parser.add_argument("--prediction-type", type=str, default="moneyline", 
                       choices=["moneyline", "spread", "totals"], help="Type of prediction to train for")
    parser.add_argument("--models", type=str, nargs="+", help="Specific models to train")
    
    args = parser.parse_args()
    
    # Set auto-detect as default behavior unless specifically set to false
    if '--no-auto-detect-season' not in sys.argv:
        args.auto_detect_season = True
    
    if args.force_collection and args.skip_collection:
        parser.error("--force-collection and --skip-collection cannot be used together")
    
    success = run_training_pipeline(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
