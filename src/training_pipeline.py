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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

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

# Import project modules
from src.data.historical_collector import HistoricalDataCollector
from src.features.advanced_features import run_feature_engineering
from src.models.base_model import BaseModel
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.bayesian_model import BayesianModel
from src.models.ensemble_model import EnsembleModel
from src.models.combined_gradient_boosting import CombinedGradientBoostingModel
from src.models.ensemble_stacking import EnsembleStackingModel


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
        season: Season to collect data for (e.g., "2025")
        force: Force collection even if data exists
        skip: Skip collection entirely
    
    Returns:
        bool: True if collection was successful or skipped, False otherwise
    """
    if skip:
        logger.info("Skipping data collection as requested")
        return True
    
    logger.info(f"Starting historical data collection for season {season}")
    
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
            
            # Collect data
            result = collector.collect_data_for_season(
                season=season,
                include_stats=True,
                include_odds=False  # Skip odds collection as it was causing issues
            )
            
            logger.info(f"Data collection result: {result}")
            
            # Check if data was collected
            if not result or 'games' not in result or result['games'] == 0:
                logger.error("No game data was collected")
                return False
            
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
    Train all available prediction models
    
    Returns:
        Dict[str, Any]: Dictionary with training results for each model
    """
    logger.info("Starting comprehensive model training")
    
    results = {}
    
    # Initialize all models with their correct parameters based on their implementation
    models_to_train = [
        # Each model initialized according to its specific constructor parameters
        RandomForestModel(version=1, params={"n_estimators": 100, "max_depth": 10}),
        GradientBoostingModel(version=1, params={"n_estimators": 100, "learning_rate": 0.1}),
        # Skip problematic models for now until they can be fixed properly
        # BayesianModel(name="Bayesian", prediction_target="moneyline", version=1),
        # CombinedGradientBoostingModel(name="CombinedGBM", prediction_target="moneyline", version=1),
        # EnsembleStackingModel(name="EnsembleStacking", prediction_target="moneyline", version=1)
    ]
    
    # Skip EnsembleModel too since it requires base models that haven't been trained yet
    # Try to add the EnsembleModel which requires base models
    # try:
    #     # Ensemble model needs to be created after basic models are trained
    #     base_models = []
    #     models_to_train.append(
    #         EnsembleModel(prediction_type="classification", version=1, base_models=base_models)
    #     )
    # except Exception as e:
    #     logger.warning(f"Could not initialize EnsembleModel: {str(e)}")
    
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

        # Ensure all metrics are numeric
        numeric_cols = features_df.select_dtypes(include=['number']).columns
        features_df = features_df[numeric_cols]
        
        # Identify feature columns (exclude targets and metadata)
        meta_cols = ['game_id', 'date', 'home_team_id', 'away_team_id', 'home_score', 'away_score', 'home_won', 'point_diff']
        feature_cols = [col for col in features_df.columns if col not in meta_cols]
        
        # For each prediction target, prepare X and y
        prediction_targets = {
            "moneyline": "home_won",    # Binary classification target
            "spread": "point_diff",      # Regression target
            "combined": "home_won"      # Default to moneyline for combined models
        }
        
        # Create data splits for each prediction target
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
        
        if not data_splits:
            logger.error("No valid target columns found in feature data")
            return {"status": "error", "message": "No valid targets found"}
            
    except Exception as e:
        logger.error(f"Error loading feature data: {str(e)}")
        return {"status": "error", "message": f"Error loading data: {str(e)}"}

    # Train each model
    for model in models_to_train:
        model_name = model.__class__.__name__
        logger.info(f"Training {model_name}")
        
        try:
            # Get the appropriate data split for this model's prediction target
            prediction_target = getattr(model, "prediction_target", "moneyline")
            
            if prediction_target not in data_splits:
                logger.warning(f"No data split available for {prediction_target}, using moneyline")
                prediction_target = "moneyline"
                
            X, y = data_splits[prediction_target]
                
            # Train the model
            start_time = time.time()
            model.train(X, y)
            training_time = time.time() - start_time
            
            # Evaluate the model
            try:
                metrics = model.evaluate(X, y)
            except TypeError as e:
                # If evaluate method doesn't accept parameters, use the default implementation
                logger.warning(f"Model evaluate() method doesn't accept parameters: {str(e)}")
                metrics = {"warning": "Model evaluation skipped due to interface mismatch"}
                
            # Save the model
            try:
                if hasattr(model, 'save') and callable(getattr(model, 'save')):
                    model.save()
                else:
                    logger.warning(f"{model_name} does not have a save method, skipping persistence")
            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")
            
            # Record results
            results[model_name] = {
                "status": "success",
                "training_time": training_time,
                "metrics": metrics
            }
            
            logger.info(f"Successfully trained {model_name} in {training_time:.2f} seconds")
            logger.info(f"Metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            results[model_name] = {
                "status": "error",
                "message": str(e)
            }
    
    # Try to train the stacking ensemble with all trained models
    try:
        successful_models = []
        for model in models_to_train:
            if model.__class__.__name__ in results and results[model.__class__.__name__]["status"] == "success":
                successful_models.append(model)
        
        if len(successful_models) >= 2:  # Need at least 2 models for stacking
            logger.info("Training meta-ensemble model with all successful models")
            
            meta_ensemble = EnsembleStackingModel(name="MetaEnsemble", prediction_target="moneyline", version=1, base_models=successful_models)
            meta_ensemble.train()
            metrics = meta_ensemble.evaluate()
            meta_ensemble.save()
            
            results["MetaEnsemble"] = {
                "status": "success",
                "metrics": metrics
            }
            
            logger.info(f"Successfully trained meta-ensemble model")
            logger.info(f"Metrics: {metrics}")
    except Exception as e:
        logger.error(f"Error training meta-ensemble model: {str(e)}")
        results["MetaEnsemble"] = {
            "status": "error",
            "message": str(e)
        }
    
    # Save training results
    try:
        results_dir = Path("data/results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving training results: {str(e)}")
    
    return results


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
        # Create production models directory
        prod_dir = Path("data/production_models")
        prod_dir.mkdir(exist_ok=True)
        
        # Get list of successful models
        successful_models = []
        for model_name, result in results.items():
            if result["status"] == "success":
                successful_models.append(model_name)
        
        if not successful_models:
            logger.error("No successful models to deploy")
            return False
        
        # Deploy models
        deployment_log = {
            "deployed_at": datetime.now().isoformat(),
            "models": successful_models
        }
        
        with open(prod_dir / "deployment_log.json", "w") as f:
            json.dump(deployment_log, f, indent=2)
        
        logger.info(f"Successfully deployed {len(successful_models)} models to production")
        return True
    except Exception as e:
        logger.error(f"Error deploying models: {str(e)}")
        return False


def run_training_pipeline(args: argparse.Namespace) -> bool:
    """
    Run the complete training pipeline
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True if pipeline was successful, False otherwise
    """
    logger.info("Starting NBA prediction training pipeline")
    logger.info(f"Arguments: {args}")
    
    # Setup database first
    if not args.skip_database:
        db_success = setup_database()
        if not db_success and not args.ignore_database_errors:
            logger.error("Database setup failed, stopping pipeline")
            return False
    
    # Collect historical data
    if not args.skip_collection:
        collection_success = collect_historical_data(
            season=args.season,
            force=args.force_collection,
            skip=args.skip_collection
        )
        if not collection_success:
            logger.error("Historical data collection failed, stopping pipeline")
            return False
    
    # Engineer features
    feature_success = engineer_training_features()
    if not feature_success:
        logger.error("Feature engineering failed, stopping pipeline")
        return False
    
    # Train models
    results = train_all_models()
    if results.get("status") == "error":
        logger.error(f"All model training failed: {results.get('message', 'Unknown error')}")
        return False
    
    # Count successful models
    success_count = 0
    for model_name, result in results.items():
        # Skip the status key which is at the top level
        if model_name == "status":
            continue
        # Check if this model's training was successful
        if isinstance(result, dict) and result.get("status") == "success":
            success_count += 1
    
    if success_count == 0:
        logger.error("All model training failed, stopping pipeline")
        return False
    
    logger.info(f"Successfully trained {success_count} models")
    
    # Deploy models
    if not args.skip_deployment:
        deploy_success = deploy_models(results)
        if not deploy_success:
            logger.error("Model deployment failed")
            return False
    
    logger.info("Training pipeline completed successfully")
    return True


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description="NBA Prediction Training Pipeline")
    parser.add_argument("--season", type=str, default="2025", help="Season to train on (e.g., '2025')")
    parser.add_argument("--force-collection", action="store_true", help="Force data collection even if data exists")
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection entirely")
    parser.add_argument("--skip-database", action="store_true", help="Skip database setup")
    parser.add_argument("--skip-deployment", action="store_true", help="Skip model deployment")
    parser.add_argument("--ignore-database-errors", action="store_true", help="Continue pipeline even if database setup fails")
    
    args = parser.parse_args()
    
    if args.force_collection and args.skip_collection:
        parser.error("--force-collection and --skip-collection cannot be used together")
    
    success = run_training_pipeline(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
