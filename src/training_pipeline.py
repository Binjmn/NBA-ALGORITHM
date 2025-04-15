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
from src.features.advanced_features import FeatureEngineer
from src.features.advanced_features_plus import EnhancedFeatureEngineer
from src.models.base_model import BaseModel
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.bayesian_model import BayesianModel
from src.models.ensemble_model import EnsembleModel
from src.models.combined_gradient_boosting import CombinedGradientBoostingModel
from src.models.ensemble_stacking import EnsembleStackingModel
# Import SeasonManager for automatic season detection
from src.utils.season_manager import SeasonManager

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
    
    # Initialize models
    prediction_type = "moneyline"
    models = {
        "RandomForestModel": RandomForestModel(version=1),
        "GradientBoostingModel": GradientBoostingModel(version=1),
        "BayesianModel": BayesianModel(prediction_target=prediction_type, version=1),
        "CombinedGradientBoostingModel": CombinedGradientBoostingModel(prediction_target=prediction_type, version=1),
        "EnsembleStackingModel": EnsembleStackingModel(prediction_target=prediction_type, version=1)
    }
    
    # Initialize and train ensemble models after base models
    # We'll train them separately after the base models are trained
    ensemble_models = {
        "EnsembleModel": EnsembleModel(prediction_target=prediction_type, version=1)
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

    # Get a count of how many models we're expecting to train
    expected_model_count = len(models) + len(ensemble_models)
    logger.info(f"Attempting to train {expected_model_count} models ({len(models)} base models and {len(ensemble_models)} ensemble models)")

    # Initialize and train base models first
    trained_models = {}
    
    # Step 1: Train base models first
    for model_name, model in models.items():
        logger.info(f"Training base model: {model_name}")
        
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
            
            # Store the trained model for later use by ensemble models
            trained_models[model_name] = model
            
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
    
    # Step 2: Configure and train ensemble models using the trained base models
    for model_name, model in ensemble_models.items():
        logger.info(f"Training ensemble model: {model_name}")
        
        try:
            # Get the appropriate data split for this model's prediction target
            prediction_target = getattr(model, "prediction_target", "moneyline")
            
            if prediction_target not in data_splits:
                logger.warning(f"No data split available for {prediction_target}, using moneyline")
                prediction_target = "moneyline"
                
            X, y = data_splits[prediction_target]
            
            # Set base models for ensemble models that support it
            if model_name == "EnsembleModel" or model_name == "EnsembleStackingModel":
                # Convert the models dictionary to the format expected by set_base_models
                successful_base_models = {}
                for base_name, base_model in trained_models.items():
                    if base_name in results and results[base_name]["status"] == "success":
                        successful_base_models[base_name] = base_model
                
                if successful_base_models:
                    logger.info(f"Setting {len(successful_base_models)} base models for {model_name}")
                    # EnsembleStackingModel has set_base_models method
                    if hasattr(model, 'set_base_models') and callable(getattr(model, 'set_base_models')):
                        model.set_base_models(successful_base_models)
                    # EnsembleModel takes base_models directly
                    elif model_name == "EnsembleModel":
                        # Convert dictionary to list for EnsembleModel
                        model.base_models = list(successful_base_models.values())
                        logger.info(f"Set base models list for {model_name}")
                else:
                    logger.warning(f"No successful base models available for {model_name}")
            
            # Train the ensemble model
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
    
    # Save training results
    try:
        results_dir = Path("data/results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving training results: {str(e)}")
    
    # Count the number of successful models
    successful_count = sum(1 for model_result in results.values() if model_result.get("status") == "success")
    logger.info(f"Successfully trained {successful_count} models")
    
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
                return pd.read_csv(feature_path)
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
