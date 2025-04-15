"""
Auto Train Manager for NBA Prediction System

This module schedules and manages the automatic training and retraining of
prediction models. It ensures models are kept up-to-date with the latest data
and maintains accuracy across NBA seasons.

Key Features:
- Schedules model training using APScheduler (daily at 6:00 AM EST)
- Manages retraining triggered by drift detection
- Coordinates sequential training of base and ensemble models
- Logs training status and outcomes
- Handles season transitions

Usage:
    python -m src.utils.auto_train_manager [--force] [--models MODEL1,MODEL2]
    
Options:
    --force           Force training regardless of last training time
    --models MODELS   Comma-separated list of specific models to train
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# For scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Data processing
import numpy as np
import pandas as pd

# Database
from src.database.connection import get_connection, close_connection
from src.database.models import ModelWeight, ModelPerformance, Game

# Season handling
from src.utils.season_manager import SeasonManager

# Models
from src.models.random_forest import RandomForestModel
from src.models.combined_gradient_boosting import CombinedGradientBoostingModel
from src.models.bayesian_model import BayesianModel
from src.models.anomaly_detection import AnomalyDetectionModel
from src.models.model_mixing import ModelMixingModel
from src.models.ensemble_stacking import EnsembleStackingModel
from src.models.hyperparameter_tuning import HyperparameterTuner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/auto_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
TRAINING_STATUS_FILE = "logs/training_status.txt"
MIN_TRAIN_INTERVAL = 12  # Hours between training sessions
MODEL_DEPENDENCIES = {
    "ModelMixing": ["RandomForest", "CombinedGradientBoosting", "Bayesian"],
    "EnsembleStacking": ["RandomForest", "CombinedGradientBoosting", "Bayesian"]
}
PREDICTION_TARGETS = [
    "moneyline", "spread", "totals", "player_points", 
    "player_rebounds", "player_assists", "player_threes"
]

class AutoTrainManager:
    """
    Manages automatic training and retraining of prediction models
    
    This class schedules and coordinates the training of all NBA prediction models,
    ensuring they stay accurate and up-to-date without manual intervention.
    """
    
    def __init__(self):
        """
        Initialize the auto training manager
        """
        self.scheduler = BackgroundScheduler(timezone=timezone.utc)
        self.season_manager = SeasonManager()
        self.training_queue = set()
        self.currently_training = False
        self.training_status = {}
        self.load_status()
        
    def load_status(self):
        """
        Load the current training status from file
        """
        if os.path.exists(TRAINING_STATUS_FILE):
            try:
                with open(TRAINING_STATUS_FILE, 'r') as f:
                    self.training_status = json.load(f)
                logger.info(f"Loaded training status for {len(self.training_status)} models")
            except Exception as e:
                logger.error(f"Error loading training status: {str(e)}")
                self.training_status = {}
        else:
            logger.info("No existing training status file found")
            self.training_status = {}
    
    def save_status(self):
        """
        Save the current training status to file
        """
        try:
            os.makedirs(os.path.dirname(TRAINING_STATUS_FILE), exist_ok=True)
            with open(TRAINING_STATUS_FILE, 'w') as f:
                json.dump(self.training_status, f, indent=2)
            logger.info(f"Saved training status for {len(self.training_status)} models")
        except Exception as e:
            logger.error(f"Error saving training status: {str(e)}")
    
    def start(self):
        """
        Start the scheduler to manage model training
        """
        # Schedule daily training at 6:00 AM EST (10:00 UTC)
        self.scheduler.add_job(
            self.check_and_train_models,
            CronTrigger(hour=10, minute=0),
            id='daily_training',
            replace_existing=True,
            misfire_grace_time=3600
        )
        
        # Schedule regular check for models that need retraining (every 30 minutes)
        self.scheduler.add_job(
            self.check_for_retraining,
            IntervalTrigger(minutes=30),
            id='retraining_check',
            replace_existing=True
        )
        
        # Schedule regular check for odds updates (every 4 hours)
        self.scheduler.add_job(
            self.update_odds_data,
            IntervalTrigger(hours=4),
            id='odds_update',
            replace_existing=True
        )
        
        # Schedule daily performance tracking (2:00 AM EST)
        self.scheduler.add_job(
            self.track_performance,
            CronTrigger(hour=6, minute=0),  # 2:00 AM EST = 6:00 UTC
            id='performance_tracking',
            replace_existing=True
        )
        
        # Start the scheduler
        self.scheduler.start()
        logger.info("Auto Train Manager scheduler started")
    
    def stop(self):
        """
        Stop the scheduler
        """
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Auto Train Manager scheduler stopped")
    
    def check_and_train_models(self, force: bool = False, specific_models: Optional[List[str]] = None):
        """
        Check which models need training and add them to the queue
        
        Args:
            force: Force training regardless of last training time
            specific_models: Optional list of specific models to train
        """
        if self.currently_training:
            logger.info("Training already in progress, skipping check")
            return
        
        # Check if any models need training based on database records
        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                if specific_models:
                    placeholders = ",".join([f"%s" for _ in specific_models])
                    cursor.execute(f"""
                        SELECT model_name, params, version, trained_at, needs_training
                        FROM model_weights
                        WHERE active = TRUE AND (
                            needs_training = TRUE OR
                            model_name IN ({placeholders})
                        )
                    """, tuple(specific_models))
                else:
                    cursor.execute("""
                        SELECT model_name, params, version, trained_at, needs_training
                        FROM model_weights
                        WHERE active = TRUE
                    """)
                
                model_records = cursor.fetchall()
            
            close_connection(conn)
            
            # Check which models need training
            now = datetime.now(timezone.utc)
            for record in model_records:
                model_name = record[0]
                params = record[1]
                version = record[2]
                trained_at = record[3]
                needs_training = record[4]
                
                # Skip if specific models specified and this isn't one of them
                if specific_models and model_name not in specific_models:
                    continue
                
                # Determine if this model needs training
                prediction_target = params.get('prediction_target') if params else None
                needs_training = needs_training or force
                
                if not needs_training and trained_at:
                    # Check if it's been too long since last training
                    hours_since_trained = (now - trained_at).total_seconds() / 3600
                    if hours_since_trained > MIN_TRAIN_INTERVAL:
                        needs_training = True
                
                # If new season detected, retrain all models
                current_season = self.season_manager.get_current_season_info()
                season_key = f"{current_season['season_year']}_{current_season['phase']}"
                last_trained_season = self.training_status.get(f"{model_name}_{prediction_target}_season")
                
                if last_trained_season != season_key:
                    logger.info(f"Season change detected for {model_name}: {last_trained_season} -> {season_key}")
                    needs_training = True
                
                if needs_training:
                    self.training_queue.add((model_name, prediction_target))
                    logger.info(f"Added {model_name} ({prediction_target}) to training queue")
            
            # Check if dependent models need training
            for model_name, dependencies in MODEL_DEPENDENCIES.items():
                for target in PREDICTION_TARGETS:
                    if (model_name, target) in self.training_queue:
                        # Ensure dependent models are also in the queue
                        for dep_model in dependencies:
                            if (dep_model, target) not in self.training_queue:
                                self.training_queue.add((dep_model, target))
                                logger.info(f"Added {dep_model} ({target}) to training queue as dependency of {model_name}")
            
            # If we have models to train, start training
            if self.training_queue:
                logger.info(f"Starting training for {len(self.training_queue)} models")
                self.train_models()
            else:
                logger.info("No models need training at this time")
                
        except Exception as e:
            logger.error(f"Error checking models for training: {str(e)}")
    
    def check_for_retraining(self):
        """
        Check if any models have been flagged for retraining by the drift detector
        """
        if self.currently_training:
            return
        
        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT model_name, params
                    FROM model_weights
                    WHERE active = TRUE AND needs_training = TRUE
                """)
                
                model_records = cursor.fetchall()
            
            close_connection(conn)
            
            if model_records:
                for record in model_records:
                    model_name = record[0]
                    params = record[1]
                    prediction_target = params.get('prediction_target') if params else None
                    
                    self.training_queue.add((model_name, prediction_target))
                    logger.info(f"Added {model_name} ({prediction_target}) to training queue for retraining")
                
                # Start training
                self.train_models()
                
        except Exception as e:
            logger.error(f"Error checking for models that need retraining: {str(e)}")
    
    def update_odds_data(self):
        """
        Update odds data from APIs
        """
        logger.info("Updating odds data")
        try:
            # Import here to avoid circular imports
            from src.data.data_collector import update_odds_data
            update_odds_data()
            logger.info("Successfully updated odds data")
        except Exception as e:
            logger.error(f"Error updating odds data: {str(e)}")
    
    def track_performance(self):
        """
        Run the performance tracking script
        """
        logger.info("Running performance tracking")
        try:
            # Import here to avoid circular imports
            from src.utils.track_performance import PerformanceTracker
            tracker = PerformanceTracker()
            tracker.run()
            logger.info("Successfully ran performance tracking")
        except Exception as e:
            logger.error(f"Error tracking performance: {str(e)}")

    def train_models(self):
        """
        Train all models in the training queue
        """
        if not self.training_queue:
            logger.info("No models in training queue")
            return
            
        if self.currently_training:
            logger.info("Training already in progress")
            return
            
        self.currently_training = True
        logger.info(f"Starting training for {len(self.training_queue)} models")
        
        try:
            # First, determine which models are base models vs ensemble models
            base_models = set()
            ensemble_models = set()
            
            for model_name, prediction_target in self.training_queue:
                if model_name in ["ModelMixing", "EnsembleStacking"]:
                    ensemble_models.add((model_name, prediction_target))
                else:
                    base_models.add((model_name, prediction_target))
            
            # Train base models first
            for model_name, prediction_target in base_models:
                self._train_model(model_name, prediction_target)
            
            # Then train ensemble models that depend on them
            for model_name, prediction_target in ensemble_models:
                self._train_model(model_name, prediction_target)
            
            # Clear the training queue
            self.training_queue.clear()
            logger.info("Training completed for all models in queue")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
        finally:
            self.currently_training = False
    
    def _train_model(self, model_name: str, prediction_target: str) -> bool:
        """
        Train a specific model
        
        Args:
            model_name: Name of the model to train
            prediction_target: Prediction target for the model
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        logger.info(f"Training {model_name} for {prediction_target}")
        
        try:
            # Get the model parameters from the database
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, params, version
                    FROM model_weights
                    WHERE model_name = %s AND active = TRUE AND params->>'prediction_target' = %s
                """, (model_name, prediction_target))
                
                model_record = cursor.fetchone()
            
            close_connection(conn)
            
            if not model_record:
                logger.error(f"No active model found for {model_name} ({prediction_target})")
                return False
            
            model_id = model_record[0]
            params = model_record[1]
            version = model_record[2]
            
            # Load training data
            X_train, y_train = self._load_training_data(prediction_target)
            
            if X_train.empty or y_train.empty:
                logger.error(f"No training data available for {model_name} ({prediction_target})")
                return False
            
            # Create and train the model
            model = self._create_model(model_name, prediction_target, params)
            
            if not model:
                logger.error(f"Failed to create model {model_name} ({prediction_target})")
                return False
            
            # For ensemble models, we need to set the base models
            if model_name == "ModelMixing" or model_name == "EnsembleStacking":
                base_models = self._load_base_models(prediction_target, MODEL_DEPENDENCIES.get(model_name, []))
                
                if not base_models:
                    logger.error(f"Failed to load base models for {model_name} ({prediction_target})")
                    return False
                
                if model_name == "ModelMixing":
                    model.set_models(base_models)
                elif model_name == "EnsembleStacking":
                    model.set_base_models(base_models)
            
            # Train the model
            logger.info(f"Training {model_name} ({prediction_target}) with {len(X_train)} samples")
            
            # Use hyperparameter tuning for base models
            if model_name in ["RandomForest", "CombinedGradientBoosting", "Bayesian"]:
                self._train_with_hyperparameter_tuning(model, X_train, y_train)
            else:
                model.train(X_train, y_train)
            
            # Evaluate the model
            X_test, y_test = self._load_test_data(prediction_target)
            
            if not X_test.empty and not y_test.empty:
                metrics = model.evaluate(X_test, y_test)
                logger.info(f"Evaluation metrics for {model_name} ({prediction_target}): {metrics}")
            
            # Save the model to the database
            if model.save_to_db():
                logger.info(f"Successfully saved {model_name} ({prediction_target}) to database")
                
                # Update the model status in the database
                conn = get_connection()
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE model_weights
                        SET needs_training = FALSE,
                            updated_at = %s
                        WHERE id = %s
                    """, (datetime.now(timezone.utc), model_id))
                    
                    conn.commit()
                
                close_connection(conn)
                
                # Update the training status
                current_season = self.season_manager.get_current_season_info()
                season_key = f"{current_season['season_year']}_{current_season['phase']}"
                
                status_key = f"{model_name}_{prediction_target}"
                self.training_status[status_key] = {
                    "last_trained": datetime.now().isoformat(),
                    "version": version,
                    "status": "Trained",
                    "metrics": metrics if 'metrics' in locals() else {}
                }
                self.training_status[f"{status_key}_season"] = season_key
                
                self.save_status()
                return True
            else:
                logger.error(f"Failed to save {model_name} ({prediction_target}) to database")
                return False
                
        except Exception as e:
            logger.error(f"Error training {model_name} ({prediction_target}): {str(e)}")
            
            # Update training status to show failure
            status_key = f"{model_name}_{prediction_target}"
            self.training_status[status_key] = {
                "last_attempt": datetime.now().isoformat(),
                "status": "Failed",
                "error": str(e)
            }
            self.save_status()
            return False
    
    def _load_training_data(self, prediction_target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare training data for the specified prediction target
        
        Args:
            prediction_target: Type of prediction (moneyline, spread, etc.)
            
        Returns:
            Tuple of (X_train, y_train)
        """
        try:
            # This would typically call a data processing function
            # For now, we'll implement a placeholder that fetches from the database
            
            # Import data processing module to avoid circular imports
            from src.data.data_processor import get_training_data_for_target
            return get_training_data_for_target(prediction_target)
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def _load_test_data(self, prediction_target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare test data for the specified prediction target
        
        Args:
            prediction_target: Type of prediction (moneyline, spread, etc.)
            
        Returns:
            Tuple of (X_test, y_test)
        """
        try:
            # This would typically call a data processing function
            # For now, we'll implement a placeholder that fetches from the database
            
            # Import data processing module to avoid circular imports
            from src.data.data_processor import get_test_data_for_target
            return get_test_data_for_target(prediction_target)
            
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            return pd.DataFrame(), pd.Series()
    
    def _create_model(self, model_name: str, prediction_target: str, params: dict) -> Optional[Any]:
        """
        Create a model instance based on model name and parameters
        
        Args:
            model_name: Name of the model to create
            prediction_target: Prediction target for the model
            params: Model parameters
            
        Returns:
            Model instance or None if creation failed
        """
        try:
            if model_name == "RandomForest":
                return RandomForestModel(name=model_name, prediction_target=prediction_target)
            elif model_name == "CombinedGradientBoosting":
                return CombinedGradientBoostingModel(name=model_name, prediction_target=prediction_target)
            elif model_name == "Bayesian":
                return BayesianModel(name=model_name, prediction_target=prediction_target)
            elif model_name == "AnomalyDetection":
                return AnomalyDetectionModel(name=model_name, prediction_target=prediction_target)
            elif model_name == "ModelMixing":
                return ModelMixingModel(name=model_name, prediction_target=prediction_target)
            elif model_name == "EnsembleStacking":
                return EnsembleStackingModel(name=model_name, prediction_target=prediction_target)
            else:
                logger.error(f"Unknown model type: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {str(e)}")
            return None
    
    def _load_base_models(self, prediction_target: str, base_model_names: List[str]) -> Dict[str, Any]:
        """
        Load base models for ensemble models
        
        Args:
            prediction_target: Prediction target for the models
            base_model_names: List of base model names
            
        Returns:
            Dictionary mapping model names to model instances
        """
        base_models = {}
        
        try:
            for model_name in base_model_names:
                # Get the latest version of this model from the database
                conn = get_connection()
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, weights, params
                        FROM model_weights
                        WHERE model_name = %s AND active = TRUE AND params->>'prediction_target' = %s
                        ORDER BY version DESC
                        LIMIT 1
                    """, (model_name, prediction_target))
                    
                    model_record = cursor.fetchone()
                
                close_connection(conn)
                
                if not model_record:
                    logger.warning(f"No active model found for {model_name} ({prediction_target})")
                    continue
                
                # Create the model
                model = self._create_model(model_name, prediction_target, model_record[2])
                
                if not model:
                    logger.warning(f"Failed to create model {model_name} ({prediction_target})")
                    continue
                
                # Load the weights (this would typically be done by the model itself)
                # For now, we'll just add the model to the dictionary
                base_models[model_name] = model
            
            return base_models
                
        except Exception as e:
            logger.error(f"Error loading base models: {str(e)}")
            return {}
    
    def _train_with_hyperparameter_tuning(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> bool:
        """
        Train a model with hyperparameter tuning
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training targets
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            model_type = model.__class__.__name__.replace('Model', '')
            prediction_target = model.prediction_target
            
            logger.info(f"Running hyperparameter tuning for {model_type} ({prediction_target})")
            
            # Create a tuner and run tuning
            tuner = HyperparameterTuner(model_type=model_type, prediction_target=prediction_target)
            tuning_results = tuner.tune(X_train, y_train, n_iter=20, cv=5)
            
            # Create and train a model with the best parameters
            tuned_model = tuner.create_tuned_model()
            
            if tuned_model:
                tuned_model.train(X_train, y_train)
                
                # Copy the trained model's parameters and data to the original model
                # This is a bit of a hack, but it allows us to keep the original model's name
                model.model = tuned_model.model
                model.feature_names = tuned_model.feature_names
                model.feature_importances_ = getattr(tuned_model, 'feature_importances_', None)
                model.trained_at = tuned_model.trained_at
                model.is_trained = tuned_model.is_trained
                model.params = tuned_model.params
                
                return True
            else:
                logger.error(f"Failed to create tuned model for {model_type} ({prediction_target})")
                # Fall back to regular training
                model.train(X_train, y_train)
                return True
                
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            # Fall back to regular training
            logger.info(f"Falling back to regular training without tuning")
            model.train(X_train, y_train)
            return True


def main():
    """
    Main function for running the auto train manager
    """
    parser = argparse.ArgumentParser(description="Manage automatic training of prediction models")
    parser.add_argument("--force", action="store_true", help="Force training regardless of last training time")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to train")
    parser.add_argument("--daemon", action="store_true", help="Run as a daemon with scheduled training")
    args = parser.parse_args()
    
    logger.info("Starting Auto Train Manager")
    
    try:
        manager = AutoTrainManager()
        
        if args.daemon:
            # Start the scheduler and keep running
            manager.start()
            
            try:
                # Keep the main thread alive
                while True:
                    time.sleep(60)
            except (KeyboardInterrupt, SystemExit):
                logger.info("Stopping Auto Train Manager")
                manager.stop()
        else:
            # Run once without the scheduler
            specific_models = args.models.split(',') if args.models else None
            manager.check_and_train_models(force=args.force, specific_models=specific_models)
            
    except Exception as e:
        logger.error(f"Error in Auto Train Manager: {str(e)}")
        sys.exit(1)
    
    logger.info("Auto Train Manager completed successfully")


if __name__ == "__main__":
    main()
