"""
Model Drift Detection for NBA Prediction System

This script monitors model performance and triggers retraining when accuracy
drops below predefined thresholds. It compares the current 7-day performance
metrics against baseline values to detect any significant degradation in model
accuracy.

Key Features:
- Compares current 7-day performance against baseline thresholds
- Triggers retraining when accuracy drops below 5% of baseline
- Logs drift detection results and retraining triggers
- Supports multiple prediction targets (moneyline, spread, totals, player props)

Usage:
    python -m src.utils.check_model_drift [--check_only]
    
Options:
    --check_only  Only check for drift and report, don't trigger retraining
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import pandas as pd

# Add the project root to the path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.database.models import ModelPerformance, ModelWeight
from src.database.connection import get_connection, close_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/model_drift.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DRIFT_THRESHOLD = 0.05  # 5% drop in accuracy triggers retraining
MIN_SAMPLES = 10  # Minimum number of predictions needed for reliable drift detection
PERFORMANCE_WINDOW_DAYS = 7  # Look at last 7 days of performance


class ModelDriftDetector:
    """
    Detects drift in model performance and triggers retraining when necessary.
    
    This class monitors the performance of prediction models over time and
    compares current performance to established baselines. When performance
    drops significantly, it can trigger model retraining.
    """
    
    def __init__(self):
        """
        Initialize the model drift detector.
        """
        self.model_types = [
            "RandomForest",
            "CombinedGradientBoosting",
            "Bayesian",
            "ModelMixing",
            "EnsembleStacking"
        ]
        
        self.prediction_targets = [
            "moneyline",
            "spread",
            "totals",
            "player_points",
            "player_rebounds",
            "player_assists",
            "player_threes"
        ]
        
        # Metrics to monitor for each target type
        self.metrics = {
            "moneyline": ["accuracy", "f1_score", "log_loss"],
            "spread": ["accuracy", "rmse", "mae"],
            "totals": ["accuracy", "rmse", "mae"],
            "player_points": ["rmse", "mae", "r2_score"],
            "player_rebounds": ["rmse", "mae", "r2_score"],
            "player_assists": ["rmse", "mae", "r2_score"],
            "player_threes": ["rmse", "mae", "r2_score"]
        }
        
        # Primary metric for each target type
        self.primary_metric = {
            "moneyline": "accuracy",
            "spread": "accuracy",
            "totals": "accuracy",
            "player_points": "rmse",
            "player_rebounds": "rmse",
            "player_assists": "rmse",
            "player_threes": "rmse"
        }
        
        # Store baseline performance for each model/target combination
        self.baselines = self._load_baselines()
        
        # Tracking current performance and drift detection results
        self.current_performance = {}
        self.drift_detected = {}
        
    def _load_baselines(self) -> Dict[str, Dict[str, float]]:
        """
        Load baseline performance metrics from the database or file.
        
        Returns:
            Dictionary mapping model/target combinations to baseline metrics
        """
        baselines = {}
        
        try:
            # First try to load from the database
            conn = get_connection()
            with conn.cursor() as cursor:
                # Get the baseline performance for each model type and prediction target
                cursor.execute("""
                    SELECT 
                        model_name,
                        prediction_target,
                        metrics,
                        created_at
                    FROM model_performance
                    WHERE is_baseline = True
                    ORDER BY created_at DESC
                """)
                
                records = cursor.fetchall()
            
            close_connection(conn)
            
            # Process the records
            for record in records:
                model_name = record[0]
                prediction_target = record[1]
                metrics = record[2]  # This is a JSONB column
                
                key = f"{model_name}_{prediction_target}"
                if key not in baselines:
                    baselines[key] = metrics
            
            if baselines:
                logger.info(f"Loaded {len(baselines)} baseline metrics from database")
                return baselines
            
        except Exception as e:
            logger.warning(f"Error loading baselines from database: {str(e)}")
        
        # If database load failed or no baselines found, try loading from file
        baseline_file = Path("data/model_baselines.json")
        if baseline_file.exists():
            try:
                with open(baseline_file, "r") as f:
                    baselines = json.load(f)
                logger.info(f"Loaded {len(baselines)} baseline metrics from file")
                return baselines
            except Exception as e:
                logger.warning(f"Error loading baselines from file: {str(e)}")
        
        # If no baselines found, return empty dict
        logger.warning("No baseline metrics found. Will use first run performance as baseline.")
        return {}
    
    def _save_baselines(self):
        """
        Save baseline performance metrics to file as a backup.
        """
        try:
            baseline_file = Path("data/model_baselines.json")
            baseline_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(baseline_file, "w") as f:
                json.dump(self.baselines, f, indent=2)
            
            logger.info(f"Saved {len(self.baselines)} baseline metrics to file")
        except Exception as e:
            logger.error(f"Error saving baselines to file: {str(e)}")
    
    def get_current_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get the current performance metrics for all models and prediction targets.
        
        This fetches the last 7 days of performance data from the database and
        calculates average metrics for each model/target combination.
        
        Returns:
            Dictionary mapping model/target combinations to current metrics
        """
        performance = {}
        
        try:
            # Calculate the date range (last 7 days)
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=PERFORMANCE_WINDOW_DAYS)
            
            # Query the database for recent performance metrics
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        model_name,
                        prediction_target,
                        metrics,
                        sample_count,
                        created_at
                    FROM model_performance
                    WHERE created_at BETWEEN %s AND %s
                      AND is_baseline = False
                    ORDER BY created_at DESC
                """, (start_date, end_date))
                
                records = cursor.fetchall()
            
            close_connection(conn)
            
            # Group by model_name and prediction_target
            grouped_records = {}
            for record in records:
                model_name = record[0]
                prediction_target = record[1]
                metrics = record[2]  # JSONB
                sample_count = record[3]
                
                key = f"{model_name}_{prediction_target}"
                if key not in grouped_records:
                    grouped_records[key] = []
                
                grouped_records[key].append((metrics, sample_count))
            
            # Calculate weighted average metrics for each model/target
            for key, records in grouped_records.items():
                # Skip if we don't have enough samples
                total_samples = sum(sample_count for _, sample_count in records)
                if total_samples < MIN_SAMPLES:
                    logger.warning(f"Not enough samples for {key} ({total_samples}/{MIN_SAMPLES})")
                    continue
                
                # Calculate weighted average for each metric
                avg_metrics = {}
                for metric_name in self.metrics.get(key.split("_")[-1], []):
                    weighted_sum = sum(metrics.get(metric_name, 0) * sample_count 
                                     for metrics, sample_count in records)
                    avg_metrics[metric_name] = weighted_sum / total_samples
                
                performance[key] = avg_metrics
            
            logger.info(f"Calculated current performance for {len(performance)} model/target combinations")
            
        except Exception as e:
            logger.error(f"Error getting current performance: {str(e)}")
        
        self.current_performance = performance
        return performance
    
    def detect_drift(self) -> Dict[str, bool]:
        """
        Detect drift in model performance by comparing current metrics to baselines.
        
        Returns:
            Dictionary mapping model/target combinations to drift detection results (True if drift detected)
        """
        if not self.current_performance:
            self.get_current_performance()
        
        drift_results = {}
        
        for key, current_metrics in self.current_performance.items():
            # If we don't have a baseline for this model/target, create one
            if key not in self.baselines:
                logger.info(f"Creating new baseline for {key}")
                self.baselines[key] = current_metrics
                drift_results[key] = False
                continue
            
            # Get the baseline metrics
            baseline_metrics = self.baselines[key]
            
            # Get the prediction target
            prediction_target = key.split("_")[-1]
            
            # Get the primary metric for this target
            primary_metric = self.primary_metric.get(prediction_target)
            if not primary_metric or primary_metric not in current_metrics:
                logger.warning(f"No primary metric found for {key}")
                drift_results[key] = False
                continue
            
            # Compare current performance to baseline
            baseline_value = baseline_metrics.get(primary_metric, 0)
            current_value = current_metrics.get(primary_metric, 0)
            
            # For metrics where lower is better (like RMSE), invert the comparison
            if primary_metric in ["rmse", "mae", "log_loss"]:
                # Avoid division by zero
                if baseline_value == 0:
                    drift_detected = False
                else:
                    # Check if error increased by more than threshold
                    drift_detected = (current_value - baseline_value) / baseline_value > DRIFT_THRESHOLD
            else:
                # For metrics where higher is better (like accuracy)
                # Avoid division by zero
                if baseline_value == 0:
                    drift_detected = False
                else:
                    # Check if accuracy decreased by more than threshold
                    drift_detected = (baseline_value - current_value) / baseline_value > DRIFT_THRESHOLD
            
            drift_results[key] = drift_detected
            
            # Log the results
            if drift_detected:
                logger.warning(f"Drift detected for {key}: {primary_metric} degraded from {baseline_value:.4f} to {current_value:.4f}")
            else:
                logger.info(f"No drift detected for {key}: {primary_metric} changed from {baseline_value:.4f} to {current_value:.4f}")
        
        self.drift_detected = drift_results
        return drift_results
    
    def trigger_retraining(self, check_only: bool = False) -> Dict[str, bool]:
        """
        Trigger retraining for models where drift was detected.
        
        Args:
            check_only: If True, only check for drift but don't trigger retraining
            
        Returns:
            Dictionary mapping model/target combinations to retraining status
        """
        if not self.drift_detected:
            self.detect_drift()
        
        retraining_status = {}
        
        for key, drift_detected in self.drift_detected.items():
            if not drift_detected:
                retraining_status[key] = False
                continue
            
            # Split the key into model name and prediction target
            parts = key.split("_")
            model_name = "_".join(parts[:-1])  # Handle model names with underscores
            prediction_target = parts[-1]
            
            if check_only:
                logger.info(f"Drift detected for {model_name} ({prediction_target}), would trigger retraining")
                retraining_status[key] = True
                continue
            
            # Trigger retraining by updating the needs_training flag in the database
            try:
                conn = get_connection()
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE model_weights
                        SET needs_training = TRUE,
                            updated_at = %s
                        WHERE model_name = %s
                          AND active = TRUE
                          AND params->>'prediction_target' = %s
                        RETURNING id
                    """, (datetime.now(timezone.utc), model_name, prediction_target))
                    
                    updated = cursor.fetchone()
                    conn.commit()
                
                close_connection(conn)
                
                if updated:
                    logger.info(f"Triggered retraining for {model_name} ({prediction_target})")
                    retraining_status[key] = True
                else:
                    logger.warning(f"Failed to trigger retraining for {model_name} ({prediction_target}) - model not found")
                    retraining_status[key] = False
                    
            except Exception as e:
                logger.error(f"Error triggering retraining for {model_name} ({prediction_target}): {str(e)}")
                retraining_status[key] = False
        
        return retraining_status
    
    def update_baselines(self, new_baselines: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Update baseline metrics, either with provided values or current performance.
        
        Args:
            new_baselines: Optional dictionary of new baseline metrics
        """
        if new_baselines:
            self.baselines.update(new_baselines)
        elif self.current_performance:
            # Use current performance as new baseline for models that need it
            # This typically happens after retraining
            for key, current_metrics in self.current_performance.items():
                if key in self.drift_detected and self.drift_detected[key]:
                    logger.info(f"Updating baseline for {key} after retraining")
                    self.baselines[key] = current_metrics
        
        # Save to database and file
        try:
            conn = get_connection()
            
            for key, metrics in self.baselines.items():
                # Split the key into model name and prediction target
                parts = key.split("_")
                model_name = "_".join(parts[:-1])
                prediction_target = parts[-1]
                
                # Check if this baseline already exists
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id FROM model_performance
                        WHERE model_name = %s
                          AND prediction_target = %s
                          AND is_baseline = TRUE
                    """, (model_name, prediction_target))
                    
                    existing = cursor.fetchone()
                
                now = datetime.now(timezone.utc)
                
                # Update or insert the baseline
                with conn.cursor() as cursor:
                    if existing:
                        cursor.execute("""
                            UPDATE model_performance
                            SET metrics = %s,
                                updated_at = %s
                            WHERE id = %s
                        """, (json.dumps(metrics), now, existing[0]))
                    else:
                        cursor.execute("""
                            INSERT INTO model_performance
                            (model_name, prediction_target, metrics, is_baseline, sample_count, created_at, updated_at)
                            VALUES (%s, %s, %s, TRUE, %s, %s, %s)
                        """, (model_name, prediction_target, json.dumps(metrics), MIN_SAMPLES, now, now))
                
            conn.commit()
            close_connection(conn)
            
            logger.info(f"Updated {len(self.baselines)} baseline metrics in database")
            
            # Save to file as backup
            self._save_baselines()
            
        except Exception as e:
            logger.error(f"Error updating baselines in database: {str(e)}")


def main():
    """
    Main function for running the model drift detection process.
    """
    parser = argparse.ArgumentParser(description="Check for model drift and trigger retraining if necessary")
    parser.add_argument("--check_only", action="store_true", help="Only check for drift, don't trigger retraining")
    args = parser.parse_args()
    
    logger.info("Starting model drift detection")
    
    try:
        # Create the detector
        detector = ModelDriftDetector()
        
        # Get current performance
        detector.get_current_performance()
        
        # Detect drift
        drift_results = detector.detect_drift()
        
        # Trigger retraining if needed
        retraining_status = detector.trigger_retraining(check_only=args.check_only)
        
        # Update baselines if needed (after retraining)
        if not args.check_only and any(retraining_status.values()):
            logger.info("Retraining triggered for some models, baselines will be updated after retraining")
        
        # Summarize results
        num_models_checked = len(drift_results)
        num_drift_detected = sum(1 for v in drift_results.values() if v)
        num_retraining = sum(1 for v in retraining_status.values() if v)
        
        logger.info(f"Drift detection complete: {num_drift_detected}/{num_models_checked} models showed drift")
        if args.check_only:
            logger.info(f"Check-only mode: {num_retraining} models would be retrained")
        else:
            logger.info(f"Retraining triggered for {num_retraining} models")
        
        # Write a summary file
        summary = {
            "timestamp": datetime.now().isoformat(),
            "models_checked": num_models_checked,
            "drift_detected": num_drift_detected,
            "retraining_triggered": num_retraining,
            "details": {
                key: {
                    "drift_detected": drift,
                    "retraining_triggered": retraining_status.get(key, False)
                } for key, drift in drift_results.items()
            }
        }
        
        with open("logs/model_drift_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error in model drift detection: {str(e)}")
        sys.exit(1)
    
    logger.info("Model drift detection completed successfully")


if __name__ == "__main__":
    main()
