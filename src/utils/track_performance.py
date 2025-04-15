"""
Performance Tracking for NBA Prediction System

This script calculates and tracks the performance of prediction models over time.
It runs daily (scheduled for 2:00 AM EST) and calculates 7-day accuracy metrics
for all prediction types (moneyline, spread, totals, player props).

Key Features:
- Calculates 7-day accuracy metrics for all prediction types
- Stores results in the model_performance database table
- Generates summary reports for model performance
- Compares performance against target thresholds

Usage:
    python -m src.utils.track_performance [--summary] [--days DAYS]
    
Options:
    --summary   Generate a summary report only, don't update the database
    --days DAYS Number of days to look back for performance data (default: 7)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_squared_error, mean_absolute_error, r2_score

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.database.models import ModelPerformance
from src.database.connection import get_connection, close_connection
from src.api.theodds_client import TheOddsClient
from src.api.balldontlie_client import BallDontLieClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/performance_tracking.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
PERFORMANCE_WINDOW_DAYS = 7  # Default days to look back
PERFORMANCE_TARGETS = {  # Target performance thresholds by prediction type
    "moneyline": {"accuracy": 0.60, "f1_score": 0.60},
    "spread": {"accuracy": 0.55, "rmse": 5.0},
    "totals": {"accuracy": 0.55, "rmse": 8.0},
    "player_points": {"rmse": 4.0, "r2_score": 0.6},
    "player_rebounds": {"rmse": 2.0, "r2_score": 0.5},
    "player_assists": {"rmse": 1.5, "r2_score": 0.5},
    "player_threes": {"rmse": 1.0, "r2_score": 0.5}
}


class PerformanceTracker:
    """
    Tracks and analyzes model performance over time.
    
    This class calculates performance metrics for various prediction models and targets,
    compares them against defined thresholds, and stores the results in the database.
    """
    
    def __init__(self, days_back: int = PERFORMANCE_WINDOW_DAYS):
        """
        Initialize the performance tracker.
        
        Args:
            days_back: Number of days to look back for performance data
        """
        self.days_back = days_back
        self.end_date = datetime.now(timezone.utc)
        self.start_date = self.end_date - timedelta(days=days_back)
        
        # Initialize API clients
        self.odds_client = TheOddsClient()
        self.balldontlie_client = BallDontLieClient()
        
        # Storage for predictions and actual outcomes
        self.predictions = {}
        self.outcomes = {}
        
        # Performance metrics
        self.performance = {}
        
        # Model types to track
        self.model_types = [
            "RandomForest",
            "CombinedGradientBoosting",
            "Bayesian",
            "ModelMixing",
            "EnsembleStacking"
        ]
        
        # Prediction targets to track
        self.prediction_targets = [
            "moneyline",
            "spread",
            "totals",
            "player_points",
            "player_rebounds",
            "player_assists",
            "player_threes"
        ]
    
    def load_predictions(self):
        """
        Load predictions made within the time window from the database.
        """
        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        p.model_name,
                        p.prediction_target,
                        p.game_id,
                        p.player_id,
                        p.prediction_data,
                        p.created_at
                    FROM predictions p
                    WHERE p.created_at BETWEEN %s AND %s
                    ORDER BY p.created_at DESC
                """, (self.start_date, self.end_date))
                
                records = cursor.fetchall()
            
            close_connection(conn)
            
            # Group predictions by model and target
            for record in records:
                model_name = record[0]
                prediction_target = record[1]
                game_id = record[2]
                player_id = record[3]
                prediction_data = record[4]  # JSONB data
                
                key = f"{model_name}_{prediction_target}"
                if key not in self.predictions:
                    self.predictions[key] = []
                
                # Create a prediction entry
                entry = {
                    "game_id": game_id,
                    "player_id": player_id,
                    "prediction": prediction_data,
                    "timestamp": record[5]
                }
                
                self.predictions[key].append(entry)
            
            logger.info(f"Loaded {len(records)} predictions from database")
            
        except Exception as e:
            logger.error(f"Error loading predictions: {str(e)}")
    
    def load_outcomes(self):
        """
        Load actual game and player outcomes from the database.
        """
        try:
            # Get unique game IDs from predictions
            game_ids = set()
            player_ids = set()
            
            for predictions in self.predictions.values():
                for pred in predictions:
                    if pred["game_id"]:
                        game_ids.add(pred["game_id"])
                    if pred["player_id"]:
                        player_ids.add(pred["player_id"])
            
            # Load game outcomes from the database
            conn = get_connection()
            
            # Get game results
            if game_ids:
                placeholder_str = ",".join([f"%s" for _ in game_ids])
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT 
                            g.id,
                            g.home_team_id,
                            g.away_team_id,
                            g.home_score,
                            g.away_score,
                            g.home_spread,
                            g.total_score
                        FROM games g
                        WHERE g.id IN ({placeholder_str})
                          AND g.status = 'closed'
                    """, tuple(game_ids))
                    
                    game_records = cursor.fetchall()
                
                # Process game records
                for record in game_records:
                    game_id = record[0]
                    home_score = record[3]
                    away_score = record[4]
                    home_spread = record[5]
                    total_score = record[6]
                    
                    # Moneyline outcome (did home team win?)
                    home_win = home_score > away_score if home_score is not None and away_score is not None else None
                    
                    # Spread outcome (did home team cover?)
                    if home_spread is not None and home_score is not None and away_score is not None:
                        home_cover = (home_score + home_spread) > away_score
                    else:
                        home_cover = None
                    
                    # Totals outcome (over/under)
                    if total_score is not None and home_score is not None and away_score is not None:
                        over = (home_score + away_score) > total_score
                    else:
                        over = None
                    
                    self.outcomes[f"game_{game_id}"] = {
                        "moneyline": home_win,
                        "spread": home_cover,
                        "totals": over,
                        "home_score": home_score,
                        "away_score": away_score,
                        "total": home_score + away_score if home_score is not None and away_score is not None else None
                    }
            
            # Get player stats
            if player_ids and game_ids:
                placeholders_players = ",".join([f"%s" for _ in player_ids])
                placeholders_games = ",".join([f"%s" for _ in game_ids])
                
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT 
                            ps.game_id,
                            ps.player_id,
                            ps.points,
                            ps.rebounds,
                            ps.assists,
                            ps.three_pointers_made
                        FROM player_stats ps
                        WHERE ps.player_id IN ({placeholders_players})
                          AND ps.game_id IN ({placeholders_games})
                    """, tuple(list(player_ids) + list(game_ids)))
                    
                    player_records = cursor.fetchall()
                
                # Process player records
                for record in player_records:
                    game_id = record[0]
                    player_id = record[1]
                    points = record[2]
                    rebounds = record[3]
                    assists = record[4]
                    threes = record[5]
                    
                    self.outcomes[f"player_{player_id}_{game_id}"] = {
                        "player_points": points,
                        "player_rebounds": rebounds,
                        "player_assists": assists,
                        "player_threes": threes
                    }
            
            close_connection(conn)
            
            logger.info(f"Loaded outcomes for {len(self.outcomes)} games/players")
            
        except Exception as e:
            logger.error(f"Error loading outcomes: {str(e)}")
    
    def calculate_metrics(self):
        """
        Calculate performance metrics for each model and prediction target.
        """
        for key, predictions in self.predictions.items():
            # Split the key into model name and prediction target
            parts = key.split("_")
            model_name = "_".join(parts[:-1])
            prediction_target = parts[-1]
            
            # Skip if we don't have any predictions
            if not predictions:
                continue
            
            # Prepare data for metrics calculation
            y_true = []
            y_pred = []
            y_proba = []
            regression_targets = ["player_points", "player_rebounds", "player_assists", "player_threes"]
            
            # Separate logic for game vs player predictions
            if prediction_target in regression_targets:
                # Player props (regression task)
                for pred in predictions:
                    game_id = pred["game_id"]
                    player_id = pred["player_id"]
                    
                    if not game_id or not player_id:
                        continue
                    
                    outcome_key = f"player_{player_id}_{game_id}"
                    if outcome_key not in self.outcomes:
                        continue
                    
                    actual_value = self.outcomes[outcome_key].get(prediction_target)
                    if actual_value is None:
                        continue
                    
                    predicted_value = pred["prediction"].get("predicted_value")
                    if predicted_value is None:
                        continue
                    
                    y_true.append(actual_value)
                    y_pred.append(predicted_value)
            else:
                # Game predictions (classification or regression)
                for pred in predictions:
                    game_id = pred["game_id"]
                    if not game_id:
                        continue
                    
                    outcome_key = f"game_{game_id}"
                    if outcome_key not in self.outcomes:
                        continue
                    
                    if prediction_target in ["moneyline", "spread", "totals"]:
                        # Binary classification
                        actual_value = self.outcomes[outcome_key].get(prediction_target)
                        if actual_value is None:
                            continue
                        
                        if prediction_target == "moneyline":
                            # Probability that home team wins
                            predicted_prob = pred["prediction"].get("home_win_probability")
                            predicted_class = predicted_prob > 0.5 if predicted_prob is not None else None
                        elif prediction_target == "spread":
                            # Probability that home team covers
                            predicted_prob = pred["prediction"].get("home_cover_probability")
                            predicted_class = predicted_prob > 0.5 if predicted_prob is not None else None
                        elif prediction_target == "totals":
                            # Probability of over
                            predicted_prob = pred["prediction"].get("over_probability")
                            predicted_class = predicted_prob > 0.5 if predicted_prob is not None else None
                        
                        if predicted_class is not None and predicted_prob is not None:
                            y_true.append(int(actual_value))
                            y_pred.append(int(predicted_class))
                            y_proba.append([1-predicted_prob, predicted_prob])
                    else:
                        # Regression for other targets (e.g., predicted score)
                        actual_total = self.outcomes[outcome_key].get("total")
                        predicted_total = pred["prediction"].get("predicted_total")
                        
                        if actual_total is not None and predicted_total is not None:
                            y_true.append(actual_total)
                            y_pred.append(predicted_total)
            
            # Calculate metrics based on prediction type
            metrics = {}
            sample_count = len(y_true)
            
            if sample_count < 5:  # Need minimum samples for meaningful metrics
                logger.warning(f"Not enough samples for {key}: {sample_count}/5")
                continue
            
            if prediction_target in ["moneyline", "spread", "totals"]:
                # Classification metrics
                if y_true and y_pred:
                    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
                    metrics["f1_score"] = float(f1_score(y_true, y_pred, average="binary"))
                
                # Log loss if we have probability predictions
                if y_true and y_proba:
                    try:
                        metrics["log_loss"] = float(log_loss(y_true, y_proba))
                    except Exception as e:
                        logger.warning(f"Error calculating log_loss for {key}: {str(e)}")
            else:
                # Regression metrics
                if y_true and y_pred:
                    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
                    
                    try:
                        metrics["r2_score"] = float(r2_score(y_true, y_pred))
                    except Exception as e:
                        logger.warning(f"Error calculating r2_score for {key}: {str(e)}")
            
            # Add comparison to targets
            target_metrics = PERFORMANCE_TARGETS.get(prediction_target, {})
            for metric_name, target_value in target_metrics.items():
                if metric_name in metrics:
                    current_value = metrics[metric_name]
                    
                    # For metrics where lower is better (like RMSE), invert the comparison
                    if metric_name in ["rmse", "mae", "log_loss"]:
                        metrics[f"{metric_name}_vs_target"] = (target_value - current_value) / target_value
                    else:
                        metrics[f"{metric_name}_vs_target"] = (current_value - target_value) / target_value
            
            # Store the performance data
            self.performance[key] = {
                "model_name": model_name,
                "prediction_target": prediction_target,
                "metrics": metrics,
                "sample_count": sample_count,
                "start_date": self.start_date,
                "end_date": self.end_date
            }
        
        logger.info(f"Calculated performance metrics for {len(self.performance)} model/target combinations")
    
    def save_performance(self):
        """
        Save the calculated performance metrics to the database.
        """
        try:
            conn = get_connection()
            now = datetime.now(timezone.utc)
            
            for key, perf_data in self.performance.items():
                model_name = perf_data["model_name"]
                prediction_target = perf_data["prediction_target"]
                metrics = perf_data["metrics"]
                sample_count = perf_data["sample_count"]
                
                # Insert into the database
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO model_performance
                        (model_name, prediction_target, metrics, sample_count, is_baseline, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, FALSE, %s, %s)
                        RETURNING id
                    """, (model_name, prediction_target, json.dumps(metrics), sample_count, now, now))
                    
                    inserted_id = cursor.fetchone()[0]
                
                logger.info(f"Saved performance metrics for {key} with ID {inserted_id}")
            
            conn.commit()
            close_connection(conn)
            
        except Exception as e:
            logger.error(f"Error saving performance metrics: {str(e)}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a performance summary report.
        
        Returns:
            Dictionary containing the performance summary
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "period": f"{self.start_date.isoformat()} to {self.end_date.isoformat()}",
            "models": {},
            "targets": {},
            "overall": {}
        }
        
        # Calculate summary by model type
        for key, perf_data in self.performance.items():
            model_name = perf_data["model_name"]
            prediction_target = perf_data["prediction_target"]
            metrics = perf_data["metrics"]
            
            # Add to model summary
            if model_name not in summary["models"]:
                summary["models"][model_name] = {
                    "targets": {},
                    "overall": {}
                }
            
            summary["models"][model_name]["targets"][prediction_target] = metrics
            
            # Add to target summary
            if prediction_target not in summary["targets"]:
                summary["targets"][prediction_target] = {
                    "models": {},
                    "overall": {}
                }
            
            summary["targets"][prediction_target]["models"][model_name] = metrics
        
        # Calculate overall averages by model
        for model_name, model_data in summary["models"].items():
            overall_metrics = {}
            
            for target, metrics in model_data["targets"].items():
                for metric_name, value in metrics.items():
                    if metric_name not in overall_metrics:
                        overall_metrics[metric_name] = []
                    overall_metrics[metric_name].append(value)
            
            # Calculate averages
            for metric_name, values in overall_metrics.items():
                if values:
                    model_data["overall"][metric_name] = sum(values) / len(values)
        
        # Calculate overall averages by target
        for target, target_data in summary["targets"].items():
            overall_metrics = {}
            
            for model_name, metrics in target_data["models"].items():
                for metric_name, value in metrics.items():
                    if metric_name not in overall_metrics:
                        overall_metrics[metric_name] = []
                    overall_metrics[metric_name].append(value)
            
            # Calculate averages
            for metric_name, values in overall_metrics.items():
                if values:
                    target_data["overall"][metric_name] = sum(values) / len(values)
        
        # Calculate overall system performance
        overall_metrics = {}
        for key, perf_data in self.performance.items():
            metrics = perf_data["metrics"]
            
            for metric_name, value in metrics.items():
                if metric_name not in overall_metrics:
                    overall_metrics[metric_name] = []
                overall_metrics[metric_name].append(value)
        
        # Calculate averages
        for metric_name, values in overall_metrics.items():
            if values:
                summary["overall"][metric_name] = sum(values) / len(values)
        
        return summary
    
    def save_summary(self, summary: Dict[str, Any]):
        """
        Save the performance summary to file.
        
        Args:
            summary: The performance summary dictionary
        """
        try:
            # Create the logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            # Save to JSON file
            with open("logs/model_performance_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            # Save to text file for easier viewing
            with open("logs/model_performance_summary.txt", "w") as f:
                f.write(f"NBA Prediction System - Performance Report\n")
                f.write(f"Generated: {summary['timestamp']}\n")
                f.write(f"Period: {summary['period']}\n\n")
                
                f.write("Overall System Performance:\n")
                for metric, value in summary["overall"].items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
                
                f.write("Performance by Model:\n")
                for model, data in summary["models"].items():
                    f.write(f"  {model}:\n")
                    for metric, value in data["overall"].items():
                        f.write(f"    {metric}: {value:.4f}\n")
                    f.write("\n")
                
                f.write("Performance by Target:\n")
                for target, data in summary["targets"].items():
                    f.write(f"  {target}:\n")
                    for metric, value in data["overall"].items():
                        f.write(f"    {metric}: {value:.4f}\n")
                    f.write("\n")
                
                f.write("Detailed Metrics by Model and Target:\n")
                for model, model_data in summary["models"].items():
                    for target, metrics in model_data["targets"].items():
                        f.write(f"  {model} - {target}:\n")
                        for metric, value in metrics.items():
                            f.write(f"    {metric}: {value:.4f}\n")
                        f.write("\n")
            
            logger.info("Performance summary saved to logs/model_performance_summary.json and .txt")
            
        except Exception as e:
            logger.error(f"Error saving performance summary: {str(e)}")
    
    def run(self, summary_only: bool = False):
        """
        Run the performance tracking process.
        
        Args:
            summary_only: If True, only generate summary, don't update database
        """
        # Load predictions and outcomes
        self.load_predictions()
        self.load_outcomes()
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Save to database if not summary only
        if not summary_only and self.performance:
            self.save_performance()
        
        # Generate and save summary
        summary = self.generate_summary()
        self.save_summary(summary)
        
        return summary


def main():
    """
    Main function for running the performance tracking process.
    """
    parser = argparse.ArgumentParser(description="Track model performance metrics")
    parser.add_argument("--summary", action="store_true", help="Generate summary only, don't update database")
    parser.add_argument("--days", type=int, default=PERFORMANCE_WINDOW_DAYS, help="Number of days to look back")
    args = parser.parse_args()
    
    logger.info(f"Starting performance tracking (looking back {args.days} days)")
    
    try:
        # Create the tracker
        tracker = PerformanceTracker(days_back=args.days)
        
        # Run the tracking process
        summary = tracker.run(summary_only=args.summary)
        
        # Print a brief summary to console
        if summary and "overall" in summary:
            print("\nPerformance Summary:")
            for metric, value in summary["overall"].items():
                print(f"  {metric}: {value:.4f}")
            print("\nFor details, see logs/model_performance_summary.txt")
        
    except Exception as e:
        logger.error(f"Error in performance tracking: {str(e)}")
        sys.exit(1)
    
    logger.info("Performance tracking completed successfully")


if __name__ == "__main__":
    main()
