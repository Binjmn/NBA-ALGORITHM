#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Tracking Module for NBA Prediction System

This module tracks prediction performance over time, evaluates model accuracy,
and provides historical performance metrics for the prediction system.

It uses real NBA game outcomes to calculate accuracy metrics for different
prediction types and models, enabling continuous improvement of the system.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Tracks and analyzes prediction performance for continuous improvement
    
    This class stores prediction results, compares them with actual outcomes,
    calculates performance metrics, and provides historical performance analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance tracker
        
        Args:
            config: Configuration dictionary with performance tracking settings
        """
        self.config = config or {}
        
        # Set up data storage paths
        base_dir = Path(__file__).parent.parent.parent  # Go up to project root
        self.data_dir = os.path.join(base_dir, "results", "performance")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize performance metrics tracking
        self.current_predictions = []
        self.historical_predictions = []
        self.performance_metrics = {
            "moneyline": {"correct": 0, "total": 0, "accuracy": 0.0},
            "spread": {"correct": 0, "total": 0, "accuracy": 0.0},
            "total": {"correct": 0, "total": 0, "accuracy": 0.0},
            "player_props": {"correct": 0, "total": 0, "accuracy": 0.0}
        }
        
        # Load historical performance data if available
        self._load_historical_data()
        
    def _load_historical_data(self) -> None:
        """
        Load historical performance data from storage
        """
        history_file = os.path.join(self.data_dir, "prediction_history.json")
        metrics_file = os.path.join(self.data_dir, "performance_metrics.json")
        
        try:
            # Load historical predictions
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.historical_predictions = json.load(f)
                logger.info(f"Loaded {len(self.historical_predictions)} historical predictions")
                
            # Load performance metrics
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    self.performance_metrics = json.load(f)
                logger.info(f"Loaded historical performance metrics")
                
        except Exception as e:
            logger.error(f"Error loading historical performance data: {str(e)}")
    
    def track_prediction(self, prediction_data: Dict[str, Any]) -> None:
        """
        Track a new prediction for future performance analysis
        
        Args:
            prediction_data: Dictionary with prediction details
        """
        # Make sure prediction has required fields
        required_fields = ["date", "game_id", "prediction_type", "predicted_value"]
        missing_fields = [field for field in required_fields if field not in prediction_data]
        
        if missing_fields:
            logger.warning(f"Cannot track prediction: missing required fields: {missing_fields}")
            return
        
        # Add timestamp if not present
        if "timestamp" not in prediction_data:
            prediction_data["timestamp"] = datetime.now().isoformat()
        
        # Add to current predictions list
        self.current_predictions.append(prediction_data)
        
        # Save current predictions to disk periodically
        if len(self.current_predictions) >= 10:  # Save after every 10 predictions
            self._save_current_predictions()
    
    def track_batch_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        """
        Track a batch of predictions at once
        
        Args:
            predictions: List of prediction dictionaries
        """
        for prediction in predictions:
            self.track_prediction(prediction)
    
    def update_with_actual_outcome(self, game_id: str, actual_outcomes: Dict[str, Any]) -> None:
        """
        Update tracked predictions with actual outcomes
        
        Args:
            game_id: ID of the game to update
            actual_outcomes: Dictionary with actual outcomes
        """
        updated_count = 0
        
        # Update current predictions
        for prediction in self.current_predictions:
            if prediction.get("game_id") == game_id:
                prediction["actual_value"] = actual_outcomes.get(prediction["prediction_type"])
                prediction["correct"] = self._is_prediction_correct(prediction)
                updated_count += 1
        
        # Update historical predictions if needed
        for prediction in self.historical_predictions:
            if prediction.get("game_id") == game_id and "actual_value" not in prediction:
                prediction["actual_value"] = actual_outcomes.get(prediction["prediction_type"])
                prediction["correct"] = self._is_prediction_correct(prediction)
                updated_count += 1
        
        if updated_count > 0:
            logger.info(f"Updated {updated_count} predictions with actual outcomes for game {game_id}")
            # Calculate updated performance metrics
            self._update_performance_metrics()
            # Save updated data
            self._save_current_predictions()
            self._save_performance_metrics()
    
    def _is_prediction_correct(self, prediction: Dict[str, Any]) -> bool:
        """
        Check if a prediction was correct based on actual outcome
        
        Args:
            prediction: Dictionary with prediction details including actual_value
            
        Returns:
            True if prediction was correct, False otherwise
        """
        if "actual_value" not in prediction:
            return False
            
        pred_type = prediction["prediction_type"]
        predicted = prediction["predicted_value"]
        actual = prediction["actual_value"]
        
        # Different logic based on prediction type
        if pred_type == "moneyline":
            # For moneyline, check if predicted winner matches actual winner
            return bool(predicted) == bool(actual)
            
        elif pred_type == "spread":
            # For spread, check if spread prediction was correct
            if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
                # Some complex spread logic here...
                # Example: if spread is -5.5, home team must win by 6 or more
                spread = prediction.get("line", 0)
                point_diff = prediction.get("actual_point_diff", 0)
                
                if predicted > 0:  # Predicted home team covers
                    return point_diff > spread
                else:  # Predicted away team covers
                    return point_diff < spread
            return False
            
        elif pred_type == "total":
            # For total, check if over/under prediction was correct
            if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
                line = prediction.get("line", 0)
                if predicted > line:  # Predicted over
                    return actual > line
                else:  # Predicted under
                    return actual <= line
            return False
            
        elif pred_type.startswith("player_"):
            # For player props, check if over/under prediction was correct
            if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
                line = prediction.get("line", 0)
                if predicted > line:  # Predicted over
                    return actual > line
                else:  # Predicted under
                    return actual <= line
            return False
            
        # Default case
        return predicted == actual
    
    def _update_performance_metrics(self) -> None:
        """
        Update performance metrics based on all predictions with outcomes
        """
        # Reset metrics
        for pred_type in self.performance_metrics:
            self.performance_metrics[pred_type] = {"correct": 0, "total": 0, "accuracy": 0.0}
        
        # Process all predictions with actual values
        all_predictions = self.current_predictions + self.historical_predictions
        predictions_with_outcomes = [p for p in all_predictions if "actual_value" in p]
        
        for prediction in predictions_with_outcomes:
            pred_type = prediction["prediction_type"]
            correct = prediction.get("correct", False)
            
            # Map player props to the general category
            if pred_type.startswith("player_"):
                pred_type = "player_props"
            
            # Make sure we have this prediction type in our metrics
            if pred_type not in self.performance_metrics:
                self.performance_metrics[pred_type] = {"correct": 0, "total": 0, "accuracy": 0.0}
            
            # Update counts
            self.performance_metrics[pred_type]["total"] += 1
            if correct:
                self.performance_metrics[pred_type]["correct"] += 1
        
        # Calculate accuracy percentages
        for pred_type in self.performance_metrics:
            total = self.performance_metrics[pred_type]["total"]
            correct = self.performance_metrics[pred_type]["correct"]
            
            accuracy = correct / total if total > 0 else 0.0
            self.performance_metrics[pred_type]["accuracy"] = accuracy
    
    def _save_current_predictions(self) -> None:
        """
        Save current predictions to historical storage
        """
        try:
            # Combine current and historical predictions
            all_predictions = self.historical_predictions + self.current_predictions
            
            # Convert to DataFrame for easy processing
            df = pd.DataFrame(all_predictions)
            
            # Remove duplicates if any
            if not df.empty and "game_id" in df.columns and "prediction_type" in df.columns:
                df = df.drop_duplicates(subset=["game_id", "prediction_type"])
            
            # Convert back to list of dicts
            all_predictions = df.to_dict(orient="records") if not df.empty else []
            
            # Save to disk
            history_file = os.path.join(self.data_dir, "prediction_history.json")
            with open(history_file, 'w') as f:
                json.dump(all_predictions, f, indent=2)
            
            # Update historical predictions and reset current
            self.historical_predictions = all_predictions
            self.current_predictions = []
            
            logger.info(f"Saved {len(all_predictions)} predictions to history")
            
        except Exception as e:
            logger.error(f"Error saving prediction history: {str(e)}")
    
    def _save_performance_metrics(self) -> None:
        """
        Save performance metrics to disk
        """
        try:
            metrics_file = os.path.join(self.data_dir, "performance_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
            
            logger.info("Saved performance metrics to disk")
            
        except Exception as e:
            logger.error(f"Error saving performance metrics: {str(e)}")
    
    def get_performance_metrics(self, prediction_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current performance metrics
        
        Args:
            prediction_type: Optional specific prediction type to get metrics for
            
        Returns:
            Dictionary with performance metrics
        """
        if prediction_type is not None:
            # For player props, map to the general category
            if prediction_type.startswith("player_"):
                prediction_type = "player_props"
                
            if prediction_type in self.performance_metrics:
                return self.performance_metrics[prediction_type]
            else:
                logger.warning(f"No metrics available for prediction type: {prediction_type}")
                return {"correct": 0, "total": 0, "accuracy": 0.0}
        else:
            return self.performance_metrics
    
    def get_recent_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Get performance metrics for the recent time period
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary with recent performance metrics
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Combine current and historical predictions
        all_predictions = self.current_predictions + self.historical_predictions
        
        # Filter to recent predictions with outcomes
        recent_predictions = [
            p for p in all_predictions 
            if "timestamp" in p and p["timestamp"] >= cutoff_date and "actual_value" in p
        ]
        
        # Calculate metrics for recent predictions
        recent_metrics = {
            "moneyline": {"correct": 0, "total": 0, "accuracy": 0.0},
            "spread": {"correct": 0, "total": 0, "accuracy": 0.0},
            "total": {"correct": 0, "total": 0, "accuracy": 0.0},
            "player_props": {"correct": 0, "total": 0, "accuracy": 0.0}
        }
        
        for prediction in recent_predictions:
            pred_type = prediction["prediction_type"]
            correct = prediction.get("correct", False)
            
            # Map player props to the general category
            if pred_type.startswith("player_"):
                pred_type = "player_props"
            
            # Make sure we have this prediction type in our metrics
            if pred_type not in recent_metrics:
                recent_metrics[pred_type] = {"correct": 0, "total": 0, "accuracy": 0.0}
            
            # Update counts
            recent_metrics[pred_type]["total"] += 1
            if correct:
                recent_metrics[pred_type]["correct"] += 1
        
        # Calculate accuracy percentages
        for pred_type in recent_metrics:
            total = recent_metrics[pred_type]["total"]
            correct = recent_metrics[pred_type]["correct"]
            
            accuracy = correct / total if total > 0 else 0.0
            recent_metrics[pred_type]["accuracy"] = accuracy
        
        return recent_metrics
    
    def get_prediction_history(self, game_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get prediction history for analysis
        
        Args:
            game_id: Optional game ID to filter by
            
        Returns:
            List of prediction dictionaries
        """
        # Combine current and historical predictions
        all_predictions = self.current_predictions + self.historical_predictions
        
        if game_id is not None:
            return [p for p in all_predictions if p.get("game_id") == game_id]
        else:
            return all_predictions
