#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Tracking Module

This module provides comprehensive tracking and analysis of model performance over time.
It records prediction accuracy, maintains historical performance metrics, and identifies
underperforming models.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Track and analyze the performance of prediction models over time"""
    
    def __init__(self, storage_dir: str = "data/performance"):
        """
        Initialize the performance tracker
        
        Args:
            storage_dir: Directory to store performance data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        # Load existing performance data if available
        self.performance_history = self._load_performance_history()
        
        # Performance thresholds for alerts
        self.thresholds = {
            "moneyline": 0.52,  # Win rate threshold for moneyline predictions
            "spread": 0.52,    # Win rate threshold for spread predictions
            "total": 0.52,     # Win rate threshold for total predictions
            "player_props": 0.52  # Win rate threshold for player prop predictions
        }
    
    def _load_performance_history(self) -> Dict[str, Any]:
        """
        Load historical performance data from storage
        
        Returns:
            Dictionary containing performance history by model and type
        """
        history_file = self.storage_dir / "performance_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading performance history: {str(e)}")
                return self._initialize_history()
        else:
            return self._initialize_history()
    
    def _initialize_history(self) -> Dict[str, Any]:
        """
        Initialize a new performance history structure
        
        Returns:
            New performance history dictionary
        """
        return {
            "models": {},
            "prediction_types": {
                "moneyline": {
                    "daily": [],
                    "weekly": [],
                    "monthly": [],
                    "season": {}
                },
                "spread": {
                    "daily": [],
                    "weekly": [],
                    "monthly": [],
                    "season": {}
                },
                "total": {
                    "daily": [],
                    "weekly": [],
                    "monthly": [],
                    "season": {}
                },
                "player_props": {
                    "daily": [],
                    "weekly": [],
                    "monthly": [],
                    "season": {}
                }
            },
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_performance_history(self) -> None:
        """
        Save performance history to disk
        """
        history_file = self.storage_dir / "performance_history.json"
        
        try:
            # Update last updated timestamp
            self.performance_history["last_updated"] = datetime.now().isoformat()
            
            with open(history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            
            logger.info(f"Performance history saved to {history_file}")
        except Exception as e:
            logger.error(f"Error saving performance history: {str(e)}")
    
    def record_prediction_results(self, 
                                 date: str,
                                 model_name: str,
                                 prediction_type: str,
                                 predictions: List[Dict[str, Any]],
                                 actual_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Record prediction results and compute accuracy metrics
        
        Args:
            date: Date of predictions in YYYY-MM-DD format
            model_name: Name of the model making predictions
            prediction_type: Type of prediction (moneyline, spread, total, player_props)
            predictions: List of prediction dictionaries
            actual_results: List of actual result dictionaries
            
        Returns:
            Dictionary with performance metrics
        """
        if prediction_type not in self.performance_history["prediction_types"]:
            logger.warning(f"Unknown prediction type: {prediction_type}. Skipping recording.")
            return {}
        
        # Ensure model exists in history
        if model_name not in self.performance_history["models"]:
            self.performance_history["models"][model_name] = {
                "prediction_types": {},
                "version_history": []
            }
        
        if prediction_type not in self.performance_history["models"][model_name]["prediction_types"]:
            self.performance_history["models"][model_name]["prediction_types"][prediction_type] = {
                "daily": [],
                "weekly": [],
                "monthly": [],
                "season": {}
            }
        
        # Compute performance metrics based on prediction type
        metrics = self._compute_performance_metrics(prediction_type, predictions, actual_results)
        
        # Add daily performance record
        daily_record = {
            "date": date,
            "metrics": metrics,
            "prediction_count": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to overall type history
        self.performance_history["prediction_types"][prediction_type]["daily"].append(daily_record)
        
        # Add to model-specific history
        self.performance_history["models"][model_name]["prediction_types"][prediction_type]["daily"].append(daily_record)
        
        # Update aggregated metrics (weekly, monthly, season)
        self._update_aggregated_metrics(prediction_type)
        self._update_aggregated_metrics(prediction_type, model_name=model_name)
        
        # Save updated history
        self._save_performance_history()
        
        return metrics
    
    def _compute_performance_metrics(self,
                                   prediction_type: str,
                                   predictions: List[Dict[str, Any]],
                                   actual_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute performance metrics based on prediction type
        
        Args:
            prediction_type: Type of prediction
            predictions: List of predictions
            actual_results: List of actual results
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        if prediction_type == "moneyline":
            correct = 0
            total = 0
            roi = 0
            positive_clv = 0
            total_clv = 0
            
            for pred, actual in zip(predictions, actual_results):
                if pred.get("game_id") != actual.get("game_id"):
                    continue
                    
                total += 1
                
                # Check if prediction was correct
                pred_winner = pred.get("predicted_winner")
                actual_winner = actual.get("winner")
                
                if pred_winner == actual_winner:
                    correct += 1
                    
                    # Calculate ROI if bet amount and odds available
                    if "bet_amount" in pred and "bet_odds" in pred:
                        bet_amount = pred["bet_amount"]
                        bet_odds = pred["bet_odds"]
                        
                        # Calculate winnings based on odds format
                        if isinstance(bet_odds, str) and bet_odds.startswith("+"):
                            # American odds format positive
                            winnings = bet_amount * (int(bet_odds[1:]) / 100)
                        elif isinstance(bet_odds, str) and bet_odds.startswith("-"):
                            # American odds format negative
                            winnings = bet_amount * (100 / abs(int(bet_odds)))
                        else:
                            # Decimal odds format
                            winnings = bet_amount * (float(bet_odds) - 1)
                            
                        roi += winnings
                else:
                    # Loss
                    if "bet_amount" in pred:
                        roi -= pred["bet_amount"]
                
                # Check for CLV if closing line available
                if "opening_odds" in pred and "closing_odds" in pred:
                    opening = pred["opening_odds"]
                    closing = pred["closing_odds"]
                    
                    if (pred_winner == "home" and closing["home"] > opening["home"]) or \
                       (pred_winner == "away" and closing["away"] > opening["away"]):
                        positive_clv += 1
                    
                    total_clv += 1
            
            # Calculate final metrics
            metrics["accuracy"] = correct / total if total > 0 else 0
            metrics["roi_pct"] = (roi / total) * 100 if total > 0 else 0
            metrics["positive_clv_rate"] = positive_clv / total_clv if total_clv > 0 else 0
            
        elif prediction_type == "spread":
            correct = 0
            total = 0
            roi = 0
            
            for pred, actual in zip(predictions, actual_results):
                if pred.get("game_id") != actual.get("game_id"):
                    continue
                    
                total += 1
                
                # Check if prediction was correct (beat the spread)
                pred_spread = pred.get("predicted_spread")
                actual_spread = actual.get("actual_spread")
                pred_team = pred.get("spread_bet_team")
                
                if (pred_team == "home" and actual_spread > pred_spread) or \
                   (pred_team == "away" and actual_spread < pred_spread):
                    correct += 1
                    
                    # Calculate ROI
                    if "bet_amount" in pred:
                        roi += pred["bet_amount"] * 0.91  # Typical -110 odds payout
                else:
                    # Loss
                    if "bet_amount" in pred:
                        roi -= pred["bet_amount"]
            
            # Calculate final metrics
            metrics["accuracy"] = correct / total if total > 0 else 0
            metrics["roi_pct"] = (roi / total) * 100 if total > 0 else 0
            
        elif prediction_type == "total":
            correct = 0
            total = 0
            roi = 0
            
            for pred, actual in zip(predictions, actual_results):
                if pred.get("game_id") != actual.get("game_id"):
                    continue
                    
                total += 1
                
                # Check if prediction was correct (over/under hit)
                pred_total = pred.get("predicted_total")
                actual_total = actual.get("actual_total")
                pred_bet = pred.get("total_bet")  # "over" or "under"
                
                if (pred_bet == "over" and actual_total > pred_total) or \
                   (pred_bet == "under" and actual_total < pred_total):
                    correct += 1
                    
                    # Calculate ROI
                    if "bet_amount" in pred:
                        roi += pred["bet_amount"] * 0.91  # Typical -110 odds payout
                else:
                    # Loss
                    if "bet_amount" in pred:
                        roi -= pred["bet_amount"]
            
            # Calculate final metrics
            metrics["accuracy"] = correct / total if total > 0 else 0
            metrics["roi_pct"] = (roi / total) * 100 if total > 0 else 0
            
        elif prediction_type == "player_props":
            correct = 0
            total = 0
            roi = 0
            
            for pred, actual in zip(predictions, actual_results):
                if pred.get("player_id") != actual.get("player_id"):
                    continue
                    
                total += 1
                
                # Check if prediction was correct
                prop_type = pred.get("prop_type")  # "points", "rebounds", "assists"
                pred_value = pred.get("predicted_value")
                actual_value = actual.get("actual_value")
                pred_bet = pred.get("bet_type")  # "over" or "under"
                
                if (pred_bet == "over" and actual_value > pred_value) or \
                   (pred_bet == "under" and actual_value < pred_value):
                    correct += 1
                    
                    # Calculate ROI
                    if "bet_amount" in pred:
                        roi += pred["bet_amount"] * 0.91  # Typical -110 odds payout
                else:
                    # Loss
                    if "bet_amount" in pred:
                        roi -= pred["bet_amount"]
            
            # Calculate final metrics
            metrics["accuracy"] = correct / total if total > 0 else 0
            metrics["roi_pct"] = (roi / total) * 100 if total > 0 else 0
        
        return metrics
    
    def _update_aggregated_metrics(self, prediction_type: str, model_name: Optional[str] = None) -> None:
        """
        Update weekly, monthly, and season metrics based on daily records
        
        Args:
            prediction_type: Type of prediction
            model_name: Optional name of model for model-specific updates
        """
        # Determine which history to update
        if model_name:
            history = self.performance_history["models"][model_name]["prediction_types"][prediction_type]
        else:
            history = self.performance_history["prediction_types"][prediction_type]
        
        # Get daily records
        daily_records = history["daily"]
        
        if not daily_records:
            return
        
        # Update weekly metrics (last 7 days)
        today = datetime.now().date()
        week_start = (today - timedelta(days=7)).isoformat()
        
        weekly_records = [r for r in daily_records if r["date"] >= week_start]
        
        if weekly_records:
            weekly_metrics = self._aggregate_metrics(weekly_records)
            history["weekly"] = [
                {
                    "start_date": week_start,
                    "end_date": today.isoformat(),
                    "metrics": weekly_metrics,
                    "prediction_count": sum(r["prediction_count"] for r in weekly_records),
                    "timestamp": datetime.now().isoformat()
                }
            ]
        
        # Update monthly metrics (last 30 days)
        month_start = (today - timedelta(days=30)).isoformat()
        
        monthly_records = [r for r in daily_records if r["date"] >= month_start]
        
        if monthly_records:
            monthly_metrics = self._aggregate_metrics(monthly_records)
            history["monthly"] = [
                {
                    "start_date": month_start,
                    "end_date": today.isoformat(),
                    "metrics": monthly_metrics,
                    "prediction_count": sum(r["prediction_count"] for r in monthly_records),
                    "timestamp": datetime.now().isoformat()
                }
            ]
        
        # Update season metrics
        # Group by season (simple approach - assume season is calendar year)
        seasons = {}
        for record in daily_records:
            year = record["date"].split("-")[0]
            if year not in seasons:
                seasons[year] = []
            seasons[year].append(record)
        
        # Update each season's metrics
        for season, records in seasons.items():
            season_metrics = self._aggregate_metrics(records)
            history["season"][season] = {
                "metrics": season_metrics,
                "prediction_count": sum(r["prediction_count"] for r in records),
                "timestamp": datetime.now().isoformat()
            }
    
    def _aggregate_metrics(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate metrics from multiple records
        
        Args:
            records: List of performance records
            
        Returns:
            Aggregated metrics dictionary
        """
        if not records:
            return {}
        
        # Get all metric keys from the first record
        metrics_keys = records[0]["metrics"].keys()
        
        # Initialize aggregated metrics
        aggregated = {}
        
        for key in metrics_keys:
            # Weight metrics by prediction count
            weighted_sum = sum(r["metrics"].get(key, 0) * r["prediction_count"] for r in records)
            total_predictions = sum(r["prediction_count"] for r in records)
            
            aggregated[key] = weighted_sum / total_predictions if total_predictions > 0 else 0
        
        return aggregated
    
    def get_model_performance(self, model_name: str, prediction_type: str, timeframe: str = "season", season: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for a specific model
        
        Args:
            model_name: Name of the model
            prediction_type: Type of prediction
            timeframe: Timeframe for metrics (daily, weekly, monthly, season)
            season: Optional season identifier for season timeframe
            
        Returns:
            Dictionary with performance metrics
        """
        if model_name not in self.performance_history["models"]:
            logger.warning(f"Model {model_name} not found in performance history")
            return {}
        
        if prediction_type not in self.performance_history["models"][model_name]["prediction_types"]:
            logger.warning(f"Prediction type {prediction_type} not found for model {model_name}")
            return {}
        
        model_history = self.performance_history["models"][model_name]["prediction_types"][prediction_type]
        
        if timeframe == "season" and season:
            if season in model_history["season"]:
                return model_history["season"][season]
            else:
                logger.warning(f"Season {season} not found for model {model_name}")
                return {}
        elif timeframe in ["daily", "weekly", "monthly"]:
            records = model_history[timeframe]
            if records:
                return records[-1]  # Return most recent record
            else:
                logger.warning(f"No {timeframe} records found for model {model_name}")
                return {}
        else:
            logger.warning(f"Invalid timeframe: {timeframe}")
            return {}
    
    def get_prediction_type_performance(self, prediction_type: str, timeframe: str = "season", season: Optional[str] = None) -> Dict[str, Any]:
        """
        Get overall performance metrics for a prediction type
        
        Args:
            prediction_type: Type of prediction
            timeframe: Timeframe for metrics (daily, weekly, monthly, season)
            season: Optional season identifier for season timeframe
            
        Returns:
            Dictionary with performance metrics
        """
        if prediction_type not in self.performance_history["prediction_types"]:
            logger.warning(f"Prediction type {prediction_type} not found in performance history")
            return {}
        
        type_history = self.performance_history["prediction_types"][prediction_type]
        
        if timeframe == "season" and season:
            if season in type_history["season"]:
                return type_history["season"][season]
            else:
                logger.warning(f"Season {season} not found for prediction type {prediction_type}")
                return {}
        elif timeframe in ["daily", "weekly", "monthly"]:
            records = type_history[timeframe]
            if records:
                return records[-1]  # Return most recent record
            else:
                logger.warning(f"No {timeframe} records found for prediction type {prediction_type}")
                return {}
        else:
            logger.warning(f"Invalid timeframe: {timeframe}")
            return {}
    
    def detect_underperforming_models(self) -> List[Dict[str, Any]]:
        """
        Detect models that are underperforming based on thresholds
        
        Returns:
            List of dictionaries with underperforming model details
        """
        underperforming = []
        
        for model_name, model_data in self.performance_history["models"].items():
            for pred_type, type_data in model_data["prediction_types"].items():
                # Check monthly performance
                if type_data["monthly"]:
                    monthly_metrics = type_data["monthly"][-1]["metrics"]
                    threshold = self.thresholds.get(pred_type, 0.52)
                    
                    if monthly_metrics.get("accuracy", 0) < threshold:
                        underperforming.append({
                            "model_name": model_name,
                            "prediction_type": pred_type,
                            "accuracy": monthly_metrics.get("accuracy", 0),
                            "threshold": threshold,
                            "timeframe": "monthly",
                            "prediction_count": type_data["monthly"][-1]["prediction_count"]
                        })
        
        return underperforming
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance summary across all models and prediction types
        
        Returns:
            Dictionary with performance summary
        """
        summary = {
            "prediction_types": {},
            "models": {},
            "underperforming_models": self.detect_underperforming_models(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Summarize by prediction type
        for pred_type in self.performance_history["prediction_types"]:
            type_data = self.performance_history["prediction_types"][pred_type]
            
            # Get most recent metrics for each timeframe
            daily = type_data["daily"][-1]["metrics"] if type_data["daily"] else {}
            weekly = type_data["weekly"][-1]["metrics"] if type_data["weekly"] else {}
            monthly = type_data["monthly"][-1]["metrics"] if type_data["monthly"] else {}
            
            # Get current season metrics
            current_year = datetime.now().year
            season = type_data["season"].get(str(current_year), {}).get("metrics", {})
            
            summary["prediction_types"][pred_type] = {
                "daily": daily,
                "weekly": weekly,
                "monthly": monthly,
                "season": season
            }
        
        # Summarize by model
        for model_name, model_data in self.performance_history["models"].items():
            model_summary = {}
            
            for pred_type, type_data in model_data["prediction_types"].items():
                # Get most recent metrics for monthly timeframe
                if type_data["monthly"]:
                    model_summary[pred_type] = type_data["monthly"][-1]["metrics"]
            
            summary["models"][model_name] = model_summary
        
        return summary
    
    def export_performance_data(self, format: str = "json") -> str:
        """
        Export performance data in various formats
        
        Args:
            format: Format to export (json, csv)
            
        Returns:
            Path to exported file
        """
        if format == "json":
            export_path = self.storage_dir / "performance_export.json"
            with open(export_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            return str(export_path)
        elif format == "csv":
            # Create summary DataFrame
            summary = []
            
            for model_name, model_data in self.performance_history["models"].items():
                for pred_type, type_data in model_data["prediction_types"].items():
                    for timeframe in ["daily", "weekly", "monthly"]:
                        for record in type_data[timeframe]:
                            entry = {
                                "model": model_name,
                                "prediction_type": pred_type,
                                "timeframe": timeframe
                            }
                            
                            # Add date information
                            if timeframe == "daily":
                                entry["date"] = record["date"]
                            else:
                                entry["start_date"] = record["start_date"]
                                entry["end_date"] = record["end_date"]
                            
                            # Add metrics
                            for metric, value in record["metrics"].items():
                                entry[metric] = value
                            
                            entry["prediction_count"] = record["prediction_count"]
                            
                            summary.append(entry)
            
            if not summary:
                logger.warning("No performance data to export")
                return ""
            
            # Create DataFrame and export
            df = pd.DataFrame(summary)
            export_path = self.storage_dir / "performance_export.csv"
            df.to_csv(export_path, index=False)
            return str(export_path)
        else:
            logger.warning(f"Unsupported export format: {format}")
            return ""

# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = PerformanceTracker()
    
    # Example recording of prediction results
    sample_predictions = [
        {"game_id": 1, "predicted_winner": "home", "bet_amount": 100, "bet_odds": "+150"}
    ]
    
    sample_results = [
        {"game_id": 1, "winner": "home"}
    ]
    
    # Record results
    metrics = tracker.record_prediction_results(
        date="2025-04-17",
        model_name="EnsembleModel",
        prediction_type="moneyline",
        predictions=sample_predictions,
        actual_results=sample_results
    )
    
    print("Recorded metrics:", metrics)
    
    # Get performance summary
    summary = tracker.get_performance_summary()
    print("\nPerformance Summary:")
    print(json.dumps(summary, indent=2))
