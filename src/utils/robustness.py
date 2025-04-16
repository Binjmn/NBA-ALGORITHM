#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robustness Utilities for NBA Prediction System

This module contains utilities to make the training pipeline more robust,
including data validation, outlier detection, feature drift monitoring,
model checkpointing, and hyperparameter tuning scheduling.
"""

import os
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logger = logging.getLogger(__name__)


def validate_training_data(features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate training data before processing
    
    Args:
        features_df: DataFrame containing features for training
    
    Returns:
        Dict with validation status and any issues detected
    """
    validation_results = {
        "status": "valid",
        "issues": []
    }
    
    # Check for sufficient sample size
    if len(features_df) < 50:
        validation_results["status"] = "warning"
        validation_results["issues"].append(f"Small sample size: {len(features_df)} rows (recommended: 50+)")
    
    # Check for class imbalance in target
    if 'home_won' in features_df.columns:
        win_rate = features_df['home_won'].mean()
        if win_rate < 0.3 or win_rate > 0.7:
            validation_results["status"] = "warning"
            validation_results["issues"].append(f"Class imbalance detected: {win_rate:.2f} win rate (recommended: 0.3-0.7)")
    
    # Check for missing values in critical columns
    critical_columns = ['home_team_id', 'away_team_id', 'home_won', 'point_diff']
    for col in critical_columns:
        if col in features_df.columns and features_df[col].isna().sum() > 0:
            validation_results["status"] = "error"
            validation_results["issues"].append(f"Missing values in critical column: {col}")
    
    # Check for outliers in key metrics
    if 'point_diff' in features_df.columns:
        point_diff_std = features_df['point_diff'].std()
        if point_diff_std > 30:
            validation_results["status"] = "warning"
            validation_results["issues"].append(f"Unusual variance in point_diff: std={point_diff_std:.2f}")
    
    # Log validation results
    if validation_results["status"] == "valid":
        logger.info("Data validation passed with no issues")
    elif validation_results["status"] == "warning":
        logger.warning(f"Data validation warnings: {', '.join(validation_results['issues'])}")
    else:
        logger.error(f"Data validation errors: {', '.join(validation_results['issues'])}")
    
    return validation_results


def train_with_checkpointing(model_name: str, model: Any, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, bool]:
    """
    Train a model with checkpointing to allow recovery from failures
    
    Args:
        model_name: Name of the model for checkpointing
        model: Model instance to train
        X: Feature matrix
        y: Target variable
    
    Returns:
        Tuple of (trained model, success boolean)
    """
    checkpoint_dir = Path("data/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    try:
        # Try to load from checkpoint first
        checkpoint_path = checkpoint_dir / f"{model_name}_checkpoint.pkl"
        if checkpoint_path.exists():
            # Check if checkpoint is recent (within 7 days)
            checkpoint_age = (datetime.now() - datetime.fromtimestamp(checkpoint_path.stat().st_mtime)).days
            if checkpoint_age <= 7:
                logger.info(f"Loading {model_name} from checkpoint (age: {checkpoint_age} days)")
                with open(checkpoint_path, 'rb') as f:
                    model = pickle.load(f)
                return model, True
            else:
                logger.info(f"Checkpoint for {model_name} is {checkpoint_age} days old, retraining")
        
        # Train the model
        start_time = time.time()
        model.train(X, y)
        training_time = time.time() - start_time
        logger.info(f"Trained {model_name} in {training_time:.2f} seconds")
        
        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved checkpoint for {model_name} to {checkpoint_path}")
        
        return model, True
    except Exception as e:
        logger.error(f"Error training {model_name}: {str(e)}")
        return model, False


def analyze_feature_importance(models_dict: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
    """
    Analyze feature importance across all models
    
    Args:
        models_dict: Dictionary of trained models
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importance analysis
    """
    importance_df = pd.DataFrame(index=feature_names)
    
    for model_name, model in models_dict.items():
        if hasattr(model, 'get_feature_importance'):
            try:
                importances = model.get_feature_importance()
                # Handle different return types from get_feature_importance
                if isinstance(importances, dict):
                    # Convert dict to series
                    importance_series = pd.Series(importances)
                    # Align with feature names
                    importance_series = importance_series.reindex(feature_names, fill_value=0)
                    importance_df[model_name] = importance_series
                elif isinstance(importances, (np.ndarray, list)):
                    # Assuming the array aligns with feature_names
                    if len(importances) == len(feature_names):
                        importance_df[model_name] = importances
                    else:
                        logger.warning(f"Feature importance length mismatch for {model_name}: {len(importances)} vs {len(feature_names)}")
            except Exception as e:
                logger.warning(f"Error getting feature importance for {model_name}: {str(e)}")
    
    # Average importance across models
    if importance_df.empty or importance_df.isna().all().all():
        logger.warning("No valid feature importance data available")
        return pd.DataFrame(index=feature_names)
    
    importance_df['avg_importance'] = importance_df.mean(axis=1, skipna=True)
    
    # Normalize to sum to 1.0
    if importance_df['avg_importance'].sum() > 0:
        importance_df['avg_importance_normalized'] = importance_df['avg_importance'] / importance_df['avg_importance'].sum()
    
    # Sort by average importance
    return importance_df.sort_values('avg_importance', ascending=False)


def prune_low_importance_features(X: pd.DataFrame, importance_df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    Remove features with importance below threshold
    
    Args:
        X: Feature matrix
        importance_df: DataFrame with feature importance analysis
        threshold: Threshold for pruning (features with lower importance will be removed)
    
    Returns:
        Pruned feature matrix
    """
    if 'avg_importance_normalized' in importance_df.columns:
        low_importance = importance_df[importance_df['avg_importance_normalized'] < threshold].index
    elif 'avg_importance' in importance_df.columns:
        low_importance = importance_df[importance_df['avg_importance'] < threshold].index
    else:
        logger.warning("No valid importance metric found in importance_df")
        return X
    
    # Only consider columns that are actually in X
    low_importance = [col for col in low_importance if col in X.columns]
    
    if not low_importance:
        logger.info("No low importance features to prune")
        return X
    
    logger.info(f"Pruning {len(low_importance)} low importance features: {', '.join(low_importance)}")
    return X.drop(columns=low_importance, errors='ignore')


def _train_one_model(model_name, model_pickle, X_pickle, y_pickle):
    """
    Train a single model (used by parallel training)
    
    Args:
        model_name: Name of the model
        model_pickle: Pickled model
        X_pickle: Pickled feature matrix
        y_pickle: Pickled target variable
    
    Returns:
        Tuple of (model_name, trained_model, success, duration, error)
    """
    try:
        # Unpickle the model and data
        model = pickle.loads(model_pickle)
        X = pickle.loads(X_pickle)
        y = pickle.loads(y_pickle)
        
        start_time = time.time()
        model.train(X, y)
        duration = time.time() - start_time
        
        # Pickle the trained model
        trained_model_pickle = pickle.dumps(model)
        
        return model_name, trained_model_pickle, True, duration, None
    except Exception as e:
        return model_name, model_pickle, False, 0, str(e)


def train_models_parallel(models_dict: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple models in parallel
    
    Args:
        models_dict: Dictionary of models to train
        X: Feature matrix
        y: Target variable
    
    Returns:
        Dictionary with training results for each model
    """
    results = {}
    
    # Check if we have enough models to warrant parallel training
    if len(models_dict) < 2:
        logger.info("Not enough models for parallel training, using sequential")
        for model_name, model in models_dict.items():
            try:
                start_time = time.time()
                model.train(X, y)
                duration = time.time() - start_time
                results[model_name] = {
                    "model": model,
                    "success": True,
                    "duration": duration
                }
                logger.info(f"Trained {model_name} sequentially in {duration:.2f} seconds")
            except Exception as e:
                logger.error(f"Error training {model_name} sequentially: {str(e)}")
                results[model_name] = {
                    "model": model,
                    "success": False,
                    "duration": 0,
                    "error": str(e)
                }
        return results
    
    # Check if multiprocessing is supported
    try:
        # Test if we can create a process pool
        with ProcessPoolExecutor(max_workers=1) as executor:
            pass
    except Exception as e:
        logger.warning(f"Parallel training not supported: {str(e)}. Falling back to sequential training.")
        # Fall back to sequential training
        return train_models_parallel({next(iter(models_dict.items()))}, X, y)
    
    # Train in parallel, limited by CPU cores and number of models
    max_workers = min(os.cpu_count() or 4, len(models_dict))
    logger.info(f"Training {len(models_dict)} models in parallel with {max_workers} workers")
    
    # For models that support parallel training, use them directly
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for model_name, model in models_dict.items():
            # Pickle the model and data to avoid shared memory issues
            model_pickle = pickle.dumps(model)
            X_pickle = pickle.dumps(X)
            y_pickle = pickle.dumps(y)
            
            future = executor.submit(_train_one_model, model_name, model_pickle, X_pickle, y_pickle)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                model_name, model_pickle, success, duration, error = future.result()
                
                # Unpickle the model
                model = pickle.loads(model_pickle) if success else models_dict[model_name]
                
                results[model_name] = {
                    "model": model,
                    "success": success,
                    "duration": duration
                }
                if error:
                    results[model_name]["error"] = error
                
                if success:
                    logger.info(f"Successfully trained {model_name} in parallel ({duration:.2f} seconds)")
                else:
                    logger.error(f"Failed to train {model_name} in parallel: {error}")
            except Exception as e:
                logger.error(f"Error processing parallel training result: {str(e)}")
                # Try to identify which model caused the error
                for name, model in models_dict.items():
                    if name not in results:
                        results[name] = {
                            "model": model,
                            "success": False,
                            "duration": 0,
                            "error": f"Error in parallel processing: {str(e)}"
                        }
    
    # If we failed to train any models in parallel, try sequential as fallback
    if not any(res["success"] for res in results.values()):
        logger.warning("All parallel training attempts failed, falling back to sequential training")
        
        # Try sequential training for models that failed
        for model_name, model in models_dict.items():
            if model_name not in results or not results[model_name]["success"]:
                try:
                    start_time = time.time()
                    model.train(X, y)
                    duration = time.time() - start_time
                    results[model_name] = {
                        "model": model,
                        "success": True,
                        "duration": duration
                    }
                    logger.info(f"Trained {model_name} sequentially (fallback) in {duration:.2f} seconds")
                except Exception as e:
                    logger.error(f"Error training {model_name} sequentially (fallback): {str(e)}")
                    results[model_name] = {
                        "model": model,
                        "success": False,
                        "duration": 0,
                        "error": str(e)
                    }
    
    return results


def should_tune_hyperparameters() -> bool:
    """
    Determine if hyperparameter tuning should be done
    
    Returns:
        Boolean indicating whether to perform hyperparameter tuning
    """
    last_tuning_file = Path("data/tuning_history.json")
    
    # Default to tuning if no history exists
    if not last_tuning_file.exists():
        logger.info("No tuning history found, will perform hyperparameter tuning")
        # Create directory if it doesn't exist
        last_tuning_file.parent.mkdir(exist_ok=True)
        
        # Initialize with current date
        with open(last_tuning_file, 'w') as f:
            json.dump({"last_tuning": datetime.now().isoformat()}, f)
        return True
    
    # Load tuning history
    try:
        with open(last_tuning_file, 'r') as f:
            history = json.load(f)
        
        # Check if it's been more than 7 days since last tuning
        last_tuning = datetime.fromisoformat(history['last_tuning'])
        days_since_tuning = (datetime.now() - last_tuning).days
        
        should_tune = days_since_tuning >= 7
        if should_tune:
            logger.info(f"Last hyperparameter tuning was {days_since_tuning} days ago, will perform tuning")
            # Update tuning history with current date
            with open(last_tuning_file, 'w') as f:
                json.dump({"last_tuning": datetime.now().isoformat()}, f)
        else:
            logger.info(f"Last hyperparameter tuning was {days_since_tuning} days ago, skipping tuning")
        
        return should_tune
    except Exception as e:
        logger.error(f"Error checking hyperparameter tuning schedule: {str(e)}")
        return True  # Default to tuning if there's an error


def detect_and_handle_outliers(features_df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Detect and handle outliers in the data
    
    Args:
        features_df: DataFrame to process
        columns: Specific columns to check (if None, will use all numeric columns except targets and IDs)
    
    Returns:
        DataFrame with outliers handled
    """
    # Create a copy to avoid modifying the original
    df = features_df.copy()
    
    if columns is None:
        # Use numeric columns except targets and IDs
        exclude_cols = ['game_id', 'home_team_id', 'away_team_id', 'home_won', 'date', 'point_diff']
        columns = [c for c in df.select_dtypes(include=['number']).columns 
                   if c not in exclude_cols]
    
    outliers_detected = 0
    outliers_by_column = {}
    
    for col in columns:
        # Skip columns with insufficient data or all identical values
        if df[col].nunique() <= 1 or df[col].isna().all():
            logger.warning(f"Skipping outlier detection for column {col} (insufficient unique values)")
            continue
        
        # Simple IQR-based outlier detection
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:  # Avoid division by zero
            logger.warning(f"Skipping outlier detection for column {col} (zero IQR)")
            continue
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        lower_outliers = df[df[col] < lower_bound]
        upper_outliers = df[df[col] > upper_bound]
        outliers = pd.concat([lower_outliers, upper_outliers])
        
        if not outliers.empty:
            outlier_count = len(outliers)
            outliers_detected += outlier_count
            outliers_by_column[col] = outlier_count
            
            # Replace outliers with boundary values (winsorization)
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
            
            logger.info(f"Handled {outlier_count} outliers in {col} (bounds: {lower_bound:.2f} to {upper_bound:.2f})")
    
    if outliers_detected > 0:
        outlier_pct = outliers_detected / (len(df) * len(columns)) * 100 if len(columns) > 0 else 0
        logger.info(f"Total outliers handled: {outliers_detected} ({outlier_pct:.2f}% of data)")
        for col, count in outliers_by_column.items():
            logger.debug(f"  - {col}: {count} outliers ({count/len(df)*100:.2f}% of rows)")
    else:
        logger.info("No outliers detected in the data")
    
    return df


def check_feature_drift(current_df: pd.DataFrame, historical_stats_path: str = "data/feature_stats.json") -> Tuple[bool, Dict[str, Any]]:
    """
    Check if current features have drifted significantly from historical patterns
    
    Args:
        current_df: Current DataFrame to check for drift
        historical_stats_path: Path to saved historical feature statistics
    
    Returns:
        Tuple of (drift_detected boolean, drift report dictionary)
    """
    drift_detected = False
    drift_report = {}
    
    # Calculate current statistics
    current_stats = {}
    for col in current_df.select_dtypes(include=['number']).columns:
        # Skip columns with all NaN or single value
        if current_df[col].isna().all() or current_df[col].nunique() <= 1:
            continue
            
        current_stats[col] = {
            "mean": float(current_df[col].mean()),
            "std": float(current_df[col].std()),
            "min": float(current_df[col].min()),
            "max": float(current_df[col].max()),
            "median": float(current_df[col].median()),
            "sample_size": len(current_df)
        }
    
    # Ensure directory exists
    Path(os.path.dirname(historical_stats_path)).mkdir(exist_ok=True)
    
    # Check if historical stats exist
    if not os.path.exists(historical_stats_path):
        # First run, save current stats as historical
        logger.info(f"No historical feature statistics found, saving current statistics as baseline")
        with open(historical_stats_path, 'w') as f:
            json.dump({"last_updated": datetime.now().isoformat(), "stats": current_stats}, f, indent=2)
        return False, {}
    
    # Load historical stats
    try:
        with open(historical_stats_path, 'r') as f:
            historical_data = json.load(f)
            historical_stats = historical_data.get("stats", {})
    except Exception as e:
        logger.error(f"Error loading historical feature statistics: {str(e)}")
        return False, {}
    
    # Compare current to historical
    for col, stats in current_stats.items():
        if col in historical_stats:
            hist_stats = historical_stats[col]
            
            # Skip if historical data has very different sample size
            if "sample_size" in hist_stats and abs(stats["sample_size"] - hist_stats["sample_size"]) / hist_stats["sample_size"] > 0.5:
                logger.warning(f"Skipping drift check for {col}: Sample size too different (historical: {hist_stats['sample_size']}, current: {stats['sample_size']})")
                continue
                
            drift_metrics = {}
            
            # Check mean drift
            if abs(hist_stats["mean"]) > 1e-10:  # Avoid division by zero
                mean_change = (stats["mean"] - hist_stats["mean"]) / abs(hist_stats["mean"])
                if abs(mean_change) > 0.2:  # >20% change in mean
                    drift_metrics["mean"] = {
                        "historical": hist_stats["mean"],
                        "current": stats["mean"],
                        "percent_change": mean_change * 100
                    }
            
            # Check std drift
            if hist_stats["std"] > 1e-10:  # Avoid division by zero
                std_change = (stats["std"] - hist_stats["std"]) / hist_stats["std"]
                if abs(std_change) > 0.2:  # >20% change in std
                    drift_metrics["std"] = {
                        "historical": hist_stats["std"],
                        "current": stats["std"],
                        "percent_change": std_change * 100
                    }
            
            # Check range expansion
            hist_range = hist_stats["max"] - hist_stats["min"]
            curr_range = stats["max"] - stats["min"]
            if hist_range > 1e-10:  # Avoid division by zero
                range_change = (curr_range - hist_range) / hist_range
                if range_change > 0.3:  # >30% expansion in range
                    drift_metrics["range"] = {
                        "historical": [hist_stats["min"], hist_stats["max"]],
                        "current": [stats["min"], stats["max"]],
                        "percent_change": range_change * 100
                    }
            
            if drift_metrics:
                drift_detected = True
                drift_report[col] = drift_metrics
    
    # Log drift detection results
    if drift_detected:
        logger.warning(f"Feature drift detected in {len(drift_report)} columns")
        for col, metrics in drift_report.items():
            metric_msgs = []
            for metric, values in metrics.items():
                if metric == "range":
                    msg = f"{metric}: {values['percent_change']:.1f}% expansion"
                else:
                    msg = f"{metric}: {values['percent_change']:.1f}% change"
                metric_msgs.append(msg) 
            logger.warning(f"  - {col}: {', '.join(metric_msgs)}")
    else:
        logger.info("No significant feature drift detected")
        
        # Periodically update the historical stats (every 30 days)
        last_updated = datetime.fromisoformat(historical_data.get("last_updated", "2000-01-01T00:00:00"))
        days_since_update = (datetime.now() - last_updated).days
        
        if days_since_update > 30:
            logger.info(f"Updating historical feature statistics (last updated {days_since_update} days ago)")
            with open(historical_stats_path, 'w') as f:
                json.dump({"last_updated": datetime.now().isoformat(), "stats": current_stats}, f, indent=2)
    
    return drift_detected, drift_report
