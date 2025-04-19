
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility Functions Module for NBA Model Training Pipeline

Responsibilities:
- Provide shared helper functions for the training pipeline
- Implement data validation and cleaning utilities
- Handle logging and error reporting consistently
- Facilitate data formatting and conversion
- Support file I/O operations and path management
- Provide serialization and deserialization tools
"""

import os
import logging
import json
import pandas as pd
import numpy as np
import hashlib
import re
import time
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import traceback
import warnings

from .config import logger


def setup_logger(log_file: str = None, log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting
    
    Args:
        log_file: Path to log file
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    custom_logger = logging.getLogger(f"nba_pipeline_{int(time.time())}")
    custom_logger.setLevel(log_level)
    custom_logger.propagate = False
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    custom_logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        custom_logger.addHandler(file_handler)
    
    return custom_logger


def validate_data_structure(data: Any, expected_structure: Dict[str, type], 
                           allow_missing: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validate data structure against expected types
    
    Args:
        data: Data to validate
        expected_structure: Dictionary mapping keys to expected types
        allow_missing: Whether to allow missing keys
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, f"Data is not a dictionary: {type(data)}"
    
    for key, expected_type in expected_structure.items():
        if key not in data:
            if allow_missing:
                continue
            return False, f"Missing required key: {key}"
        
        if not isinstance(data[key], expected_type):
            return False, f"Invalid type for {key}: expected {expected_type}, got {type(data[key])}"
    
    return True, None


def clean_dataframe(df: pd.DataFrame, drop_missing_threshold: float = 0.5, 
                    fill_method: str = 'mean') -> pd.DataFrame:
    """
    Clean and prepare DataFrame for model training
    
    Args:
        df: Input DataFrame
        drop_missing_threshold: Drop columns with more than this fraction of missing values
        fill_method: Method to fill missing values ('mean', 'median', 'zero', or 'none')
        
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for cleaning")
        return df
    
    # Initial shape
    initial_shape = df.shape
    logger.info(f"Initial DataFrame shape: {initial_shape}")
    
    # Drop columns with excessive missing values
    missing_fractions = df.isnull().mean()
    cols_to_drop = missing_fractions[missing_fractions > drop_missing_threshold].index.tolist()
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} columns with > {drop_missing_threshold*100}% missing values")
        df = df.drop(columns=cols_to_drop)
    
    # Fill missing values based on specified method
    if fill_method != 'none' and not df.empty:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if fill_method == 'mean' and len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif fill_method == 'median' and len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif fill_method == 'zero':
            df = df.fillna(0)
    
    # Final shape
    final_shape = df.shape
    logger.info(f"Final DataFrame shape after cleaning: {final_shape}")
    
    return df


def generate_train_test_split(data: Union[pd.DataFrame, np.ndarray], 
                             test_size: float = 0.2, 
                             validation_size: Optional[float] = None, 
                             random_state: int = 42, 
                             chronological: bool = False,
                             date_column: Optional[str] = None) -> Tuple:
    """
    Generate train/test or train/validation/test split
    
    Args:
        data: DataFrame or ndarray to split
        test_size: Fraction of data for testing
        validation_size: Optional fraction for validation
        random_state: Random seed for reproducibility
        chronological: Whether to split data chronologically (requires DataFrame with date column)
        date_column: Name of date column for chronological split
        
    Returns:
        Tuple of (train_data, test_data) or (train_data, validation_data, test_data)
    """
    if isinstance(data, pd.DataFrame) and chronological and date_column:
        if date_column not in data.columns:
            logger.warning(f"Date column '{date_column}' not found in DataFrame. Falling back to random split.")
            return generate_train_test_split(data, test_size, validation_size, random_state, False, None)
        
        # Sort by date
        data = data.sort_values(date_column)
        
        # Calculate split indices
        total_size = len(data)
        test_idx = int(total_size * (1 - test_size))
        
        if validation_size is not None:
            val_idx = int(total_size * (1 - test_size - validation_size))
            train_data = data.iloc[:val_idx]
            validation_data = data.iloc[val_idx:test_idx]
            test_data = data.iloc[test_idx:]
            logger.info(f"Chronological split: train={len(train_data)}, validation={len(validation_data)}, test={len(test_data)}")
            return train_data, validation_data, test_data
        else:
            train_data = data.iloc[:test_idx]
            test_data = data.iloc[test_idx:]
            logger.info(f"Chronological split: train={len(train_data)}, test={len(test_data)}")
            return train_data, test_data
    else:
        # Use scikit-learn for random split
        from sklearn.model_selection import train_test_split
        
        if validation_size is not None:
            # Calculate adjusted test_fraction for sklearn
            test_fraction = test_size / (1 - validation_size)
            
            # First split: separate (train+val) from test
            train_val, test = train_test_split(data, test_size=test_size, random_state=random_state)
            
            # Second split: separate train from validation
            train, val = train_test_split(train_val, test_size=validation_size/(1-test_size), random_state=random_state)
            
            if isinstance(data, pd.DataFrame):
                logger.info(f"Random split: train={len(train)}, validation={len(val)}, test={len(test)}")
            else:
                logger.info(f"Random split: train={train.shape[0]}, validation={val.shape[0]}, test={test.shape[0]}")
            
            return train, val, test
        else:
            train, test = train_test_split(data, test_size=test_size, random_state=random_state)
            
            if isinstance(data, pd.DataFrame):
                logger.info(f"Random split: train={len(train)}, test={len(test)}")
            else:
                logger.info(f"Random split: train={train.shape[0]}, test={test.shape[0]}")
            
            return train, test


def normalize_features(X_train: np.ndarray, X_test: np.ndarray = None, method: str = 'standard') -> Tuple:
    """
    Normalize features using specified method
    
    Args:
        X_train: Training features to normalize
        X_test: Optional test features to normalize using same parameters
        method: Normalization method ('standard', 'minmax', or 'robust')
        
    Returns:
        Tuple of (normalized_X_train, scaler, normalized_X_test) or (normalized_X_train, scaler)
    """
    # Import appropriate scaler based on method
    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        logger.warning(f"Unknown normalization method '{method}'. Using standard scaling.")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    
    # Fit scaler on training data and transform
    X_train_normalized = scaler.fit_transform(X_train)
    
    if X_test is not None:
        # Transform test data using same scaler
        X_test_normalized = scaler.transform(X_test)
        return X_train_normalized, scaler, X_test_normalized
    else:
        return X_train_normalized, scaler


def handle_class_imbalance(X: np.ndarray, y: np.ndarray, method: str = 'smote', 
                          sampling_strategy: Union[str, float, dict] = 'auto',
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle class imbalance in training data
    
    Args:
        X: Feature matrix
        y: Target labels
        method: Resampling method ('smote', 'adasyn', 'random_oversampling', 'random_undersampling')
        sampling_strategy: Strategy for resampling
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (resampled_X, resampled_y)
    """
    # Check if we have imbalanced classes
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_ratio = min(class_counts) / max(class_counts)
    
    if class_ratio > 0.8:
        logger.info(f"Class ratio is {class_ratio:.2f}, which is relatively balanced. Skipping resampling.")
        return X, y
    
    logger.info(f"Class imbalance detected (ratio: {class_ratio:.2f}). Using {method} to balance classes.")
    
    try:
        if method == 'smote':
            from imblearn.over_sampling import SMOTE
            resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'adasyn':
            from imblearn.over_sampling import ADASYN
            resampler = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'random_oversampling':
            from imblearn.over_sampling import RandomOverSampler
            resampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'random_undersampling':
            from imblearn.under_sampling import RandomUnderSampler
            resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        else:
            logger.warning(f"Unknown resampling method '{method}'. Using SMOTE.")
            from imblearn.over_sampling import SMOTE
            resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        
        # Log resampling results
        _, new_counts = np.unique(y_resampled, return_counts=True)
        logger.info(f"Original class distribution: {dict(zip(unique_classes, class_counts))}")
        logger.info(f"Resampled class distribution: {dict(zip(unique_classes, new_counts))}")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logger.error(f"Error during class rebalancing: {str(e)}")
        logger.error(traceback.format_exc())
        return X, y


def calculate_feature_importance(model: Any, X: np.ndarray, feature_names: List[str] = None) -> pd.DataFrame:
    """
    Calculate and format feature importance for a trained model
    
    Args:
        model: Trained model with feature_importances_ attribute
        X: Feature matrix used for training
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importances
    """
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not support direct feature importance calculation")
        
        # Try to use permutation importance as fallback
        try:
            from sklearn.inspection import permutation_importance
            logger.info("Attempting to calculate permutation importance instead")
            
            # Generate dummy target data of appropriate shape
            if hasattr(model, 'predict'):
                y_dummy = model.predict(X)
                result = permutation_importance(model, X, y_dummy, n_repeats=10, random_state=42)
                importances = result.importances_mean
            else:
                logger.error("Model does not support prediction, cannot calculate permutation importance")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {str(e)}")
            return pd.DataFrame()
    else:
        importances = model.feature_importances_
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    elif len(feature_names) != X.shape[1]:
        logger.warning(f"Feature names count ({len(feature_names)}) doesn't match feature count ({X.shape[1]})")
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create DataFrame with importances
    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return importance_df


def save_numpy_arrays(arrays: Dict[str, np.ndarray], filepath: str) -> bool:
    """
    Save multiple numpy arrays to a single file
    
    Args:
        arrays: Dictionary mapping names to numpy arrays
        filepath: Path to save file
        
    Returns:
        Success status
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save arrays
        np.savez_compressed(filepath, **arrays)
        logger.info(f"Successfully saved {len(arrays)} arrays to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving numpy arrays: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def load_numpy_arrays(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load multiple numpy arrays from a single file
    
    Args:
        filepath: Path to load file from
        
    Returns:
        Dictionary mapping names to numpy arrays
    """
    try:
        loaded = np.load(filepath)
        result = {key: loaded[key] for key in loaded.files}
        logger.info(f"Successfully loaded {len(result)} arrays from {filepath}")
        return result
    except Exception as e:
        logger.error(f"Error loading numpy arrays: {str(e)}")
        logger.error(traceback.format_exc())
        return {}


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Generate a hash for a DataFrame to track data versions
    
    Args:
        df: DataFrame to hash
        
    Returns:
        Hash string
    """
    try:
        # Convert to string and hash
        df_str = pd.util.hash_pandas_object(df).sum()
        hash_str = hashlib.md5(str(df_str).encode()).hexdigest()
        return hash_str
    except Exception as e:
        logger.error(f"Error hashing DataFrame: {str(e)}")
        return f"error_{int(time.time())}"


def format_time_elapsed(seconds: float) -> str:
    """
    Format time elapsed in seconds to human-readable string
    
    Args:
        seconds: Time elapsed in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for logging
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        'cpu_count': psutil.cpu_count(logical=False),
        'logical_cpu_count': psutil.cpu_count(logical=True),
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Try to get GPU info if available
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available']:
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else ''
    except (ImportError, Exception):
        info['cuda_available'] = False
    
    return info


def validate_config(config: Dict[str, Any], template: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration dictionary against a template
    
    Args:
        config: Configuration to validate
        template: Template with expected keys and types
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    def _validate_section(config_section, template_section, path=""):
        for key, expected in template_section.items():
            current_path = f"{path}.{key}" if path else key
            
            # Check if key exists
            if key not in config_section:
                errors.append(f"Missing required key: {current_path}")
                continue
            
            # If expected is a dict, recursively validate
            if isinstance(expected, dict) and isinstance(config_section[key], dict):
                _validate_section(config_section[key], expected, current_path)
            
            # If expected is a tuple of (type, validator_func), check both
            elif isinstance(expected, tuple) and len(expected) == 2:
                expected_type, validator = expected
                
                # Check type
                if not isinstance(config_section[key], expected_type):
                    errors.append(f"Invalid type for {current_path}: expected {expected_type.__name__}, got {type(config_section[key]).__name__}")
                    continue
                
                # Apply validator
                valid, error = validator(config_section[key])
                if not valid:
                    errors.append(f"Validation failed for {current_path}: {error}")
            
            # If expected is just a type, check type
            elif isinstance(expected, type):
                if not isinstance(config_section[key], expected):
                    errors.append(f"Invalid type for {current_path}: expected {expected.__name__}, got {type(config_section[key]).__name__}")
    
    _validate_section(config, template)
    
    return len(errors) == 0, errors