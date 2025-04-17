# -*- coding: utf-8 -*-
"""
Persistence Module

This module provides functions for saving prediction outputs to various formats,
including CSV and JSON files.
"""

import os
import json
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)


def save_predictions(predictions_df, output_dir="predictions", prediction_date=None, 
                    prefix="nba_predictions") -> Tuple[str, str]:
    """
    Save predictions to CSV and JSON files
    
    Args:
        predictions_df: DataFrame of predictions
        output_dir: Directory to save files to
        prediction_date: Date string for the predictions
        prefix: Prefix for the output filenames
        
    Returns:
        Tuple[str, str]: Paths to the saved CSV and JSON files
        
    Raises:
        RuntimeError: If predictions cannot be saved
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Format date for filenames
        if prediction_date is None:
            prediction_date = datetime.now().strftime("%Y-%m-%d")
            
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filenames
        csv_filename = f"{prefix}_{prediction_date}_{timestamp}.csv"
        json_filename = f"{prefix}_{prediction_date}_{timestamp}.json"
        
        csv_path = os.path.join(output_dir, csv_filename)
        json_path = os.path.join(output_dir, json_filename)
        
        # Save to CSV
        predictions_df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")
        
        # Save to JSON (with date handling)
        json_data = predictions_df.to_dict(orient="records")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)  # Use default=str to handle dates
        logger.info(f"Saved predictions to {json_path}")
        
        return csv_path, json_path
        
    except Exception as e:
        logger.error(f"Failed to save predictions: {str(e)}")
        raise RuntimeError(f"Failed to save predictions: {str(e)}")


def save_prediction_schema(prediction_data, output_dir="predictions", 
                          prediction_date=None, filename=None) -> str:
    """
    Save the full prediction schema to a JSON file
    
    Args:
        prediction_data: Dictionary with structured prediction data
        output_dir: Directory to save file to
        prediction_date: Date string for the predictions
        filename: Optional filename override
        
    Returns:
        str: Path to the saved JSON file
        
    Raises:
        RuntimeError: If schema cannot be saved
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Format date for filenames
        if prediction_date is None:
            prediction_date = datetime.now().strftime("%Y-%m-%d")
            
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename
        if filename is None:
            filename = f"nba_prediction_schema_{prediction_date}_{timestamp}.json"
        
        json_path = os.path.join(output_dir, filename)
        
        # Save to JSON
        with open(json_path, "w") as f:
            json.dump(prediction_data, f, indent=2, default=str)  # Use default=str to handle dates
        logger.info(f"Saved prediction schema to {json_path}")
        
        return json_path
        
    except Exception as e:
        logger.error(f"Failed to save prediction schema: {str(e)}")
        raise RuntimeError(f"Failed to save prediction schema: {str(e)}")


def load_saved_predictions(file_path) -> pd.DataFrame:
    """
    Load saved predictions from a file
    
    Args:
        file_path: Path to the predictions file (CSV or JSON)
        
    Returns:
        pandas.DataFrame: Loaded predictions
        
    Raises:
        ValueError: If the file format is not supported or file cannot be loaded
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == ".csv":
            return pd.read_csv(file_path)
        elif file_extension == ".json":
            with open(file_path, "r") as f:
                json_data = json.load(f)
            return pd.DataFrame(json_data)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
    except Exception as e:
        logger.error(f"Failed to load predictions from {file_path}: {str(e)}")
        raise ValueError(f"Failed to load predictions: {str(e)}")
