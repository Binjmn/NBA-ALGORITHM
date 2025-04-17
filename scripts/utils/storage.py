#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Storage Utilities Module

This module provides file storage and retrieval functions for the NBA prediction system.

Author: Cascade
Date: April 2025
"""

import os
import json
import logging
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd

from ..config import OUTPUT_DIR

# Configure logger
logger = logging.getLogger(__name__)


def save_prediction_results(results: Any, base_filename: str) -> bool:
    """
    Save prediction results to CSV and JSON files with proper error handling
    
    Args:
        results: Prediction results (DataFrame or dictionary)
        base_filename: Base filename for output files
        
    Returns:
        True if saving was successful, False otherwise
    """
    try:
        # Create timestamps and filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = OUTPUT_DIR / f"{base_filename}_{timestamp}.csv"
        json_filename = OUTPUT_DIR / f"{base_filename}_{timestamp}.json"
        
        # Ensure the output directory exists
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        
        # Convert results to DataFrame if it's a dictionary
        if isinstance(results, dict):
            if all(isinstance(value, dict) for value in results.values()):
                # Convert dictionary of dictionaries to DataFrame
                df = pd.DataFrame.from_dict(results, orient='index')
            else:
                # Handle special case for player props or other nested structures
                df = pd.json_normalize(results)
        elif isinstance(results, pd.DataFrame):
            df = results
        else:
            logger.error(f"Unsupported results type: {type(results)}")
            return False
        
        # Save as CSV
        try:
            df.to_csv(csv_filename, index=False)
            logger.info(f"Successfully saved prediction results to CSV: {csv_filename}")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {str(e)}")
            return False
        
        # Save as JSON
        try:
            if isinstance(results, dict):
                # Save original dictionary structure to preserve all data
                with open(json_filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            else:
                # Convert DataFrame to dictionary and save
                results_dict = df.to_dict(orient='records')
                with open(json_filename, 'w') as f:
                    json.dump(results_dict, f, indent=2, default=str)
            
            logger.info(f"Successfully saved prediction results to JSON: {json_filename}")
        except Exception as e:
            logger.error(f"Error saving results to JSON: {str(e)}")
            return False
        
        # Return success if we got this far
        return True
    
    except Exception as e:
        logger.error(f"Error saving prediction results: {str(e)}")
        return False


def load_cached_data(cache_file: Union[str, Path], max_age_hours: int = 24) -> Optional[Any]:
    """
    Load data from a cache file if it exists and is not too old
    
    Args:
        cache_file: Path to the cache file
        max_age_hours: Maximum age of the cache file in hours
        
    Returns:
        Cached data or None if cache is invalid or too old
    """
    try:
        cache_path = Path(cache_file) if isinstance(cache_file, str) else cache_file
        
        if not cache_path.exists():
            return None
        
        # Check the file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if file_age.total_seconds() > max_age_hours * 3600:
            logger.info(f"Cache file {cache_path} is too old ({file_age.total_seconds() / 3600:.1f} hours)")
            return None
        
        # Load the data based on file extension
        if cache_path.suffix.lower() == '.json':
            with open(cache_path, 'r') as f:
                data = json.load(f)
        elif cache_path.suffix.lower() == '.csv':
            data = pd.read_csv(cache_path)
        else:
            logger.warning(f"Unsupported cache file type: {cache_path.suffix}")
            return None
        
        logger.info(f"Successfully loaded cached data from {cache_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading cached data from {cache_file}: {str(e)}")
        return None


def save_to_cache(data: Any, cache_file: Union[str, Path]) -> bool:
    """
    Save data to a cache file with proper error handling
    
    Args:
        data: Data to save (dictionary, list, or DataFrame)
        cache_file: Path to the cache file
        
    Returns:
        True if saving was successful, False otherwise
    """
    try:
        cache_path = Path(cache_file) if isinstance(cache_file, str) else cache_file
        
        # Create parent directories if they don't exist
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save based on file extension
        if cache_path.suffix.lower() == '.json':
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif cache_path.suffix.lower() == '.csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(cache_path, index=False)
            else:
                # Try to convert to DataFrame
                pd.DataFrame(data).to_csv(cache_path, index=False)
        else:
            logger.warning(f"Unsupported cache file type: {cache_path.suffix}")
            return False
        
        logger.info(f"Successfully saved data to cache: {cache_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving data to cache {cache_file}: {str(e)}")
        return False


def clean_old_cache_files(cache_dir: Union[str, Path], max_age_days: int = 7) -> int:
    """
    Clean up old cache files to prevent disk space issues
    
    Args:
        cache_dir: Directory containing cache files
        max_age_days: Maximum age of cache files in days
        
    Returns:
        Number of files deleted
    """
    try:
        cache_path = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        
        if not cache_path.exists() or not cache_path.is_dir():
            logger.warning(f"Cache directory {cache_path} does not exist or is not a directory")
            return 0
        
        # Get all cache files
        cache_files = list(cache_path.glob("*.json")) + list(cache_path.glob("*.csv"))
        
        # Check each file's age
        deleted_count = 0
        for file in cache_files:
            file_age = datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)
            if file_age.days > max_age_days:
                logger.info(f"Deleting old cache file {file} (age: {file_age.days} days)")
                try:
                    file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {file}: {str(e)}")
        
        logger.info(f"Deleted {deleted_count} old cache files from {cache_path}")
        return deleted_count
    
    except Exception as e:
        logger.error(f"Error cleaning old cache files: {str(e)}")
        return 0
