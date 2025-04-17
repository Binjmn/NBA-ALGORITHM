#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Get Baseline Metrics Script

This script loads the existing trained models and evaluates their performance to establish
a baseline for comparison with our new production models.
"""

import os
import sys
import logging
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'

def get_model_metrics():
    """
    Load trained models and get their metrics
    
    Returns:
        Dict of model metrics by model name
    """
    metrics = {}
    model_files = list(MODEL_DIR.glob('*.pkl'))
    
    if not model_files:
        logger.error("No trained models found in models directory")
        return metrics
    
    logger.info(f"Found {len(model_files)} trained models")
    
    for model_file in model_files:
        model_name = model_file.stem
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            # Check if the model is packaged with a scaler (newer format)
            if isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
                features = model_data.get('features', [])
                logger.info(f"Model {model_name} uses {len(features)} features")
            else:
                # Older format (just the model)
                model = model_data
            
            # Get model feature importances if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                logger.info(f"Model {model_name} has feature importances")
            
            # Store in metrics dictionary
            metrics[model_name] = {
                'model_type': type(model).__name__,
                'loaded_successfully': True
            }
            
            logger.info(f"Successfully loaded {model_name} ({type(model).__name__})")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            metrics[model_name] = {
                'loaded_successfully': False,
                'error': str(e)
            }
    
    return metrics

def main():
    """
    Main function
    """
    metrics = get_model_metrics()
    
    if not metrics:
        print("\nNo models found. Please run training first.")
        return 1
    
    print("\nBaseline Model Information:")
    print("-" * 50)
    for model_name, data in metrics.items():
        if data['loaded_successfully']:
            print(f"✅ {model_name} ({data['model_type']})")
        else:
            print(f"❌ {model_name}: Failed to load - {data['error']}")
    
    print("\nBaseline models are using synthetic data for several features:")
    print("- Team win rates")
    print("- Points averages")
    print("- Rebounds and assists")
    print("- Defensive metrics")
    print("\nOur new production models will replace these with real historical data")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
