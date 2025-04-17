#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full NBA Prediction Pipeline

This script runs the full pipeline:
1. Trains all models (if needed)
2. Generates predictions for today's games
3. Displays the results in a user-friendly format

Usage:
    python run_full_pipeline.py --props

Author: Cascade
Date: April 2025
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="NBA Full Prediction Pipeline")
    parser.add_argument(
        "--date",
        type=str,
        help="Date to generate predictions for (YYYY-MM-DD format). Defaults to today."
    )
    parser.add_argument(
        "--props",
        action="store_true",
        help="Include player prop predictions"
    )
    parser.add_argument(
        "--props-only",
        action="store_true",
        help="Show only player prop predictions"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save prediction results to files"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training and use existing models"
    )
    
    return parser.parse_args()


def run_model_training():
    """
    Run the model training pipeline
    
    Returns:
        bool: True if training succeeded, False otherwise
    """
    try:
        logger.info("Running model training pipeline...")
        from nba_algorithm.setup.train_models_no_db import train_all_models
        
        # Train all models
        results = train_all_models()
        if not results:
            logger.error("Model training failed. No results returned.")
            return False
        
        # Check if all models were trained successfully
        success_count = sum(1 for model, metrics in results.items() if metrics)
        logger.info(f"Successfully trained {success_count} models.")
        
        return success_count > 0
    
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def run_predictions(args):
    """
    Run predictions using the scripts in the scripts directory
    
    Args:
        args: Command-line arguments
        
    Returns:
        bool: True if predictions succeeded, False otherwise
    """
    try:
        logger.info("Running prediction pipeline...")
        # Construct command arguments for the prediction script
        cmd_args = [sys.executable, "scripts/user_friendly_predictions.py"]
        
        if args.date:
            cmd_args.extend(["--date", args.date])
        if args.props:
            cmd_args.append("--props")
        if args.props_only:
            cmd_args.append("--props-only")
        if args.verbose:
            cmd_args.append("--verbose")
        if args.no_save:
            cmd_args.append("--no-save")
        
        # Run the prediction script
        import subprocess
        logger.info(f"Executing command: {' '.join(cmd_args)}")
        
        process = subprocess.run(cmd_args, cwd=str(Path(__file__).parent))
        
        if process.returncode != 0:
            logger.error(f"Prediction script failed with return code {process.returncode}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error running predictions: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main():
    """
    Main function to run the full NBA prediction pipeline
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Run model training unless skipped
        if not args.skip_training:
            training_success = run_model_training()
            if not training_success:
                logger.error("Model training failed. Cannot proceed with predictions.")
                return 1
        else:
            logger.info("Skipping model training as requested.")
        
        # Run predictions
        prediction_success = run_predictions(args)
        if not prediction_success:
            logger.error("Prediction generation failed.")
            return 1
        
        logger.info("Full prediction pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
