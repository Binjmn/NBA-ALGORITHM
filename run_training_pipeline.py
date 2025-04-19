#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Algorithm Training Pipeline Runner

This script runs the complete NBA model training pipeline with proper error handling,
logging, and metrics tracking. It provides a CLI interface for running the pipeline
with different configuration options.

Usage:
    python run_training_pipeline.py --config path/to/config.json
    python run_training_pipeline.py --seasons 4 --target moneyline
"""

import os
import sys
import json
import argparse
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("nba_training")

# Import training pipeline components
from src.Model_training_pipeline.pipeline import ModelTrainingPipeline
from src.Model_training_pipeline.config import get_default_config


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command line arguments
    
    Returns:
        Dictionary of parsed arguments
    """
    parser = argparse.ArgumentParser(description="NBA Model Training Pipeline Runner")
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration JSON file"
    )
    
    parser.add_argument(
        "--seasons",
        type=int,
        default=4,
        help="Number of seasons to collect data for (default: 4)"
    )
    
    parser.add_argument(
        "--target",
        type=str,
        choices=["moneyline", "spread", "totals", "all"],
        default="all",
        help="Prediction target to train models for"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        choices=["random_forest", "gradient_boosting", "ensemble", "all"],
        default="all",
        help="Models to train"
    )
    
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results and models"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--compatibility-check",
        action="store_true",
        help="Enable backward compatibility checks with existing models"
    )
    
    return vars(parser.parse_args())


def load_config(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load configuration from file and/or command line arguments
    
    Args:
        args: Command line arguments
        
    Returns:
        Configuration dictionary
    """
    # Start with default config
    config = get_default_config()
    
    # Override with config file if provided
    if args.get("config"):
        config_path = args["config"]
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                
                # Deep merge configs
                for key, value in file_config.items():
                    if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                        config[key].update(value)
                    else:
                        config[key] = value
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Override with command line arguments
    if args.get("seasons"):
        config["data_collection"]["num_seasons"] = args["seasons"]
    
    if args.get("target") and args["target"] != "all":
        targets_map = {
            "moneyline": [{"name": "moneyline", "column": "home_win", "type": "classification"}],
            "spread": [{"name": "spread", "column": "spread_diff", "type": "regression"}],
            "totals": [{"name": "totals", "column": "total_points", "type": "regression"}]
        }
        config["training"]["target_types"] = targets_map.get(args["target"], config["training"]["target_types"])
    
    if args.get("models") and args["models"] != "all":
        config["training"]["models"] = [args["models"]]
    
    if args.get("tune"):
        for model in config["models"]:
            config["models"][model]["optimize_hyperparams"] = True
    
    if args.get("output"):
        config["paths"]["results_dir"] = args["output"]
        config["paths"]["models_dir"] = os.path.join(args["output"], "models")
    
    if args.get("verbose"):
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.get("compatibility_check"):
        config["results"]["check_backward_compatibility"] = True
    
    return config


def setup_environment(config: Dict[str, Any]) -> None:
    """
    Set up environment for training
    
    Args:
        config: Configuration dictionary
    """
    # Create output directories
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)
    os.makedirs(config["paths"]["models_dir"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Set environment variables from config
    if config["data_collection"]["api"].get("key"):
        os.environ["NBA_API_KEY"] = config["data_collection"]["api"]["key"]
    
    # Log key configuration settings
    logger.info(f"Training with {config['data_collection']['num_seasons']} seasons of data")
    logger.info(f"Target types: {[t['name'] for t in config['training']['target_types']]}")
    logger.info(f"Models: {config['training']['models']}")
    logger.info(f"Results directory: {config['paths']['results_dir']}")


def run_pipeline(config: Dict[str, Any]) -> None:
    """
    Run the training pipeline
    
    Args:
        config: Configuration dictionary
    """
    try:
        # Initialize the pipeline
        logger.info("Initializing training pipeline")
        pipeline = ModelTrainingPipeline(config)
        
        # Run the pipeline
        logger.info("Running training pipeline")
        results = pipeline.run()
        
        # Log results summary
        if results.get("status") == "success":
            logger.info("Pipeline completed successfully")
            logger.info(f"Trained {len(results.get('models_trained', []))} models")
            logger.info(f"Targets processed: {results.get('targets_processed', [])}")
            
            # Log evaluation metrics if available
            if "evaluation" in results:
                for model_id, metrics in results["evaluation"].items():
                    logger.info(f"Model {model_id} metrics: {metrics.get('metrics', {})}")
        else:
            logger.error(f"Pipeline failed: {results.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Unexpected error running pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "failed", "error": str(e)}


def main() -> None:
    """
    Main entry point
    """
    # Parse arguments and load config
    logger.info("Starting NBA training pipeline runner")
    start_time = datetime.now()
    
    args = parse_arguments()
    config = load_config(args)
    
    # Set up environment
    setup_environment(config)
    
    # Run the pipeline
    results = run_pipeline(config)
    
    # Finalize
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Pipeline run completed in {duration:.2f} seconds")
    
    # Print summary to console
    if results.get("status") == "success":
        print("\n=== NBA Model Training Pipeline - Run Summary ===")
        print(f"Status: {results['status']}")
        print(f"Models trained: {len(results.get('models_trained', []))}")
        print(f"Training data: {results.get('training_details', {}).get('total_samples', 0)} samples")
        print(f"Results saved to: {config['paths']['results_dir']}")
        print("==================================================\n")
    else:
        print("\n=== NBA Model Training Pipeline - Run Failed ===")
        print(f"Error: {results.get('error', 'Unknown error')}")
        print("Please check logs for details.")
        print("==============================================\n")


if __name__ == "__main__":
    main()
