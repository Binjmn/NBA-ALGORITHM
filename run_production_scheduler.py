#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Production Scheduler Runner

This script sets up and starts the production scheduler to automatically generate
predictions using our new production models without synthetic data.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Import scheduler module
from nba_algorithm.scheduling.scheduler import start_scheduler, configure_scheduled_jobs, get_scheduler
from nba_algorithm.utils.logger import setup_logging

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"

setup_logging(log_file, log_level="INFO")
logger = logging.getLogger(__name__)


def setup_production_scheduler():
    """
    Set up the production scheduler with our new models
    """
    logger.info("Setting up production scheduler with real data models")
    
    # Configure the models to be used by the scheduler
    model_config = {
        "use_production_models": True,
        "game_prediction": {
            "primary": "production_gradient_boosting.pkl",
            "secondary": "production_random_forest.pkl"
        },
        "player_props": {
            "points": "gradient_boost_points.pkl",
            "rebounds": "gradient_boost_rebounds.pkl",
            "assists": "gradient_boost_assists.pkl"
        }
    }
    
    # Save the model configuration for the scheduler to use
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "prediction_models.json"
    
    import json
    with open(config_file, "w") as f:
        json.dump(model_config, f, indent=4)
    
    logger.info(f"Saved model configuration to {config_file}")
    
    # Get the scheduler
    scheduler = get_scheduler(background=True)
    
    # Configure the scheduled jobs
    configure_scheduled_jobs(scheduler)
    
    return scheduler


def main():
    """
    Main function to run the production scheduler
    """
    print("NBA Algorithm Production Scheduler")
    print("==================================")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Using production models with real data (no synthetic data)")
    print("")
    print("Schedule Information:")
    print("- Generate predictions: Daily at 6:00 AM EST")
    print("- Update predictions: 10:00 AM, 2:00 PM, 6:00 PM EST")
    print("- Track accuracy: Daily at 2:00 AM EST")
    print("- Retrain models: Every other Monday at 3:00 AM EST")
    print("")
    print("Press Ctrl+C to stop the scheduler")
    print("")
    
    try:
        # Set up the scheduler
        scheduler = setup_production_scheduler()
        
        # Start the scheduler (blocking call)
        scheduler.start()
        
        return 0
    except KeyboardInterrupt:
        print("\nScheduler stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error running scheduler: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
