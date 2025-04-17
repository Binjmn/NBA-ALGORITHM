#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Prediction System Scheduler

This script serves as the main entry point for running the NBA prediction system
scheduler in production environments. It will maintain the defined schedule:

- 6:00 AM Daily: Generate first predictions for games and props
- 2:00 AM Daily: Track 7-day prediction accuracy 
- Every 4 Hours (10:00 AM, 2:00 PM, 6:00 PM): Update stats and odds
- Every 15 Minutes in 2-Hour Pre-Game Windows: Refresh injuries
- Every 10 Minutes During Games: Update live scores and odds
- Hourly: Track CLV

Usage:
    python run_scheduler.py [--background]

Options:
    --background    Run in background mode (non-blocking)
    --blocking      Run in blocking mode (default)

Author: Cascade
Date: April 2025
"""

import sys
import argparse
import logging
import time
from datetime import datetime

from nba_algorithm.scheduling import start_scheduler, shutdown_scheduler
from nba_algorithm.utils.logger import setup_logging

# Set up logging
logger = setup_logging("scheduler_main")


def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="NBA Prediction System Scheduler")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--background", 
        action="store_true", 
        help="Run in background mode (non-blocking)"
    )
    group.add_argument(
        "--blocking", 
        action="store_true", 
        default=True, 
        help="Run in blocking mode (default)"
    )
    return parser.parse_args()


def main():
    """
    Main entry point for the scheduler
    """
    args = parse_arguments()
    
    logger.info("Starting NBA Prediction System Scheduler")
    logger.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Start the scheduler in the appropriate mode
        background_mode = args.background
        blocking_mode = not background_mode
        
        if background_mode:
            logger.info("Starting scheduler in background mode")
        else:
            logger.info("Starting scheduler in blocking mode")
        
        start_scheduler(background=background_mode)
        
        # If we're in background mode, keep the script running
        if background_mode:
            logger.info("Scheduler running in background. Press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received. Shutting down.")
                shutdown_scheduler()
    except Exception as e:
        logger.error(f"Error in scheduler main: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
