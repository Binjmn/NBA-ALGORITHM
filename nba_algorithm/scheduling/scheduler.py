# -*- coding: utf-8 -*-
"""
NBA Prediction System Scheduler

This module implements a production-ready scheduling system for the NBA prediction
algorithm, ensuring that predictions are generated and updated on the specified
schedules.

Schedule (All Times in EST):
- 6:00 AM Daily: Generate first predictions for games and props
- 2:00 AM Daily: Track 7-day prediction accuracy 
- Every 4 Hours (10:00 AM, 2:00 PM, 6:00 PM): Update stats and odds
- Every 15 Minutes in 2-Hour Pre-Game Windows (Hourly otherwise): Refresh injuries
- Every 10 Minutes During Games: Update live scores and odds
- Hourly: Track CLV

Author: Cascade
Date: April, 2025
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from pytz import timezone

# Import prediction modules
from nba_algorithm.scripts.production_prediction import run_production_prediction
from nba_algorithm.data.game_data import fetch_nba_games
from nba_algorithm.data.team_data import fetch_team_stats
from nba_algorithm.data.odds_data import fetch_betting_odds
from nba_algorithm.data.player_data import fetch_player_stats, check_player_injuries
from nba_algorithm.utils.season_manager import get_season_manager
from nba_algorithm.utils.logger import setup_logging
from nba_algorithm.utils.performance import track_prediction_accuracy, track_clv
from nba_algorithm.models.trainer import train_models_if_needed

# Set up logging
logger = setup_logging("scheduler")

# Define EST timezone for scheduling
EST = timezone('US/Eastern')

# Singleton scheduler instance
_scheduler = None


def get_scheduler(background=True):
    """
    Get or create the scheduler instance
    
    Args:
        background (bool): If True, use a background scheduler, otherwise blocking
        
    Returns:
        apscheduler.schedulers.base.BaseScheduler: The scheduler instance
    """
    global _scheduler
    
    if _scheduler is None:
        # Job stores
        jobstores = {
            'default': MemoryJobStore(),
        }
        
        # Executors
        executors = {
            'default': ThreadPoolExecutor(20),
            'processpool': ProcessPoolExecutor(5)
        }
        
        # Job defaults
        job_defaults = {
            'coalesce': False,
            'max_instances': 3,
            'misfire_grace_time': 60
        }
        
        # Create the appropriate scheduler
        if background:
            _scheduler = BackgroundScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=EST
            )
        else:
            _scheduler = BlockingScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=EST
            )
            
        # Configure all the scheduled jobs
        configure_scheduled_jobs(_scheduler)
        
    return _scheduler


def configure_scheduled_jobs(scheduler):
    """
    Configure all the scheduled jobs according to the defined schedule
    
    Args:
        scheduler: The APScheduler instance to configure
    """
    # 6:00 AM Daily - Generate first predictions
    scheduler.add_job(
        generate_daily_predictions,
        CronTrigger(hour=6, minute=0, timezone=EST),
        id='daily_predictions',
        name='Generate Daily Predictions',
        replace_existing=True
    )
    
    # 2:00 AM Daily - Track 7-day prediction accuracy
    scheduler.add_job(
        track_weekly_accuracy,
        CronTrigger(hour=2, minute=0, timezone=EST),
        id='track_accuracy',
        name='Track Weekly Prediction Accuracy',
        replace_existing=True
    )
    
    # Every 4 Hours (10:00 AM, 2:00 PM, 6:00 PM) - Update team/player stats and odds
    for hour in [10, 14, 18]:
        scheduler.add_job(
            update_stats_and_odds,
            CronTrigger(hour=hour, minute=0, timezone=EST),
            id=f'update_stats_{hour}',
            name=f'Update Stats and Odds at {hour}:00',
            replace_existing=True
        )
    
    # Every hour - Track CLV
    scheduler.add_job(
        track_hourly_clv,
        IntervalTrigger(hours=1, timezone=EST),
        id='track_clv',
        name='Track Closing Line Value',
        replace_existing=True
    )
    
    # Dynamic scheduling for pre-game and in-game updates
    scheduler.add_job(
        update_dynamic_schedules,
        IntervalTrigger(minutes=30, timezone=EST),
        id='update_dynamic_schedules',
        name='Update Dynamic Schedules',
        replace_existing=True
    )


def update_dynamic_schedules():
    """
    Update dynamic schedules based on game times
    
    This function fetches today's games and adjusts the schedule for:
    - Pre-game injury updates (every 15 min in 2-hour pre-game window, hourly otherwise)
    - In-game updates (every 10 min during games)
    """
    try:
        scheduler = get_scheduler()
        today = datetime.now(EST).strftime('%Y-%m-%d')
        
        # Fetch today's games
        games = fetch_nba_games(date=today)
        if not games:
            logger.info("No games found for today. Maintaining default schedules.")
            return
        
        # Current time in EST
        now = datetime.now(EST)
        
        # Remove existing dynamic jobs
        for job_id in list(scheduler.get_jobs()):
            if job_id.id.startswith('injury_') or job_id.id.startswith('in_game_'):
                scheduler.remove_job(job_id.id)
        
        # Create new dynamic jobs based on game times
        for game in games:
            game_time_str = game.get('status', {}).get('scheduled')
            if not game_time_str:
                continue
                
            # Convert game time to datetime object in EST
            game_time = datetime.fromisoformat(game_time_str.replace('Z', '+00:00')).astimezone(EST)
            
            # Define pre-game window (2 hours before game)
            pre_game_start = game_time - timedelta(hours=2)
            
            # Define game window (estimated 3 hours duration)
            game_end = game_time + timedelta(hours=3)
            
            # Current status relative to the game
            game_id = game.get('id')
            
            # Schedule pre-game injury updates
            if now < game_time:
                # If we're in the 2-hour pre-game window, check injuries every 15 minutes
                if now >= pre_game_start:
                    scheduler.add_job(
                        check_player_injuries,
                        IntervalTrigger(minutes=15, timezone=EST, end_date=game_time),
                        id=f'injury_pregame_{game_id}',
                        args=[game_id],
                        name=f'Pre-game Injury Updates for Game {game_id}',
                        replace_existing=True
                    )
                else:
                    # Otherwise check hourly until we hit the pre-game window
                    scheduler.add_job(
                        check_player_injuries,
                        IntervalTrigger(hours=1, timezone=EST, end_date=pre_game_start),
                        id=f'injury_hourly_{game_id}',
                        args=[game_id],
                        name=f'Hourly Injury Updates for Game {game_id}',
                        replace_existing=True
                    )
            
            # Schedule in-game updates (if game is currently in progress)
            if now >= game_time and now <= game_end:
                scheduler.add_job(
                    update_live_game,
                    IntervalTrigger(minutes=10, timezone=EST, end_date=game_end),
                    id=f'in_game_{game_id}',
                    args=[game_id],
                    name=f'In-game Updates for Game {game_id}',
                    replace_existing=True
                )
                
        logger.info(f"Updated dynamic schedules for {len(games)} games")
    except Exception as e:
        logger.error(f"Error updating dynamic schedules: {str(e)}")


def generate_daily_predictions():
    """
    Generate daily predictions at 6:00 AM EST
    
    This function:
    1. Checks if the current date is within an NBA season
    2. Trains models if needed based on recent performance
    3. Generates comprehensive predictions for today's games
    4. Outputs predictions to file and database
    """
    try:
        logger.info("Generating daily predictions")
        
        # Check if we're in an NBA season using the season manager
        season_manager = get_season_manager()
        season_info = season_manager.get_current_season_info()
        
        if season_info['phase'] == 'off_season':
            logger.info("Currently in NBA off-season. Skipping daily predictions.")
            return
        
        # Train models if needed based on performance metrics
        train_models_if_needed()
        
        # Run the main prediction function
        run_production_prediction()
        
        logger.info("Daily predictions generated successfully")
    except Exception as e:
        logger.error(f"Error generating daily predictions: {str(e)}")


def track_weekly_accuracy():
    """
    Track 7-day prediction accuracy at 2:00 AM EST daily
    
    This function analyzes the accuracy of predictions from the past 7 days,
    including moneyline, spread, totals, and player props.
    """
    try:
        logger.info("Tracking 7-day prediction accuracy")
        # Calculate the date range (7 days ago to yesterday)
        end_date = datetime.now(EST).date() - timedelta(days=1)
        start_date = end_date - timedelta(days=7)
        
        # Track accuracy for different prediction types
        accuracy_metrics = track_prediction_accuracy(
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"Prediction accuracy tracked successfully: {accuracy_metrics}")
    except Exception as e:
        logger.error(f"Error tracking prediction accuracy: {str(e)}")


def update_stats_and_odds():
    """
    Update team/player stats and odds, recalculate predictions if significant changes
    
    This function runs at 10:00 AM, 2:00 PM, and 6:00 PM EST daily.
    It checks for significant changes in odds (>5%) and updates predictions accordingly.
    """
    try:
        logger.info("Updating team/player stats and odds")
        
        # Fetch updated data
        games = fetch_nba_games()
        team_stats = fetch_team_stats()
        betting_odds = fetch_betting_odds(games)
        significant_changes = check_for_significant_changes(betting_odds)
        
        # If significant changes detected, regenerate predictions
        if significant_changes:
            logger.info("Significant odds changes detected. Regenerating predictions.")
            run_production_prediction()
        else:
            logger.info("No significant changes detected. Skipping prediction update.")
    except Exception as e:
        logger.error(f"Error updating stats and odds: {str(e)}")


def check_for_significant_changes(current_odds):
    """
    Check if there are significant changes in odds
    
    Args:
        current_odds: Current betting odds data
        
    Returns:
        bool: True if significant changes detected (>5% shift), False otherwise
    """
    try:
        # Load previous odds from storage
        from nba_algorithm.utils.storage import load_previous_odds
        previous_odds = load_previous_odds()
        
        if not previous_odds:
            logger.info("No previous odds data found for comparison")
            return False
        
        # Check for significant changes (>5% shift in any market)
        threshold = 0.05
        for game_id, markets in current_odds.items():
            if game_id not in previous_odds:
                continue
            
            for market, odds in markets.items():
                if market not in previous_odds[game_id]:
                    continue
                
                previous = previous_odds[game_id][market]
                
                # Calculate percentage change for moneyline, spread, or totals
                if isinstance(odds, dict) and isinstance(previous, dict):
                    for key in odds:
                        if key in previous and previous[key] != 0:
                            change = abs((odds[key] - previous[key]) / previous[key])
                            if change > threshold:
                                logger.info(f"Significant odds change detected for game {game_id}, market {market}: {change:.2%}")
                                return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking for significant odds changes: {str(e)}")
        return False


def update_live_game(game_id):
    """
    Update live scores and odds for a specific game
    
    This function runs every 10 minutes while a game is in progress.
    
    Args:
        game_id: ID of the game to update
    """
    try:
        logger.info(f"Updating live data for game {game_id}")
        
        # Fetch live game data
        from nba_algorithm.data.live_data import fetch_live_game_data
        live_data = fetch_live_game_data(game_id)
        
        # Fetch current odds
        from nba_algorithm.data.odds_data import fetch_live_odds
        live_odds = fetch_live_odds(game_id)
        
        # Generate live predictions
        from nba_algorithm.predictions.live_prediction import generate_live_predictions
        live_predictions = generate_live_predictions(game_id, live_data, live_odds)
        
        logger.info(f"Live predictions updated for game {game_id}")
    except Exception as e:
        logger.error(f"Error updating live game data for game {game_id}: {str(e)}")


def track_hourly_clv():
    """
    Track Closing Line Value hourly
    
    This function compares opening odds to current odds to calculate betting advantages.
    """
    try:
        logger.info("Tracking Closing Line Value")
        
        # Track CLV for today's games
        clv_metrics = track_clv()
        
        logger.info(f"CLV tracked successfully: {clv_metrics}")
    except Exception as e:
        logger.error(f"Error tracking CLV: {str(e)}")


def start_scheduler(background=True):
    """
    Start the scheduler
    
    Args:
        background (bool): If True, run in background mode, otherwise blocking
    """
    scheduler = get_scheduler(background=background)
    
    try:
        scheduler.start()
        logger.info("Scheduler started successfully")
    except (KeyboardInterrupt, SystemExit):
        shutdown_scheduler()
        raise
    except Exception as e:
        logger.error(f"Error starting scheduler: {str(e)}")
        raise


def shutdown_scheduler():
    """
    Shutdown the scheduler gracefully
    """
    global _scheduler
    
    if _scheduler and _scheduler.running:
        logger.info("Shutting down scheduler...")
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler shut down successfully")
