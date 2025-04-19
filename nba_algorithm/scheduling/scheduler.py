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
- Sunday at 1:00 AM: Weekly feature evolution analysis
- Every other Monday at 3:00 AM: Bi-weekly model retraining
- Daily at 5:00 AM: Daily performance decline check

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

# Import our new advanced systems
from nba_algorithm.utils.feature_evolution import FeatureEvolution
from nba_algorithm.models.model_registry import ModelRegistry
from nba_algorithm.utils.performance_tracker import PerformanceTracker

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
    
    # Sunday at 1:00 AM - Weekly feature evolution analysis
    scheduler.add_job(
        run_weekly_feature_evolution,
        CronTrigger(day_of_week='sun', hour=1, minute=0, timezone=EST),
        id='feature_evolution',
        name='Weekly Feature Evolution Analysis',
        replace_existing=True
    )
    
    # Every other Monday at 3:00 AM - Bi-weekly model retraining
    scheduler.add_job(
        retrain_models_with_feature_evolution,
        CronTrigger(day_of_week='mon', hour=3, minute=0, timezone=EST),
        id='model_retraining',
        name='Bi-weekly Model Retraining',
        replace_existing=True
    )
    
    # Daily performance decline check at 5:00 AM
    scheduler.add_job(
        check_model_performance,
        CronTrigger(hour=5, minute=0, timezone=EST),
        id='performance_check',
        name='Daily Model Performance Check',
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


def run_weekly_feature_evolution():
    """
    Run weekly feature evolution analysis
    
    This function:
    1. Analyzes recent game and player data for pattern changes
    2. Discovers and evaluates new potential features
    3. Updates the feature evolution system with findings
    """
    logger.info("Starting weekly feature evolution analysis")
    
    try:
        # Initialize feature evolution system
        feature_evolution = FeatureEvolution()
        
        # Get recent and historical data for analysis
        today = datetime.now(EST).strftime('%Y-%m-%d')
        recent_days = 14  # Last two weeks for current patterns
        historical_days = 90  # Last ~3 months for historical context
        
        logger.info(f"Fetching recent data ({recent_days} days) and historical data ({historical_days} days)")
        recent_data = fetch_historical_games(days=recent_days, end_date=today)
        historical_data = fetch_historical_games(days=historical_days, end_date=today)
        
        if recent_data.empty or historical_data.empty:
            logger.warning("Insufficient data for feature evolution analysis")
            return
        
        # Detect changes in league patterns
        logger.info("Analyzing changes in NBA patterns")
        prediction_types = ["moneyline", "spread", "total", "player_props"]
        
        for prediction_type in prediction_types:
            # Get current feature set
            current_features = feature_evolution.get_production_feature_set(prediction_type)
            
            # Detect changes in league patterns
            changes = feature_evolution.detect_league_changes(
                current_data=recent_data,
                historical_data=historical_data,
                features=current_features,
                threshold=0.1  # 10% change is significant
            )
            
            if changes:
                logger.info(f"Detected {len(changes)} significant changes for {prediction_type} prediction")
                
                # Discover potential new features based on these changes
                new_candidates = feature_evolution.discover_candidate_features(
                    data=recent_data,
                    existing_features=current_features,
                    prediction_type=prediction_type
                )
                
                if new_candidates:
                    logger.info(f"Discovered {len(new_candidates)} candidate features for {prediction_type}")
                    
                    # Engineer the new features
                    enhanced_data = feature_evolution.engineer_discovered_features(
                        data=recent_data,
                        candidates=new_candidates,
                        existing_features=current_features
                    )
                    
                    if not enhanced_data.empty:
                        # Determine target variable based on prediction type
                        target_var = 'result'
                        is_classification = prediction_type in ["moneyline"]
                        
                        if target_var in enhanced_data.columns:
                            # Compare the new feature set with current production features
                            logger.info(f"Evaluating new feature set for {prediction_type}")
                            comparison = feature_evolution.compare_feature_sets(
                                base_features=current_features,
                                candidate_features=list(enhanced_data.columns),
                                data=enhanced_data,
                                target=enhanced_data[target_var],
                                prediction_type=prediction_type,
                                is_classification=is_classification
                            )
                            
                            if comparison["is_better"]:
                                # Register the improved feature set
                                logger.info(f"Found improved feature set for {prediction_type} with " +
                                           f"{len(comparison['candidate_set']['features'])} features")
                                
                                # Register the new feature set
                                feature_set_id = feature_evolution.register_feature_set(
                                    prediction_type=prediction_type,
                                    features=comparison["candidate_set"]["features"],
                                    performance_metrics=comparison["candidate_set"]["metrics"],
                                    model_id="pending",  # Will be set when model is trained
                                    description=f"Evolved feature set from {today}"
                                )
                                
                                # Set as production feature set
                                feature_evolution.set_production_feature_set(prediction_type, feature_set_id)
                                logger.info(f"Set new feature set as production for {prediction_type}")
                            else:
                                logger.info(f"Current feature set for {prediction_type} is still optimal")
                        else:
                            logger.warning(f"Target variable '{target_var}' not found in data")
        
        # Generate a report on feature evolution
        report = feature_evolution.generate_report()
        logger.info("Feature evolution analysis completed - Report generated")
        
        # TODO: Save report to file or send as notification
        
    except Exception as e:
        logger.error(f"Error in feature evolution process: {str(e)}")


def retrain_models_with_feature_evolution():
    """
    Retrain models using the latest optimized feature sets
    
    This function:
    1. Checks if retraining is needed based on performance and data drift
    2. Uses optimized feature sets from the feature evolution system
    3. Retrains models and registers them with the model registry
    """
    logger.info("Starting bi-weekly model retraining process")
    
    try:
        # Initialize our systems
        feature_evolution = FeatureEvolution()
        model_registry = ModelRegistry()
        performance_tracker = PerformanceTracker()
        
        # Get current date in EST timezone
        today = datetime.now(EST).strftime('%Y-%m-%d')
        
        # Determine which models need retraining
        prediction_types = ["moneyline", "spread", "total", "player_props"]
        models_to_retrain = []
        
        for pred_type in prediction_types:
            # Check if performance has declined
            if performance_tracker.detect_performance_decline(pred_type):
                logger.info(f"Performance decline detected for {pred_type} model - scheduling retraining")
                models_to_retrain.append(pred_type)
                continue
            
            # Check if feature set has changed since last training
            current_features = feature_evolution.get_production_feature_set(pred_type)
            current_model = model_registry.get_production_model(pred_type)
            
            if current_model:
                # Get model metadata to check features used during training
                model_info = model_registry.get_model_by_id(model_registry.production_models[pred_type]["id"])
                if "metadata" in model_info and "features" in model_info["metadata"]:
                    trained_features = set(model_info["metadata"]["features"])
                    current_features_set = set(current_features)
                    
                    # If feature sets differ significantly, retrain
                    feature_diff = len(trained_features.symmetric_difference(current_features_set))
                    feature_total = len(trained_features.union(current_features_set))
                    feature_change_ratio = feature_diff / feature_total if feature_total > 0 else 0
                    
                    if feature_change_ratio > 0.1:  # More than 10% change in features
                        logger.info(f"Feature set for {pred_type} has changed significantly - scheduling retraining")
                        models_to_retrain.append(pred_type)
                        continue
            else:
                # No production model exists, definitely needs training
                logger.info(f"No production model found for {pred_type} - scheduling training")
                models_to_retrain.append(pred_type)
        
        if models_to_retrain:
            logger.info(f"Models scheduled for retraining: {', '.join(models_to_retrain)}")
            
            # Use our enhanced training pipeline for all retraining
            training_script = os.path.join(PROJECT_ROOT, "src", "models", "training_pipeline.py")
            
            if os.path.exists(training_script):
                try:
                    # Prepare command line arguments
                    cmd = [sys.executable, training_script]
                    
                    # Add specific configuration arguments based on what needs retraining
                    if "player_props" in models_to_retrain:
                        # Ensure player props are trained
                        pass  # Player props are automatically included in enhanced pipeline
                    else:
                        # Skip player props training if not needed
                        cmd.append("--skip-player-props")
                    
                    # Run the enhanced training pipeline
                    logger.info(f"Running enhanced training pipeline: {training_script}")
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=PROJECT_ROOT
                    )
                    
                    if result.returncode == 0:
                        logger.info("Enhanced training pipeline completed successfully")
                        # Log abbreviated output to avoid excessive logging
                        log_output = result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout
                        logger.info(f"Training output: {log_output}")
                    else:
                        logger.error(f"Enhanced training pipeline failed with code {result.returncode}")
                        logger.error(f"Error: {result.stderr}")
                except Exception as e:
                    logger.error(f"Error running enhanced training pipeline: {str(e)}")
            else:
                logger.error(f"Enhanced training pipeline not found at {training_script}")
                # Fallback to the older method
                train_models_if_needed()
        
        # Generate summary of retraining
        registry_summary = model_registry.get_registry_summary()
        logger.info(f"Model retraining complete. Registry now has {registry_summary['models_count']} models")
        
    except Exception as e:
        logger.error(f"Error in model retraining process: {str(e)}")


def check_model_performance():
    """
    Check model performance and send alerts for any issues
    
    This function:
    1. Analyzes recent prediction accuracy across all model types
    2. Compares current performance against historical benchmarks
    3. Sends alerts for any significant performance declines
    4. Logs performance metrics for monitoring and visualization
    """
    logger.info("Running daily model performance check")
    
    try:
        # Initialize performance tracker
        performance_tracker = PerformanceTracker()
        
        # Get current date and recent date range
        today = datetime.now(EST).strftime('%Y-%m-%d')
        last_week = (datetime.now(EST) - timedelta(days=7)).strftime('%Y-%m-%d')
        last_month = (datetime.now(EST) - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Check performance for each model type
        model_types = ["moneyline", "spread", "total", "player_props"]
        alerts = []
        
        for model_type in model_types:
            # Get recent performance (last 7 days)
            recent_metrics = performance_tracker.get_performance_metrics(
                model_name=f"{model_type}_ensemble",  # Standard naming convention
                prediction_type=model_type,
                date_range=(last_week, today)
            )
            
            # Get historical performance (last 30 days)
            historical_metrics = performance_tracker.get_performance_metrics(
                model_name=f"{model_type}_ensemble",
                prediction_type=model_type,
                date_range=(last_month, last_week)
            )
            
            if not recent_metrics or not historical_metrics:
                logger.warning(f"Insufficient data to evaluate {model_type} model performance")
                continue
            
            # Calculate performance change
            if model_type in ["moneyline"]:  # Classification models
                recent_acc = recent_metrics.get("accuracy", 0)
                hist_acc = historical_metrics.get("accuracy", 0)
                
                if hist_acc > 0:
                    change_pct = ((recent_acc - hist_acc) / hist_acc) * 100
                    if change_pct < -5:  # More than 5% drop in accuracy
                        alert = f"⚠️ {model_type.capitalize()} model accuracy has dropped by {abs(change_pct):.1f}% " + \
                                f"(from {hist_acc:.1%} to {recent_acc:.1%})"
                        alerts.append(alert)
                        logger.warning(alert)
            else:  # Regression models
                recent_mse = recent_metrics.get("mse", float('inf'))
                hist_mse = historical_metrics.get("mse", float('inf'))
                
                if hist_mse > 0 and hist_mse != float('inf'):
                    change_pct = ((recent_mse - hist_mse) / hist_mse) * 100
                    if change_pct > 10:  # More than 10% increase in error
                        alert = f"⚠️ {model_type.capitalize()} model error has increased by {change_pct:.1f}% " + \
                                f"(MSE from {hist_mse:.2f} to {recent_mse:.2f})"
                        alerts.append(alert)
                        logger.warning(alert)
        
        # Additional checks for specific issues
        for model_type in model_types:
            # Check for consistent under/over predictions
            bias_metrics = performance_tracker.check_prediction_bias(model_type)
            if bias_metrics.get("has_bias", False):
                bias_direction = "high" if bias_metrics.get("bias_direction", 0) > 0 else "low"
                alert = f"⚠️ {model_type.capitalize()} model shows consistent {bias_direction} bias " + \
                        f"({bias_metrics.get('bias_magnitude', 0):.1%})"
                alerts.append(alert)
                logger.warning(alert)
        
        # Log summary of performance check
        if alerts:
            logger.info(f"Performance check complete - {len(alerts)} issues detected")
            # TODO: Send alerts to appropriate channels (email, Slack, etc.)           
        else:
            logger.info("Performance check complete - all models performing normally")
        
    except Exception as e:
        logger.error(f"Error in model performance check: {str(e)}")


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
