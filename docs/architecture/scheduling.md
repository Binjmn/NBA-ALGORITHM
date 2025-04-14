# Scheduling Architecture

## Overview
This document details the scheduling architecture of the NBA Betting Prediction System, focusing on how various tasks are scheduled and executed automatically without manual intervention.

## Scheduling Technology

### APScheduler
- Primary scheduling system used throughout the application
- Configured in `scheduler.py`
- Provides robust job scheduling with error handling and persistence
- All times standardized to Eastern Standard Time (EST)

## Scheduled Tasks

### Data Collection

#### Stats and Odds Updates
- **Schedule**: Every 4 hours (10:00 AM, 2:00 PM, 6:00 PM EST)
- **Component**: `get_nba_data.py`
- **Purpose**: Collect latest game statistics and betting odds
- **Process**:
  - Makes API calls to fetch fresh data
  - Validates data integrity
  - Stores in database
  - Triggers prediction updates if significant changes detected

#### Live Game Data
- **Schedule**: Every 10 minutes during active games
- **Component**: `update_live_predictions.py`
- **Purpose**: Update predictions based on in-game events
- **Process**:
  - Fetches live scores and statistics
  - Updates models with live data
  - Generates revised predictions

#### News and Injury Updates
- **Schedule**: Every 15 minutes in 2-hour pre-game windows (e.g., 5:00–7:00 PM for 7:00 PM games), hourly otherwise
- **Component**: `get_news_data.py`
- **Purpose**: Track player injuries and team news that may impact predictions
- **Process**:
  - Fetches news from reliable sources
  - Extracts relevant information
  - Updates player availability status
  - Triggers prediction updates if necessary

### Model Training

#### Initial Daily Predictions
- **Schedule**: 6:00 AM EST daily
- **Component**: `make_predictions.py`
- **Purpose**: Generate first set of predictions for the day
- **Process**:
  - Applies trained models to current data
  - Generates predictions for all of the day's games
  - Outputs results in standardized JSON format

#### Model Training
- **Schedule**: 6:00 AM EST daily
- **Component**: `auto_train_manager.py`
- **Purpose**: Keep models up-to-date with latest data
- **Process**:
  - Retrieves historical data
  - Trains all models
  - Updates model weights
  - Outputs status to training_status.txt

#### Performance Tracking
- **Schedule**: 2:00 AM EST daily
- **Component**: `track_performance.py`
- **Purpose**: Evaluate model performance
- **Process**:
  - Calculates 7-day accuracy metrics
  - Compares to target thresholds
  - Stores results in database and model_performance.txt

#### CLV Tracking
- **Schedule**: Hourly
- **Component**: `track_clv.py`
- **Purpose**: Track closing line value to evaluate prediction quality
- **Process**:
  - Compares initial odds to closing odds
  - Calculates CLV for each prediction
  - Updates CLV metrics in database

### Database Maintenance

#### Backups
- **Schedule**: Daily at 3:00 AM EST
- **Component**: `backup_db.py`
- **Purpose**: Ensure data integrity and recoverability
- **Process**:
  - Creates full database backup
  - Implements rolling retention policy
  - Verifies backup integrity

## Schedule Management

### Scheduling Logic
- All schedules defined in `scheduler.py`
- Schedule times optimized to balance data freshness and API usage
- Staggered schedules to prevent resource contention

### Error Handling
- Failed jobs are logged and retried with exponential backoff
- Critical failures trigger notifications
- System maintains operation despite individual task failures

### Schedule Customization
- Schedules can be adjusted in configuration without code changes
- Special schedules for playoffs or other events are supported

## Version History

### Version 1.0.0 (2025-04-14)
- Initial scheduling architecture documentation
