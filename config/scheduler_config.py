"""
Scheduler Configuration

This module provides configuration settings for the scheduler system, including
job definitions, frequencies, and dependencies.

Jobs are defined with:
- name: Unique identifier for the job
- type: The type of job (data_collection, prediction, model_training, etc.)
- schedule: When the job should run (cron expressions or intervals)
- depends_on: List of jobs that must complete before this one
- active_phases: List of season phases where this job should be active
- description: Human-readable description of what the job does
"""

import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum, auto

from config.season_config import SeasonPhase

logger = logging.getLogger(__name__)

class JobType(Enum):
    """Types of scheduled jobs"""
    DATA_COLLECTION = auto()    # Jobs that collect data from APIs
    DATA_PROCESSING = auto()    # Jobs that process and transform data
    PREDICTION = auto()         # Jobs that make predictions
    MODEL_TRAINING = auto()     # Jobs that train models
    MODEL_EVALUATION = auto()   # Jobs that evaluate model performance
    NOTIFICATION = auto()       # Jobs that send notifications/alerts
    MAINTENANCE = auto()        # Jobs that perform system maintenance
    CUSTOM = auto()             # Custom job types

class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

# Define the job configuration schema
JOB_CONFIG_SCHEMA = {
    "name": str,                # Unique job identifier
    "type": JobType,            # Type of job
    "schedule": str,            # Cron expression or interval
    "enabled": bool,            # Whether the job is enabled
    "module": str,              # Python module containing the job function
    "function": str,            # Function name to execute
    "args": list,               # Positional arguments
    "kwargs": dict,             # Keyword arguments
    "depends_on": list,         # Jobs that must complete before this one
    "priority": JobPriority,    # Job priority
    "timeout": int,             # Max runtime in seconds
    "retry": dict,              # Retry configuration
    "active_phases": list,      # Season phases when this job should run
    "description": str          # Human-readable description
}

# Default values for job configuration
JOB_CONFIG_DEFAULTS = {
    "enabled": True,
    "args": [],
    "kwargs": {},
    "depends_on": [],
    "priority": JobPriority.NORMAL,
    "timeout": 3600,  # 1 hour
    "retry": {
        "max_retries": 3,
        "delay": 300  # 5 minutes
    },
    "active_phases": list(SeasonPhase),  # Active in all phases by default
}

# Job schedule definitions
# Using APScheduler syntax: https://apscheduler.readthedocs.io/en/stable/modules/triggers/cron.html
# For "interval" jobs, use {"seconds": X} or {"minutes": X} etc.

# -------------------------------------------------------------------------------
# DATA COLLECTION JOBS
# -------------------------------------------------------------------------------

# Daily initial data collection (runs at 6:00 AM EST)
DAILY_DATA_COLLECTION = {
    "name": "daily_data_collection",
    "type": JobType.DATA_COLLECTION,
    "schedule": {"hour": 6, "minute": 0},  # 6:00 AM EST
    "module": "src.tasks.data_collection",
    "function": "collect_daily_data",
    "description": "Collects initial daily data for teams, players, and games",
    "active_phases": [
        SeasonPhase.PRESEASON,
        SeasonPhase.REGULAR_SEASON,
        SeasonPhase.ALL_STAR_BREAK,
        SeasonPhase.PLAY_IN_TOURNAMENT,
        SeasonPhase.PLAYOFFS,
        SeasonPhase.FINALS
    ]
}

# Regular odds updates (every 4 hours)
ODDS_UPDATE = {
    "name": "odds_update",
    "type": JobType.DATA_COLLECTION,
    "schedule": {"hours": 4},  # Every 4 hours
    "module": "src.tasks.data_collection",
    "function": "update_odds_data",
    "description": "Updates betting odds data every 4 hours",
    "active_phases": [
        SeasonPhase.PRESEASON,
        SeasonPhase.REGULAR_SEASON,
        SeasonPhase.PLAY_IN_TOURNAMENT,
        SeasonPhase.PLAYOFFS,
        SeasonPhase.FINALS
    ]
}

# Live game updates (every 10 minutes during games)
LIVE_GAME_UPDATE = {
    "name": "live_game_update",
    "type": JobType.DATA_COLLECTION,
    "schedule": {"minutes": 10},  # Every 10 minutes
    "module": "src.tasks.data_collection",
    "function": "update_live_game_data",
    "description": "Updates live game data every 10 minutes during active games",
    "active_phases": [
        SeasonPhase.PRESEASON,
        SeasonPhase.REGULAR_SEASON,
        SeasonPhase.PLAY_IN_TOURNAMENT,
        SeasonPhase.PLAYOFFS,
        SeasonPhase.FINALS
    ]
}

# -------------------------------------------------------------------------------
# DATA PROCESSING JOBS
# -------------------------------------------------------------------------------

# Process collected data (runs after daily data collection)
PROCESS_DAILY_DATA = {
    "name": "process_daily_data",
    "type": JobType.DATA_PROCESSING,
    "schedule": {"hour": 6, "minute": 30},  # 6:30 AM EST
    "depends_on": ["daily_data_collection"],
    "module": "src.tasks.data_processing",
    "function": "process_daily_data",
    "description": "Processes and transforms daily collected data",
    "active_phases": [
        SeasonPhase.PRESEASON,
        SeasonPhase.REGULAR_SEASON,
        SeasonPhase.ALL_STAR_BREAK,
        SeasonPhase.PLAY_IN_TOURNAMENT,
        SeasonPhase.PLAYOFFS,
        SeasonPhase.FINALS
    ]
}

# -------------------------------------------------------------------------------
# PREDICTION JOBS
# -------------------------------------------------------------------------------

# Generate daily predictions (runs after data processing)
DAILY_PREDICTIONS = {
    "name": "daily_predictions",
    "type": JobType.PREDICTION,
    "schedule": {"hour": 7, "minute": 0},  # 7:00 AM EST
    "depends_on": ["process_daily_data"],
    "module": "src.tasks.predictions",
    "function": "generate_daily_predictions",
    "description": "Generates daily game predictions",
    "active_phases": [
        SeasonPhase.PRESEASON,
        SeasonPhase.REGULAR_SEASON,
        SeasonPhase.PLAY_IN_TOURNAMENT,
        SeasonPhase.PLAYOFFS,
        SeasonPhase.FINALS
    ]
}

# Update live predictions (every 30 minutes during games)
LIVE_PREDICTIONS = {
    "name": "live_predictions",
    "type": JobType.PREDICTION,
    "schedule": {"minutes": 30},  # Every 30 minutes
    "module": "src.tasks.predictions",
    "function": "update_live_predictions",
    "description": "Updates predictions during live games",
    "active_phases": [
        SeasonPhase.PRESEASON,
        SeasonPhase.REGULAR_SEASON,
        SeasonPhase.PLAY_IN_TOURNAMENT,
        SeasonPhase.PLAYOFFS,
        SeasonPhase.FINALS
    ]
}

# -------------------------------------------------------------------------------
# MODEL TRAINING JOBS
# -------------------------------------------------------------------------------

# Weekly model retraining (runs on weekends)
MODEL_RETRAINING = {
    "name": "model_retraining",
    "type": JobType.MODEL_TRAINING,
    "schedule": {"day_of_week": "sun", "hour": 1, "minute": 0},  # Sunday at 1:00 AM EST
    "module": "src.tasks.model_training",
    "function": "retrain_models",
    "description": "Retrains prediction models with latest data",
    "active_phases": [
        SeasonPhase.PRESEASON,
        SeasonPhase.REGULAR_SEASON,
        SeasonPhase.ALL_STAR_BREAK,
        SeasonPhase.PLAY_IN_TOURNAMENT,
        SeasonPhase.PLAYOFFS,
        SeasonPhase.FINALS,
        SeasonPhase.OFFSEASON
    ]
}

# Season transition model training (runs at season changes)
SEASON_TRANSITION_TRAINING = {
    "name": "season_transition_training",
    "type": JobType.MODEL_TRAINING,
    "schedule": None,  # Triggered by events, not scheduled
    "module": "src.tasks.model_training",
    "function": "season_transition_training",
    "description": "Retrains models when transitioning to a new season",
    "active_phases": list(SeasonPhase)  # Active in all phases
}

# -------------------------------------------------------------------------------
# MODEL EVALUATION JOBS
# -------------------------------------------------------------------------------

# Daily model performance evaluation
MODEL_EVALUATION = {
    "name": "model_evaluation",
    "type": JobType.MODEL_EVALUATION,
    "schedule": {"hour": 10, "minute": 0},  # 10:00 AM EST
    "module": "src.tasks.model_evaluation",
    "function": "evaluate_model_performance",
    "description": "Evaluates prediction model performance",
    "active_phases": [
        SeasonPhase.PRESEASON,
        SeasonPhase.REGULAR_SEASON,
        SeasonPhase.PLAY_IN_TOURNAMENT,
        SeasonPhase.PLAYOFFS,
        SeasonPhase.FINALS
    ]
}

# -------------------------------------------------------------------------------
# MAINTENANCE JOBS
# -------------------------------------------------------------------------------

# Database cleanup (daily)
DATABASE_CLEANUP = {
    "name": "database_cleanup",
    "type": JobType.MAINTENANCE,
    "schedule": {"hour": 2, "minute": 0},  # 2:00 AM EST
    "module": "src.tasks.maintenance",
    "function": "cleanup_database",
    "description": "Cleans up and optimizes the database",
    "active_phases": list(SeasonPhase)  # Active in all phases
}

# Data backup (daily)
DATA_BACKUP = {
    "name": "data_backup",
    "type": JobType.MAINTENANCE,
    "schedule": {"hour": 3, "minute": 0},  # 3:00 AM EST
    "module": "src.tasks.maintenance",
    "function": "backup_data",
    "description": "Creates backups of important data",
    "active_phases": list(SeasonPhase)  # Active in all phases
}

# Define the active jobs
ACTIVE_JOBS = [
    DAILY_DATA_COLLECTION,
    ODDS_UPDATE,
    LIVE_GAME_UPDATE,
    PROCESS_DAILY_DATA,
    DAILY_PREDICTIONS,
    LIVE_PREDICTIONS,
    MODEL_RETRAINING,
    MODEL_EVALUATION,
    DATABASE_CLEANUP,
    DATA_BACKUP
]
