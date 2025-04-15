# NBA Prediction System Auto-Update Scheduler

## Overview

The automatic update scheduler (`scripts/auto_update_scheduler.py`) provides a robust system for keeping NBA prediction models up-to-date with the latest game data, ensuring models always reflect current team performance and trends.

## Features

### Flexible Scheduling Options
- **Daily Updates**: Run at a specified time each day
- **Weekly Updates**: Run on a specific day of the week
- **Post-Game Updates**: Check for new games and retrain when found

### Smart Update Logic
- Checks if new games have been played since last update
- Avoids unnecessary retraining when no new data is available
- Force collection option available when needed

### Production-Quality Implementation
- Comprehensive error handling and logging
- Database integration for tracking update history
- Email notifications for update results

## Usage

```bash
# Create default configuration file
python scripts/auto_update_scheduler.py --create-config --output auto_update_config.json

# Run with a specific configuration file
python scripts/auto_update_scheduler.py --config auto_update_config.json

# Run update immediately then start scheduler
python scripts/auto_update_scheduler.py --config auto_update_config.json --run-now
```

## Configuration Options

The configuration file supports these options:

```json
{
    "update_schedule": {
        "frequency": "daily",  # daily, weekly, after_games
        "time": "02:00",  # 24-hour format
        "day_of_week": 1,  # 0=Monday for weekly updates
        "check_for_games": true  # Only update if new games
    },
    "data_collection": {
        "historical_days": 30,  # Days of data to fetch
        "force_collection": false  # Force regardless of new games
    },
    "email_notifications": {
        "enabled": false,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "",
        "sender_password": "",
        "receiver_emails": []
    },
    "performance_thresholds": {
        "accuracy_min": 0.58,  # Minimum accuracy for deployment
        "rmse_max": 7.5,  # Maximum RMSE for deployment
        "auc_min": 0.65  # Minimum AUC for deployment
    },
    "retrain_options": {
        "models": ["RandomForestModel", "GradientBoostingModel", 
                 "BayesianModel", "CombinedGradientBoostingModel", 
                 "EnsembleModel", "EnsembleStackingModel"],
        "skip_deployment": false
    }
}
```

## Email Notifications

When configured, the system can send email notifications about:
- Successful model retraining
- Training failures or errors
- Performance metrics below thresholds

## Database Integration

All update activities are logged to the database in the `system_logs` table with:
- Timestamp of update attempt
- Success/failure status
- Detailed update logs
- Configuration settings used

## Production Deployment

For production deployment, consider:
1. Running as a service or with a process manager like `supervisord`
2. Setting up monitoring to ensure the scheduler stays running
3. Configuring secure credential management for API and email access
4. Adding additional notification channels (Slack, SMS, etc.)
