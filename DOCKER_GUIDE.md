# NBA Betting Prediction System - Docker Guide

This guide will help you get started with the NBA Betting Prediction System using Docker, allowing you to run the entire system with minimal technical knowledge.

## What is Docker?

Docker is a platform that allows you to package and run applications in isolated environments called containers. Using Docker means you don't need to worry about installing specific versions of Python, databases, or other dependencies on your computer. Everything runs in a self-contained environment.

## Prerequisites

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your operating system
2. Make sure Docker is running on your computer (you should see the Docker icon in your system tray)

## Quick Start

### Step 1: Open a terminal or command prompt

- On Windows: Press `Windows + R`, type `cmd` and press Enter
- On Mac: Open Terminal from Applications/Utilities

### Step 2: Navigate to the project directory

```
cd path\to\NBA ALGORITHM
```

### Step 3: Start the system

```
nba-control.bat start
```

This single command will:
1. Build all containers if it's your first time
2. Start the prediction system
3. Start the database for storing predictions
4. Start the web interface for monitoring

### Step 4: Access the dashboard

Open your web browser and go to:
```
http://localhost:8080
```

## Useful Commands

All commands are provided through the user-friendly `nba-control.bat` script:

| Command | Description |
|---------|-------------|
| `nba-control.bat start` | Start the entire system |
| `nba-control.bat stop` | Stop the system |
| `nba-control.bat restart` | Restart the system |
| `nba-control.bat status` | Check if the system is running |
| `nba-control.bat logs` | View system logs |
| `nba-control.bat logs nba-prediction` | View prediction service logs |

## System Components

The NBA Prediction System consists of:

1. **Prediction Engine** - Core machine learning system running 7 prediction models
2. **Database** - Storing game data, predictions, and model performance metrics
3. **Web Dashboard** - User-friendly interface for monitoring and control

## Web Interface Guide

### Dashboard
The main dashboard shows:
- System status and health
- Today's predictions with confidence levels
- Status of all 7 prediction models
- Quick actions like refreshing data

### Models Page
View and configure each of the 7 prediction models:
1. **Random Forests**: Captures patterns like rest advantage
2. **Combined (XGBoost + LightGBM)**: Blends deep analysis approaches
3. **Bayesian**: Updates probabilities with new data
4. **Anomaly Detection**: Flags statistical outliers
5. **Model Mixing**: Weights models by recent performance
6. **Ensemble Stacking**: Creates meta-model from all outputs
7. **Hyperparameter Tuning**: Optimizes settings across models

You can:
- Enable/disable each model
- Adjust confidence thresholds
- Set model weights in the ensemble
- View performance metrics

### Configuration Page
Configure system settings:
- API Keys for data sources
- Scheduling preferences
- Email notifications
- Data retention policies

### Logs Page
Monitor system activity:
- View logs from all components
- Filter by service or severity
- Troubleshoot any issues

## Data Management

The system stores all data in persistent volumes:
- `/data` - Game data, team stats, odds
- `/models` - Trained machine learning models
- `/logs` - System logs and activity records

This data persists even when you stop and restart the system.

## Updating the System

When new features are released:

1. Stop the system: `nba-control.bat stop`
2. Pull the latest code: `git pull`
3. Restart the system: `nba-control.bat start`

The system will automatically update with the latest improvements.

## Troubleshooting

### System won't start
- Check if Docker is running
- Ensure ports 8080 and 5432 aren't being used by other applications
- View logs with `nba-control.bat logs` to identify specific errors

### Can't access the dashboard
- Verify the system is running with `nba-control.bat status`
- Check if your firewall is blocking port 8080
- Try accessing http://127.0.0.1:8080 instead of localhost

### Models aren't updating
- Check your API keys in the Configuration page
- View the logs for any API rate limit issues
- Ensure the system has internet connectivity

## Getting Help

If you encounter any issues:
1. Check the logs using `nba-control.bat logs`
2. Look for specific error messages
3. Contact support with these error details
