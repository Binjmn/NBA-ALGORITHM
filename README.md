# NBA Betting Prediction System

## Overview
A modular, minimal, and scalable NBA betting prediction system using real data from odds and stats APIs. The system auto-detects NBA seasons (current and future), trains models, and generates predictions for the current day's games, including game outcomes, player stats, bankroll recommendations, and CLV analysis.

## Project Structure
This project follows a modular and minimal architecture:

- `src/` - Source code
- `data/` - Data storage
- `logs/` - Application logs
- `docs/` - Documentation
- `config/` - Configuration files
- `tests/` - Unit and integration tests

## Project Rules
This project adheres to specific rules and guidelines documented in [PROJECT_RULES.md](PROJECT_RULES.md). Please review these rules before contributing to the project.

## Setup
1. Ensure Python 3.9+ is installed
2. Run `setup_env.py` to set up the virtual environment
3. Configure API keys in the secured configuration file

## Usage
The system runs automatically on scheduled intervals. No manual execution is required.

### Scheduling
- Model training and first predictions: 6:00 AM EST daily
- Model performance summary: 2:00 AM EST daily
- Data updates: Every 4 hours (10:00 AM, 2:00 PM, 6:00 PM EST)
- Live game data: Every 10 minutes during active games
- News/injury updates: Every 15 minutes in 2-hour pre-game windows

## Models
The system uses multiple models for predictions:
1. Random Forests
2. Combined (XGBoost + LightGBM)
3. Bayesian
4. Anomaly Detection
5. Model Mixing
6. Ensemble Stacking

## Predictions
Predictions are generated for the current day's games only, including:
- Game outcomes (moneyline, spread, totals)
- Player stats (points, rebounds, assists, threes)
- Bankroll recommendations
- CLV analysis

## Performance Tracking
Model performance is tracked daily and stored in the database. A summary is generated daily at 2:00 AM EST.

## Maintenance
The system is designed for long-term operation with minimal maintenance. It automatically adapts to new NBA seasons and retrains models when performance drops below thresholds.

## Documentation
Detailed documentation is available in the `docs/` directory, including:
- Architecture details
- Data flow diagrams
- Model designs
- API integrations

## Changelog

### Version 1.0.0 (2025-04-14)
- Initial project setup
