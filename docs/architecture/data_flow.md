# Data Flow Architecture

## Overview
This document details the data flow architecture of the NBA Betting Prediction System, focusing on how data moves through the system from acquisition to prediction generation.

## Data Sources

### Primary Sources
- Stats API (provider TBD) for game and player statistics
- Odds API (provider TBD) for betting odds

### Backup Sources
- Secondary public basketball stats websites for failover when primary sources are unavailable

## Data Flow Process

### 1. Data Acquisition
- **Responsible Components**: `get_nba_data.py`, `get_news_data.py`
- **Process**:
  - Stats and odds data refreshed every 4 hours (10:00 AM, 2:00 PM, 6:00 PM EST)
  - Live game data updated every 10 minutes during active games
  - News/injury updates every 15 minutes in 2-hour pre-game windows, hourly otherwise
  - All acquired data is validated for authenticity and completeness
  - Data is cached to reduce API calls and manage rate limits

### 2. Data Processing
- **Responsible Components**: `process_data.py`, `update_team_features.py`, `update_player_stats.py`
- **Process**:
  - Raw data is cleaned and normalized
  - Features are extracted and engineered
  - Team and player statistics are updated
  - All times are standardized to EST

### 3. Feature Storage
- **Responsible Components**: PostgreSQL database
- **Process**:
  - Processed features are stored in appropriate database tables
  - JSONB format used for flexible storage of odds, features, and predictions
  - Historical data is preserved for model training

### 4. Model Consumption
- **Responsible Components**: Auto-training models in `auto_train_manager.py`
- **Process**:
  - Models retrieve features from the database
  - Training is performed on historical data
  - Current features are used for prediction generation

### 5. Prediction Output
- **Responsible Components**: `make_predictions.py`, `update_live_predictions.py`
- **Process**:
  - Predictions are generated in standardized JSON format
  - Outputs include game outcomes, player stats, bankroll recommendations, and CLV analysis
  - Predictions are stored in the database and made available via the API

## Data Validation and Security

### Validation
- `validate_data.py` performs automated checks on all incoming data
- Validation includes checks for completeness, consistency, and format
- Failed validations trigger alerts and potential fallback to secondary sources

### Security
- API keys and sensitive data are encrypted using `secure_data.py`
- All data transfers use secure protocols
- Access to data is logged for audit purposes

## Backup and Recovery

- Database backups are performed daily via `backup_db.py`
- If primary data sources fail, the system automatically switches to secondary sources using `get_backup_data.py`
- Data integrity is maintained throughout recovery processes

## Version History

### Version 1.0.0 (2025-04-14)
- Initial data flow architecture documentation
