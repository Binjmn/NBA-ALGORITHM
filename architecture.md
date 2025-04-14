# NBA Betting Prediction System Architecture

## Overview
This document provides a high-level overview of the NBA Betting Prediction System architecture. For detailed information on specific components, please refer to the individual architecture documents in the `docs/architecture/` directory.

## System Components

### 1. Data Acquisition
Responsible for fetching real data from the odds and stats APIs, with proper validation and scheduling.

### 2. Data Processing
Transforms raw data into features suitable for model training, including team and player statistics.

### 3. Model Development
Implements various predictive models that auto-train without manual intervention.

### 4. Prediction Generation
Generates predictions for current day's games, including game outcomes, player stats, and bankroll recommendations.

### 5. Performance Tracking
Monitors model performance and CLV over time, with automatic retraining when performance degrades.

### 6. API Integration
Provides a REST API for future UI/website integration.

## Detailed Architecture Documents

For detailed information on specific architecture components, please refer to the following documents:

1. [Data Flow](docs/architecture/data_flow.md)
2. [Model Design](docs/architecture/model_design.md)
3. [Scheduling](docs/architecture/scheduling.md)
4. [Season Management](docs/architecture/season_management.md)
5. [API Integration](docs/architecture/api_integration.md)
6. [Performance Tracking](docs/architecture/performance_tracking.md)

## File Structure

```
nba-prediction-system/
├── src/                # Source code
│   ├── api_clients.py  # API client wrappers
│   ├── get_nba_data.py # Data collection
│   ├── process_data.py # Data processing
│   └── ...
├── data/               # Data storage
├── logs/               # Application logs
├── docs/               # Documentation
│   └── architecture/   # Detailed architecture docs
├── config/             # Configuration files
└── tests/              # Unit and integration tests
```

## Technology Stack

- Python 3.9+
- PostgreSQL (for data storage)
- XGBoost, LightGBM, scikit-learn (for modeling)
- APScheduler (for task scheduling)
- Flask (for REST API)

## Version History

### Version 1.0.0 (2025-04-14)
- Initial architecture documentation
