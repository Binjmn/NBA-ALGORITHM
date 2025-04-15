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

## Enhanced Prediction System Architecture (April 2025)

### 1. Enhanced Feature Engineering

The system now includes advanced feature engineering capabilities that significantly improve prediction accuracy:

- **Path**: `src/features/advanced_features_plus.py`
- **Component**: `EnhancedFeatureEngineer` class
- **Documentation**: See `docs/enhanced_features.md`

This component adds sophisticated metrics including fatigue modeling, team chemistry analysis, venue-specific factors, and matchup-specific historical performance. These features are integrated with the training pipeline through the `--use-enhanced-features` flag.

### 2. Prediction Testing Framework

A comprehensive testing framework has been implemented to evaluate model performance on upcoming games:

- **Path**: `scripts/test_predictions.py`
- **Purpose**: Test predictions for upcoming NBA games using trained models
- **Documentation**: See `docs/prediction_testing.md`

The testing framework loads all trained models from the database, applies feature engineering to upcoming games, generates predictions, and provides detailed visualizations and analysis reports.

### 3. Performance Monitoring System

A new monitoring system tracks model performance over time to identify opportunities for improvement:

- **Path**: `scripts/monitor_performance.py`
- **Purpose**: Track and visualize model performance metrics
- **Documentation**: See `docs/performance_monitoring.md`

This component extracts performance metrics from the database, analyzes trends, generates visualizations, and provides actionable recommendations based on model performance data.

### 4. Automatic Retraining System

An automatic update scheduler ensures models stay current with the latest NBA data:

- **Path**: `scripts/auto_update_scheduler.py`
- **Purpose**: Automatically retrain models on a configurable schedule
- **Documentation**: See `docs/auto_update_system.md`

The scheduler supports various update frequencies (daily, weekly, post-game), includes smart update logic to avoid unnecessary retraining, and provides email notifications for update status.

### 5. Integrated Training Pipeline

The training pipeline has been enhanced to improve model training success rates and integration with new features:

- **Path**: `src/training_pipeline.py`
- **Updates**: Added support for enhanced features, improved model instantiation, fixed parameter mismatch issues

All models (RandomForestModel, GradientBoostingModel, BayesianModel, CombinedGradientBoostingModel, EnsembleModel, and EnsembleStackingModel) now train successfully with proper error handling.

## Architecture Diagram

```
┌────────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  Data Acquisition  │────▶│  Feature Engineering │────▶│  Model Training  │
└────────────────────┘     └─────────────────────┘     └──────────────────┘
         │                           ▲                          │
         │                           │                          │
         ▼                           │                          ▼
┌────────────────────┐              │              ┌──────────────────┐
│     Live APIs      │              │              │ Prediction System │
└────────────────────┘              │              └──────────────────┘
         │                           │                          │
         │                           │                          │
         ▼                           │                          ▼
┌────────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  Database Storage  │◀────│ Performance Monitor │◀────│ Testing Framework │
└────────────────────┘     └─────────────────────┘     └──────────────────┘
         │                           ▲                          
         │                           │                          
         ▼                           │                          
┌────────────────────┐              │              
│  Auto-Update System│──────────────┘              
└────────────────────┘                             
```

## Component Integration

All new components are designed for seamless integration with the existing system:

1. **Enhanced features** integrate with the existing feature engineering pipeline
2. **Prediction testing** uses the same data sources and model loading logic as the main system
3. **Performance monitoring** extracts data from the existing database tables
4. **Automatic updates** use the same training pipeline with configurable options

This modular approach allows for easy maintenance and future extensions to the system.

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
