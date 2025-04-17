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

### 7. Season Management
Automatically detects and manages NBA season transitions, ensuring the system adapts to different season phases.

## Modular Code Structure (April 2025 Update)

The codebase has been restructured into a modular organization to improve maintainability, readability, and compatibility with context-aware tools. Each module focuses on a specific functionality area.

### Directory Structure

```
scripts/
├── main.py                    # Entry point that orchestrates the prediction workflow
├── config.py                  # Configuration, logging setup, and constants
├── models/                    # Model management
│   ├── __init__.py            
│   ├── loader.py              # Model loading functions
│   └── predictor.py           # Core prediction logic
├── data/                      # Data acquisition 
│   ├── __init__.py
│   ├── game_data.py           # Game data fetching
│   ├── player_data.py         # Player data fetching
│   ├── team_data.py           # Team statistics fetching
│   └── odds_data.py           # Betting odds fetching
├── features/                  # Feature engineering
│   ├── __init__.py
│   ├── game_features.py       # Game feature extraction
│   └── player_features.py     # Player feature extraction
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── math_utils.py          # Mathematical helper functions
│   ├── string_utils.py        # String processing helpers
│   └── storage.py             # File saving/loading functions
├── presentation/              # Output formatting
│   ├── __init__.py
│   ├── display.py             # User-friendly display functions
│   └── explanations.py        # Natural language explanations
├── season_management/         # Season management
│   ├── __init__.py
│   ├── season_config.py       # NBA season definitions and phase transitions
│   └── season_manager.py      # Season detection and management
```

### Module Responsibilities

#### Main Controller
- **main.py**: Central orchestration point that coordinates all other modules

#### Configuration
- **config.py**: Environment configuration, logging setup, and global constants

#### Data Acquisition
- **game_data.py**: Fetches NBA games data with robust error handling
- **player_data.py**: Retrieves player information and statistics
- **team_data.py**: Gets team statistics and contextual information
- **odds_data.py**: Acquires betting odds and player props from odds providers

#### Feature Engineering
- **game_features.py**: Extracts and prepares game-level features
- **player_features.py**: Processes player-specific features for prop predictions

#### Model Management
- **loader.py**: Handles loading of trained models from storage
- **predictor.py**: Core prediction logic for both games and player props

#### Utilities
- **math_utils.py**: Mathematical functions for predictions and processing
- **string_utils.py**: String manipulation helpers for team/player names
- **storage.py**: File operations for saving/loading data and results

#### Presentation
- **display.py**: User-friendly formatting of prediction results
- **explanations.py**: Natural language generation for explaining predictions

#### Season Management
- **season_config.py**: Defines NBA season date ranges, phases, and provides utilities for identifying seasons based on dates
- **season_manager.py**: Implements the `SeasonManager` class that automatically detects and manages season transitions, adapting the system's behavior based on current season context

### Key Improvements

1. **Enhanced Modularity**: Each file is focused on a specific functionality area
2. **Better Error Handling**: Comprehensive error handling throughout all modules
3. **Improved Readability**: Logical organization makes code easier to navigate
4. **Windsurf Compatibility**: Files are kept under 600 lines for better context-aware tool support
5. **Production Readiness**: Removed fallback mechanisms for synthetic data, ensuring real data usage
6. **Proper Documentation**: Comprehensive docstrings and code comments

## Detailed Architecture Documents

For detailed information on specific architecture components, please refer to the following documents:

1. [Data Flow](docs/architecture/data_flow.md)
2. [Model Design](docs/architecture/model_design.md)
3. [Scheduling](docs/architecture/scheduling.md)
4. [Season Management](docs/architecture/season_management.md)
5. [API Integration](docs/architecture/api_integration.md)
6. [Performance Tracking](docs/architecture/performance_tracking.md)

## Enhanced Prediction System Architecture 

### 1. Enhanced Feature Engineering

The system now includes advanced feature engineering capabilities that significantly improve prediction accuracy:

- **Path**: `scripts/features/game_features.py` and `scripts/features/player_features.py`
- **Component**: Feature extraction and processing functions
- **Documentation**: Comprehensive docstrings in code

These components add sophisticated metrics including matchup analysis, rest factors, and contextual performance indicators.

### 2. Prediction Testing Framework

A comprehensive testing framework has been implemented to evaluate model performance on upcoming games:

- **Path**: `scripts/test_predictions.py`
- **Purpose**: Test predictions for upcoming NBA games using trained models

### 3. Performance Monitoring System

A monitoring system tracks model performance over time to identify opportunities for improvement:

- **Path**: `scripts/monitor_performance.py`
- **Purpose**: Track and visualize model performance metrics

### 4. Automatic Retraining System

An automatic update scheduler ensures models stay current with the latest NBA data:

- **Path**: `scripts/auto_update_scheduler.py`
- **Purpose**: Automatically retrain models on a configurable schedule

## Architecture Diagram

```
┌────────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  Data Acquisition  │────▶│  Feature Engineering │────▶│  Model Prediction │
└────────────────────┘     └─────────────────────┘     └──────────────────┘
         │                           ▲                          │
         │                           │                          │
         ▼                           │                          ▼
┌────────────────────┐              │              ┌──────────────────┐
│     External APIs  │              │              │ Results Display  │
└────────────────────┘              │              └──────────────────┘
         │                           │                          │
         │                           │                          │
         ▼                           │                          ▼
┌────────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  Data Storage      │◀────│ Utilities           │◀────│ Output Formatter │
└────────────────────┘     └─────────────────────┘     └──────────────────┘
```

## Technology Stack

- Python 3.9+
- scikit-learn, XGBoost, LightGBM (for modeling)
- pandas, numpy (for data processing)
- requests (for API communication)
- logging (for robust error tracking)

## Version History

### Version 2.0.0 (2025-04-16)
- Restructured codebase into modular organization
- Improved error handling and removed synthetic data fallbacks
- Enhanced compatibility with context-aware tools

### Version 1.0.0 (2025-04-14)
- Initial architecture documentation
