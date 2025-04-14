# Model Design Architecture

## Overview
This document details the model architecture of the NBA Betting Prediction System, focusing on the various models used for predictions, their training process, and performance monitoring.

## Model Types

### 1. Random Forests
- **Purpose**: Captures non-linear patterns in team and player performance
- **Features**: Team statistics, player efficiency, rest days, home/away splits
- **Implementation**: `models/random_forest.py`

### 2. Combined Models (XGBoost + LightGBM)
- **Purpose**: Provides deep analysis of complex interactions between features
- **Features**: All available features including advanced team and player metrics
- **Implementation**: `models/combined_models.py`

### 3. Bayesian Models
- **Purpose**: Updates probabilities based on new information, provides confidence intervals
- **Features**: Prior probabilities, likelihood functions based on historical performance
- **Implementation**: `models/bayesian.py`

### 4. Anomaly Detection
- **Purpose**: Flags outliers and unusual patterns in game and player performance
- **Features**: Historical performance distributions, deviation from expected values
- **Implementation**: `models/anomaly_detection.py`

### 5. Model Mixing
- **Purpose**: Weights models by their recent performance to optimize overall prediction accuracy
- **Features**: Model performance metrics over the last 7 days
- **Implementation**: `models/model_mixing.py`

### 6. Ensemble Stacking
- **Purpose**: Meta-model that combines outputs from all other models
- **Features**: Predictions from all other models
- **Implementation**: `models/ensemble_stacking.py`

## Training Process

### Automated Training
- **Responsible Component**: `auto_train_manager.py`
- **Schedule**: Daily at 6:00 AM EST
- **Process**:
  - Retrieves historical data for training
  - Splits data into training and validation sets
  - Trains each model independently
  - Evaluates performance on validation data
  - Updates model weights
  - Stores trained models

### Hyperparameter Tuning
- **Responsible Component**: `models/hyperparameter_tuning.py`
- **Process**:
  - Uses Bayesian optimization to find optimal hyperparameters
  - Tests multiple parameter combinations
  - Selects parameters that maximize prediction accuracy
  - Applies tuned parameters to models

## Drift Detection and Retraining

### Drift Detection
- **Responsible Component**: `check_model_drift.py`
- **Process**:
  - Monitors model performance daily
  - Calculates accuracy metrics (e.g., moneyline, spread success rate)
  - Compares to historical performance
  - Triggers retraining if accuracy drops below 5% threshold

### Performance Tracking
- **Responsible Component**: `track_performance.py`
- **Schedule**: Daily at 2:00 AM EST
- **Process**:
  - Calculates 7-day accuracy for each model
  - Compares to target thresholds (e.g., 75% accuracy)
  - Stores results in database and model_performance.txt
  - Updates model weights for model mixing

## Season Management

- **Responsible Component**: `season_manager.py`
- **Process**:
  - Auto-detects current NBA season
  - Prepares for future seasons (e.g., 2026, 2027, 2028)
  - Archives models from prior seasons
  - Adjusts features for season-specific changes

## Model Output Format

- All predictions are in standardized JSON format
- Includes team predictions (moneyline, spread, totals)
- Includes player predictions (points, rebounds, assists, threes)
- Includes bankroll recommendations and CLV analysis
- Includes confidence levels and model contributions

## Version History

### Version 1.0.0 (2025-04-14)
- Initial model design architecture documentation
