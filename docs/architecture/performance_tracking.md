# Performance Tracking Architecture

## Overview
This document details the performance tracking architecture of the NBA Betting Prediction System, focusing on how the system monitors, measures, and improves prediction accuracy over time.

## Performance Metrics

### Game Prediction Metrics
- **Accuracy**: Percentage of correct predictions (moneyline, spread, totals)
- **Profit/Loss**: Calculated based on recommended bankroll percentages
- **CLV (Closing Line Value)**: Difference between prediction odds and closing odds
- **Kelly Criterion Performance**: Evaluation of bankroll optimization effectiveness
- **Brier Score**: Measures calibration of probability estimates

### Player Prediction Metrics
- **Accuracy**: Percentage of correct over/under predictions
- **Average Deviation**: Mean absolute error between predicted and actual stats
- **Profit/Loss**: Calculated based on recommended prop bets
- **CLV**: Difference between prediction odds and closing odds for player props

### Model-Specific Metrics
- **Individual Model Accuracy**: Performance of each model independently
- **Feature Importance**: Impact of different features on prediction accuracy
- **Calibration**: Reliability of probability estimates
- **Training Efficiency**: Time and resources required for model training

## Tracking Components

### Performance Tracking System
- **Responsible Component**: `track_performance.py`
- **Schedule**: Daily at 2:00 AM EST
- **Purpose**: Evaluate overall system performance
- **Process**:
  - Calculates 7-day accuracy metrics for all prediction types
  - Compares to target thresholds (e.g., 75% accuracy)
  - Stores results in database and model_performance.txt
  - Generates visualizations of performance trends
  - Outputs summaries for review

### CLV Tracking System
- **Responsible Component**: `track_clv.py`
- **Schedule**: Hourly
- **Purpose**: Measure the value of predictions compared to market movement
- **Process**:
  - Captures initial odds when predictions are made
  - Tracks odds movement throughout the day
  - Records closing odds before game start
  - Calculates CLV for each prediction
  - Aggregates CLV metrics by prediction type and time period

### Model Drift Detection
- **Responsible Component**: `check_model_drift.py`
- **Purpose**: Identify when models are becoming less accurate
- **Process**:
  - Monitors accuracy trends over time
  - Compares recent performance to historical baselines
  - Triggers retraining when accuracy drops below 5% threshold
  - Documents drift patterns for analysis

## Performance Improvement Mechanisms

### Automated Model Retraining
- **Responsible Component**: `auto_train_manager.py`
- **Trigger**: Scheduled daily or when drift is detected
- **Process**:
  - Retrieves latest performance metrics
  - Adjusts training parameters based on performance
  - Retrains underperforming models
  - Updates model weights in the ensemble
  - Verifies improvement through validation

### Feature Importance Analysis
- **Responsible Component**: `analyze_features.py`
- **Schedule**: Weekly
- **Purpose**: Identify most influential features
- **Process**:
  - Calculates feature importance for each model
  - Identifies underutilized or noisy features
  - Recommends feature adjustments
  - Informs feature engineering improvements

### Hyperparameter Optimization
- **Responsible Component**: `optimize_hyperparameters.py`
- **Schedule**: Monthly or when significant performance drops occur
- **Process**:
  - Uses Bayesian optimization to find better hyperparameters
  - Tests multiple parameter combinations
  - Selects parameters that maximize prediction accuracy
  - Applies optimized parameters to models

## Reporting and Visualization

### Performance Dashboard
- **Responsible Component**: `generate_dashboard.py`
- **Output**: HTML dashboard in `data/reports/`
- **Content**:
  - 7-day, 30-day, and season-to-date accuracy
  - Profit/loss trends
  - CLV analysis
  - Model contribution breakdown
  - Feature importance visualization

### Long-term Trend Analysis
- **Responsible Component**: `analyze_trends.py`
- **Schedule**: Monthly
- **Purpose**: Identify long-term patterns in performance
- **Process**:
  - Analyzes performance across multiple time scales
  - Identifies seasonal or situational patterns
  - Correlates performance with external factors
  - Recommends strategic adjustments

## Storage and Access

### Performance Data Storage
- **Database Table**: `model_performance`
- **Structure**: Stores daily performance metrics for all models and prediction types
- **Retention**: Complete history maintained for long-term analysis

### Performance API Access
- **Endpoint**: `/api/v1/performance`
- **Purpose**: Provide programmatic access to performance metrics
- **Authentication**: Required to protect sensitive performance data

## Version History

### Version 1.0.0 (2025-04-14)
- Initial performance tracking architecture documentation
