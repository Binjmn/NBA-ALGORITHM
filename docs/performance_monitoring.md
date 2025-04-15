# NBA Model Performance Monitoring System

## Overview

The model performance monitoring system (`scripts/monitor_performance.py`) provides comprehensive tracking of prediction model performance over time, enabling data-driven decisions about model retraining, deployment, and configuration adjustments.

## Features

### Performance History Tracking
- Extracts performance metrics from the database
- Organizes metrics by model, prediction type, and time period
- Calculates trend analysis to identify model drift

### Visualization Capabilities
- Generates performance trend charts over time
- Creates model comparison visualizations
- Produces heatmaps of metric changes

### Automated Reporting
- Generates Markdown reports with key findings
- Highlights models requiring attention
- Provides actionable recommendations based on performance data

## Usage

```bash
# Basic usage - analyzes all available performance data
python scripts/monitor_performance.py

# Analyze a specific time window
python scripts/monitor_performance.py --time-window 30d  # 7d, 30d, 90d, all

# Save reports and visualizations to a specific directory
python scripts/monitor_performance.py --output reports/performance

# Generate report only without visualizations
python scripts/monitor_performance.py --report-only
```

## Report Components

### Summary Statistics
- Latest performance metrics for each model
- Model version information
- Prediction targets and sample counts

### Performance Trends
- Metric changes over time (percentage and absolute)
- Improvement/decline indicators
- Statistical significance of changes

### Recommendations
- Models requiring retraining
- Suggested configuration adjustments
- Best-performing models for production use

## Database Integration

The monitoring system automatically logs its activity to the database's system_logs table, providing an audit trail of when monitoring was performed and what findings were discovered.

## Technical Details

### Metric Calculations
- Classification models: accuracy, precision, recall, F1, AUC
- Regression models: MAE, MSE, RMSE, R²
- Calibration metrics for probability outputs

### Visualization Methods
- Time series charts for metric tracking
- Bar charts for model comparisons
- Heatmaps for multi-dimensional performance analysis
