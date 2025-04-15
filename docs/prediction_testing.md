# NBA Prediction Testing System

## Overview

The prediction testing system (`scripts/test_predictions.py`) enables comprehensive evaluation of all trained models on upcoming NBA games, allowing for comparison of predictions against actual outcomes and between different model types.

## Features

### Prediction Generation
- Loads all trained models from the database
- Collects upcoming game data from BallDontLie API
- Applies feature engineering to prepare prediction inputs
- Runs predictions using all available models

### Performance Visualization
- Generates probability distribution charts for moneyline predictions
- Creates comparison visualizations between different models
- Produces game-by-game prediction breakdowns

### Result Analysis
- Exports predictions to CSV for detailed analysis
- Calculates confidence metrics for each prediction
- Identifies high-value betting opportunities

## Usage

```bash
# Basic usage - predicts games for the next 7 days
python scripts/test_predictions.py

# Specify number of days to predict
python scripts/test_predictions.py --days 14

# Save visualizations and CSV files to a specific directory
python scripts/test_predictions.py --output reports/predictions

# Test only specific models
python scripts/test_predictions.py --models RandomForestModel GradientBoostingModel
```

## Output Examples

### CSV Output
The script generates a CSV file with predictions from all models including:
- Game details (teams, date, time)
- Win probability for each team
- Point spread predictions
- Confidence metrics

### Visualization Output
- Model comparison charts
- Probability distribution graphs
- Game-by-game prediction breakdowns

## Technical Details

### Real Data Integration
The prediction system only uses real data from the APIs, never falling back to synthetic or sample data. This ensures production-quality predictions and adheres to best practices for real-world deployment.

### Error Handling
Comprehensive error handling ensures the system can gracefully recover from API failures, model loading issues, or data inconsistencies without crashing.

### Performance Considerations
Model loading and feature engineering are optimized for performance, making prediction generation quick enough for interactive use.
