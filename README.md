# NBA Betting Prediction System

## Overview
A modular, minimal, and scalable NBA betting prediction system using real data from odds and stats APIs. The system auto-detects NBA seasons (current and future), trains models, and generates predictions for the current day's games, including game outcomes, player stats, bankroll recommendations, and CLV analysis.

## Key Features
- **Multi-Season Training**: Uses 4 seasons of NBA data (current + 3 previous) for more robust and accurate predictions
- **Real Data APIs**: Integrates with BallDontLie API for game statistics and The Odds API for betting markets
- **Enhanced Model Architecture**: Ensemble stacking with robust cross-validation for superior prediction accuracy
- **Production-Quality Implementation**: Enterprise-grade logging, error handling, and health monitoring
- **Reliable Data Collection**: Robust API integration with proper error handling and fallback mechanisms
- **Pipeline Robustness**: Automatic data validation, outlier detection, feature drift monitoring, and model checkpointing
- **Automatic Season Detection**: System automatically detects NBA seasons and transitions, adapting predictions and models accordingly

## Project Structure (Updated April 2025)
This project now follows a clean Python package structure for better organization, maintainability, and importability:

```
nba_algorithm/          # Main package directory
├── __init__.py         # Package initialization
├── api/                # API client implementations
│   ├── __init__.py
│   ├── balldontlie_client.py
│   ├── theodds_client.py
│   └── ... 
├── config/             # Configuration modules
│   ├── __init__.py
│   └── season_config.py # NBA season definitions and phase transitions
├── data/               # Data acquisition modules
│   ├── __init__.py
│   ├── game_data.py
│   ├── player_data.py
│   ├── team_data.py
│   └── odds_data.py
├── database/           # Database models and connections
│   ├── __init__.py
│   ├── connection.py
│   ├── models.py
│   └── ...
├── features/           # Feature engineering modules
│   ├── __init__.py
│   ├── game_features.py
│   └── player_features.py
├── models/             # ML model management 
│   ├── __init__.py
│   ├── loader.py       # Model loading with season awareness
│   ├── predictor.py    # Prediction engine
│   ├── player_predictor.py # Player prop predictions
│   └── model_registry.py # Model versioning and registry system
├── output/             # Output formatting and storage
│   ├── __init__.py
│   ├── display.py      # Formatted display of predictions
│   └── persistence.py  # Saving predictions to files
├── predictions/        # Prediction engine modules
│   ├── __init__.py
│   └── prediction_engine.py # Core prediction functionality
├── scripts/            # Entry point scripts
│   ├── __init__.py
│   └── production_prediction.py # Main prediction script
├── utils/              # Utility modules
│   ├── __init__.py
│   ├── settings.py     # Settings management
│   ├── season_manager.py # Season detection and management
│   ├── logger.py       # Logging utilities
│   ├── performance_tracker.py # Model performance tracking
│   └── feature_evolution.py # Feature discovery and optimization
└── scheduler/          # Scheduling module
    ├── __init__.py
    └── scheduler.py    # Scheduling logic

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nba-algorithm.git
cd nba-algorithm

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run predictions for today's games
python run_prediction.py

# Run predictions for a specific date
python run_prediction.py --date 2025-04-16

# Include player prop predictions
python run_prediction.py --include_players

# Adjust risk tolerance for bankroll management
python run_prediction.py --risk_level aggressive --bankroll 1000
```

## Season Detection & Management

The NBA Prediction System now features automatic season detection and management, ensuring it stays accurate and relevant across multiple seasons without manual intervention.

### How Season Detection Works

1. **Automatic Season Detection**: The system automatically determines the current NBA season based on the date, including the specific phase (pre-season, regular season, playoffs, etc.).

2. **Season Transitions**: When transitioning between seasons (e.g., from 2024-25 to 2025-26), the system automatically adapts its models and predictions to the new season context.

3. **Phase-Aware Predictions**: The prediction engine adjusts its confidence levels and analysis based on the current season phase - predictions during the playoffs use different considerations than those during the regular season.

4. **Future-Proof Configuration**: Season date ranges are pre-configured through the 2026-27 season, with a robust fallback mechanism for dates beyond the configured ranges.

### Season Manager Features

- **Season Phase Detection**: Accurately identifies the current phase (pre-season, regular season, all-star break, playoffs, finals, off-season)
- **Contextual Warning System**: Provides warnings when making predictions during the off-season
- **Season Information Enrichment**: Adds season context to all prediction outputs
- **Season Transition Handling**: Includes callback system for season and phase changes
- **Regular Updates**: Periodically checks for season changes to ensure predictions stay current

## Scheduling System

The NBA Prediction System includes a robust scheduling module that runs predictions and updates on a defined schedule. This ensures the system always provides timely and accurate predictions for NBA games and player props.

### Production Schedule

The system follows this schedule (all times in EST):

- **6:00 AM Daily**: Generates first predictions for NBA games and player props (moneyline, spread, totals, points, rebounds, assists, threes), including bankroll recommendations, CLV, and confidence scores.

- **2:00 AM Daily**: Tracks 7-day prediction accuracy (moneyline, spread, totals, props) using game outcomes, storing results to guide future model weighting.

- **Every 4 Hours (10:00 AM, 2:00 PM, 6:00 PM)**: Updates team/player stats and odds, recalculating predictions if significant changes occur (e.g., odds shift >5%).

- **Every 15 Minutes in 2-Hour Pre-Game Windows, Else Hourly**: Refreshes player injury statuses, checking frequently before games (e.g., 5:00–7:00 PM for 7:00 PM game) and hourly otherwise, to catch roster changes.

- **Every 10 Minutes During Games**: Updates live scores and odds for real-time predictions, adjusting moneyline, spread, totals, and prop bets.

- **Hourly**: Tracks CLV by comparing early odds to current odds, calculating betting advantages for value bets.

### Running the Scheduler

The scheduler can be run using the provided entry point script:

```bash
# Run in blocking mode (default)
python run_scheduler.py

# Run in background mode
python run_scheduler.py --background
```

## API Keys

This system requires API keys for data collection:

1. BallDontLie API: https://www.balldontlie.io/
2. The Odds API: https://the-odds-api.com/

Create a `.env` file in the project root with your API keys:

```
BALLDONTLIE_API_KEY=your_api_key_here
THEODDS_API_KEY=your_api_key_here
```

## Components

### Data Collection Modules

#### game_data.py
Retrieves NBA game data from BallDontLie API with robust error handling and rate limit management.

#### team_data.py
Fetches comprehensive team statistics for predictions.

#### player_data.py
Gathers player statistics, injuries, and game participation data.

#### odds_data.py
Retrieves betting odds from The Odds API with market analysis functionality.

### Feature Engineering

#### game_features.py
Transforms raw NBA data into predictive features for machine learning models.

#### player_features.py
Generates player-specific features for prop betting predictions.

#### feature_evolution.py
Automatically discovers, evaluates, and optimizes feature sets for different prediction types. Detects changes in league patterns and adapts features accordingly.

### Model Management Modules

#### loader.py
Handles loading of trained prediction models from storage with proper validation and error handling.

#### predictor.py
Applies loaded models to features to generate predictions with confidence scoring.

#### player_predictor.py
Specialized prediction logic for player prop betting.

#### model_registry.py
Manages model versioning, tracks model lineage, and provides a centralized registry for all trained models. Supports production model designation, A/B testing, and model comparison.

### Performance Tracking

#### performance_tracker.py
Tracks and analyzes model performance over time, maintaining historical records of prediction accuracy. Detects underperforming models and provides metrics for model improvement.

## Advanced Systems

### Model Registry System

The NBA Prediction System now features a comprehensive Model Registry system that handles model versioning, tracking, and deployment:

#### Key Features

- **Model Versioning**: Maintains a history of all trained models with version information
- **Performance Tracking**: Records and compares performance metrics across model versions
- **Production Model Management**: Designates specific model versions as production-ready
- **Model Lineage**: Tracks relationships between models and their training data
- **A/B Testing**: Supports comparison testing between different model versions
- **Model Retrieval**: Fast loading of models by name, version, or other criteria

#### Usage

```python
# Example: Registering a newly trained model
from src.models.model_registry import ModelRegistry

registry = ModelRegistry()
registry.register_model(
    model_name="moneyline_rf",
    model_type="moneyline",
    version="20250510",
    model_path="models/moneyline_rf_20250510.pkl",
    metrics={"accuracy": 0.72, "f1": 0.68},
    metadata={"features": ["home_win_rate", "away_defense_rating"]},
    register_as_production=True
)

# Example: Loading a production model
model = registry.get_production_model("moneyline")
```

### Feature Evolution System

The Feature Evolution system automatically discovers, evaluates, and optimizes feature sets for all prediction types:

#### Key Features

- **Feature Discovery**: Identifies potential new predictive features through pattern analysis
- **Automatic Engineering**: Generates interaction terms, rolling averages, and other derived features
- **Performance Evaluation**: Tests new feature sets against baseline to measure improvements
- **League Pattern Detection**: Identifies changes in NBA playing patterns that affect feature importance
- **Optimal Feature Selection**: Selects the best feature combinations for each prediction type

#### Usage

```python
# Example: Discovering optimal features for a prediction task
from src.utils.feature_evolution import FeatureEvolution

fe = FeatureEvolution()
optimal_features = fe.select_optimal_features(
    data=training_data,
    target=targets,
    prediction_type="spread",
    is_classification=False
)

# Example: Detecting changes in league patterns
changes = fe.detect_league_changes(
    current_data=current_season_data,
    historical_data=previous_seasons_data,
    features=current_feature_set
)
```

### Performance Tracking System

The Performance Tracking system monitors and analyzes model accuracy over time:

#### Key Features

- **Prediction Recording**: Records all predictions and actual outcomes
- **Metric Calculation**: Computes accuracy, ROI, and other performance metrics
- **Historical Tracking**: Maintains time-series performance data
- **Early Warning**: Detects deteriorating model performance
- **Performance Reporting**: Generates detailed performance reports

#### Usage

```python
# Example: Recording prediction results
from src.utils.performance_tracker import PerformanceTracker

tracker = PerformanceTracker()
tracker.record_prediction_results(
    date="2025-05-10",
    model_name="spread_xgboost",
    prediction_type="spread",
    predictions=predictions_list,
    actual_results=actual_outcomes
)

# Example: Getting performance metrics
metrics = tracker.get_performance_metrics(
    model_name="spread_xgboost",
    prediction_type="spread",
    date_range=("2025-04-01", "2025-05-01")
)
```

### Continuous Learning Pipeline

The system implements a continuous learning pipeline that ensures models are regularly updated with the latest data and features:

1. **Automatic Data Collection**: Daily gathering of game, player, and odds data
2. **Feature Evolution**: Weekly analysis of feature effectiveness and discovery of new patterns
3. **Model Retraining**: Biweekly retraining of models with optimized feature sets
4. **Performance Monitoring**: Daily tracking of prediction accuracy and model health
5. **Production Deployment**: Automatic deployment of improved models to production

This pipeline ensures that the prediction system adapts to changes in the NBA landscape and maintains high accuracy throughout the season.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### Version 1.2.0 (2025-04-16)
- Added automatic season detection and management system
- Integrated season-aware prediction engine
- Updated model loader to support season-specific models
- Improved output schema with season context

### Version 1.1.0 (2025-04-15)
- Fully modularized codebase for improved maintainability
- Added bankroll management module with Kelly criterion support
- Implemented CLV tracking for betting performance analysis
- Created robust output and persistence modules

### Version 1.0.0 (2025-04-10)
- Complete refactoring to package-based architecture
- Implemented robust database connection manager
- Added direct API data access for improved reliability

### Version 0.1.0 (2025-04-01)
- Initial release with basic prediction functionality
