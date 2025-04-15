# NBA Betting Prediction System

## Overview
A modular, minimal, and scalable NBA betting prediction system using real data from odds and stats APIs. The system auto-detects NBA seasons (current and future), trains models, and generates predictions for the current day's games, including game outcomes, player stats, bankroll recommendations, and CLV analysis.

## Project Structure
This project follows a modular and minimal architecture:

- `src/` - Source code
  - `api/` - API client implementations
    - `balldontlie_client.py` - BallDontLie API client
    - `theodds_client.py` - The Odds API client
    - `server.py` - API server module
    - `direct_data_access.py` - Direct API data access without database dependency
    - `model_predictions.py` - Connects trained models to API endpoints
  - `database/` - Database models and connection management
    - `connection.py` - PostgreSQL connection pool management
    - `robust_connection.py` - Enhanced connection manager with pooling and retry logic
    - `models.py` - Database models (Game, Player, ModelWeight, ModelPerformance)
    - `init_db.py` - Database initialization script
  - `utils/` - Utility functions and classes
  - `models/` - Prediction models
    - `base_model.py` - Base model interface for all prediction models
    - `random_forest_model.py` - Random Forest classifier for moneyline predictions
    - `gradient_boosting_model.py` - Gradient Boosting regressor for spread predictions
    - `bayesian_model.py` - Bayesian model for probabilistic predictions
    - `ensemble_model.py` - Stacking ensemble that combines multiple models
    - `training_pipeline.py` - End-to-end training system for all models
    - `model_deployer.py` - Production deployment system for trained models
  - `data/` - Data processing and feature engineering
    - `historical_collector.py` - Collects historical NBA data from APIs
    - `feature_engineering.py` - Creates features for model training
  - `examples/` - Example scripts demonstrating usage
- `data/` - Data storage
  - `api_cache/` - Cache for API responses
  - `processed/` - Processed data ready for model training
  - `models/` - Trained model files
  - `historical/` - Historical NBA data
  - `features/` - Engineered features
  - `production_models/` - Production-ready deployed models
- `logs/` - Application logs
- `docs/` - Documentation
  - `architecture.md` - Detailed architecture documentation
  - `api_server_guide.md` - API server documentation
  - `api_setup_guide.md` - API setup guide
  - `docker_guide.md` - Docker deployment guide
- `config/` - Configuration files
  - `api_keys.py` - API key configuration (not committed to git)
  - `api_keys.py.sample` - Sample API key configuration template
- `tests/` - Unit and integration tests

## Project Rules
This project adheres to specific rules and guidelines documented in [PROJECT_RULES.md](PROJECT_RULES.md). Please review these rules before contributing to the project.

## Setup
1. Ensure Python 3.9+ is installed
2. Run `setup_env.py` to set up the virtual environment
3. Configure API keys in the config/api_keys.py file
4. Activate the virtual environment:
   - Windows: `venv\Scripts\activate.ps1`
   - Unix/Mac: `source venv/bin/activate`

## Usage
The system runs automatically on scheduled intervals. No manual execution is required.

### Database Integration

The NBA Prediction System uses PostgreSQL to store and manage data for long-term operation:

### Database Setup

The system includes a production database setup script that populates your database with real NBA data:

```bash
# Set up the database with real NBA data from APIs
python -m src.database.setup_production_db
```

This script will:
1. Create all necessary database tables and indexes
2. Fetch real teams, players and games from the BallDontLie API
3. Configure model templates for training

### Advanced Model Training

The system now includes an advanced model training pipeline that uses real NBA data:

```bash
# Train all prediction models with real NBA data
python -m src.models.training_pipeline
```

Features:
- Fetches historical game data from BallDontLie and The Odds APIs
- Performs comprehensive feature engineering on real NBA statistics
- Trains multiple model types:
  - Random Forest Classifier for moneyline (win/loss) predictions
  - Gradient Boosting Regressor for spread predictions
  - Bayesian Model for probabilistic predictions
  - Ensemble Stacking Model for optimal combined predictions
- Cross-validation and hyperparameter optimization for all models
- Evaluates model performance with proper validation
- Saves trained models with version control

### Model Deployment

To deploy trained models to production for use in the API:

```bash
# Deploy trained models to production
python -m src.models.model_deployer --deploy --model ensemble --target moneyline
```

Features:
- Manages model versions and deployment
- Supports rollback to previous model versions
- Tracks model performance metrics
- Enables multiple models for different prediction targets

### Database Models
- **Games** - Stores game information, odds, features, and predictions
- **Players** - Stores player information, statistics, and performance predictions
- **ModelWeights** - Stores trained model weights and parameters with version control
- **ModelPerformance** - Tracks model accuracy and performance metrics over time

#### Database Setup

1. Ensure PostgreSQL (version 12 or higher) is installed
2. Create a new database named `nba_prediction`:
   ```sql
   CREATE DATABASE nba_prediction;
   ```
3. Run the database initialization script:
   ```bash
   python -m src.database.init_db --verbose
   ```

The database connection is configured through environment variables:
- `POSTGRES_HOST` - Database host (default: localhost)
- `POSTGRES_PORT` - Database port (default: 5432)
- `POSTGRES_DB` - Database name (default: nba_prediction)
- `POSTGRES_USER` - Database username (default: postgres)
- `POSTGRES_PASSWORD` - Database password (default: postgres)

### API Components

#### BallDontLie API Client
We use the BallDontLie API (GOAT plan) to access comprehensive NBA data:
- Teams and players information
- Game schedules, results, and statistics
- Player and team season averages
- Advanced statistics and metrics

Example usage:
```python
from src.api.balldontlie_client import BallDontLieClient

# Initialize client
client = BallDontLieClient()

# Get today's games
todays_games = client.get_todays_games()
```

#### The Odds API Client
We use The Odds API to access comprehensive sports betting odds and live scores data:
- Betting odds for games
- Live scores for games
- Historical odds data

Example usage:
```python
from src.api.theodds_client import TheOddsClient

# Initialize client
client = TheOddsClient()

# Check if NBA is available
nba_available = client.is_nba_available()

# Get today's NBA odds
odds = client.get_todays_odds()
```

#### Direct Data Access
The system provides direct API access functionality that doesn't rely on database connectivity:

```python
from src.api.direct_data_access import DirectDataAccess

# Initialize the direct data access
data_access = DirectDataAccess()

# Get upcoming NBA games
upcoming_games = data_access.get_upcoming_games(days=7)
```

### API Integration

The system provides a comprehensive REST API for monitoring and controlling the prediction models through the `src/api/server.py` module:

#### Training Control Endpoints

- `GET /api/health` - Health check endpoint to verify API is operational
- `POST /api/training/start` - Start model training (requires authentication)
- `GET /api/training/status` - Check current training status
- `POST /api/training/cancel` - Cancel ongoing training (requires authentication)

#### Model Management Endpoints

- `GET /api/models/list` - List all available models
- `GET /api/models/{model_name}/details` - Get detailed information about a specific model
- `POST /api/models/{model_name}/retrain` - Trigger retraining for a specific model (requires authentication)

#### Performance Monitoring Endpoints

- `GET /api/models/performance` - Get performance metrics for all models
- `GET /api/models/drift` - Check for model drift and see which models need retraining

#### Prediction Endpoints

- `GET /api/predictions/today` - Get predictions for today's games
- `GET /api/predictions/upcoming` - Get predictions for upcoming games
- `GET /api/predictions/recent` - Get recent predictions and their outcomes

To run the API server:

```bash
python -m src.api.server
```

### Key Components

### Automatic Season Detection
- **Season Manager**: Sophisticated season detection system that automatically identifies the current NBA season
- **Future-Proof Design**: Training pipeline automatically adapts to season transitions without manual updates
- **SeasonPhase Detection**: Identifies current phase (preseason, regular season, playoffs, etc.)
- **Season Transition Handling**: Manages data archiving and initialization during season transitions
- **Command-Line Override**: Allows manual season specification when needed, but defaults to auto-detection

### Model Training Pipeline
- **Modular Design**: Allows for easy addition of new models and training algorithms
- **Real Data Integration**: Trains models using real NBA data from BallDontLie and The Odds APIs
- **Comprehensive Feature Engineering**: Generates features for model training using real NBA statistics
- **Cross-Validation and Hyperparameter Tuning**: Optimizes model performance with proper validation
- **Model Versioning and Deployment**: Manages model versions and deployment for production use

### Feature Engineering

The system includes a robust feature engineering pipeline that transforms raw NBA data into features suitable for machine learning models:

```python
from src.data.feature_engineering import NBAFeatureEngineer

# Initialize feature engineer
engineer = NBAFeatureEngineer()

# Load games data
games_df = engineer.load_games()

# Generate features for all games
features = engineer.generate_features_for_all_games(games_df)

# Prepare training data
X_train, y_train = engineer.prepare_training_data(features, target='home_win')
```

Features generated include:
- Team performance metrics (recent form, scoring trends, etc.)
- Head-to-head history between teams
- Rest days and travel impact
- Home court advantage effects
- Comprehensive comparative metrics

### Performance Tracking

The system tracks prediction performance to evaluate model accuracy over time using the `src/utils/performance_tracker.py` module.

To run performance tracking manually or generate a summary:
```bash
python -m src.utils.track_performance [--summary] [--days DAYS]
```

## Recent Enhancements (April 2025)

### 1. Complete Model Training Pipeline
- **Fixed All Models**: All 6 prediction models now train successfully with proper error handling
- **Parameter Compatibility**: Resolved constructor parameter mismatches across model classes
- **Flexible Training**: Added support for different prediction targets (moneyline, spread, totals)

### 2. Enhanced Feature Engineering
- **Advanced Features**: Implemented `EnhancedFeatureEngineer` with fatigue modeling, team chemistry metrics, and venue-specific advantages
- **Context Awareness**: Added matchup history, travel distance, and style compatibility metrics
- **Quality Improvements**: Enhanced missing value handling and feature preprocessing

### 3. Prediction Testing & Monitoring
- **Test Script**: Added `scripts/test_predictions.py` for evaluating model performance on upcoming games
- **Visualization**: Implemented confidence metrics and comparative model analysis
- **Performance Monitoring**: Created `scripts/monitor_performance.py` to track model performance over time

### 4. Automated Retraining System
- **Scheduler**: Implemented `scripts/auto_update_scheduler.py` for automatic model updates
- **Configurable**: Support for daily, weekly, or post-game training schedules
- **Production Quality**: Added error handling, notifications, and performance thresholds

### 5. Real Data Integration
- **Live Data Only**: Removed all synthetic data fallbacks for production quality
- **API Integration**: Enhanced BallDontLie and TheOdds API clients with better error handling
- **Data Validation**: Added integrity checks for features and model inputs

## Recent Codebase Cleanup (April 2025)

The following changes were made to clean up the codebase and improve organization:

1. **Removed Redundant Files**:
   - `diagnose_data_collection.py` - Obsolete with the new comprehensive error handling
   - `test_odds_api.py` - Superseded by the robust `scripts/test_predictions.py`
   - `initialize_database.bat` - No longer needed with improved database management
   - `ui/templates/dashboard.html.bak` - Removed backup file

2. **Reorganized Example Files**:
   - Moved example files from `src/examples/` to `docs/examples/` for better organization
   - Examples are preserved for reference while keeping the production code clean
   - Available examples include API usage, database operations, and model implementations

## Troubleshooting

### Database Connectivity Issues
If you experience database connectivity problems:

1. Verify PostgreSQL is running:
   ```bash
   # On Linux/Mac
   pg_isready
   
   # On Windows
   pg_isready -U postgres
   ```

2. Check connection parameters in environment variables or `.env` file
3. The system includes a robust connection manager that handles connection errors and retries

### API Rate Limits
The system is designed to respect API rate limits, but you may encounter issues if running multiple instances or making frequent manual requests.

1. BallDontLie API: 1000 requests per day (standard plan)
2. The Odds API: 500 requests per month (standard plan)

The system implements caching to minimize API calls. Check the logs for rate limit warnings.

## Acknowledgements
- [BallDontLie API](https://www.balldontlie.io/) for NBA data
- [The Odds API](https://the-odds-api.com/) for betting data

## Changelog

### Version 1.0.0 (2025-04-14)
- Full production implementation with real-time API data
- Four advanced prediction models with ensemble stacking
- Robust feature engineering pipeline
- Comprehensive training and deployment system
- Enhanced API endpoints for predictions

### Version 0.2.0 (2025-04-14)
- Added The Odds API integration for comprehensive betting odds data
- Implemented robust database connection manager
- Added direct API data access for improved reliability

### Version 0.1.0 (2025-04-01)
- Initial release with basic prediction functionality
- PostgreSQL database integration
- BallDontLie API integration

## NBA Season Management

The system automatically detects NBA seasons and handles the transition between seasons, including preseason, regular season, and playoffs. This ensures continuous operation without manual intervention.

## License
This project is licensed under the terms of the MIT license.
