# NBA Betting Prediction System

## Overview
A modular, minimal, and scalable NBA betting prediction system using real data from odds and stats APIs. The system auto-detects NBA seasons (current and future), trains models, and generates predictions for the current day's games, including game outcomes, player stats, bankroll recommendations, and CLV analysis.

## Project Structure
This project follows a modular and minimal architecture:

- `src/` - Source code
  - `api/` - API client implementations
    - `balldontlie_client.py` - BallDontLie API client
    - `theodds_client.py` - The Odds API client
  - `utils/` - Utility functions and classes
  - `models/` - Prediction models
  - `data/` - Data processing and feature engineering
  - `examples/` - Example scripts demonstrating usage
- `data/` - Data storage
  - `api_cache/` - Cache for API responses
  - `processed/` - Processed data ready for model training
- `logs/` - Application logs
- `docs/` - Documentation
  - `architecture/` - Detailed architecture documentation
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

### API Components

#### BallDontLie API Client
We use the BallDontLie API (GOAT plan) to access comprehensive NBA data:
- Teams and players information
- Game schedules, results, and statistics
- Player and team season averages
- Advanced statistics and metrics
- Betting odds for games

Example usage:
```python
from src.api.balldontlie_client import BallDontLieClient

# Initialize client
client = BallDontLieClient()

# Get today's games
todays_games = client.get_todays_games()

# Get odds for today's games
todays_odds = client.get_todays_odds()
```

#### The Odds API Client
We use The Odds API to access comprehensive sports betting odds and live scores data:
- Betting odds for games
- Live scores for games

Example usage:
```python
from src.api.theodds_client import TheOddsClient

# Initialize client
client = TheOddsClient()

# Check if NBA is available
nba_available = client.is_nba_available()

# Get today's NBA odds
odds = client.get_todays_odds()

# Get live scores
scores = client.get_live_scores()
```

#### NBA Data Processor
The data processor provides higher-level functions to organize and process data from the API:
- Team and player data retrieval and caching
- Comprehensive game statistics collection
- Season statistics for teams and players
- Today's games with odds and team information

Example usage:
```python
from src.utils.data_processor import NBADataProcessor

# Initialize processor
processor = NBADataProcessor()

# Get today's games with comprehensive data
todays_games = processor.get_todays_games_with_data()

# Save processed data
processor.save_processed_data(todays_games, "games_today")
```

### Scheduling
- Model training and first predictions: 6:00 AM EST daily
- Model performance summary: 2:00 AM EST daily
- Data updates: Every 4 hours (10:00 AM, 2:00 PM, 6:00 PM EST)
- Live game data: Every 10 minutes during active games
- News/injury updates: Every 15 minutes in 2-hour pre-game windows

## Models
The system uses multiple models for predictions:
1. Random Forests
2. Combined (XGBoost + LightGBM)
3. Bayesian
4. Anomaly Detection
5. Model Mixing
6. Ensemble Stacking

## Predictions
Predictions are generated for the current day's games only, including:
- Game outcomes (moneyline, spread, totals)
- Player stats (points, rebounds, assists, threes)
- Bankroll recommendations
- CLV analysis

## Performance Tracking
Model performance is tracked daily and stored in the database. A summary is generated daily at 2:00 AM EST.

## Maintenance
The system is designed for long-term operation with minimal maintenance. It automatically adapts to new NBA seasons and retrains models when performance drops below thresholds.

## Documentation
Detailed documentation is available in the `docs/` directory, including:
- [Architecture details](architecture.md)
- [Data flow diagrams](docs/architecture/data_flow.md)
- [Model designs](docs/architecture/model_design.md)
- [API integrations](docs/architecture/api_integration.md)

## Examples
Example scripts are provided in the `src/examples/` directory:
- `balldontlie_example.py` - Demonstrates how to use the BallDontLie API client and data processor
- `theodds_example.py` - Demonstrates how to use The Odds API client

To run an example:
```bash
python -m src.examples.balldontlie_example
python -m src.examples.theodds_example
```

## Changelog

### Version 1.0.1 (2025-04-14)
- Added BallDontLie API client implementation
- Created data processor for organizing and processing NBA data
- Added example script for demonstrating API usage
- Updated documentation with API usage instructions

### Version 1.0.0 (2025-04-14)
- Initial project setup

### Version 0.2.0 (2025-04-14)
- Added The Odds API integration for comprehensive betting odds data

## NBA Season Management

The system includes a robust season management component that:

- Automatically detects the current NBA season and phase
- Handles transitions between seasons
- Maintains season-specific data in organized directories
- Validates season data with the BallDontLie API

```python
from src.utils.season_manager import SeasonManager
from config.season_config import SeasonPhase

# Create a season manager
season_manager = SeasonManager()

# Get current season information
season_year = season_manager.get_current_season_year()
season_display = season_manager.get_current_season_display()
current_phase = season_manager.get_current_phase()

# Check if we're in an active season
is_in_season = season_manager.is_in_season()

# Get days until next phase
days_till_playoffs = season_manager.days_until_phase(SeasonPhase.PLAYOFFS)
```

## License

Proprietary - All rights reserved
