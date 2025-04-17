# NBA Prediction System: Automation Documentation

*Last Updated: April 17, 2025*

## Overview

This document provides detailed information about all automated components of the NBA Prediction System. Any new automation features added to the codebase should be documented here.

## Table of Contents
- [Scheduled Tasks](#scheduled-tasks)
- [Automated Systems](#automated-systems)
- [File Locations](#file-locations)
- [Setup Instructions](#setup-instructions)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)
- [Season Handling](#season-handling)
- [Data Sources and Reliability](#data-sources-and-reliability)
- [Player Prop Prediction System](#player-prop-prediction-system)
- [Visualization and User Interface](#visualization-and-user-interface)
- [Notification and Alert System](#notification-and-alert-system)
- [Backup and Recovery](#backup-and-recovery)
- [Security Considerations](#security-considerations)
- [Version Control and Rollback](#version-control-and-rollback)
- [Command Line Options](#command-line-options)
- [Cache Management](#cache-management)
- [Error Handling and Fallbacks](#error-handling-and-fallbacks)

## Scheduled Tasks

### Daily Tasks

| Task | Time (EST) | Description | File Location |
|------|------------|-------------|---------------|
| Generate Predictions | 6:00 AM | Creates predictions for today's NBA games | `nba_algorithm/scripts/production_prediction.py` |
| Track Accuracy | 2:00 AM | Records 7-day prediction accuracy | `nba_algorithm/utils/performance_tracker.py` |
| Model Performance Check | 5:00 AM | Monitors model health and sends alerts | `nba_algorithm/scheduling/scheduler.py` |
| Update Stats & Odds | 10:00 AM, 2:00 PM, 6:00 PM | Updates team/player stats and odds | `nba_algorithm/data/game_data.py`, `nba_algorithm/data/odds_data.py` |
| Track CLV | Hourly | Tracks Closing Line Value metrics | `nba_algorithm/utils/performance.py` |

### Weekly Tasks

| Task | Time (EST) | Description | File Location |
|------|------------|-------------|---------------|
| Feature Evolution | Sunday 1:00 AM | Discovers and optimizes prediction features | `src/utils/feature_evolution.py` |

### Bi-Weekly Tasks

| Task | Time (EST) | Description | File Location |
|------|------------|-------------|---------------|
| Model Retraining | Every other Monday 3:00 AM | Retrains models with optimal features | `src/training_pipeline.py` |

### Dynamic Tasks

| Task | Frequency | Description | File Location |
|------|-----------|-------------|---------------|
| Pre-Game Updates | 15 min intervals before games | Updates player injuries and lineups | `nba_algorithm/data/player_data.py` |
| In-Game Updates | 10 min intervals during games | Updates live scores and odds | `nba_algorithm/data/game_data.py` |

## Automated Systems

### Model Registry System

The Model Registry automatically manages model versioning, deployment, and A/B testing.

- **Primary File**: `src/models/model_registry.py`
- **Functions**:
  - `register_model()`: Registers a new model version
  - `get_production_model()`: Retrieves the current production model
  - `set_production_model()`: Sets a model as the production model
  - `setup_ab_test()`: Creates A/B tests between model versions
  - `promote_ab_test_winner()`: Promotes the winning model to production

### Feature Evolution System

The Feature Evolution system automatically discovers, evaluates, and optimizes prediction features.

- **Primary File**: `src/utils/feature_evolution.py`
- **Functions**:
  - `discover_candidate_features()`: Discovers potential new features
  - `engineer_discovered_features()`: Creates new derived features
  - `compare_feature_sets()`: Evaluates feature set performance
  - `select_optimal_features()`: Selects best features for prediction
  - `detect_league_changes()`: Identifies changes in NBA patterns

### Performance Tracking System

The Performance Tracking system monitors model accuracy and provides alerts.

- **Primary File**: `src/utils/performance_tracker.py`
- **Functions**:
  - `record_prediction_results()`: Records prediction outcomes
  - `get_performance_metrics()`: Calculates performance metrics
  - `detect_performance_decline()`: Identifies declining models
  - `export_performance_data()`: Exports data for analysis

## Player Prop Prediction System

The player prop prediction system uses real player data and statistical models to generate accurate predictions.

### Overview

- **Primary Files**: 
  - `nba_algorithm/models/player_props_loader.py`
  - `nba_algorithm/models/player_props_predictor.py`

### Machine Learning Models

The system leverages multiple model types for optimal prediction:

- **GradientBoosting**: Primary model for points, rebounds and assists predictions
- **RandomForest**: Used for specialized prop types and categorical outcomes
- **EnsembleStacking**: Meta-model that combines predictions from multiple models

### Context-Aware Features

The system incorporates sophisticated features for higher accuracy:

- **Matchup Data**: How players perform against specific teams/defenders
- **Defensive Ratings**: Team defensive metrics against specific positions
- **Recent Form**: Weighted recent performance indicators
- **Team Rotation**: Minutes and usage projections

### Automation Schedule

- **Daily Updates**: Player stats and projections updated at 2:00 PM EST
- **Pre-Game Refinement**: Final projections generated 1 hour before game time
- **Model Evaluation**: Performance tracking for each prop type runs daily at 2:00 AM

### Output Format

Player prop predictions are formatted for easy consumption:

- **JSON Structure**: Organized by player, prop type, and projection
- **Confidence Levels**: Each prediction includes a confidence score
- **Storage**: Saved to `predictions/player_props_YYYY-MM-DD.json`

## Visualization and User Interface

The prediction system includes user-friendly display components for viewing predictions and results.

### Console Output

- **Primary File**: `nba_algorithm/output/display.py`
- **Features**:
  - Formatted table display of game matchups
  - Color-coded win probabilities
  - Clear betting insights with recommended plays
  - Summary statistics for model performance

### Output Files

All predictions are automatically saved in multiple formats:

- **CSV Format**: `predictions/nba_predictions_YYYY-MM-DD.csv`
  - Optimized for spreadsheet viewing and analysis
  - Contains all prediction details in tabular format

- **JSON Format**: `predictions/nba_predictions_YYYY-MM-DD.json`
  - Complete structured data including metadata
  - Useful for programmatic access and integration

### Visualization Files

Performance metrics are visualized in:

- **Performance Charts**: `reports/performance_charts_YYYY-MM-DD.png`
  - Rolling accuracy over time
  - ROI tracking for different bet types
  - Model confidence calibration plots

## File Locations

### Core Automation Files

- **Scheduler**: `run_scheduler.py` (project root)
- **Scheduler Implementation**: `nba_algorithm/scheduling/scheduler.py`
- **Production Prediction**: `nba_algorithm/scripts/production_prediction.py`

### Data Storage

- **Logs**: `logs/` directory
  - `scheduler.log`: Scheduler activity
  - `predictions.log`: Prediction generation logs
  - `performance.log`: Model performance logs
- **Predictions**: `predictions/` directory
  - Daily files in format: `nba_predictions_YYYY-MM-DD.json`
- **Models**: `models/` directory
  - `models/registry/`: Model registry storage
- **Feature Data**: `data/features/evolution/` directory

## Setup Instructions

### Initial Setup

1. **API Keys**:
   - Create `.env` file in project root
   - Add: `BALLDONTLIE_API_KEY=your_key` and `THEODDS_API_KEY=your_key`

2. **Windows Service Setup**:
   ```
   # Install NSSM from https://nssm.cc/download
   # Then run these commands as Administrator:
   nssm install NBAPredictor python.exe
   nssm set NBAPredictor AppDirectory C:\Users\bcada\CascadeProjects\NBA ALGORITHM
   nssm set NBAPredictor AppParameters "run_scheduler.py --background"
   nssm set NBAPredictor DisplayName "NBA Prediction System"
   nssm set NBAPredictor Description "Automated NBA prediction system"
   nssm set NBAPredictor Start SERVICE_AUTO_START
   ```

3. **Starting the Service**:
   ```
   nssm start NBAPredictor
   ```

### Alternative: Task Scheduler Setup

If not using Windows Service, set up Task Scheduler:

1. Create a basic task named "NBA Prediction Scheduler"
2. Trigger: When computer starts
3. Action: Start program
4. Program: `python`
5. Arguments: `C:\Users\bcada\CascadeProjects\NBA ALGORITHM\run_scheduler.py --background`
6. Set to run whether user is logged on or not
7. Configure for automatic restart if fails

## Monitoring

### Log Files

Primary logs to check for system status:

1. `logs/scheduler.log`: Overall scheduler activity
2. `logs/predictions.log`: Daily prediction details
3. `logs/performance.log`: Model performance tracking

### Health Checks

To verify the system is running properly:

1. Check Windows Services or Task Manager for running process
2. Check `predictions/` directory for recent prediction files
3. Review log files for any error messages

## Troubleshooting

### Common Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| No predictions generated | 1. No NBA games scheduled<br>2. API key issues<br>3. Scheduler not running | 1. Check if games are scheduled<br>2. Verify API keys in `.env`<br>3. Check service status |
| API errors | 1. Rate limits<br>2. Expired API key<br>3. Network issue | 1. Check API usage<br>2. Renew API key<br>3. Check internet connection |
| Models not improving | 1. Feature evolution issue<br>2. Training pipeline error<br>3. Insufficient data | 1. Check feature evolution logs<br>2. Verify training pipeline<br>3. Wait for more games data |

### Restarting the System

```
# Stop service
nssm stop NBAPredictor

# Start service
nssm start NBAPredictor
```

## Maintenance

### Regular Maintenance Tasks

| Task | Frequency | Instructions |
|------|-----------|-------------|
| Check API key validity | Monthly | Verify keys in `.env` are valid and not near expiration |
| Restart service | Monthly | `nssm restart NBAPredictor` for clean state |
| Check disk space | Monthly | Ensure sufficient space for logs and models |
| Review performance | Quarterly | Check accuracy trends in performance tracker |

### Adding New Automated Components

When adding new automated components to the system:

1. Update this documentation file with:
   - Component name and purpose
   - File location
   - Scheduling information (if applicable)
   - Any configuration requirements

2. Ensure the component follows these automation principles:
   - Robust error handling
   - Logging for all activities
   - Season-aware behavior
   - Fallback mechanisms

3. Test the component thoroughly before deployment

## Season Handling

The prediction system automatically detects and adapts to the NBA season schedule.

### Season Detection

- **Primary File**: `nba_algorithm/utils/season_manager.py`
- **Season States**:
  - **Regular Season**: Full prediction schedule with daily updates
  - **Playoffs**: Enhanced prediction frequency with special playoff features
  - **Off-Season**: Reduced activity focused on model improvement
- **Transition Handling**: Season transitions are detected automatically without requiring manual configuration

### Off-Season Operations

During the NBA off-season (typically June-September), the system:

1. Suspends daily game predictions (no games to predict)
2. Continues running model improvement tasks:
   - Feature evolution research runs weekly
   - Historical performance analysis runs monthly
   - Model architecture experiments run bi-weekly
3. Prepares for the upcoming season by:
   - Analyzing draft impacts
   - Incorporating roster changes
   - Adjusting for rule modifications

## Data Sources and Reliability

The prediction system relies entirely on real data from official APIs with no fallbacks to synthetic or sample data.

### Primary Data Sources

- **The Odds API**: Used for betting odds data
  - **Configuration**: API key required in `.env` file
  - **Usage**: Provides live betting odds for moneyline, spread, and totals markets
  - **Error Handling**: System will alert rather than use synthetic data when API is unavailable

- **BallDontLie API**: Used for NBA statistics
  - **Configuration**: API key required in `.env` file
  - **Usage**: Provides player statistics, game data, and team performance metrics
  - **Rate Limiting**: System respects API rate limits with appropriate throttling

### Data Quality Assurance

- **Validation**: All incoming data is validated before use in predictions
- **Consistency Checks**: Data is checked for consistency with previous records
- **Missing Data Handling**: Predictions require complete data; no interpolation of missing values
- **Error Alerting**: System generates alerts when data quality issues are detected

### Live Data Priority

The system is configured to prioritize data reliability:

1. Will **not** fall back to synthetic data when APIs are unavailable
2. Will delay predictions rather than use incomplete information
3. Includes comprehensive error messages when data cannot be retrieved

## Notification and Alert System

The prediction system includes automated notifications for important events and errors.

### Alert Types

- **API Failures**: Generated when API requests fail consistently
- **Model Performance**: Generated when model accuracy drops below thresholds
- **System Health**: Generated for disk space, memory usage, or process issues
- **Prediction Opportunities**: Generated for high-confidence betting opportunities

### Alert Methods

- **Log Files**: All alerts are recorded in `logs/alerts.log`
- **Email Notifications**: Critical alerts can be configured to send emails
- **Console Messages**: Displayed when running in interactive mode

### Configuration

Alert thresholds and recipients can be configured in:

- **Primary File**: `nba_algorithm/utils/alerts.py`
- **Configuration File**: `config/alerts_config.json`

## Backup and Recovery

The system includes automated backup procedures to prevent data loss.

### Model Backups

- **Frequency**: Daily backups of all model files
- **Location**: `backups/models/YYYY-MM-DD/`
- **Retention**: 30 days of rolling backups

### Data Backups

- **Frequency**: Weekly backups of all collected data
- **Location**: `backups/data/YYYY-MM-DD/`
- **Retention**: 8 weeks of rolling backups

### Recovery Procedure

To restore from backup:

1. Stop the prediction service: `nssm stop NBAPredictor`
2. Copy files from relevant backup directory to production location
3. Restart the service: `nssm start NBAPredictor`

## Security Considerations

The prediction system implements several security measures.

### API Key Protection

- API keys are stored in `.env` file (not in source control)
- Environment variables are used for access instead of hardcoded values
- Access to the prediction system should be restricted to authorized users

### Update Procedures

1. Make system code changes in a controlled manner
2. Test changes thoroughly before deploying to production
3. Use version control system to track all code modifications
4. Document all system changes in a changelog

## Version Control and Rollback

The system uses Git for version control and includes rollback procedures.

### Git Repository Structure

- **Main Branch**: Contains stable production code
- **Development Branch**: For new features before production deployment
- **Feature Branches**: For individual feature development

### Rollback Procedure

To roll back to a previous version:

1. Stop the prediction service: `nssm stop NBAPredictor`
2. Switch to the desired Git version: `git checkout [tag/commit]`
3. Restart the service: `nssm start NBAPredictor`

### Version Tagging

Major system versions are tagged in Git for easy reference:

- Tags follow semantic versioning (e.g., v1.2.3)
- Each tag has a corresponding entry in `CHANGELOG.md`
- Production deployments should always use a tagged version

## Command Line Options

The prediction system supports numerous command line options for flexibility when running manually or from automation.

### Main Production Script Options

The `nba_algorithm/scripts/production_prediction.py` script accepts these key parameters:

| Option | Description | Usage |
|--------|-------------|-------|
| `--date` | Specific date for predictions (YYYY-MM-DD) | `--date 2025-04-20` |
| `--output-dir` | Directory for saving prediction results | `--output-dir predictions` |
| `--log-level` | Sets logging verbosity | `--log-level DEBUG` |
| `--include-players` | Include player props predictions | `--include-players` |
| `--no-cache` | Bypass all caching and fetch fresh data | `--no-cache` |
| `--clear-cache` | Clear all cache before running | `--clear-cache` |
| `--history_days` | Days of historical data to use | `--history_days 45` |
| `--risk_level` | Sets risk tolerance for bet recommendations | `--risk_level moderate` |

### Running Manually

Example of running the script with options:

```bash
python -m nba_algorithm.scripts.production_prediction --date 2025-04-20 --include-players --risk_level aggressive
```

## Cache Management

The prediction system implements a sophisticated caching system to optimize performance.

### Cache Tiers

- **Volatile Cache**: Short-lived data (odds, lineups)
  - Cleared automatically daily
  - Can be cleared with `--clear-volatile-cache`

- **Persistent Cache**: Longer-term data (team stats, historical games)
  - Retained across runs
  - Only cleared with explicit `--clear-cache` flag

### Cache Locations

- Primary location: `data/cache/`
- Cache metadata: `data/cache/cache_index.json`

### Automated Cache Management

- Volatile cache is cleared nightly at 1:00 AM
- Full cache is rebuilt weekly on Saturdays at 4:00 AM
- Cache verification runs hourly to detect corrupt entries

## Error Handling and Fallbacks

The prediction system implements robust error handling to ensure continuous operation.

### Error Response Hierarchy

1. **API Errors**: If an API fails to respond
   - System retries up to 3 times with exponential backoff
   - Alerts are generated after continuous failures
   - System will not fall back to synthetic data (uses last valid data)

2. **Model Loading Errors**: If models cannot be loaded from registry
   - System attempts to load from traditional location
   - If both fail, system logs error and exits

3. **Feature Selection Errors**: If feature optimization fails
   - System falls back to production feature set
   - If that fails, uses base feature set

4. **Prediction Generation Errors**: If prediction process fails
   - Each prediction type (moneyline, spread, total) handled separately
   - Partial results returned when possible

---

*This document should be updated whenever new automation features are added to the codebase.*
