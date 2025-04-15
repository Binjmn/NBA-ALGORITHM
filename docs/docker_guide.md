# Docker Implementation Guide for NBA Prediction System

This guide explains how to use Docker to deploy the NBA prediction system with auto-training capabilities. The Docker implementation allows you to run the entire system, including the database, prediction models, and training scheduler in a containerized environment.

## Architecture

The Docker implementation consists of three main components:

1. **NBA Prediction Service (`nba-prediction`)**: The core prediction system that includes all the models and training logic.
2. **PostgreSQL Database (`nba-db`)**: Stores game data, player information, model weights, and performance metrics.
3. **Web UI (`nba-ui`)**: A simple web interface for monitoring and interacting with the prediction system.

## Prerequisites

- Docker and Docker Compose installed on your system
- API keys for The Odds API and BallDontLie API (optional but recommended)

## Environment Variables

The following environment variables can be set before starting the containers:

- `ODDS_API_KEY`: Your API key for The Odds API
- `BALLDONTLIE_API_KEY`: Your API key for BallDontLie API

You can set these environment variables in your shell before running Docker Compose:

```bash
export ODDS_API_KEY="your_odds_api_key"
export BALLDONTLIE_API_KEY="your_balldontlie_api_key"
```

Alternatively, you can create a `.env` file in the project root with these variables.

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/yourusername/nba-prediction.git
cd nba-prediction
```

2. Set your API keys (optional but recommended):

```bash
export ODDS_API_KEY="your_odds_api_key"
export BALLDONTLIE_API_KEY="your_balldontlie_api_key"
```

3. Build and start the containers:

```bash
docker-compose up -d --build
```

4. Check the status of the containers:

```bash
docker-compose ps
```

5. View the logs:

```bash
docker-compose logs -f nba-prediction
```

## Auto-Training Schedule

The system is configured with the following automatic training and update schedule:

- **Daily Model Training**: 6:00 AM EST - Trains all models with new data
- **Performance Tracking**: 2:00 AM EST - Calculates accuracy metrics for all models
- **Model Drift Detection**: Every 6 hours - Checks if models need retraining
- **Odds Data Updates**: Every 4 hours - Updates betting odds data
- **Live Game Updates**: Every 10 minutes during game hours (6 PM - 1 AM EST)
- **Injury Report Updates**: Every 15 minutes during pre-game windows (3 PM - 7 PM EST)

## Manual Training

To manually trigger model training, you can run:

```bash
docker-compose exec nba-prediction python -m src.utils.auto_train_manager --force
```

To train specific models only:

```bash
docker-compose exec nba-prediction python -m src.utils.auto_train_manager --force --models RandomForest,Bayesian
```

## Accessing the UI

The web UI is available at http://localhost:8080

## Checking Model Performance

To view the latest model performance metrics:

```bash
docker-compose exec nba-prediction python -m src.utils.track_performance --summary
```

## Database Management

You can connect to the PostgreSQL database using the following credentials:

- Host: localhost
- Port: 5432 (or the port you've mapped in docker-compose.yml)
- Database: nba_predictions
- Username: nbauser
- Password: nba_secure_password

## Volumes and Data Persistence

The Docker setup uses several volumes to ensure data persistence:

- `./data`: Stores processed data and cached API responses
- `./logs`: Contains application logs
- `./config`: Stores configuration files
- `./models`: Stores serialized model files
- `nba-db-data`: PostgreSQL database volume

## Troubleshooting

### Container Fails to Start

Check the logs for error messages:

```bash
docker-compose logs nba-prediction
```

### API Connection Issues

Ensure your API keys are set correctly and that the APIs are accessible:

```bash
docker-compose exec nba-prediction python -m src.api.theodds_client --test
docker-compose exec nba-prediction python -m src.api.balldontlie_client --test
```

### Database Connectivity Issues

Check if the database is running and accessible:

```bash
docker-compose exec nba-db pg_isready -U nbauser
```

## Production Deployment

For production deployments, consider the following recommendations:

1. Use a more secure password for the database
2. Enable TLS/SSL for database connections
3. Set up a reverse proxy (like Nginx) in front of the web UI
4. Implement proper backup strategies for the database
5. Configure monitoring and alerting for the containers

## Scaling

The system can be scaled horizontally by deploying multiple prediction services. However, this requires additional configuration for load balancing and database connection pooling.

## Updates and Maintenance

To update the system to a newer version:

1. Pull the latest code changes
2. Rebuild the containers:

```bash
docker-compose down
docker-compose up -d --build
```

The database schema will be automatically updated during startup if needed.
