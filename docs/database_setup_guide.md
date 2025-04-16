# NBA Prediction System Database Setup Guide

This guide provides step-by-step instructions for setting up the PostgreSQL database for the NBA Prediction System.

## Prerequisites

- PostgreSQL 13+ installed on your system
- NBA Algorithm project code

## Installation Steps

### 1. Install PostgreSQL

1. Download PostgreSQL from the official website: [PostgreSQL Downloads](https://www.postgresql.org/download/windows/)
2. Run the installer and follow the setup wizard
3. When prompted, set the password to `ALGO123` for simplicity
4. Complete the installation with default options
5. Verify the installation by launching pgAdmin or using psql in command line

### 2. Create the Database

```sql
-- Connect to PostgreSQL using psql or pgAdmin
-- Run these commands:

CREATE DATABASE nba_prediction;
```

### 3. Set Environment Variables

Create a file named `.env` in the project root with the following content:

```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=nba_prediction
POSTGRES_USER=postgres
POSTGRES_PASSWORD=ALGO123
BALLDONTLIE_API_KEY=d0e93357-b9b0-4a62-bed1-920eeab5db50
ODDS_API_KEY=12186f096bb2e6a9f9b472391323893d
```

Or set these as system environment variables.

### 4. Initialize the Database Schema

For development or regular use, initialize the database with the production schema:

```bash
psql -U postgres -d nba_prediction -f scripts/production_schema.sql
```

If you encounter SQL reserved keyword issues, apply the fixes:

```bash
psql -U postgres -d nba_prediction -f scripts/fix_window_reserved.sql
```

For Docker deployment, the initialization script is automatically applied:
```
docker/init-db/01-init.sql
```

### 5. Import NBA Data

Run the data import script to populate the database with real NBA data:

```bash
python -m src.database.setup_production_db
```

This will fetch teams, players, and games data from the BallDontLie API and store it in your database.

### 6. Train Models

Run the model training script to train prediction models using real NBA data:

```bash
python -m src.models.advanced_trainer
```

### 7. Start the API Server

Start the API server to serve predictions and dashboard data:

```bash
python -m src.api.server
```

Access the dashboard at http://localhost:5000/dashboard.

## Troubleshooting

### Database Connection Issues

If you encounter database connection issues:

1. Verify PostgreSQL is running using Task Manager or Services
2. Test connection: `psql -U postgres -d postgres -c "SELECT version();"`
3. Check environment variables are set correctly
4. Ensure firewall isn't blocking PostgreSQL port (5432)

### API Key Issues

If API requests fail:

1. Verify API keys are correctly set in environment variables
2. Check API rate limits haven't been exceeded
3. Test API manually using curl or Postman

## Database Schema

The NBA Prediction System uses the following tables:

- `teams` - NBA team information
- `players` - Player information and statistics
- `games` - Game schedules, results, and predictions
- `game_stats` - Detailed player statistics per game
- `model_weights` - Trained prediction models
- `model_performance` - Model accuracy metrics

Full schema details available in `scripts/init_nba_schema.sql`.
