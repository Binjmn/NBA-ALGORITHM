# Database Architecture

## Overview
The NBA Prediction System utilizes PostgreSQL as its primary database system for storing and managing all persistent data. The database architecture is designed to support long-term operation and scale efficiently as data accumulates over multiple NBA seasons.

## Design Principles

1. **Modularity** - Database tables are organized by domain (games, players, models)
2. **Scalability** - Schema designed to handle multiple years of data efficiently
3. **Performance** - Proper indexes and query optimization for common access patterns
4. **Flexibility** - JSONB columns for dynamic data that may change over time
5. **Data Integrity** - Constraints and triggers to ensure data consistency

## Schema Design

### Games Table
Stores information about NBA games, including team data, odds, features, and predictions.

```sql
CREATE TABLE games (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(100) UNIQUE NOT NULL,
    season_year INTEGER NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    home_team_id VARCHAR(100) NOT NULL,
    away_team_id VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    odds JSONB NOT NULL DEFAULT '{}'::jsonb,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    predictions JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

#### Key Features:
- `game_id`: Unique identifier for each game
- `season_year`: The NBA season year (e.g., 2025 for the 2024-2025 season)
- `data`: JSONB field for full game data (scores, stats, etc.)
- `odds`: JSONB field for betting odds data
- `features`: JSONB field for calculated features used in prediction models
- `predictions`: JSONB field for model predictions and results

### Players Table
Stores information about NBA players, including stats, features, and predictions.

```sql
CREATE TABLE players (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    team_id VARCHAR(100),
    position VARCHAR(50),
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    predictions JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

#### Key Features:
- `player_id`: Unique identifier for each player
- `data`: JSONB field for player statistics
- `features`: JSONB field for calculated features
- `predictions`: JSONB field for player performance predictions

### Model Weights Table
Stores the trained model weights and parameters with version control.

```sql
CREATE TABLE model_weights (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    weights JSONB NOT NULL,
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    version INTEGER NOT NULL,
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT TRUE,
    UNIQUE (model_name, version)
);
```

#### Key Features:
- `model_name`: Identifies the specific model (e.g., "RandomForest", "Bayesian")
- `model_type`: The type of model (e.g., "classification", "regression")
- `weights`: JSONB field containing serialized model weights
- `params`: JSONB field containing model hyperparameters
- `version`: Integer representing the model version
- `active`: Boolean flag indicating if this is the currently active version

### Model Performance Table
Tracks model accuracy and performance metrics over time.

```sql
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    metrics JSONB NOT NULL,
    prediction_type VARCHAR(100) NOT NULL,
    num_predictions INTEGER NOT NULL,
    window VARCHAR(50) DEFAULT '7d'
);
```

#### Key Features:
- `model_name`: Identifies the specific model being evaluated
- `metrics`: JSONB field containing performance metrics (accuracy, precision, etc.)
- `prediction_type`: The type of prediction (e.g., "moneyline", "spread", "player_points")
- `window`: Time window for the evaluation (e.g., "7d", "30d", "season")

## Indexes
The following indexes are created to optimize query performance:

```sql
CREATE INDEX idx_games_date ON games(date);
CREATE INDEX idx_games_season_year ON games(season_year);
CREATE INDEX idx_model_performance_date ON model_performance(date);
CREATE INDEX idx_model_performance_model_name ON model_performance(model_name);
```

## Triggers
A trigger is set up to automatically update the `updated_at` timestamp whenever a record is updated:

```sql
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_games_modtime
BEFORE UPDATE ON games
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_players_modtime
BEFORE UPDATE ON players
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();
```

## Connection Management
The database module includes a connection pool to efficiently manage database connections:

1. **Connection Pool** - Uses ThreadedConnectionPool from psycopg2 to maintain a pool of database connections
2. **Context Managers** - The `get_db_connection()` and `get_dict_cursor()` functions provide context managers for safe connection handling
3. **Environment Variables** - Database connection parameters are configured via environment variables

## Data Access Layer
The data access layer provides ORM-style model classes for interacting with the database:

1. **BaseModel** - Common functionality for all models (find_by_id, find_all)
2. **Game** - Methods for working with game data (create, update, find_by_date_range, etc.)
3. **Player** - Methods for working with player data (create, update, find_by_team, etc.)
4. **ModelWeight** - Methods for managing model weights (create, find_latest_active, etc.)
5. **ModelPerformance** - Methods for tracking model performance (create, get_latest_performance, etc.)
