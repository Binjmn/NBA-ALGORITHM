-- NBA Prediction System Database Initialization
-- This script runs when the PostgreSQL container first starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create the database user with appropriate permissions
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'nbauser') THEN
        CREATE USER nbauser WITH PASSWORD 'nba_secure_password';
    END IF;
END
$$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE nba_predictions TO nbauser;
ALTER ROLE nbauser WITH CREATEDB;

-- Create necessary tables if they don't exist
-- These will be initialized properly by the application

-- Games table
CREATE TABLE IF NOT EXISTS games (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(255) UNIQUE NOT NULL,
    season INT NOT NULL,
    game_date TIMESTAMP WITH TIME ZONE NOT NULL,
    home_team_id VARCHAR(255) NOT NULL,
    away_team_id VARCHAR(255) NOT NULL,
    home_score INT,
    away_score INT,
    home_spread FLOAT,
    total_score FLOAT,
    status VARCHAR(50) NOT NULL DEFAULT 'scheduled',
    game_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Players table
CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    team_id VARCHAR(255),
    position VARCHAR(50),
    player_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model weights table
CREATE TABLE IF NOT EXISTS model_weights (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    weights JSONB NOT NULL,
    params JSONB,
    version INT NOT NULL DEFAULT 1,
    trained_at TIMESTAMP WITH TIME ZONE,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    needs_training BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, version)
);

-- Model performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    prediction_target VARCHAR(50) NOT NULL,
    metrics JSONB NOT NULL,
    sample_count INT NOT NULL,
    is_baseline BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    prediction_target VARCHAR(50) NOT NULL,
    game_id VARCHAR(255) REFERENCES games(game_id),
    player_id VARCHAR(255) REFERENCES players(player_id),
    prediction_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Add timestamps trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE 'plpgsql';

-- Add triggers for updated_at timestamps
DROP TRIGGER IF EXISTS update_games_updated_at ON games;
CREATE TRIGGER update_games_updated_at
BEFORE UPDATE ON games
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

DROP TRIGGER IF EXISTS update_players_updated_at ON players;
CREATE TRIGGER update_players_updated_at
BEFORE UPDATE ON players
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

DROP TRIGGER IF EXISTS update_model_weights_updated_at ON model_weights;
CREATE TRIGGER update_model_weights_updated_at
BEFORE UPDATE ON model_weights
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

DROP TRIGGER IF EXISTS update_model_performance_updated_at ON model_performance;
CREATE TRIGGER update_model_performance_updated_at
BEFORE UPDATE ON model_performance
FOR EACH ROW
EXECUTE PROCEDURE update_updated_at_column();

-- Create indices for better performance
CREATE INDEX IF NOT EXISTS idx_games_game_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);
CREATE INDEX IF NOT EXISTS idx_players_team_id ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_model_weights_active ON model_weights(active);
CREATE INDEX IF NOT EXISTS idx_model_weights_name_version ON model_weights(model_name, version);
CREATE INDEX IF NOT EXISTS idx_predictions_game_id ON predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_player_id ON predictions(player_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);

-- Grant permissions to nbauser
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO nbauser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO nbauser;
GRANT ALL PRIVILEGES ON SCHEMA public TO nbauser;
