-- NBA Prediction System - Simple Database Initialization
-- This script creates the essential tables without complex constraints

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    team_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    abbreviation VARCHAR(10) NOT NULL,
    city VARCHAR(100) NOT NULL,
    conference VARCHAR(50),
    division VARCHAR(50),
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Players table
CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    team_id VARCHAR(100),
    position VARCHAR(50),
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Games table
CREATE TABLE IF NOT EXISTS games (
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

-- Game Stats table
CREATE TABLE IF NOT EXISTS game_stats (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(100) NOT NULL,
    team_id VARCHAR(100) NOT NULL,
    player_id VARCHAR(100),
    stat_type VARCHAR(50) NOT NULL,
    value NUMERIC NOT NULL,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model weights table (simplified)
CREATE TABLE IF NOT EXISTS model_weights (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    prediction_target VARCHAR(50) NOT NULL,
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    weight_data BYTEA,
    is_active BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Add unique constraint separately 
ALTER TABLE model_weights ADD CONSTRAINT unique_model_version_target 
    UNIQUE(model_name, version, prediction_target);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    prediction_target VARCHAR(50) NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_baseline BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Add unique constraint separately
ALTER TABLE model_performance ADD CONSTRAINT unique_model_performance_target 
    UNIQUE(model_name, prediction_target);

-- System logs
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    log_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create basic indexes
CREATE INDEX idx_teams_conference ON teams(conference);
CREATE INDEX idx_teams_division ON teams(division);
CREATE INDEX idx_players_team_id ON players(team_id);
CREATE INDEX idx_players_position ON players(position);
CREATE INDEX idx_games_season_year ON games(season_year);
CREATE INDEX idx_games_date ON games(date);
CREATE INDEX idx_games_home_team_id ON games(home_team_id);
CREATE INDEX idx_games_away_team_id ON games(away_team_id);
CREATE INDEX idx_games_status ON games(status);

-- Pre-populate model_weights table with initial entries
INSERT INTO model_weights (model_name, model_type, prediction_target, params, weight_data, is_active)
VALUES
('RandomForest', 'classification', 'moneyline', '{"n_estimators": 100, "max_depth": 10}'::jsonb, '\x00'::bytea, true),
('GradientBoosting', 'regression', 'spread', '{"n_estimators": 150, "learning_rate": 0.1}'::jsonb, '\x00'::bytea, true),
('BayesianModel', 'probability', 'moneyline', '{}'::jsonb, '\x00'::bytea, true),
('EnsembleStack', 'meta-model', 'combined', '{"base_models": ["RandomForest", "GradientBoosting"]}'::jsonb, '\x00'::bytea, true);

-- Pre-populate model_performance table with initial entries
INSERT INTO model_performance (model_name, prediction_target, metrics, is_baseline)
VALUES
('RandomForest', 'moneyline', '{"accuracy": 0.5, "recent_accuracy": 0.5, "f1_score": 0.5, "precision": 0.5, "recall": 0.5}'::jsonb, true),
('GradientBoosting', 'spread', '{"mae": 140, "mape": 8, "r2": 0, "rmse": 170}'::jsonb, true),
('BayesianModel', 'moneyline', '{"accuracy": 0.0, "log_loss": 0.698, "brier_score": 0.25}'::jsonb, true),
('EnsembleStack', 'combined', '{"accuracy": 0.0, "mae": 185, "combined_score": 0.5}'::jsonb, true);

-- Add system log entry for initialization
INSERT INTO system_logs (log_type, message, details)
VALUES ('INIT', 'Database schema initialized', '{"timestamp": "' || NOW()::text || '", "version": "1.0.0"}'::jsonb);
