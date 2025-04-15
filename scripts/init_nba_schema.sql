-- NBA Prediction System Database Schema Initialization
-- This script creates all tables, indexes, and constraints required
-- for the NBA Prediction System to operate with real NBA data.

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable text search capabilities
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
    team_id VARCHAR(100) REFERENCES teams(team_id),
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
    home_team_id VARCHAR(100) NOT NULL REFERENCES teams(team_id),
    away_team_id VARCHAR(100) NOT NULL REFERENCES teams(team_id),
    status VARCHAR(50) NOT NULL,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    odds JSONB NOT NULL DEFAULT '{}'::jsonb,
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    predictions JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Game Statistics table
CREATE TABLE IF NOT EXISTS game_stats (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(100) NOT NULL REFERENCES games(game_id),
    player_id VARCHAR(100) NOT NULL REFERENCES players(player_id),
    team_id VARCHAR(100) NOT NULL REFERENCES teams(team_id),
    minutes INTEGER,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    field_goals_made INTEGER,
    field_goals_attempted INTEGER,
    three_pointers_made INTEGER,
    three_pointers_attempted INTEGER,
    free_throws_made INTEGER,
    free_throws_attempted INTEGER,
    plus_minus INTEGER,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(game_id, player_id)
);

-- Model Weights table
CREATE TABLE IF NOT EXISTS model_weights (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    weights BYTEA NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    trained_at TIMESTAMP WITH TIME ZONE,
    active BOOLEAN NOT NULL DEFAULT true,
    needs_training BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, (params->>'prediction_target'), version)
);

-- Model Performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    prediction_target VARCHAR(100) NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_baseline BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System Logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    log_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_games_date ON games (date);
CREATE INDEX IF NOT EXISTS idx_games_status ON games (status);
CREATE INDEX IF NOT EXISTS idx_games_season ON games (season_year);
CREATE INDEX IF NOT EXISTS idx_games_prediction_target ON games ((predictions->>'moneyline'));
CREATE INDEX IF NOT EXISTS idx_teams_conference ON teams (conference);
CREATE INDEX IF NOT EXISTS idx_players_team ON players (team_id);
CREATE INDEX IF NOT EXISTS idx_game_stats_game ON game_stats (game_id);
CREATE INDEX IF NOT EXISTS idx_game_stats_player ON game_stats (player_id);
CREATE INDEX IF NOT EXISTS idx_model_weights_active ON model_weights (active);

-- Add model templates for training
INSERT INTO model_weights (model_name, model_type, params, weights, needs_training)
VALUES 
('RandomForest', 'classification', '{"n_estimators": 100, "max_depth": 10, "prediction_target": "moneyline"}'::jsonb, '\x00'::bytea, true),
('GradientBoosting', 'regression', '{"n_estimators": 150, "learning_rate": 0.1, "prediction_target": "spread"}'::jsonb, '\x00'::bytea, true),
('BayesianModel', 'probability', '{"prediction_target": "moneyline"}'::jsonb, '\x00'::bytea, true),
('EnsembleStack', 'meta-model', '{"prediction_target": "combined", "base_models": ["RandomForest", "GradientBoosting"]}'::jsonb, '\x00'::bytea, true)
ON CONFLICT (model_name, (params->>'prediction_target'), version) DO NOTHING;

-- Pre-populate model_performance table with initial entries
INSERT INTO model_performance (model_name, prediction_target, metrics, is_baseline)
VALUES
('RandomForest', 'moneyline', '{"accuracy": 0.5, "recent_accuracy": 0.5, "f1_score": 0.5, "precision": 0.5, "recall": 0.5}'::jsonb, true),
('GradientBoosting', 'spread', '{"mae": 140, "mape": 8, "r2": 0, "rmse": 170}'::jsonb, true),
('BayesianModel', 'moneyline', '{"accuracy": 0.0, "log_loss": 0.698, "brier_score": 0.25}'::jsonb, true),
('EnsembleStack', 'combined', '{"accuracy": 0.0, "mae": 185, "combined_score": 0.5}'::jsonb, true)
ON CONFLICT DO NOTHING;

-- Add system log entry for initialization
INSERT INTO system_logs (log_type, message, details)
VALUES ('INIT', 'Database schema initialized', '{"timestamp": "' || NOW() || '", "version": "1.0.0"}'::jsonb);
