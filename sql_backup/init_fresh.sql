-- NBA Prediction System Database Schema Initialization
-- Fresh script with corrected syntax

-- Enable necessary extensions
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

-- Game Stats table
CREATE TABLE IF NOT EXISTS game_stats (
    id SERIAL PRIMARY KEY,
    game_id VARCHAR(100) NOT NULL REFERENCES games(game_id),
    team_id VARCHAR(100) NOT NULL REFERENCES teams(team_id),
    player_id VARCHAR(100) REFERENCES players(player_id),
    stat_type VARCHAR(50) NOT NULL,
    value NUMERIC NOT NULL,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model weights table
CREATE TABLE IF NOT EXISTS model_weights (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    weight_data BYTEA,
    is_active BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, version, (params->>'prediction_target'))
);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    prediction_target VARCHAR(50) NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_baseline BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, prediction_target)
);

-- System logs
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    log_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_teams_conference ON teams(conference);
CREATE INDEX idx_teams_division ON teams(division);
CREATE INDEX idx_players_team_id ON players(team_id);
CREATE INDEX idx_players_position ON players(position);
CREATE INDEX idx_games_season_year ON games(season_year);
CREATE INDEX idx_games_date ON games(date);
CREATE INDEX idx_games_home_team_id ON games(home_team_id);
CREATE INDEX idx_games_away_team_id ON games(away_team_id);
CREATE INDEX idx_games_status ON games(status);
CREATE INDEX idx_game_stats_game_id ON game_stats(game_id);
CREATE INDEX idx_game_stats_team_id ON game_stats(team_id);
CREATE INDEX idx_game_stats_player_id ON game_stats(player_id);
CREATE INDEX idx_game_stats_stat_type ON game_stats(stat_type);
CREATE INDEX idx_model_weights_model_name ON model_weights(model_name);
CREATE INDEX idx_model_weights_is_active ON model_weights(is_active);
CREATE INDEX idx_model_performance_model_name ON model_performance(model_name);

-- Pre-populate model_weights table with initial entries
INSERT INTO model_weights (model_name, model_type, params, weight_data, is_active)
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
