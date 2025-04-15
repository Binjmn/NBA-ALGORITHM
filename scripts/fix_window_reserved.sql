-- Fix model_performance table to handle 'window' reserved keyword

-- Drop the model_performance table if it exists
DROP TABLE IF EXISTS model_performance CASCADE;

-- Recreate model_performance with properly quoted reserved word
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    prediction_target VARCHAR(50) NOT NULL,
    metrics JSONB DEFAULT '{}'::jsonb,
    is_baseline BOOLEAN NOT NULL DEFAULT false,
    "time_window" VARCHAR(50) DEFAULT '7d',  -- Renamed from 'window' to avoid reserved word issue
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, prediction_target)
);

-- Create index on model_performance
CREATE INDEX idx_model_performance_model_name ON model_performance(model_name);

-- Pre-populate model_performance with baseline metrics
INSERT INTO model_performance (model_name, prediction_target, metrics, is_baseline, "time_window")
VALUES
('RandomForest', 'moneyline', '{"accuracy": 0.5, "recent_accuracy": 0.5, "f1_score": 0.5, "precision": 0.5, "recall": 0.5}'::jsonb, true, '7d'),
('GradientBoosting', 'spread', '{"mae": 140, "mape": 8, "r2": 0, "rmse": 170}'::jsonb, true, '7d'),
('BayesianModel', 'moneyline', '{"accuracy": 0.0, "log_loss": 0.698, "brier_score": 0.25}'::jsonb, true, '7d'),
('CombinedGradientBoosting', 'moneyline', '{"accuracy": 0.5, "f1_score": 0.5, "precision": 0.5, "recall": 0.5}'::jsonb, true, '7d'),
('EnsembleModel', 'combined', '{"accuracy": 0.5, "mae": 140, "combined_score": 0.5}'::jsonb, true, '7d'),
('EnsembleStacking', 'combined', '{"accuracy": 0.5, "mae": 140, "combined_score": 0.5}'::jsonb, true, '7d');
