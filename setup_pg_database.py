#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple PostgreSQL Database Setup Script for NBA Prediction System

This script assumes PostgreSQL is already installed and creates:
1. The NBA prediction database
2. Required tables and schema
3. Initial data structure for real NBA data
"""

import os
import sys
import subprocess
import logging
import json
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/db_setup.log')
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'nba_prediction',  # Target database name
    'user': 'postgres',
    'password': 'ALGO123',         # Default password
    'maintenance_db': 'postgres'   # Default database for initial connection
}

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)


def run_psql_command(sql_command, db_name=None):
    """Run a SQL command using psql command line"""
    target_db = db_name or DB_CONFIG['maintenance_db']
    
    # Construct the command
    cmd = [
        'psql',
        '-h', DB_CONFIG['host'],
        '-p', DB_CONFIG['port'],
        '-U', DB_CONFIG['user'],
        '-d', target_db,
        '-c', sql_command
    ]
    
    # Set PGPASSWORD environment variable
    env = os.environ.copy()
    env['PGPASSWORD'] = DB_CONFIG['password']
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing SQL command: {sql_command}")
        logger.error(f"Error details: {e.stderr.strip() if e.stderr else str(e)}")
        return None


def run_psql_script(script_path, db_name=None):
    """Run a SQL script file using psql command line"""
    target_db = db_name or DB_CONFIG['maintenance_db']
    
    # Construct the command
    cmd = [
        'psql',
        '-h', DB_CONFIG['host'],
        '-p', DB_CONFIG['port'],
        '-U', DB_CONFIG['user'],
        '-d', target_db,
        '-f', script_path
    ]
    
    # Set PGPASSWORD environment variable
    env = os.environ.copy()
    env['PGPASSWORD'] = DB_CONFIG['password']
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing SQL script: {script_path}")
        logger.error(f"Error details: {e.stderr.strip() if e.stderr else str(e)}")
        return None


def check_postgres_connection():
    """Check if PostgreSQL is installed and accessible"""
    logger.info("Checking PostgreSQL connection...")
    
    try:
        version = run_psql_command("SELECT version();")
        if version:
            logger.info(f"Successfully connected to PostgreSQL: {version}")
            return True
        else:
            logger.error("Failed to connect to PostgreSQL.")
            print("\n⚠️ Error connecting to PostgreSQL database.")
            print("Make sure PostgreSQL is installed and running.")
            print("Default connection parameters:")
            print(f"  Host: {DB_CONFIG['host']}")
            print(f"  Port: {DB_CONFIG['port']}")
            print(f"  User: {DB_CONFIG['user']}")
            print(f"  Password: {DB_CONFIG['password']}")
            return False
    except Exception as e:
        logger.error(f"Error checking PostgreSQL connection: {str(e)}")
        return False


def create_database():
    """Create the NBA prediction database"""
    logger.info(f"Creating database '{DB_CONFIG['database']}'")
    
    # Check if database exists
    db_exists_query = f"SELECT 1 FROM pg_database WHERE datname = '{DB_CONFIG['database']}'"
    result = run_psql_command(db_exists_query)
    
    if result and '1' in result:
        logger.info(f"Database '{DB_CONFIG['database']}' already exists")
        return True
    
    # Create database
    create_db_sql = f"CREATE DATABASE {DB_CONFIG['database']} WITH ENCODING 'UTF8'"
    result = run_psql_command(create_db_sql)
    
    if result is not None or result == '':
        logger.info(f"Successfully created database '{DB_CONFIG['database']}'")
        return True
    else:
        logger.error(f"Failed to create database '{DB_CONFIG['database']}'")
        return False


def create_schema():
    """Create the database schema"""
    logger.info("Creating database schema")
    
    # Define schema as a string for simplicity
    schema_sql = """
    -- Enable UUID extension
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    
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
    CREATE INDEX IF NOT EXISTS idx_teams_conference ON teams (conference);
    CREATE INDEX IF NOT EXISTS idx_players_team ON players (team_id);
    CREATE INDEX IF NOT EXISTS idx_game_stats_game ON game_stats (game_id);
    CREATE INDEX IF NOT EXISTS idx_game_stats_player ON game_stats (player_id);
    CREATE INDEX IF NOT EXISTS idx_model_weights_active ON model_weights (active);
    
    -- Add model templates
    INSERT INTO model_weights (model_name, model_type, params, weights, needs_training) VALUES
    ('RandomForest', 'classification', '{"n_estimators": 100, "max_depth": 10, "prediction_target": "moneyline"}'::jsonb, '\\x00'::bytea, true),
    ('GradientBoosting', 'regression', '{"n_estimators": 150, "learning_rate": 0.1, "prediction_target": "spread"}'::jsonb, '\\x00'::bytea, true)
    ON CONFLICT DO NOTHING;
    """
    
    # Save schema to file
    schema_file = "db_schema.sql"
    with open(schema_file, "w") as f:
        f.write(schema_sql)
    
    # Run the schema file
    result = run_psql_script(schema_file, DB_CONFIG['database'])
    
    if result is not None or result == '':
        logger.info("Successfully created database schema")
        return True
    else:
        logger.error("Failed to create database schema")
        return False


def set_environment_variables():
    """Set environment variables for database connection"""
    os.environ['POSTGRES_HOST'] = DB_CONFIG['host']
    os.environ['POSTGRES_PORT'] = DB_CONFIG['port']
    os.environ['POSTGRES_DB'] = DB_CONFIG['database']
    os.environ['POSTGRES_USER'] = DB_CONFIG['user']
    os.environ['POSTGRES_PASSWORD'] = DB_CONFIG['password']
    
    # Also set API keys
    os.environ['BALLDONTLIE_API_KEY'] = 'd0e93357-b9b0-4a62-bed1-920eeab5db50'
    os.environ['ODDS_API_KEY'] = '12186f096bb2e6a9f9b472391323893d'
    
    logger.info("Environment variables set for database connection")


def main():
    """Main function to set up the database"""
    print("\n===============================================")
    print("NBA PREDICTION SYSTEM - DATABASE SETUP")
    print("===============================================\n")
    
    print("This script will set up the PostgreSQL database for NBA predictions.")
    print("Make sure PostgreSQL is installed and running.\n")
    
    # Step 1: Check PostgreSQL connection
    print("Step 1: Checking PostgreSQL connection...")
    if not check_postgres_connection():
        return False
    
    # Step 2: Create database
    print("\nStep 2: Creating NBA prediction database...")
    if not create_database():
        return False
    
    # Step 3: Create schema
    print("\nStep 3: Creating database schema...")
    if not create_schema():
        return False
    
    # Step 4: Set environment variables
    print("\nStep 4: Setting environment variables...")
    set_environment_variables()
    
    print("\n===============================================")
    print("DATABASE SETUP COMPLETED SUCCESSFULLY!")
    print("===============================================\n")
    print(f"Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print(f"Username: {DB_CONFIG['user']}")
    print(f"Password: {DB_CONFIG['password']}")
    
    print("\nNext steps:")
    print("1. Import real NBA data: python -m src.database.setup_production_db")
    print("2. Train models: python -m src.models.advanced_trainer")
    print("3. Start the API server: python -m src.api.server")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️ Database setup encountered errors. Please check the logs.")
        sys.exit(1)
