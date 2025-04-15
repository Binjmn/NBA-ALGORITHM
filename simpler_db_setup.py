#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simpler Database Setup Script for NBA Prediction System

This script provides a more reliable approach to set up the database
for the NBA Prediction System. It offers step-by-step guidance and
verification to ensure the database is properly configured.
"""

import os
import sys
import subprocess
import time
import platform
import logging
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

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'nba_prediction',
    'user': 'postgres',
    'password': 'ALGO123'
}

# API Keys
API_KEYS = {
    'balldontlie': 'd0e93357-b9b0-4a62-bed1-920eeab5db50',
    'theodds': '12186f096bb2e6a9f9b472391323893d'
}


def print_header():
    """Print a header for the setup script"""
    print("\n" + "=" * 80)
    print(" " * 25 + "NBA PREDICTION SYSTEM - DATABASE SETUP")
    print("=" * 80 + "\n")


def print_section(title):
    """Print a section header"""
    print("\n" + "-" * 80)
    print(f" {title}")
    print("-" * 80)


def run_command(command, shell=True, env=None):
    """Run a command and return the output and status"""
    try:
        if env is None:
            env = os.environ.copy()
        
        result = subprocess.run(
            command,
            shell=shell,
            env=env,
            check=False,  # Don't raise exception on non-zero exit
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return {
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr,
            'code': result.returncode
        }
    except Exception as e:
        return {
            'success': False,
            'output': '',
            'error': str(e),
            'code': -1
        }


def check_postgres_installed():
    """Check if PostgreSQL is installed"""
    print_section("Checking PostgreSQL Installation")
    
    # Check for psql command
    if platform.system() == 'Windows':
        result = run_command('where psql', shell=True)
    else:
        result = run_command('which psql', shell=True)
    
    if result['success']:
        print("✅ PostgreSQL client (psql) is installed.")
        # Try to get version
        version_result = run_command('psql --version', shell=True)
        if version_result['success']:
            print(f"   {version_result['output'].strip()}")
        return True
    else:
        print("❌ PostgreSQL client (psql) is NOT installed or not in PATH.")
        print("   Please install PostgreSQL and ensure it's in your system PATH.")
        print("   Download from: https://www.postgresql.org/download/")
        return False


def check_postgres_service():
    """Check if PostgreSQL service is running"""
    print_section("Checking PostgreSQL Service")
    
    # Different commands based on OS
    if platform.system() == 'Windows':
        result = run_command('sc query postgresql', shell=True)
        running = 'RUNNING' in result['output']
    else:
        result = run_command('ps aux | grep postgres[q]l', shell=True)
        running = result['success'] and result['output'].strip() != ''
    
    if running:
        print("✅ PostgreSQL service is running.")
        return True
    else:
        print("❌ PostgreSQL service is NOT running.")
        print("   Please start the PostgreSQL service.")
        
        if platform.system() == 'Windows':
            print("   Try running: net start postgresql")
        else:
            print("   Try running: sudo service postgresql start")
        
        return False


def check_postgres_connection():
    """Check if we can connect to PostgreSQL"""
    print_section("Testing PostgreSQL Connection")
    
    # Set environment variables for psql
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_CONFIG['password']
    
    # Try connecting to default postgres database
    command = f"psql -h {DB_CONFIG['host']} -p {DB_CONFIG['port']} -U {DB_CONFIG['user']} -d postgres -c \"SELECT version();\""
    result = run_command(command, env=env)
    
    if result['success'] and 'PostgreSQL' in result['output']:
        print("✅ Successfully connected to PostgreSQL server.")
        print(f"   {result['output'].strip()}")
        return True
    else:
        print("❌ Could not connect to PostgreSQL server.")
        print(f"   Error: {result['error']}")
        print("\nTroubleshooting tips:")
        print("   1. Ensure PostgreSQL is installed and running")
        print("   2. Check if the credentials are correct:")
        print(f"      Host: {DB_CONFIG['host']}")
        print(f"      Port: {DB_CONFIG['port']}")
        print(f"      User: {DB_CONFIG['user']}")
        print(f"      Password: {'*' * len(DB_CONFIG['password'])}")
        print("   3. Make sure PostgreSQL is configured to accept connections")
        return False


def create_database():
    """Create the NBA prediction database"""
    print_section("Creating NBA Prediction Database")
    
    # First check if database already exists
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_CONFIG['password']
    
    check_command = f"psql -h {DB_CONFIG['host']} -p {DB_CONFIG['port']} -U {DB_CONFIG['user']} -d postgres -c \"SELECT 1 FROM pg_database WHERE datname = '{DB_CONFIG['database']}';\""
    result = run_command(check_command, env=env)
    
    if result['success'] and '1 row' in result['output']:
        print(f"✅ Database '{DB_CONFIG['database']}' already exists.")
        return True
    
    # Create the database
    create_command = f"psql -h {DB_CONFIG['host']} -p {DB_CONFIG['port']} -U {DB_CONFIG['user']} -d postgres -c \"CREATE DATABASE {DB_CONFIG['database']};\""
    result = run_command(create_command, env=env)
    
    if result['success']:
        print(f"✅ Database '{DB_CONFIG['database']}' created successfully.")
        return True
    else:
        print(f"❌ Failed to create database '{DB_CONFIG['database']}'.")
        print(f"   Error: {result['error']}")
        return False


def create_schema():
    """Create the database schema"""
    print_section("Creating Database Schema")
    
    # Create a temporary schema file
    schema_file = Path('temp_schema.sql')
    schema_sql = """
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
    
    schema_file.write_text(schema_sql)
    
    # Run the schema file
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_CONFIG['password']
    
    schema_command = f"psql -h {DB_CONFIG['host']} -p {DB_CONFIG['port']} -U {DB_CONFIG['user']} -d {DB_CONFIG['database']} -f {schema_file.absolute()}"
    result = run_command(schema_command, env=env)
    
    # Clean up the temp file
    schema_file.unlink(missing_ok=True)
    
    if result['success']:
        print("✅ Database schema created successfully.")
        return True
    else:
        print("❌ Failed to create database schema.")
        print(f"   Error: {result['error']}")
        return False


def setup_environment_variables():
    """Set up environment variables for the application"""
    print_section("Setting Environment Variables")
    
    # Set environment variables
    os.environ["POSTGRES_HOST"] = DB_CONFIG['host']
    os.environ["POSTGRES_PORT"] = DB_CONFIG['port']
    os.environ["POSTGRES_DB"] = DB_CONFIG['database']
    os.environ["POSTGRES_USER"] = DB_CONFIG['user']
    os.environ["POSTGRES_PASSWORD"] = DB_CONFIG['password']
    os.environ["BALLDONTLIE_API_KEY"] = API_KEYS['balldontlie']
    os.environ["ODDS_API_KEY"] = API_KEYS['theodds']
    
    # Create a .env file for environment variables
    with open(".env", "w") as f:
        f.write(f"POSTGRES_HOST={DB_CONFIG['host']}\n")
        f.write(f"POSTGRES_PORT={DB_CONFIG['port']}\n")
        f.write(f"POSTGRES_DB={DB_CONFIG['database']}\n")
        f.write(f"POSTGRES_USER={DB_CONFIG['user']}\n")
        f.write(f"POSTGRES_PASSWORD={DB_CONFIG['password']}\n")
        f.write(f"BALLDONTLIE_API_KEY={API_KEYS['balldontlie']}\n")
        f.write(f"ODDS_API_KEY={API_KEYS['theodds']}\n")
    
    # Create a batch file for setting environment variables
    with open("set_env.bat", "w") as f:
        f.write("@echo off\n")
        f.write("echo Setting environment variables for NBA Prediction System...\n")
        f.write(f"set POSTGRES_HOST={DB_CONFIG['host']}\n")
        f.write(f"set POSTGRES_PORT={DB_CONFIG['port']}\n")
        f.write(f"set POSTGRES_DB={DB_CONFIG['database']}\n")
        f.write(f"set POSTGRES_USER={DB_CONFIG['user']}\n")
        f.write(f"set POSTGRES_PASSWORD={DB_CONFIG['password']}\n")
        f.write(f"set BALLDONTLIE_API_KEY={API_KEYS['balldontlie']}\n")
        f.write(f"set ODDS_API_KEY={API_KEYS['theodds']}\n")
        f.write("echo Environment variables set successfully!\n")
    
    print("✅ Environment variables set successfully.")
    print("✅ Created .env file with configuration.")
    print("✅ Created set_env.bat for setting environment variables.")
    return True


def test_api_connection():
    """Test connection to BallDontLie API"""
    print_section("Testing BallDontLie API Connection")
    
    try:
        # Check if API module exists
        api_module = Path('src/api/balldontlie_client.py')
        if not api_module.exists():
            print("❌ BallDontLie API client module not found.")
            return False
        
        # Use the API module to test connection
        sys.path.insert(0, str(Path().absolute()))
        try:
            from src.api.balldontlie_client import BallDontLieClient
            client = BallDontLieClient()
            teams = client.get_teams(per_page=1)
            
            if teams and len(teams) > 0:
                print("✅ Successfully connected to BallDontLie API.")
                print(f"   Retrieved {len(teams)} team(s).")
                return True
            else:
                print("❌ Connected to BallDontLie API but received no data.")
                return False
        except Exception as e:
            print(f"❌ Error testing BallDontLie API: {str(e)}")
            return False
    except Exception as e:
        print(f"❌ Error testing API connection: {str(e)}")
        return False


def main():
    """Main function to run the database setup"""
    print_header()
    
    print("This script will set up the PostgreSQL database for the NBA Prediction System.")
    print("It will guide you through the process step by step and verify each component.")
    print("\nPress Enter to begin...")
    input()
    
    # Step 1: Check if PostgreSQL is installed
    postgres_installed = check_postgres_installed()
    if not postgres_installed:
        print("\nPlease install PostgreSQL and run this script again.")
        return False
    
    # Step 2: Check if PostgreSQL service is running
    postgres_running = check_postgres_service()
    if not postgres_running:
        print("\nPlease start the PostgreSQL service and run this script again.")
        return False
    
    # Step 3: Test PostgreSQL connection
    postgres_connection = check_postgres_connection()
    if not postgres_connection:
        print("\nPlease resolve the connection issues and run this script again.")
        return False
    
    # Step 4: Create the database
    database_created = create_database()
    if not database_created:
        print("\nPlease resolve the database creation issues and run this script again.")
        return False
    
    # Step 5: Create the schema
    schema_created = create_schema()
    if not schema_created:
        print("\nPlease resolve the schema creation issues and run this script again.")
        return False
    
    # Step 6: Set up environment variables
    env_setup = setup_environment_variables()
    if not env_setup:
        print("\nPlease resolve the environment variable issues and run this script again.")
        return False
    
    # Step 7: Test API connection
    api_connection = test_api_connection()
    if not api_connection:
        print("\nWarning: BallDontLie API connection test failed, but you can still proceed.")
    
    # Success!
    print_section("Database Setup Complete")
    print("✅ PostgreSQL database has been successfully set up for the NBA Prediction System!")
    print("\nDatabase Configuration:")
    print(f"   Host: {DB_CONFIG['host']}")
    print(f"   Port: {DB_CONFIG['port']}")
    print(f"   Database: {DB_CONFIG['database']}")
    print(f"   Username: {DB_CONFIG['user']}")
    print(f"   Password: {'*' * len(DB_CONFIG['password'])}")
    
    print("\nNext Steps:")
    print("1. Run the environment setup script:")
    print("   - Windows: .\\set_env.bat")
    print("2. Import NBA data:")
    print("   - python -m src.database.setup_production_db")
    print("3. Train prediction models:")
    print("   - python -m src.models.advanced_trainer")
    print("4. Start the API server:")
    print("   - python -m src.api.server")
    print("5. Access the dashboard at http://localhost:5000/dashboard")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ Database setup encountered issues. Please resolve them and try again.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSetup canceled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ An unexpected error occurred: {str(e)}")
        logger.exception("Unexpected error in setup script")
        sys.exit(1)
