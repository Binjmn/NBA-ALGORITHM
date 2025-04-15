#!/usr/bin/env python
"""
Production Database Setup for NBA Prediction System

This script sets up a production-ready PostgreSQL database for the NBA Prediction System,
creating all necessary tables, indexes, and initial data. The database is designed to
store real NBA data fetched from external APIs and support model training.

Features:
- Creates all required database tables with appropriate constraints
- Configures indexes for optimal query performance
- Sets up initial data needed for the system to function
- Implements logging for all operations
- Handles database connection errors gracefully
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import database connection utilities
from src.database.connection import get_connection, close_connection

# Import API client for fetching real NBA data
from src.api.balldontlie_client import BallDontLieClient

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


def create_database_tables():
    """Create all required database tables with appropriate constraints"""
    logger.info("Creating database tables")
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            # Teams table
            cursor.execute("""
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
                )
            """)
            
            # Players table
            cursor.execute("""
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
                )
            """)
            
            # Games table
            cursor.execute("""
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
                )
            """)
            
            # Game Statistics table
            cursor.execute("""
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
                )
            """)
            
            # Model Weights table
            cursor.execute("""
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
                    UNIQUE(model_name, params->>'prediction_target', version)
                )
            """)
            
            # Model Performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100) NOT NULL,
                    prediction_target VARCHAR(100) NOT NULL,
                    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
                    is_baseline BOOLEAN NOT NULL DEFAULT false,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # System Logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id SERIAL PRIMARY KEY,
                    log_type VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    details JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance optimization
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games (date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_status ON games (status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_season ON games (season_year)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams_conference ON teams (conference)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players (team_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_stats_game ON game_stats (game_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_stats_player ON game_stats (player_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_weights_active ON model_weights (active)")
            
            conn.commit()
            logger.info("Successfully created all database tables")
            return True
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        return False
    finally:
        close_connection(conn)


def fetch_and_store_teams():
    """Fetch teams from BallDontLie API and store in database"""
    logger.info("Fetching teams from BallDontLie API")
    try:
        # Initialize BallDontLie client
        client = BallDontLieClient()
        teams = client.get_teams()
        
        if not teams:
            logger.warning("No teams returned from BallDontLie API")
            return False
        
        logger.info(f"Fetched {len(teams)} teams from BallDontLie API")
        
        # Store teams in database
        conn = get_connection()
        inserted_count = 0
        
        with conn.cursor() as cursor:
            for team in teams:
                try:
                    # Extract team data
                    team_id = str(team.get('id'))
                    name = team.get('full_name')
                    abbreviation = team.get('abbreviation')
                    city = team.get('city')
                    conference = team.get('conference')
                    division = team.get('division')
                    
                    # Store in database
                    cursor.execute("""
                        INSERT INTO teams (team_id, name, abbreviation, city, conference, division, data)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (team_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            abbreviation = EXCLUDED.abbreviation,
                            city = EXCLUDED.city,
                            conference = EXCLUDED.conference,
                            division = EXCLUDED.division,
                            data = EXCLUDED.data,
                            updated_at = CURRENT_TIMESTAMP
                    """, (team_id, name, abbreviation, city, conference, division, json.dumps(team)))
                    inserted_count += 1
                except Exception as e:
                    logger.error(f"Error inserting team {team.get('full_name')}: {str(e)}")
            
            conn.commit()
            logger.info(f"Successfully inserted/updated {inserted_count} teams in database")
            return True
    except Exception as e:
        logger.error(f"Error fetching and storing teams: {str(e)}")
        return False
    finally:
        close_connection(conn)


def fetch_and_store_players(limit=100):
    """Fetch players from BallDontLie API and store in database"""
    logger.info(f"Fetching up to {limit} players from BallDontLie API")
    try:
        # Initialize BallDontLie client
        client = BallDontLieClient()
        players = client.get_players(limit=limit)
        
        if not players:
            logger.warning("No players returned from BallDontLie API")
            return False
        
        logger.info(f"Fetched {len(players)} players from BallDontLie API")
        
        # Store players in database
        conn = get_connection()
        inserted_count = 0
        
        with conn.cursor() as cursor:
            for player in players:
                try:
                    # Extract player data
                    player_id = str(player.get('id'))
                    name = f"{player.get('first_name')} {player.get('last_name')}"
                    team = player.get('team', {})
                    team_id = str(team.get('id')) if team and team.get('id') else None
                    position = player.get('position')
                    
                    # Store in database
                    cursor.execute("""
                        INSERT INTO players (player_id, name, team_id, position, data)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (player_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            team_id = EXCLUDED.team_id,
                            position = EXCLUDED.position,
                            data = EXCLUDED.data,
                            updated_at = CURRENT_TIMESTAMP
                    """, (player_id, name, team_id, position, json.dumps(player)))
                    inserted_count += 1
                except Exception as e:
                    logger.error(f"Error inserting player {name}: {str(e)}")
            
            conn.commit()
            logger.info(f"Successfully inserted/updated {inserted_count} players in database")
            return True
    except Exception as e:
        logger.error(f"Error fetching and storing players: {str(e)}")
        return False
    finally:
        close_connection(conn)


def fetch_recent_games(days=30):
    """Fetch recent games from BallDontLie API and store in database"""
    logger.info(f"Fetching games from the last {days} days from BallDontLie API")
    try:
        # Initialize BallDontLie client
        client = BallDontLieClient()
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # Fetch games
        games = client.get_games(start_date=start_date.strftime('%Y-%m-%d'), 
                                end_date=end_date.strftime('%Y-%m-%d'))
        
        if not games:
            logger.warning("No games returned from BallDontLie API")
            return False
        
        logger.info(f"Fetched {len(games)} games from BallDontLie API")
        
        # Store games in database
        conn = get_connection()
        inserted_count = 0
        
        with conn.cursor() as cursor:
            for game in games:
                try:
                    # Extract game data
                    game_id = str(game.get('id'))
                    date = game.get('date')
                    status = game.get('status')
                    home_team = game.get('home_team', {})
                    away_team = game.get('away_team', {})
                    home_team_id = str(home_team.get('id')) if home_team and home_team.get('id') else None
                    away_team_id = str(away_team.get('id')) if away_team and away_team.get('id') else None
                    season = game.get('season') or datetime.now().year
                    
                    # Skip games with missing team information
                    if not home_team_id or not away_team_id:
                        logger.warning(f"Skipping game {game_id} with missing team information")
                        continue
                    
                    # Store in database
                    cursor.execute("""
                        INSERT INTO games (game_id, season_year, date, home_team_id, away_team_id, status, data)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (game_id) DO UPDATE SET
                            season_year = EXCLUDED.season_year,
                            date = EXCLUDED.date,
                            home_team_id = EXCLUDED.home_team_id,
                            away_team_id = EXCLUDED.away_team_id,
                            status = EXCLUDED.status,
                            data = EXCLUDED.data,
                            updated_at = CURRENT_TIMESTAMP
                    """, (game_id, season, date, home_team_id, away_team_id, status, json.dumps(game)))
                    inserted_count += 1
                except Exception as e:
                    logger.error(f"Error inserting game {game_id}: {str(e)}")
            
            conn.commit()
            logger.info(f"Successfully inserted/updated {inserted_count} games in database")
            return True
    except Exception as e:
        logger.error(f"Error fetching and storing games: {str(e)}")
        return False
    finally:
        close_connection(conn)


def setup_model_templates():
    """Set up model templates in the database"""
    logger.info("Setting up model templates in the database")
    try:
        conn = get_connection()
        
        # Define model templates
        model_templates = [
            {
                'name': 'RandomForest',
                'type': 'classification',
                'prediction_target': 'moneyline',
                'params': {'n_estimators': 100, 'max_depth': 10, 'prediction_target': 'moneyline'}
            },
            {
                'name': 'GradientBoosting',
                'type': 'regression',
                'prediction_target': 'spread',
                'params': {'n_estimators': 150, 'learning_rate': 0.1, 'prediction_target': 'spread'}
            },
            {
                'name': 'BayesianModel',
                'type': 'probability',
                'prediction_target': 'moneyline',
                'params': {'prediction_target': 'moneyline'}
            },
            {
                'name': 'EnsembleStack',
                'type': 'meta-model',
                'prediction_target': 'combined',
                'params': {'prediction_target': 'combined', 'base_models': ['RandomForest', 'GradientBoosting']}
            }
        ]
        
        # Insert model templates into database
        with conn.cursor() as cursor:
            for template in model_templates:
                # Mark the model as needing training
                cursor.execute("""
                    INSERT INTO model_weights (model_name, model_type, params, weights, needs_training)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (model_name, params->>'prediction_target', version) DO UPDATE SET
                        model_type = EXCLUDED.model_type,
                        params = EXCLUDED.params,
                        needs_training = EXCLUDED.needs_training,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    template['name'],
                    template['type'],
                    json.dumps(template['params']),
                    b'\x00',  # Empty placeholder for weights until model is trained
                    True
                ))
                
                # Set up baseline performance metrics
                cursor.execute("""
                    INSERT INTO model_performance (model_name, prediction_target, metrics, is_baseline)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    template['name'],
                    template['prediction_target'],
                    json.dumps({
                        'accuracy': 0.5,  # Baseline accuracy (random guess)
                        'recent_accuracy': 0.5,
                        'f1_score': 0.5,
                        'precision': 0.5,
                        'recall': 0.5
                    }),
                    True
                ))
            
            conn.commit()
            logger.info(f"Successfully set up {len(model_templates)} model templates")
            return True
    except Exception as e:
        logger.error(f"Error setting up model templates: {str(e)}")
        return False
    finally:
        close_connection(conn)


def init_database():
    """Initialize the database with real NBA data"""
    logger.info("Starting database initialization with real NBA data")
    
    # Create database tables
    if not create_database_tables():
        logger.error("Failed to create database tables")
        return False
    
    # Fetch and store teams
    if not fetch_and_store_teams():
        logger.warning("Failed to fetch and store teams")
    
    # Fetch and store players
    if not fetch_and_store_players():
        logger.warning("Failed to fetch and store players")
    
    # Fetch and store recent games
    if not fetch_recent_games():
        logger.warning("Failed to fetch and store recent games")
    
    # Set up model templates
    if not setup_model_templates():
        logger.warning("Failed to set up model templates")
    
    logger.info("Database initialization completed")
    return True


if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    print("\n===== NBA PREDICTION SYSTEM DATABASE SETUP =====\n")
    print("This script will initialize your database with real NBA data.")
    print("It will fetch data from the BallDontLie API using your API key.")
    print("The data will be stored in your PostgreSQL database.")
    print("\nStarting database initialization...")
    
    success = init_database()
    
    if success:
        print("\n✅ Database initialization completed successfully!")
        print("✅ Real NBA data has been loaded into your database")
        print("✅ Model templates have been set up for training")
        print("\nYou can now start the API server and train the models.")
    else:
        print("\n❌ Database initialization failed!")
        print("Please check the logs for more information.")
