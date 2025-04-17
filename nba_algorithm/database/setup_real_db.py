"""
Database Setup with Real NBA Data

This script initializes a production-ready PostgreSQL database for the NBA prediction system.
It will:
1. Create all required database tables
2. Load real NBA data from the API sources
3. Set up performance tracking tables

No mock or synthetic data is used at any point.

Usage:
    python -m src.database.setup_real_db
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import psycopg2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to make imports work when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import database and API modules
from src.database.init_db import initialize_database
from src.database.connection import get_connection, close_connection
from src.api.balldontlie_client import BallDontLieApiClient
from src.api.theodds_client import OddsApiCollector


def load_config_from_file(config_file: str = 'database.conf') -> Dict[str, str]:
    """
    Load database configuration from a file
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dict with configuration values
    """
    config = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key] = value
        logger.info(f"Loaded configuration from {config_file}")
        
        # Set environment variables
        for key, value in config.items():
            os.environ[key] = value
            
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return {}


def setup_database_user() -> bool:
    """
    Create a dedicated database user and database for the NBA prediction system
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Setting up dedicated database user...")
        
        # Get configuration
        db_name = os.environ.get('POSTGRES_DB', 'nba_prediction')
        user_name = os.environ.get('POSTGRES_USER', 'nba_user')
        password = os.environ.get('POSTGRES_PASSWORD', 'ALGO123')
        
        # Connect as postgres to create user and database
        # Note: we're using psycopg2 directly without the connection pool
        # because we need to connect as a different user initially
        default_connection = psycopg2.connect(
            dbname='postgres',  # Connect to default database initially
            user='postgres',    # Default PostgreSQL admin user
            password='',        # Empty password or your current admin password
            host=os.environ.get('POSTGRES_HOST', 'localhost'),
            port=os.environ.get('POSTGRES_PORT', '5432')
        )
        default_connection.autocommit = True  # Needed for CREATE DATABASE
        
        with default_connection.cursor() as cursor:
            # Check if user exists
            cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (user_name,))
            user_exists = cursor.fetchone() is not None
            
            if not user_exists:
                logger.info(f"Creating database user {user_name}...")
                cursor.execute(f"CREATE USER {user_name} WITH PASSWORD '{password}'")
            else:
                logger.info(f"User {user_name} already exists, updating password...")
                cursor.execute(f"ALTER USER {user_name} WITH PASSWORD '{password}'")
            
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            db_exists = cursor.fetchone() is not None
            
            if not db_exists:
                logger.info(f"Creating database {db_name}...")
                cursor.execute(f"CREATE DATABASE {db_name} OWNER {user_name}")
            
            # Grant privileges
            cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {user_name}")
        
        default_connection.close()
        logger.info(f"Database user and database setup complete: {user_name}@{db_name}")
        
        # Update environment variables to use our new user
        os.environ['POSTGRES_USER'] = user_name
        os.environ['POSTGRES_DB'] = db_name
        
        return True
    except Exception as e:
        logger.error(f"Error setting up database user: {str(e)}")
        logger.error("Please configure your PostgreSQL connection parameters manually")
        return False


def setup_database(force: bool = False) -> bool:
    """
    Initialize database structure
    
    Args:
        force: Whether to force recreation of tables
        
    Returns:
        bool: True if successful
    """
    try:
        logger.info("Initializing database structure...")
        success = initialize_database(verbose=True)
        if not success:
            logger.error("Failed to initialize database structure")
            return False
        
        logger.info("Database structure initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        return False


def load_nba_team_data() -> bool:
    """
    Load real NBA team data from BallDontLie API
    
    Returns:
        bool: True if successful
    """
    try:
        logger.info("Loading NBA team data from API...")
        
        # Initialize API client
        api_client = BallDontLieApiClient()
        
        # Fetch teams
        teams = api_client.get_teams()
        if not teams:
            logger.error("Failed to fetch teams from API")
            return False
        
        logger.info(f"Fetched {len(teams)} teams from API")
        
        # Store teams in database
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # Create teams table if not exists
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
                
                # Insert teams
                for team in teams:
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
                    """, (
                        str(team.get('id')),
                        team.get('full_name'),
                        team.get('abbreviation'),
                        team.get('city'),
                        team.get('conference'),
                        team.get('division'),
                        team
                    ))
                
                conn.commit()
                logger.info(f"Stored {len(teams)} teams in database")
                return True
        except Exception as e:
            logger.error(f"Error storing teams in database: {str(e)}")
            conn.rollback()
            return False
        finally:
            close_connection(conn)
    
    except Exception as e:
        logger.error(f"Failed to load NBA team data: {str(e)}")
        return False


def load_nba_player_data() -> bool:
    """
    Load real NBA player data from BallDontLie API
    
    Returns:
        bool: True if successful
    """
    try:
        logger.info("Loading NBA player data from API...")
        
        # Initialize API client
        api_client = BallDontLieApiClient()
        
        # Fetch players (with pagination)
        all_players = []
        page = 1
        per_page = 100
        
        # Fetch first page
        players_data = api_client.get_players(page=page, per_page=per_page)
        
        if not players_data or 'data' not in players_data:
            logger.error("Failed to fetch players from API")
            return False
            
        players = players_data.get('data', [])
        all_players.extend(players)
        
        # Get total pages (limited to 5 pages to avoid rate limits during setup)
        total_pages = min(5, players_data.get('meta', {}).get('total_pages', 1))
        
        # Fetch remaining pages
        for page in range(2, total_pages + 1):
            logger.info(f"Fetching player data page {page}/{total_pages}")
            players_data = api_client.get_players(page=page, per_page=per_page)
            if players_data and 'data' in players_data:
                all_players.extend(players_data.get('data', []))
        
        logger.info(f"Fetched {len(all_players)} players from API")
        
        # Store players in database
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # Insert players
                for player in all_players:
                    cursor.execute("""
                    INSERT INTO players (player_id, name, team_id, position, data)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (player_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        team_id = EXCLUDED.team_id,
                        position = EXCLUDED.position,
                        data = EXCLUDED.data,
                        updated_at = CURRENT_TIMESTAMP
                    """, (
                        str(player.get('id')),
                        f"{player.get('first_name')} {player.get('last_name')}",
                        str(player.get('team', {}).get('id')) if player.get('team') else None,
                        player.get('position'),
                        player
                    ))
                
                conn.commit()
                logger.info(f"Stored {len(all_players)} players in database")
                return True
        except Exception as e:
            logger.error(f"Error storing players in database: {str(e)}")
            conn.rollback()
            return False
        finally:
            close_connection(conn)
    
    except Exception as e:
        logger.error(f"Failed to load NBA player data: {str(e)}")
        return False


def load_nba_game_data() -> bool:
    """
    Load real NBA game data from BallDontLie API
    
    Returns:
        bool: True if successful
    """
    try:
        logger.info("Loading NBA game data from API...")
        
        # Initialize API client
        api_client = BallDontLieApiClient()
        
        # Fetch games from the past 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Format dates for API
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch games
        games_data = api_client.get_games(start_date=start_date_str, end_date=end_date_str)
        
        if not games_data or 'data' not in games_data:
            logger.error("Failed to fetch games from API")
            return False
            
        games = games_data.get('data', [])
        logger.info(f"Fetched {len(games)} games from API")
        
        # Store games in database
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # Insert games
                for game in games:
                    game_date = datetime.strptime(game.get('date'), '%Y-%m-%dT%H:%M:%S.%fZ') if 'date' in game else None
                    season_year = game.get('season', 0)
                    
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
                    """, (
                        str(game.get('id')),
                        season_year,
                        game_date,
                        str(game.get('home_team', {}).get('id')) if game.get('home_team') else None,
                        str(game.get('visitor_team', {}).get('id')) if game.get('visitor_team') else None,
                        game.get('status'),
                        game
                    ))
                
                conn.commit()
                logger.info(f"Stored {len(games)} games in database")
                return True
        except Exception as e:
            logger.error(f"Error storing games in database: {str(e)}")
            conn.rollback()
            return False
        finally:
            close_connection(conn)
    
    except Exception as e:
        logger.error(f"Failed to load NBA game data: {str(e)}")
        return False


def load_odds_data() -> bool:
    """
    Load real odds data from TheOdds API
    
    Returns:
        bool: True if successful
    """
    try:
        logger.info("Loading odds data from API...")
        
        # Initialize API client
        odds_collector = OddsApiCollector()
        
        # Fetch odds data
        odds_data = odds_collector.collect_game_data('basketball_nba')
        
        if not odds_data:
            logger.warning("No odds data available from API")
            return True  # Not a failure, just no data available
            
        logger.info(f"Fetched odds data for {len(odds_data)} games")
        
        # Store odds in database
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # Create odds table if not exists
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS odds (
                    id SERIAL PRIMARY KEY,
                    game_id VARCHAR(100) NOT NULL,
                    bookmaker VARCHAR(100) NOT NULL,
                    market VARCHAR(100) NOT NULL,
                    odds JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(game_id, bookmaker, market)
                )
                """)
                
                # Insert odds
                odds_inserted = 0
                for game in odds_data:
                    game_id = game.get('id')
                    
                    # Insert each bookmaker's odds
                    for bookmaker in game.get('bookmakers', []):
                        bookmaker_name = bookmaker.get('key')
                        
                        for market in bookmaker.get('markets', []):
                            market_key = market.get('key')
                            
                            try:
                                cursor.execute("""
                                INSERT INTO odds (game_id, bookmaker, market, odds)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (game_id, bookmaker, market) DO UPDATE SET
                                    odds = EXCLUDED.odds,
                                    created_at = CURRENT_TIMESTAMP
                                """, (
                                    game_id,
                                    bookmaker_name,
                                    market_key,
                                    market
                                ))
                                odds_inserted += 1
                            except Exception as e:
                                logger.warning(f"Error inserting odds for game {game_id}: {str(e)}")
                
                conn.commit()
                logger.info(f"Stored odds data ({odds_inserted} entries) in database")
                return True
        except Exception as e:
            logger.error(f"Error storing odds in database: {str(e)}")
            conn.rollback()
            return False
        finally:
            close_connection(conn)
    
    except Exception as e:
        logger.error(f"Failed to load odds data: {str(e)}")
        return False


def main():
    """
    Main function when script is run directly
    """
    parser = argparse.ArgumentParser(description='Set up the NBA prediction database with real data')
    parser.add_argument('--config', default='database.conf', help='Path to database configuration file')
    parser.add_argument('--force', action='store_true', help='Force recreation of tables')
    args = parser.parse_args()
    
    # Set up logging
    print("\n======== NBA Prediction System - Real Database Setup ========\n")
    logger.info("Starting database setup with real NBA data...")
    
    # Load configuration
    config = load_config_from_file(args.config)
    if not config:
        logger.error("Failed to load configuration - please edit the database.conf file with your credentials")
        return 1
    
    # Set up database user
    if not setup_database_user():
        logger.error("Database user setup failed - check connection details")
        return 1
    
    # Initialize database structure
    if not setup_database(args.force):
        logger.error("Database initialization failed - check connection details")
        return 1
    
    # Load real NBA data
    results = [
        ("NBA Teams", load_nba_team_data()),
        ("NBA Players", load_nba_player_data()),
        ("NBA Games", load_nba_game_data()),
        ("Betting Odds", load_odds_data())
    ]
    
    # Print summary
    print("\n======== Database Setup Results ========")
    all_successful = True
    for name, success in results:
        status = "✅ Loaded successfully" if success else "❌ Loading failed"
        print(f"{name}: {status}")
        if not success:
            all_successful = False
    
    if all_successful:
        print("\n✅ Database setup completed successfully with real NBA data!")
        print("You can now start the API server:")
        print("  python -m src.api.server")
        logger.info("Database setup completed successfully")
        return 0
    else:
        print("\n⚠️ Database setup completed with some errors.")
        print("Check the logs for details and resolve issues before continuing.")
        logger.warning("Database setup completed with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
