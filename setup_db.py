"""
Simple PostgreSQL Database Setup Script

This script creates a PostgreSQL user and database for the NBA prediction system.
It sets the password to 'ALGO123' as requested.
"""

import os
import sys
import psycopg2
from psycopg2 import sql
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_HOST = 'localhost'
DB_PORT = '5432'
ADMIN_DB = 'postgres'  # Default database to connect to initially
DB_NAME = 'nba_prediction'  # Database we want to create
DB_USER = 'nba_user'  # User we want to create/update
DB_PASSWORD = 'ALGO123'  # Password for the user

def setup_database():
    """Set up PostgreSQL database and user"""
    try:
        print("\n===== NBA Prediction System Database Setup =====\n")
        print("This script will:")
        print("1. Connect to your PostgreSQL server")
        print("2. Create a user named 'nba_user' with password 'ALGO123'")
        print("3. Create a database named 'nba_prediction'")
        print("4. Set up the required tables")
        print("\nEnter the current PostgreSQL admin password when prompted")
        
        # Get admin password securely
        admin_password = input("\nEnter your current PostgreSQL admin password: ")
        
        # Connect to PostgreSQL as admin
        logger.info(f"Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}")
        
        # Connect to default 'postgres' database first
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=ADMIN_DB,
            user='postgres',  # Default PostgreSQL admin user
            password=admin_password
        )
        conn.autocommit = True  # Needed for CREATE DATABASE
        
        # Create user and database
        with conn.cursor() as cursor:
            # Check if user exists
            cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (DB_USER,))
            user_exists = cursor.fetchone() is not None
            
            if user_exists:
                logger.info(f"User '{DB_USER}' already exists, updating password...")
                # Update user password
                cursor.execute(sql.SQL("ALTER USER {} WITH PASSWORD {}").format(
                    sql.Identifier(DB_USER),
                    sql.Literal(DB_PASSWORD)
                ))
            else:
                logger.info(f"Creating user '{DB_USER}'...")
                # Create new user
                cursor.execute(sql.SQL("CREATE USER {} WITH PASSWORD {}").format(
                    sql.Identifier(DB_USER),
                    sql.Literal(DB_PASSWORD)
                ))
            
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
            db_exists = cursor.fetchone() is not None
            
            if not db_exists:
                logger.info(f"Creating database '{DB_NAME}'...")
                # Create database
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
                
                # Grant privileges
                cursor.execute(
                    sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {} TO {}").format(
                        sql.Identifier(DB_NAME),
                        sql.Identifier(DB_USER)
                    )
                )
            else:
                logger.info(f"Database '{DB_NAME}' already exists")
        
        logger.info("Created user and database successfully")
        conn.close()
        
        # Connect to our new database to create tables
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.autocommit = True
        
        # Create tables
        with conn.cursor() as cursor:
            # Create games table
            cursor.execute("""
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
            )
            """)
            
            # Create players table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
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
            )
            """)
            
            # Create teams table
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
            
            # Create model_weights table
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
            
            # Create model_performance table
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
        
        logger.info("Created database tables successfully")
        conn.close()
        
        # Create configuration file with new credentials
        with open('database.conf', 'w') as f:
            f.write("# PostgreSQL Configuration for NBA Prediction System\n")
            f.write("# Production-quality configuration with secure credentials\n\n")
            f.write("# Database connection settings\n")
            f.write(f"POSTGRES_HOST={DB_HOST}\n")
            f.write(f"POSTGRES_PORT={DB_PORT}\n")
            f.write(f"POSTGRES_DB={DB_NAME}\n\n")
            f.write("# Authentication settings\n")
            f.write(f"POSTGRES_USER={DB_USER}\n")
            f.write(f"POSTGRES_PASSWORD={DB_PASSWORD}\n\n")
            f.write("# API configuration\n")
            f.write("API_TOKEN=nba_api_token_2025\n")
        
        print("\n\u2705 Database setup completed successfully!")
        print(f"\nDatabase: {DB_NAME}")
        print(f"User: {DB_USER}")
        print(f"Password: {DB_PASSWORD}")
        print("\nYou can now start the API server:")
        print("python -m src.api.server")
        
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        print(f"\n\u274c Error: {str(e)}")
        return False

if __name__ == "__main__":
    setup_database()
