#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Database Connection Manager

This module provides a production-ready connection manager for PostgreSQL with:
- Connection pooling with proper resource management
- Automatic reconnection on failure
- Exponential backoff retry logic
- Thread-safe connection handling
- Comprehensive error reporting
"""

import os
import time
import random
import logging
from typing import Dict, Any, Optional, Tuple
from threading import Lock
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global connection pool
_connection_pool = None
_pool_lock = Lock()

# Connection parameters
DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST', 'localhost'),
    'port': os.environ.get('POSTGRES_PORT', '5432'),
    'database': os.environ.get('POSTGRES_DB', 'nba_prediction'),
    'user': os.environ.get('POSTGRES_USER', 'postgres'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'ALGO123'),
    'min_connections': int(os.environ.get('POSTGRES_MIN_CONN', '2')),
    'max_connections': int(os.environ.get('POSTGRES_MAX_CONN', '10')),
    'max_retries': int(os.environ.get('POSTGRES_MAX_RETRIES', '3')),
    'retry_delay': int(os.environ.get('POSTGRES_RETRY_DELAY', '2'))
}

# Time of last successful database connection (for health checks)
LAST_SUCCESSFUL_CONNECTION = 0


def init_db(retry: bool = True, max_retries: int = None) -> bool:
    """Initialize the database connection pool with retry logic
    
    Args:
        retry: Whether to retry on failure
        max_retries: Maximum number of retries (None = use DB_CONFIG value)
        
    Returns:
        bool: True if successful, False otherwise
    """
    global _connection_pool, LAST_SUCCESSFUL_CONNECTION
    
    if max_retries is None:
        max_retries = DB_CONFIG['max_retries']
    
    # If pool already exists, return success
    if _connection_pool is not None:
        return True
    
    # Acquire lock to prevent multiple threads from initializing the pool simultaneously
    with _pool_lock:
        # Check again after acquiring lock
        if _connection_pool is not None:
            return True
        
        # Build connection parameters
        conn_params = {
            'host': DB_CONFIG['host'],
            'port': DB_CONFIG['port'],
            'database': DB_CONFIG['database'],
            'user': DB_CONFIG['user'],
            'password': DB_CONFIG['password']
        }
        
        logger.info(f"Initializing database connection pool to {conn_params['host']}:{conn_params['port']}/{conn_params['database']}")
        
        # Initialize with retry logic
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # Create connection pool
                _connection_pool = pool.ThreadedConnectionPool(
                    minconn=DB_CONFIG['min_connections'],
                    maxconn=DB_CONFIG['max_connections'],
                    **conn_params
                )
                
                # Test the connection
                conn = _connection_pool.getconn()
                with conn.cursor() as cursor:
                    cursor.execute("SELECT version()")
                    version = cursor.fetchone()[0]
                    logger.info(f"Connected to PostgreSQL: {version}")
                _connection_pool.putconn(conn)
                
                # Update last successful connection time
                LAST_SUCCESSFUL_CONNECTION = time.time()
                
                return True
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries or not retry:
                    logger.error(f"Failed to initialize database connection pool: {str(e)}")
                    return False
                
                # Calculate backoff with jitter to prevent thundering herd
                delay = DB_CONFIG['retry_delay'] * (2 ** (retry_count - 1)) + random.uniform(0, 0.5)
                logger.warning(f"Database connection failed (attempt {retry_count}/{max_retries}), retrying in {delay:.2f} seconds: {str(e)}")
                time.sleep(delay)
        
        return False


@contextmanager
def get_db_connection(retry: bool = True, max_retries: int = None) -> psycopg2.extensions.connection:
    """Get a database connection from the pool with retry logic
    
    Args:
        retry: Whether to retry on failure
        max_retries: Maximum number of retries (None = use DB_CONFIG value)
        
    Yields:
        connection: Database connection object
        
    Raises:
        Exception: If connection cannot be obtained
    """
    global _connection_pool, LAST_SUCCESSFUL_CONNECTION
    
    if max_retries is None:
        max_retries = DB_CONFIG['max_retries']
    
    # Initialize the pool if it doesn't exist
    if _connection_pool is None:
        if not init_db(retry=retry, max_retries=max_retries):
            raise Exception("Failed to initialize database connection pool")
    
    # Get connection with retry logic
    conn = None
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            conn = _connection_pool.getconn()
            
            # Test the connection to ensure it's still valid
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            
            # Update last successful connection time
            LAST_SUCCESSFUL_CONNECTION = time.time()
            
            # Yield connection for use
            yield conn
            
            # Commit when done
            conn.commit()
            break  # Exit the loop on success
        except Exception as e:
            last_error = e
            retry_count += 1
            
            if conn is not None:
                try:
                    conn.rollback()  # Rollback any uncommitted transactions
                except:
                    pass  # Ignore errors in rollback
                
                try:
                    _connection_pool.putconn(conn)  # Return the connection to the pool
                except:
                    pass  # Ignore errors in putconn
                conn = None
            
            if retry_count > max_retries or not retry:
                logger.error(f"Database connection failed after {retry_count} attempts: {str(e)}")
                raise
            
            # Calculate backoff with jitter
            delay = DB_CONFIG['retry_delay'] * (2 ** (retry_count - 1)) + random.uniform(0, 0.5)
            logger.warning(f"Database connection failed (attempt {retry_count}/{max_retries}), retrying in {delay:.2f} seconds: {str(e)}")
            time.sleep(delay)
    
    # Return connection to the pool
    if conn is not None:
        _connection_pool.putconn(conn)


def get_connection() -> psycopg2.extensions.connection:
    """Get a database connection (non-context manager version)
    
    Returns:
        connection: Database connection object
    
    Raises:
        Exception: If connection cannot be obtained
    """
    global _connection_pool, LAST_SUCCESSFUL_CONNECTION
    
    # Initialize the pool if it doesn't exist
    if _connection_pool is None:
        if not init_db():
            raise Exception("Failed to initialize database connection pool")
    
    # Get connection with retry logic
    retry_count = 0
    max_retries = DB_CONFIG['max_retries']
    last_error = None
    
    while retry_count <= max_retries:
        try:
            conn = _connection_pool.getconn()
            
            # Test the connection to ensure it's still valid
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            
            # Update last successful connection time
            LAST_SUCCESSFUL_CONNECTION = time.time()
            
            return conn
        except Exception as e:
            last_error = e
            retry_count += 1
            
            if retry_count > max_retries:
                logger.error(f"Database connection failed after {retry_count} attempts: {str(e)}")
                raise
            
            # Calculate backoff with jitter
            delay = DB_CONFIG['retry_delay'] * (2 ** (retry_count - 1)) + random.uniform(0, 0.5)
            logger.warning(f"Database connection failed (attempt {retry_count}/{max_retries}), retrying in {delay:.2f} seconds: {str(e)}")
            time.sleep(delay)


def close_connection(conn: psycopg2.extensions.connection) -> None:
    """Close a database connection and return it to the pool
    
    Args:
        conn: Connection to close
    """
    global _connection_pool
    
    if conn and _connection_pool:
        try:
            _connection_pool.putconn(conn)
        except Exception as e:
            logger.warning(f"Error returning connection to pool: {str(e)}")


@contextmanager
def get_dict_cursor() -> psycopg2.extras.RealDictCursor:
    """Get a cursor that returns results as dictionaries
    
    Yields:
        cursor: RealDictCursor object
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            yield cursor


def close_db_connections() -> None:
    """Close all database connections in the pool
    
    This should be called when the application is shutting down
    """
    global _connection_pool
    
    if _connection_pool:
        logger.info("Closing all database connections")
        _connection_pool.closeall()
        _connection_pool = None


def get_db_status() -> Dict[str, Any]:
    """Get the status of the database connection
    
    Returns:
        dict: Database status information
    """
    global _connection_pool, LAST_SUCCESSFUL_CONNECTION
    
    status = {
        "status": "disconnected",
        "last_connected": None,
        "uptime": None,
        "config": {
            "host": DB_CONFIG['host'],
            "port": DB_CONFIG['port'],
            "database": DB_CONFIG['database'],
            "user": DB_CONFIG['user'],
            "min_connections": DB_CONFIG['min_connections'],
            "max_connections": DB_CONFIG['max_connections']
        },
        "pool": {
            "initialized": _connection_pool is not None,
            "used_connections": 0,
            "total_connections": 0
        }
    }
    
    # Check if pool exists and is operational
    if _connection_pool:
        try:
            # Test connection
            conn = _connection_pool.getconn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT version()")
                version = cursor.fetchone()[0]
                status["version"] = version
            _connection_pool.putconn(conn)
            
            # Update status
            status["status"] = "connected"
            status["last_connected"] = time.time()
            LAST_SUCCESSFUL_CONNECTION = status["last_connected"]
            
            if LAST_SUCCESSFUL_CONNECTION > 0:
                status["uptime"] = time.time() - LAST_SUCCESSFUL_CONNECTION
            
            # Pool statistics
            if hasattr(_connection_pool, "_used") and hasattr(_connection_pool, "_pool"):
                status["pool"]["used_connections"] = len(_connection_pool._used)
                status["pool"]["total_connections"] = len(_connection_pool._used) + len(_connection_pool._pool)
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)
            
            # Try to initialize again if error
            try:
                init_db()
            except:
                pass
    
    return status


def execute_query(query: str, params: Tuple = None, fetch_one: bool = False) -> Any:
    """Execute a query and return the results
    
    Args:
        query: SQL query to execute
        params: Query parameters
        fetch_one: Whether to fetch one row or all rows
        
    Returns:
        The query results
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            if fetch_one:
                return cursor.fetchone()
            return cursor.fetchall()


def initialize_schema() -> bool:
    """Initialize the database schema
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Enable extensions
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                cursor.execute("CREATE EXTENSION IF NOT EXISTS btree_gin;")
                
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
                );
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
                );
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
                );
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
                );
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
                    UNIQUE(model_name, (params->>'prediction_target'), version)
                );
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
                );
                """)
                
                # System Logs table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id SERIAL PRIMARY KEY,
                    log_type VARCHAR(50) NOT NULL,
                    message TEXT NOT NULL,
                    details JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                # Create indexes for performance optimization
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games (date);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_status ON games (status);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_season ON games (season_year);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams_conference ON teams (conference);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_players_team ON players (team_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_stats_game ON game_stats (game_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_stats_player ON game_stats (player_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_weights_active ON model_weights (active);")
                
                # Add model templates
                cursor.execute("""
                INSERT INTO model_weights (model_name, model_type, params, weights, needs_training)
                VALUES 
                ('RandomForest', 'classification', '{"n_estimators": 100, "max_depth": 10, "prediction_target": "moneyline"}'::jsonb, '\\x00'::bytea, true),
                ('GradientBoosting', 'regression', '{"n_estimators": 150, "learning_rate": 0.1, "prediction_target": "spread"}'::jsonb, '\\x00'::bytea, true),
                ('BayesianModel', 'probability', '{"prediction_target": "moneyline"}'::jsonb, '\\x00'::bytea, true),
                ('EnsembleStack', 'meta-model', '{"prediction_target": "combined", "base_models": ["RandomForest", "GradientBoosting"]}'::jsonb, '\\x00'::bytea, true)
                ON CONFLICT (model_name, (params->>'prediction_target'), version) DO NOTHING;
                """)
                
                # Log initialization
                cursor.execute("""
                INSERT INTO system_logs (log_type, message, details)
                VALUES ('INIT', 'Database schema initialized', '{"timestamp": "' || NOW() || '", "version": "1.0.0"}'::jsonb);
                """)
        
        logger.info("Database schema initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database schema: {str(e)}")
        return False


# Module initialization
if __name__ == "__main__":
    # Test module
    print("Testing database connection...")
    success = init_db()
    if success:
        print("Successfully initialized database connection pool")
        
        # Test schema initialization
        schema_success = initialize_schema()
        if schema_success:
            print("Successfully initialized database schema")
        else:
            print("Failed to initialize database schema")
        
        # Check database status
        status = get_db_status()
        print(f"Database status: {status['status']}")
        if 'version' in status:
            print(f"PostgreSQL version: {status['version']}")
        
        # Close connections
        close_db_connections()
        print("Closed all database connections")
    else:
        print("Failed to initialize database connection pool")
