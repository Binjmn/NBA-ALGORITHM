"""
Database Connection Manager

This module provides functionality for managing PostgreSQL database connections,
including connection pooling, initialization, and transaction management.

The database structure is designed to support long-term data storage for NBA
prediction data, including games, players, model weights, and performance metrics.
"""

import logging
import os
from contextlib import contextmanager
from typing import Dict, Any, Generator, Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json

# Configure logging
logger = logging.getLogger(__name__)

# Global connection pool
_connection_pool = None

# Initialize the database
def init_db(db_config: Dict[str, Any] = None) -> bool:
    """
    Initialize the database connection pool and create tables if they don't exist
    
    Args:
        db_config: Dictionary with database configuration parameters
               If None, environment variables will be used
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global _connection_pool
    
    try:
        # Get database configuration
        if db_config is None:
            # Read from environment variables first
            db_config = {
                'host': os.environ.get('POSTGRES_HOST', 'localhost'),
                'port': os.environ.get('POSTGRES_PORT', '5433'),
                'database': os.environ.get('POSTGRES_DB', 'nba_prediction'),
                'user': os.environ.get('POSTGRES_USER', 'postgres'),
                'password': os.environ.get('POSTGRES_PASSWORD', 'ALGO123')
            }
            
            # Try to read from config file if environment variables not set
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'database.conf')
            if os.path.exists(config_path):
                logger.info(f"Reading database configuration from {config_path}")
                try:
                    with open(config_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                            key, value = line.split('=', 1)
                            if key == 'POSTGRES_HOST' and not db_config['host']:
                                db_config['host'] = value
                            elif key == 'POSTGRES_PORT' and not db_config['port']:
                                db_config['port'] = value
                            elif key == 'POSTGRES_DB' and not db_config['database']:
                                db_config['database'] = value
                            elif key == 'POSTGRES_USER' and not db_config['user']:
                                db_config['user'] = value
                            elif key == 'POSTGRES_PASSWORD' and not db_config['password']:
                                db_config['password'] = value
                except Exception as e:
                    logger.warning(f"Error reading config file: {str(e)}")
        
        # Validate configuration
        if not all([db_config.get('host'), db_config.get('port'), db_config.get('user'), db_config.get('password')]):
            logger.error("Missing required database configuration parameters")
            return False
        
        # Initialize connection pool
        logger.info(f"Initializing connection pool to PostgreSQL at {db_config['host']}:{db_config['port']}")
        
        # For production systems, keep a smaller number of connections (5) to avoid exhausting resources
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )
        
        # Test connection
        conn = _connection_pool.getconn()
        if conn:
            logger.info("Successfully connected to PostgreSQL")
            _connection_pool.putconn(conn)
        
        # Create tables if they don't exist
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Create tables
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
                );
                """)
                
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
                );
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_weights (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100) NOT NULL,
                    model_type VARCHAR(100) NOT NULL,
                    weights JSONB NOT NULL,
                    params JSONB NOT NULL DEFAULT '{}'::jsonb,
                    version INTEGER NOT NULL,
                    trained_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    active BOOLEAN DEFAULT TRUE,
                    UNIQUE (model_name, version)
                );
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100) NOT NULL,
                    prediction_target VARCHAR(50) NOT NULL,
                    metrics JSONB DEFAULT '{}'::jsonb,
                    is_baseline BOOLEAN DEFAULT false,
                    "time_window" VARCHAR(50) DEFAULT '7d',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                # Create indexes for better query performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_season_year ON games(season_year);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_performance_model_name ON model_performance(model_name);")  
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_performance_predict_target ON model_performance(prediction_target);")  
                
                # Create update trigger for updated_at timestamps
                cursor.execute("""
                CREATE OR REPLACE FUNCTION update_modified_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                """)
                
                # Apply trigger to tables
                for table in ['games', 'players']:
                    cursor.execute(f"""
                    DROP TRIGGER IF EXISTS update_{table}_modtime ON {table};
                    CREATE TRIGGER update_{table}_modtime
                    BEFORE UPDATE ON {table}
                    FOR EACH ROW
                    EXECUTE FUNCTION update_modified_column();
                    """)
                
                conn.commit()
                
        logger.info("Database tables initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        _connection_pool = None
        return False


@contextmanager
def get_db_connection() -> Generator[Any, None, None]:
    """
    Get a database connection from the pool. This function should be used with
    a context manager to ensure connections are returned to the pool.
    
    Yields:
        connection: Database connection object
    
    Example:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM games")
                results = cursor.fetchall()
    """
    global _connection_pool
    
    if _connection_pool is None:
        init_db()
        
    if _connection_pool is None:
        raise RuntimeError("Database connection pool not initialized")
    
    connection = None
    try:
        connection = _connection_pool.getconn()
        yield connection
    finally:
        if connection is not None:
            _connection_pool.putconn(connection)


# Alias for backward compatibility
def get_connection():
    """
    Alias for get_db_connection for backward compatibility
    
    Returns:
        Connection object from the pool
    """
    try:
        # Attempt to get a connection from the pool
        if _connection_pool is None:
            # Initialize the connection pool if it doesn't exist
            init_db()
            if _connection_pool is None:
                raise RuntimeError("Failed to initialize database connection pool")
        
        # Get and return a connection from the pool
        conn = _connection_pool.getconn()
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise


# Close a specific connection
def close_connection(conn):
    """
    Close a specific database connection and return it to the pool
    
    Args:
        conn: Connection to close
    """
    if conn is not None and not conn.closed:
        _connection_pool.putconn(conn)


@contextmanager
def get_dict_cursor() -> Generator[Any, None, None]:
    """
    Get a cursor that returns results as dictionaries instead of tuples.
    This is a convenience wrapper around get_db_connection().
    
    Yields:
        cursor: Database cursor object that returns dictionaries
    
    Example:
        with get_dict_cursor() as cursor:
            cursor.execute("SELECT * FROM games WHERE season_year = %s", (2025,))
            games = cursor.fetchall()
            # Each game is a dictionary with column names as keys
    """
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            yield cursor


def close_db_connections() -> None:
    """
    Close all database connections in the pool. This should be called
    when the application is shutting down.
    """
    global _connection_pool
    
    if _connection_pool is not None:
        _connection_pool.closeall()
        logger.info("All database connections closed")
        _connection_pool = None
