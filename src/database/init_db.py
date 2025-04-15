"""
Database Initialization Script

This script initializes the PostgreSQL database for the NBA prediction system.
It creates the necessary tables, indexes, and triggers if they don't already exist.

This script can be run directly or imported and used as part of the application
startup process.

Usage:
    # Run directly
    python -m src.database.init_db
    
    # Or import and use in application
    from src.database.init_db import initialize_database
    initialize_database()
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to make imports work when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.connection import init_db, close_db_connections


def initialize_database(db_config: Optional[Dict[str, Any]] = None, verbose: bool = False) -> bool:
    """
    Initialize the database with all required tables and indexes
    
    Args:
        db_config: Optional database configuration dictionary
        verbose: Whether to log detailed information
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    if verbose:
        logger.info("Initializing NBA Prediction database...")
    
    try:
        # Initialize database (creates tables, indexes, triggers)
        success = init_db(db_config)
        
        if success:
            if verbose:
                logger.info("Database initialization completed successfully.")
            return True
        else:
            logger.error("Database initialization failed.")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {str(e)}")
        return False
    finally:
        # Close any open connections
        close_db_connections()


def main():
    """
    Main function when script is run directly
    """
    parser = argparse.ArgumentParser(description="Initialize the NBA Prediction database")
    parser.add_argument(
        "--host", 
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host"
    )
    parser.add_argument(
        "--port", 
        default=os.environ.get("POSTGRES_PORT", "5432"),
        help="PostgreSQL port"
    )
    parser.add_argument(
        "--dbname", 
        default=os.environ.get("POSTGRES_DB", "nba_prediction"),
        help="PostgreSQL database name"
    )
    parser.add_argument(
        "--user", 
        default=os.environ.get("POSTGRES_USER", "postgres"),
        help="PostgreSQL username"
    )
    parser.add_argument(
        "--password", 
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create database configuration
    db_config = {
        "host": args.host,
        "port": args.port,
        "dbname": args.dbname,
        "user": args.user,
        "password": args.password
    }
    
    success = initialize_database(db_config, verbose=args.verbose)
    
    if success:
        logger.info("Database initialization completed successfully")
        sys.exit(0)
    else:
        logger.error("Database initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
