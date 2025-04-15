"""
Database Demo Setup Script

This script sets up a PostgreSQL database with demo data for the NBA prediction system.
It creates all required tables and adds sample model weights and performance data.

Run this script to initialize the database for development and testing purposes.

Usage:
    python -m src.database.setup_demo_db
"""

import argparse
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to make imports work when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import database modules
from src.database.init_db import initialize_database
from src.database.connection import get_connection, close_connection


def setup_demo_database(force: bool = False) -> bool:
    """
    Set up a demo database with sample data for testing
    
    Args:
        force: Whether to force recreation of existing tables
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Step 1: Initialize the database structure
        logger.info("Initializing database structure...")
        success = initialize_database(verbose=True)
        if not success:
            logger.error("Failed to initialize database structure")
            return False
            
        # Step 2: Create sample data
        logger.info("Adding sample model data...")
        
        # Connect to the database
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                # Check if we already have model data
                cursor.execute("SELECT COUNT(*) FROM model_weights")
                count = cursor.fetchone()[0]
                
                if count > 0 and not force:
                    logger.info(f"Database already contains {count} model records. Use --force to override.")
                    return True
                    
                # Clear existing data if force is True
                if force and count > 0:
                    logger.info("Clearing existing model data...")
                    cursor.execute("DELETE FROM model_performance")
                    cursor.execute("DELETE FROM model_weights")
                    conn.commit()
                
                # Insert sample model weights
                create_sample_models(conn)
                
                # Insert sample performance metrics
                create_sample_performance_metrics(conn)
                
                logger.info("Sample data created successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error creating sample data: {str(e)}")
            conn.rollback()
            return False
        finally:
            close_connection(conn)
            
    except Exception as e:
        logger.error(f"Error setting up demo database: {str(e)}")
        return False


def create_sample_models(conn) -> None:
    """
    Create sample model weight records
    
    Args:
        conn: Database connection
    """
    model_types = [
        "RandomForest", 
        "XGBoost", 
        "Bayesian", 
        "AnomalyDetection", 
        "ModelMixing", 
        "EnsembleStacking"
    ]
    
    prediction_targets = ["moneyline", "spread", "total", "player_props"]
    
    with conn.cursor() as cursor:
        for model_type in model_types:
            for target in prediction_targets:
                # Skip some combinations to make it realistic
                if target == "player_props" and model_type in ["AnomalyDetection", "ModelMixing"]:
                    continue
                    
                # Create model weight record
                trained_at = datetime.now() - timedelta(days=random.randint(1, 14))
                
                # Basic parameters common to all models
                params = {
                    "prediction_target": target,
                    "random_state": 42
                }
                
                # Model-specific parameters
                if model_type == "RandomForest":
                    params.update({
                        "n_estimators": 100,
                        "max_depth": 10,
                        "min_samples_split": 5
                    })
                elif model_type == "XGBoost":
                    params.update({
                        "n_estimators": 150,
                        "learning_rate": 0.1,
                        "max_depth": 6
                    })
                elif model_type == "Bayesian":
                    params.update({
                        "alpha": 1.0,
                        "beta": 1.0
                    })
                
                cursor.execute("""
                    INSERT INTO model_weights 
                    (model_name, model_type, params, weights, version, trained_at, active, needs_training)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    model_type,
                    model_type.lower(),
                    params,
                    f"Sample binary weights for {model_type}",
                    1,
                    trained_at,
                    True,
                    random.random() < 0.2  # 20% chance of needing training
                ))
        
        conn.commit()
        logger.info(f"Created {len(model_types) * len(prediction_targets)} sample model records")


def create_sample_performance_metrics(conn) -> None:
    """
    Create sample performance metric records
    
    Args:
        conn: Database connection
    """
    # Get all model weights
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT model_name, params->>'prediction_target' as prediction_target
            FROM model_weights
        """)
        models = cursor.fetchall()
        
        # Create 30 days of performance history
        for model_name, prediction_target in models:
            # Create baseline metrics
            base_accuracy = random.uniform(0.65, 0.85)
            baseline_metrics = {
                "accuracy": base_accuracy,
                "precision": base_accuracy - random.uniform(0, 0.1),
                "recall": base_accuracy - random.uniform(0, 0.1),
                "f1_score": base_accuracy - random.uniform(0, 0.05),
                "sample_count": random.randint(100, 500)
            }
            
            # Insert baseline
            cursor.execute("""
                INSERT INTO model_performance
                (model_name, prediction_target, metrics, is_baseline, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                model_name,
                prediction_target,
                baseline_metrics,
                True,
                datetime.now() - timedelta(days=30)
            ))
            
            # Create daily metrics with slight variations
            for day in range(30):
                # Skip some days to make it realistic
                if random.random() < 0.2:  # 20% chance to skip a day
                    continue
                    
                daily_accuracy = base_accuracy + random.uniform(-0.05, 0.05)
                daily_accuracy = max(0.5, min(0.95, daily_accuracy))  # Keep within reasonable bounds
                
                daily_metrics = {
                    "accuracy": daily_accuracy,
                    "precision": daily_accuracy - random.uniform(0, 0.1),
                    "recall": daily_accuracy - random.uniform(0, 0.1),
                    "f1_score": daily_accuracy - random.uniform(0, 0.05),
                    "sample_count": random.randint(5, 30)
                }
                
                cursor.execute("""
                    INSERT INTO model_performance
                    (model_name, prediction_target, metrics, is_baseline, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    model_name,
                    prediction_target,
                    daily_metrics,
                    False,
                    datetime.now() - timedelta(days=day)
                ))
        
        conn.commit()
        logger.info(f"Created performance metrics for {len(models)} models")


def main():
    """
    Main function when script is run directly
    """
    parser = argparse.ArgumentParser(description='Set up a demo database for the NBA prediction system')
    parser.add_argument('--force', action='store_true', help='Force recreation of sample data')
    args = parser.parse_args()
    
    logger.info("Setting up demo database...")
    
    # Configure database connection from environment variables
    db_host = os.environ.get('POSTGRES_HOST', 'localhost')
    db_port = os.environ.get('POSTGRES_PORT', '5432')
    db_name = os.environ.get('POSTGRES_DB', 'nba_prediction')
    db_user = os.environ.get('POSTGRES_USER', 'postgres')
    db_pass = os.environ.get('POSTGRES_PASSWORD', 'postgres')
    
    # Print connection info
    logger.info(f"Database connection: {db_user}@{db_host}:{db_port}/{db_name}")
    
    if setup_demo_database(args.force):
        logger.info("✅ Demo database setup complete!")
        print("\nDemo database has been successfully set up with sample data.")
        print("You can now start the API server:")
        print("  python -m src.api.server")
        return 0
    else:
        logger.error("❌ Demo database setup failed!")
        print("\nFailed to set up demo database. Check the logs for details.")
        print("Make sure PostgreSQL is running and the connection details are correct.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
