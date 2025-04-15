"""
Example script demonstrating the use of advanced models in the NBA prediction system.

This example shows how to:
1. Perform hyperparameter tuning on base models
2. Create an ensemble stacking model that combines multiple base models
3. Evaluate and compare the performance of different modeling approaches
"""

import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our models
from src.models.random_forest import RandomForestModel
from src.models.combined_gradient_boosting import CombinedGradientBoostingModel
from src.models.bayesian_model import BayesianModel
from src.models.ensemble_stacking import EnsembleStackingModel
from src.models.hyperparameter_tuning import HyperparameterTuner

# Import database utilities (if needed to load real data)
from src.database.connection import get_connection, close_connection
from src.database.models import Game


def load_nba_data(limit=500):
    """
    Load NBA data from the database for model training and evaluation.
    
    Args:
        limit: Maximum number of records to load
    
    Returns:
        X: Feature matrix
        y: Target variable (win/loss for the home team)
        
    Raises:
        RuntimeError: If unable to load sufficient data from the database
    """
    try:
        # Connect to database
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    g.game_data,
                    CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END as home_win
                FROM games g
                WHERE g.season >= 2020 AND g.game_data IS NOT NULL
                ORDER BY g.game_date DESC
                LIMIT %s
            """, (limit,))
            
            records = cursor.fetchall()
        
        close_connection(conn)
        
        if not records:
            raise RuntimeError("No game records found in the database. Please run data collection first.")
        
        if len(records) < 50:  # Minimum data requirement for meaningful model training
            logger.warning(f"Only {len(records)} records found. Models may not train effectively.")
        
        # Process the records
        features = []
        targets = []
        
        for record in records:
            game_data = record[0]  # JSONB data
            home_win = record[1]   # 1 if home team won, 0 otherwise
            
            # Extract features from game_data
            game_features = extract_features_from_game_data(game_data)
            
            # Only use records with sufficient features
            if game_features and len(game_features) > 5:  # Ensure we have meaningful features
                features.append(game_features)
                targets.append(home_win)
        
        if not features:
            raise RuntimeError("Failed to extract valid features from game data.")
        
        # Convert to DataFrame and Series
        X = pd.DataFrame(features)
        y = pd.Series(targets)
        
        logger.info(f"Successfully loaded {len(X)} records from database")
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading data from database: {str(e)}")
        raise RuntimeError(f"Failed to load NBA data: {str(e)}")


def extract_features_from_game_data(game_data):
    """
    Extract features from game_data JSON
    
    This is a placeholder function that would need to be implemented
    based on the actual structure of your game_data JSON.
    
    Args:
        game_data: JSON data containing game information
        
    Returns:
        Dictionary of features
    """
    # This is a placeholder - in a real system, you would extract relevant features
    # from the game_data JSON based on your specific data structure
    features = {}
    
    try:
        # Example extraction (adjust based on your actual data structure)
        if isinstance(game_data, dict):
            # Team stats
            for team_type in ['home', 'away']:
                if team_type in game_data and 'team_stats' in game_data[team_type]:
                    team_stats = game_data[team_type]['team_stats']
                    for stat_name, stat_value in team_stats.items():
                        features[f"{team_type}_{stat_name}"] = stat_value
                        
            # Game context features
            if 'context' in game_data:
                context = game_data['context']
                for context_name, context_value in context.items():
                    features[f"context_{context_name}"] = context_value
    except Exception as e:
        logger.warning(f"Error extracting features from game_data: {str(e)}")
    
    return features


def main():
    """
    Main function demonstrating the use of advanced models
    """
    logger.info("Starting advanced models example")
    
    try:
        # Load data
        logger.info("Loading data from database...")
        X, y = load_nba_data(limit=1000)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Testing set: {X_test.shape[0]} samples")
        
        # Initialize base models
        rf_model = RandomForestModel(name="RandomForest", prediction_target="moneyline")
        gbm_model = CombinedGradientBoostingModel(name="GradientBoosting", prediction_target="moneyline")
        bayes_model = BayesianModel(name="Bayesian", prediction_target="moneyline")
        
        # Train base models
        logger.info("Training base models...")
        rf_model.train(X_train, y_train)
        gbm_model.train(X_train, y_train)
        bayes_model.train(X_train, y_train)
        
        # Evaluate base models
        logger.info("Evaluating base models...")
        rf_metrics = rf_model.evaluate(X_test, y_test)
        gbm_metrics = gbm_model.evaluate(X_test, y_test)
        bayes_metrics = bayes_model.evaluate(X_test, y_test)
        
        logger.info(f"Random Forest accuracy: {rf_metrics['accuracy']:.4f}")
        logger.info(f"Gradient Boosting accuracy: {gbm_metrics['accuracy']:.4f}")
        logger.info(f"Bayesian model accuracy: {bayes_metrics['accuracy']:.4f}")
        
        # Example 1: Hyperparameter Tuning
        logger.info("\n--- Hyperparameter Tuning Example ---")
        
        # Create tuner for Random Forest model
        rf_tuner = HyperparameterTuner(model_type="RandomForest", prediction_target="moneyline")
        
        # Run tuning (with reduced iterations for example purposes)
        logger.info("Tuning Random Forest model...")
        rf_tuning_results = rf_tuner.tune(X_train, y_train, n_iter=5, cv=3, verbose=0)
        
        logger.info(f"Best parameters: {rf_tuning_results['best_params']}")
        logger.info(f"Best CV score: {rf_tuning_results['best_score']:.4f}")
        
        # Create tuned model
        tuned_rf_model = rf_tuner.create_tuned_model()
        
        # Train and evaluate tuned model
        tuned_rf_model.train(X_train, y_train)
        tuned_rf_metrics = tuned_rf_model.evaluate(X_test, y_test)
        
        logger.info(f"Tuned Random Forest accuracy: {tuned_rf_metrics['accuracy']:.4f}")
        logger.info(f"Improvement over base model: {tuned_rf_metrics['accuracy'] - rf_metrics['accuracy']:.4f}")
        
        # Example 2: Ensemble Stacking
        logger.info("\n--- Ensemble Stacking Example ---")
        
        # Create ensemble stacking model
        stack_model = EnsembleStackingModel(name="EnsembleStack", prediction_target="moneyline")
        
        # Set base models
        stack_model.set_base_models({
            "RandomForest": rf_model,
            "GradientBoosting": gbm_model,
            "Bayesian": bayes_model
        })
        
        # Train ensemble model
        logger.info("Training ensemble stacking model...")
        stack_model.train(X_train, y_train)
        
        # Evaluate ensemble model
        stack_metrics = stack_model.evaluate(X_test, y_test)
        
        logger.info(f"Ensemble Stacking accuracy: {stack_metrics['accuracy']:.4f}")
        
        # Compare with base models
        logger.info("\n--- Model Comparison ---")
        logger.info(f"Random Forest accuracy: {rf_metrics['accuracy']:.4f}")
        logger.info(f"Gradient Boosting accuracy: {gbm_metrics['accuracy']:.4f}")
        logger.info(f"Bayesian model accuracy: {bayes_metrics['accuracy']:.4f}")
        logger.info(f"Tuned Random Forest accuracy: {tuned_rf_metrics['accuracy']:.4f}")
        logger.info(f"Ensemble Stacking accuracy: {stack_metrics['accuracy']:.4f}")
        
        # Get feature importance from the ensemble model
        stack_importance = stack_model.get_feature_importance()
        logger.info("\n--- Ensemble Stacking Feature Importance ---")
        for feature, importance in list(stack_importance.items())[:10]:  # Top 10 features
            logger.info(f"{feature}: {importance:.4f}")
        
        logger.info("Advanced models example completed successfully")
        
    except RuntimeError as e:
        logger.error(f"Failed to run example: {str(e)}")
        logger.error("Please ensure the database is populated with game data before running this example.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
