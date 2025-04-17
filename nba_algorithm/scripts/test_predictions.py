#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Model Prediction Testing Script

This script evaluates the performance of all trained models on upcoming NBA games,
allowing for comparison of predictions against actual outcomes and between models.

It supports:
- Running predictions from all trained models
- Comparing model performance metrics
- Visualizing prediction confidence
- Generating detailed performance reports
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now import our modules
from src.database.connection import init_db, get_connection_pool
from src.database.models import ModelPerformance, ModelWeights, SystemLog
from src.api.theodds_client import OddsApiCollector
from src.api.balldontlie_client import HistoricalDataCollector
from src.models.random_forest_model import RandomForestModel
from src.models.gradient_boosting_model import GradientBoostingModel
from src.models.bayesian_model import BayesianModel
from src.models.combined_gradient_boosting import CombinedGradientBoostingModel
from src.models.ensemble_model import EnsembleModel
from src.models.ensemble_stacking import EnsembleStackingModel
from src.features.advanced_features import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_name, prediction_type="moneyline", version=1):
    """Load a trained model based on name and version

    Args:
        model_name: The model class name to load
        prediction_type: The prediction target ('moneyline', 'spread', etc.)
        version: The model version to load

    Returns:
        The initialized and loaded model or None if failed
    """
    try:
        # Map model names to classes
        model_classes = {
            "RandomForestModel": RandomForestModel,
            "GradientBoostingModel": GradientBoostingModel,
            "BayesianModel": BayesianModel,
            "CombinedGradientBoostingModel": CombinedGradientBoostingModel,
            "EnsembleModel": EnsembleModel,
            "EnsembleStackingModel": EnsembleStackingModel
        }

        if model_name not in model_classes:
            logger.error(f"Unknown model type: {model_name}")
            return None

        # Initialize the appropriate model class
        if model_name in ["RandomForestModel", "GradientBoostingModel"]:
            model = model_classes[model_name](version=version)
        else:
            model = model_classes[model_name](prediction_target=prediction_type, version=version)

        # Load the model weights from the database
        loaded = model.load()
        if not loaded:
            logger.error(f"Failed to load {model_name} weights from database")
            return None

        logger.info(f"Successfully loaded {model_name} version {version}")
        return model

    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        return None


def get_upcoming_games(days=7):
    """Get upcoming NBA games for prediction testing

    Args:
        days: Number of days ahead to fetch games

    Returns:
        DataFrame of upcoming games with odds data
    """
    try:
        # Initialize the API collectors
        odds_collector = OddsApiCollector()
        historical_collector = HistoricalDataCollector()

        # Fetch upcoming games
        today = datetime.now().date()
        end_date = today + timedelta(days=days)
        
        upcoming_games = historical_collector.get_games_by_date_range(
            start_date=today.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        if upcoming_games.empty:
            logger.warning(f"No upcoming games found for the next {days} days")
            return pd.DataFrame()

        # Fetch odds for these games
        game_odds = odds_collector.collect_game_data()
        
        # Merge game data with odds
        merged_data = pd.merge(
            upcoming_games,
            game_odds,
            left_on=['home_team', 'away_team'],
            right_on=['home_team', 'away_team'],
            how='left'
        )

        # Engineer features for the games
        feature_engineer = FeatureEngineer()
        game_features = feature_engineer.engineer_features(merged_data)

        return game_features

    except Exception as e:
        logger.error(f"Error fetching upcoming games: {str(e)}")
        return pd.DataFrame()


def run_model_predictions(models, game_features):
    """Run predictions on upcoming games using all models

    Args:
        models: Dictionary of loaded model objects
        game_features: DataFrame of upcoming games with features

    Returns:
        DataFrame with predictions from all models
    """
    if game_features.empty:
        logger.error("No game features provided for prediction")
        return pd.DataFrame()

    try:
        # Create a dataframe to store all predictions
        predictions = pd.DataFrame()
        predictions['game_id'] = game_features['game_id']
        predictions['date'] = game_features['date']
        predictions['home_team'] = game_features['home_team']
        predictions['away_team'] = game_features['away_team']
        
        # Extract actual odds for comparison
        if 'home_moneyline' in game_features.columns:
            predictions['actual_home_odds'] = game_features['home_moneyline']
        if 'away_moneyline' in game_features.columns:
            predictions['actual_away_odds'] = game_features['away_moneyline']

        # Run each model and collect predictions
        for model_name, model in models.items():
            logger.info(f"Running predictions with {model_name}")
            
            try:
                # For classification models (moneyline)
                if model.model_type == 'classification' and hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(game_features)
                    # Add home win probability column
                    if probs.shape[1] >= 2:  # Binary classification
                        predictions[f"{model_name}_home_win_prob"] = probs[:, 1]
                    else:
                        predictions[f"{model_name}_prediction"] = model.predict(game_features)
                
                # For regression models (spread, totals)
                else:
                    predictions[f"{model_name}_prediction"] = model.predict(game_features)
                    
            except Exception as e:
                logger.error(f"Error getting predictions from {model_name}: {str(e)}")

        return predictions

    except Exception as e:
        logger.error(f"Error running model predictions: {str(e)}")
        return pd.DataFrame()


def visualize_predictions(predictions, output_path=None):
    """Create visualization of model predictions

    Args:
        predictions: DataFrame with all model predictions
        output_path: Path to save visualization files (optional)

    Returns:
        None, saves visualizations to files if path provided
    """
    if predictions.empty:
        logger.error("No predictions to visualize")
        return

    try:
        # Create figure directory if output path is provided
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract probability columns for comparison
        prob_columns = [col for col in predictions.columns if 'prob' in col]
        
        if prob_columns:
            # Create bar chart comparing model probabilities for each game
            plt.figure(figsize=(12, len(predictions) * 1.5))
            
            for i, (_, game) in enumerate(predictions.iterrows()):
                game_label = f"{game['away_team']} @ {game['home_team']} ({game['date']:%Y-%m-%d})"
                
                # Extract probability values for this game
                probs = [game[col] for col in prob_columns]
                
                # Create subplot for this game
                plt.subplot(len(predictions), 1, i+1)
                plt.barh(range(len(prob_columns)), probs, align='center')
                plt.yticks(range(len(prob_columns)), [col.replace('_home_win_prob', '') for col in prob_columns])
                plt.xlim(0, 1.0)
                plt.title(game_label)
                plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_dir / f"model_probabilities_{timestamp}.png")
                plt.close()
            else:
                plt.show()

        # Create box plot comparing model distributions
        plt.figure(figsize=(10, 6))
        
        data_to_plot = [predictions[col].values for col in prob_columns]
        plt.boxplot(data_to_plot, labels=[col.replace('_home_win_prob', '') for col in prob_columns])
        plt.title('Model Probability Distributions')
        plt.ylabel('Home Win Probability')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if output_path:
            plt.savefig(output_dir / f"model_distributions_{timestamp}.png")
            plt.close()
        else:
            plt.show()

    except Exception as e:
        logger.error(f"Error visualizing predictions: {str(e)}")


def main():
    """Main function to run test predictions with all models"""
    parser = argparse.ArgumentParser(description='Test NBA prediction models on upcoming games')
    parser.add_argument('--days', type=int, default=7, help='Number of days ahead to predict')
    parser.add_argument('--output', type=str, help='Output directory for visualizations and reports')
    parser.add_argument('--models', type=str, nargs='+', help='Specific models to test (default: all)')
    args = parser.parse_args()

    # Initialize the database connection
    if not init_db():
        logger.error("Failed to initialize database connection")
        return

    # Get upcoming games with features
    logger.info(f"Fetching upcoming games for the next {args.days} days")
    game_features = get_upcoming_games(days=args.days)
    
    if game_features.empty:
        logger.error("No upcoming games found or feature engineering failed")
        return

    logger.info(f"Found {len(game_features)} upcoming games to predict")

    # Load all trained models
    models_to_load = [
        "RandomForestModel", 
        "GradientBoostingModel", 
        "BayesianModel",
        "CombinedGradientBoostingModel", 
        "EnsembleModel", 
        "EnsembleStackingModel"
    ]
    
    # Filter models if specified
    if args.models:
        models_to_load = [m for m in models_to_load if m in args.models]

    # Load each model
    models = {}
    for model_name in models_to_load:
        model = load_model(model_name)
        if model:
            models[model_name] = model

    if not models:
        logger.error("No models were successfully loaded")
        return

    logger.info(f"Successfully loaded {len(models)} models")

    # Run predictions with all models
    predictions = run_model_predictions(models, game_features)
    
    if predictions.empty:
        logger.error("Failed to generate predictions")
        return

    # Save predictions to CSV
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions.to_csv(output_dir / f"predictions_{timestamp}.csv", index=False)
        logger.info(f"Saved predictions to {output_dir}/predictions_{timestamp}.csv")

    # Visualize predictions
    visualize_predictions(predictions, args.output)

    # Print prediction summary
    print("\nPrediction Summary:")
    print("====================")
    
    for _, game in predictions.iterrows():
        print(f"\n{game['away_team']} @ {game['home_team']} ({game['date']:%Y-%m-%d})")
        
        # Show model probabilities
        prob_columns = [col for col in predictions.columns if 'prob' in col]
        for col in prob_columns:
            model_name = col.replace('_home_win_prob', '')
            print(f"  {model_name:25s}: {game[col]:.2f} home win probability")
    
    print("\nPrediction testing complete!")


if __name__ == "__main__":
    main()
