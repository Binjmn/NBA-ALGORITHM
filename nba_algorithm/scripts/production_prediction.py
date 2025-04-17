import os
import sys
import logging
import argparse
import traceback
from datetime import datetime

import pandas as pd
import numpy as np

# Import the new modular components
from nba_algorithm.output import (
    display_prediction_output, 
    create_prediction_schema,
    save_predictions
)
from nba_algorithm.betting import BettingAnalyzer
from nba_algorithm.utils import PredictionSettings
from nba_algorithm.utils.season_manager import get_season_manager
from nba_algorithm.data import (
    fetch_nba_games,
    fetch_historical_games,
    validate_api_data,
    check_api_keys
)
from nba_algorithm.data.team_data import fetch_team_stats
from nba_algorithm.data.odds_data import fetch_betting_odds
from nba_algorithm.features import prepare_game_features
from nba_algorithm.models import (
    calculate_confidence_level,
    get_player_predictions
)
from nba_algorithm.models.loader import load_models
from nba_algorithm.models.predictor import predict_game_outcomes, ensemble_prediction

# Import own utility functions
from nba_algorithm.utils.logger import setup_logging

# Set up logging
logger = setup_logging("production_prediction")


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="NBA Prediction System")
    parser.add_argument("--date", type=str, help="Date for prediction (YYYY-MM-DD)")
    parser.add_argument("--output_dir", type=str, help="Output directory for prediction files")
    parser.add_argument("--history_days", type=int, default=30, help="Number of days of historical data to use")
    parser.add_argument("--include_players", action="store_true", help="Include player-level predictions")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--risk_level", type=str, choices=["conservative", "moderate", "aggressive"], help="Risk tolerance level")
    parser.add_argument("--bankroll", type=float, help="Starting bankroll amount")
    parser.add_argument("--track_clv", action="store_true", help="Track closing line value")
    parser.add_argument("--output_format", type=str, choices=["standard", "minimal", "detailed"], help="Output format")
    parser.add_argument("--auto_detect_season", action="store_true", default=True, help="Automatically detect the current NBA season")
    return parser.parse_args()


def load_ensemble_models():
    """
    Load ensemble prediction models with detailed validation and error reporting
    
    Returns:
        Dict: Dictionary of loaded ensemble models by name
        
    If models can't be loaded, this function will log detailed error messages
    explaining exactly why the models couldn't be loaded, instead of silently failing.
    """
    try:
        logger.info("Loading ensemble prediction models")
        
        # Import the actual function from the module
        try:
            from nba_algorithm.models.loader import load_ensemble_models as load_ensemble_impl
            return load_ensemble_impl()
        except ImportError as e:
            logger.error(f"Failed to import ensemble model loader: {str(e)}")
            logger.error("This suggests the ensemble module may not be installed or properly configured")
            return None
        except AttributeError as e:
            logger.error(f"Failed to access ensemble model loader: {str(e)}")
            logger.error("This suggests the ensemble module exists but doesn't have the expected function")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error loading ensemble models: {str(e)}")
        logger.debug(traceback.format_exc())
        return None


def standardize_prediction_output(predictions_df, prediction_type="game"):
    """
    Standardize prediction output format to ensure consistency
    
    Args:
        predictions_df: DataFrame of predictions
        prediction_type: Type of prediction ('game' or 'player')
        
    Returns:
        pandas.DataFrame: Standardized prediction DataFrame
    """
    if predictions_df.empty:
        return predictions_df
    
    # Make a copy to avoid modifying the original
    df = predictions_df.copy()
    
    # Define standard schema based on prediction type
    if prediction_type == "game":
        # Standard game prediction schema
        required_columns = [
            "game_id", "date", "home_team", "visitor_team", 
            "win_probability", "predicted_spread", "projected_total",
            "confidence_level", "confidence_score"
        ]
        
        # Add any missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                if col in ["win_probability", "confidence_score"]:
                    df[col] = 0.5  # Default to 50% for probabilities
                elif col in ["predicted_spread", "projected_total"]:
                    df[col] = 0.0  # Default to 0 for numeric predictions
                elif col == "confidence_level":
                    df[col] = "Low"  # Default to low confidence
                else:
                    df[col] = ""  # Default to empty string for text fields
        
        # Rename columns for consistency if needed
        column_mapping = {
            "home_win_probability": "win_probability",
            "spread": "predicted_spread",
            "total": "projected_total",
            "home_team_name": "home_team",
            "visitor_team_name": "visitor_team"
        }
        
        # Apply mappings for columns that exist
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
    
    elif prediction_type == "player":
        # Standard player prediction schema
        required_columns = [
            "game_id", "player_id", "player_name", "team_id", "team_name",
            "predicted_points", "predicted_rebounds", "predicted_assists",
            "confidence_score", "confidence_level", "home_team", "visitor_team"
        ]
        
        # Add any missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                if col in ["predicted_points", "predicted_rebounds", "predicted_assists", "confidence_score"]:
                    df[col] = 0.0  # Default to 0 for numeric predictions
                else:
                    df[col] = ""  # Default to empty string for text fields
    
    return df


def main():
    """
    Main entry point for NBA predictions
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Initialize the season manager for automatic season detection
        season_manager = get_season_manager()
        current_season = season_manager.get_current_season_info()
        
        # Log season information
        logger.info(f"Current NBA season: {season_manager.get_current_season_display()} "  
                   f"({current_season['phase'].value})")
        
        # Create settings from arguments
        settings = PredictionSettings()
        settings.update_from_args(args)
        
        # Update settings with season information
        settings.season_year = current_season['season_year']
        settings.season_phase = current_season['phase'].value
        
        # Set up prediction date
        prediction_date = args.date or datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Making predictions for {prediction_date}")
        
        # Set up output directory
        output_dir = args.output_dir or "predictions"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output will be saved to {output_dir}")
        
        # Verify API keys
        missing_keys = check_api_keys()
        if missing_keys:
            logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
            return 1
            
        # Check if we're in season
        if not season_manager.is_in_season():
            logger.warning(f"Currently in the off-season ({current_season['phase'].value}). "  
                          f"Predictions may be less accurate due to roster changes and lack of recent games.")
        
        # Fetch game data
        logger.info("Fetching game data")
        games = fetch_nba_games(prediction_date)
        if not games:
            logger.error("Failed to fetch game data")
            return 1
        logger.info(f"Fetched {len(games)} games")
        
        # Fetch team stats
        logger.info("Fetching team stats")
        team_stats = fetch_team_stats()
        if not team_stats:
            logger.error("Failed to fetch team stats")
            return 1
        logger.info("Fetched team stats")
        
        # Fetch odds data
        logger.info("Fetching odds data")
        odds = fetch_betting_odds(games)
        logger.info(f"Fetched odds data for {len(odds) if odds else 0} games")
        
        # Fetch historical game data
        logger.info("Fetching historical game data")
        historical_games = fetch_historical_games(args.history_days)
        if not historical_games:
            logger.warning("No historical game data available")
        else:
            logger.info(f"Fetched {len(historical_games)} historical games")
        
        # Validate data
        logger.info("Validating API data")
        validate_api_data(games, team_stats, odds, historical_games)
        
        # Prepare features for prediction
        logger.info("Preparing game features")
        features_df = prepare_game_features(games, team_stats, odds, historical_games)
        if features_df.empty:
            logger.error("Failed to prepare game features")
            return 1
        logger.info(f"Prepared features for {len(features_df)} games")
        
        # Load prediction models for the current season
        logger.info(f"Loading prediction models for season {season_manager.get_current_season_display()}")
        models = load_models(season_year=current_season['season_year'])
        if not models:
            logger.error("Failed to load prediction models")
            return 1
        logger.info(f"Loaded {len(models)} prediction models")
        
        # Load ensemble models if available
        ensemble_models = load_ensemble_models()
        if ensemble_models:
            logger.info(f"Loaded {len(ensemble_models)} ensemble models")
            
        # Make predictions using ensemble approach for higher accuracy
        logger.info("Making predictions")
        if ensemble_models:
            # Use ensemble prediction for better accuracy
            predictions_df = ensemble_prediction(features_df, models, ensemble_models)
            logger.info("Using ensemble prediction for maximum accuracy")
        else:
            # Fall back to standard prediction if no ensemble models
            predictions_df = predict_game_outcomes(features_df, models)
            logger.info("Using standard model prediction")
            
        if predictions_df.empty:
            logger.error("Failed to make predictions")
            return 1
        logger.info(f"Made predictions for {len(predictions_df)} games")
        
        # Add season information to predictions
        predictions_df['season_year'] = current_season['season_year']
        predictions_df['season_phase'] = current_season['phase'].value
        predictions_df['season_display'] = season_manager.get_current_season_display()
        
        # Standardize prediction output
        predictions_df = standardize_prediction_output(predictions_df)
        
        # Calculate confidence levels for predictions
        predictions_df = calculate_confidence_level(predictions_df)
        
        # Get player-level predictions if requested
        player_predictions_df = pd.DataFrame()
        if settings.include_players:
            logger.info("Generating player-level predictions")
            player_predictions_df = get_player_predictions(games, team_stats, historical_games)
            
            if not player_predictions_df.empty:
                logger.info(f"Generated predictions for {len(player_predictions_df)} players")
                
                # Add season information to player predictions
                player_predictions_df['season_year'] = current_season['season_year']
                player_predictions_df['season_phase'] = current_season['phase'].value
                
                # Standardize player prediction output
                player_predictions_df = standardize_prediction_output(player_predictions_df, prediction_type="player")
        
        # Create BettingAnalyzer instance
        betting_analyzer = BettingAnalyzer(settings)
        
        # Analyze game predictions for betting opportunities
        analyzed_predictions_df = betting_analyzer.analyze_game_predictions(predictions_df, odds)
        
        # Analyze player predictions for prop betting opportunities
        analyzed_player_predictions_df = pd.DataFrame()
        if not player_predictions_df.empty:
            analyzed_player_predictions_df = betting_analyzer.analyze_player_predictions(player_predictions_df)
        
        # Save predictions
        csv_path, json_path = save_predictions(analyzed_predictions_df, output_dir, prediction_date)
        
        # Save player predictions if available
        if not analyzed_player_predictions_df.empty:
            player_csv_path, player_json_path = save_predictions(
                analyzed_player_predictions_df, 
                output_dir, 
                prediction_date,
                prefix="nba_player_predictions"
            )
        
        # Create standardized prediction schema
        prediction_data = create_prediction_schema(
            analyzed_predictions_df, 
            analyzed_player_predictions_df,
            settings,
            betting_analyzer
        )
        
        # Include season information in the schema
        prediction_data["season_info"] = {
            "season_year": current_season['season_year'],
            "season_display": season_manager.get_current_season_display(),
            "phase": current_season['phase'].value,
            "in_season": season_manager.is_in_season()
        }
        
        # Display predictions based on output format
        if settings.output_format == "minimal":
            # Just display summary
            logger.info(f"Made predictions for {len(analyzed_predictions_df)} games")
            logger.info(f"Generated {len(analyzed_player_predictions_df)} player prop recommendations")
            logger.info(f"Results saved to {csv_path} and {json_path}")
        else:
            # Display comprehensive output
            display_prediction_output(prediction_data)
            
        logger.info("Prediction process completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Prediction process interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unhandled exception in prediction process: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
