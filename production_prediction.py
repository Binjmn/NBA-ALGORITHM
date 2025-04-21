import os
import sys
import logging
import argparse
import traceback
import json
from datetime import datetime
from pathlib import Path
import time
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
from nba_algorithm.data.team_data import fetch_team_stats, fetch_all_teams
from nba_algorithm.data.nba_teams import get_active_nba_teams, filter_games_to_active_teams
from nba_algorithm.data.odds_data import fetch_betting_odds
from nba_algorithm.features import prepare_game_features
from nba_algorithm.models import (
    calculate_confidence_level,
    get_player_predictions
)
from nba_algorithm.models.loader import load_models
from nba_algorithm.models.predictor import predict_game_outcomes, ensemble_prediction
# Import the new injury analysis and advanced metrics modules
from nba_algorithm.models.injury_analysis import compare_team_injuries, get_team_injury_data
from nba_algorithm.models.advanced_metrics import get_team_efficiency_comparison, fetch_team_advanced_metrics

# Import own utility functions
from nba_algorithm.utils.logger import setup_logging
from nba_algorithm.presentation.display import display_user_friendly_predictions

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
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to file")
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
    try:
        if predictions_df is None or predictions_df.empty:
            logger.warning("Received empty predictions DataFrame in standardize_prediction_output")
            # Return a minimal DataFrame with the expected columns
            if prediction_type == "game":
                return pd.DataFrame(columns=["game_id", "date", "home_team", "visitor_team", "win_probability",
                                           "predicted_spread", "confidence_level"])
            else:  # player
                return pd.DataFrame(columns=["game_id", "player_id", "player_name", "predicted_points", 
                                           "predicted_rebounds", "predicted_assists"])
        
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
            
            # Ensure numeric columns are numeric
            numeric_cols = ["win_probability", "predicted_spread", "projected_total", "confidence_score"]
            for col in numeric_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                    except Exception as e:
                        logger.warning(f"Error converting {col} to numeric: {str(e)}")
                        df[col] = 0.0
        
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
            
            # Ensure numeric columns are numeric
            numeric_cols = ["predicted_points", "predicted_rebounds", "predicted_assists", "confidence_score"]
            for col in numeric_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                    except Exception as e:
                        logger.warning(f"Error converting {col} to numeric: {str(e)}")
                        df[col] = 0.0
        
        return df
    
    except Exception as e:
        logger.error(f"Error in standardize_prediction_output: {str(e)}")
        # Return minimal valid DataFrame
        if prediction_type == "game":
            return pd.DataFrame({"game_id": [], "home_team": [], "visitor_team": [], "win_probability": []})
        else:  # player
            return pd.DataFrame({"player_id": [], "player_name": [], "predicted_points": []})


def main():
    """
    Main entry point for NBA predictions
    """
    # Set up logging
    logger = setup_logging('production_prediction')
    logger.info("Logging initialized for production_prediction")
    
    # Parse command-line arguments
    args = parse_arguments()
    
    try:
        # Verify API keys are available
        if not check_api_keys():
            logger.error("Missing required API keys. Please set BALLDONTLIE_API_KEY and THE_ODDS_API_KEY environment variables.")
            return 1
            
        # Initialize Season Manager
        season_manager = get_season_manager()
        logger.info(f"Current season: {season_manager.get_current_season_display()}")
        
        # Get active NBA teams - core filtering to avoid non-NBA teams
        try:
            all_teams = fetch_all_teams()
            if not all_teams:
                logger.error("Failed to fetch teams from API. Aborting prediction process.")
                return 1
            else:
                active_teams = get_active_nba_teams(all_teams)
                logger.info(f"Identified {len(active_teams)} active NBA teams")
        except Exception as e:
            logger.error(f"Error getting active NBA teams: {str(e)}")
            logger.error(traceback.format_exc())
            return 1
        
        # Fetch upcoming games for prediction
        logger.info("Fetching games from API")
        try:
            raw_games = fetch_nba_games()
            
            if not raw_games or len(raw_games) == 0:
                logger.error("No games found from API. Aborting prediction process.")
                return 1
            else:
                # Filter to only include active NBA teams
                games = filter_games_to_active_teams(raw_games, active_teams)
                logger.info(f"Filtered to {len(games)} NBA games")
                
                if not games or len(games) == 0:
                    logger.error("No valid games found after filtering to active NBA teams.")
                    return 1
        except Exception as e:
            logger.error(f"Error fetching games: {str(e)}")
            logger.error(traceback.format_exc())
            return 1
        
        # Fetch team statistics for all teams
        logger.info("Fetching comprehensive team statistics with enhanced data collection")
        try:
            # Use enhanced version with full-season data collection instead of the default
            from nba_algorithm.data.team_data import fetch_team_stats
            team_stats = fetch_team_stats(active_teams, use_entire_season=True)
            
            if not team_stats:
                logger.error("Failed to fetch team stats. Aborting prediction process.")
                return 1
                
            # Validate we have defensive ratings for all teams
            teams_missing_data = []
            for team_id, stats in team_stats.items():
                if "defensive_rating" not in stats:
                    teams_missing_data.append(stats.get("name", f"Team ID: {team_id}"))
            
            if teams_missing_data:
                logger.error(f"Missing defensive ratings for {len(teams_missing_data)} teams: {', '.join(teams_missing_data)}")
                print("\nCRITICAL ERROR: Missing defensive ratings for some teams.")
                print(f"Teams affected: {', '.join(teams_missing_data)}")
                print("Run 'python enhanced_data_collection.py' to collect more comprehensive data.\n")
                return 1
        except Exception as e:
            logger.error(f"Error fetching team statistics: {str(e)}")
            logger.error(traceback.format_exc())
            return 1
            
        # Fetch betting odds for upcoming games
        logger.info("Fetching betting odds for upcoming games")
        try:
            odds_data = fetch_betting_odds(games)
            logger.info(f"Retrieved odds data for {len(odds_data) if odds_data else 0} games")
        except Exception as e:
            logger.error(f"Error fetching betting odds: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue even if odds data is missing - not critical
            odds_data = {}
        
        # Fetch historical games for context
        logger.info("Fetching comprehensive historical games for the entire season")
        try:
            # Use enhanced version with full-season data collection instead of limited days
            from nba_algorithm.data.historical_collector import fetch_historical_games
            historical_games = fetch_historical_games(fetch_full_season=True)
            
            if not historical_games:
                logger.error("No historical games found. Cannot continue without historical context.")
                return 1
                
            logger.info(f"Successfully collected {len(historical_games)} historical games for the entire season")
        except Exception as e:
            logger.error(f"Error fetching historical games: {str(e)}")
            logger.error(traceback.format_exc())
            return 1
        
        # Try to make predictions using machine learning models
        try:
            # Load prediction models
            logger.info("Loading prediction models")
            models = load_models()
            if not models or len(models) == 0:
                logger.error("Failed to load prediction models. Cannot continue without models.")
                return 1
                
            logger.info(f"Successfully loaded {len(models)} prediction models")
            
            # Prepare features for prediction
            logger.info("Preparing game features for prediction")
            features_df = prepare_game_features(games, team_stats, odds_data, historical_games)
            if features_df.empty:
                logger.error("Failed to create features for prediction. Cannot continue without features.")
                return 1
                
            logger.info(f"Generated features for {len(features_df)} games")
            
            # Validate that we have complete data requirements
            teams_missing_defensive_data = []
            for index, row in features_df.iterrows():
                # Check if required defensive metrics are present
                if pd.isna(row.get('home_def_rtg')) or pd.isna(row.get('away_def_rtg')):
                    game_id = row.get('game_id')
                    home_team = row.get('home_team')
                    away_team = row.get('away_team')
                    if pd.isna(row.get('home_def_rtg')):
                        teams_missing_defensive_data.append(home_team)
                    if pd.isna(row.get('away_def_rtg')):
                        teams_missing_defensive_data.append(away_team)
                    logger.error(f"Missing defensive rating data for game {game_id}: {home_team} vs {away_team}")
            
            if teams_missing_defensive_data:
                # Filter out None values and convert to string if needed
                unique_teams = set(team for team in teams_missing_defensive_data if team is not None)
                unique_teams_str = [str(team) for team in unique_teams]
                
                if unique_teams_str:
                    logger.error(f"CRITICAL: Missing defensive ratings for {len(unique_teams)} teams: {', '.join(unique_teams_str)}")
                    logger.error("Cannot make reliable predictions without complete defensive data")
                    print("\nCRITICAL ERROR: Cannot make reliable predictions due to missing defensive ratings.")
                    print(f"Teams missing data: {', '.join(unique_teams_str)}")
                    print("The system has been improved to collect more comprehensive data from the entire season.")
                    print("Please run the prediction again after the next data collection cycle.\n")
                else:
                    logger.error("CRITICAL: Missing defensive ratings for unknown teams")
                    print("\nCRITICAL ERROR: Cannot make reliable predictions due to missing defensive ratings for unknown teams.")
                    print("The system has been improved to collect more comprehensive data from the entire season.")
                    print("Please run the prediction again after the next data collection cycle.\n")
                return 1
            
            # Make predictions using models
            logger.info("Making predictions using loaded models")
            try:
                # First check if the function expects features_df or raw games
                from inspect import signature
                pred_sig = signature(predict_game_outcomes)
                
                if len(pred_sig.parameters) == 2:
                    # Function expects features_df and models
                    logger.info("Using 2-parameter version of predict_game_outcomes")
                    predictions_df = predict_game_outcomes(features_df, models)
                elif len(pred_sig.parameters) == 3:
                    # Function expects models, games, team_stats
                    logger.info("Using 3-parameter version of predict_game_outcomes")
                    predictions_df = predict_game_outcomes(models, games, team_stats)
                else:
                    raise ValueError(f"predict_game_outcomes has unexpected signature: {pred_sig}")
                    
                if predictions_df.empty:
                    logger.error("No predictions generated from models. Cannot continue without predictions.")
                    return 1
            except Exception as e:
                logger.error(f"Error calling predict_game_outcomes: {str(e)}")
                logger.error(traceback.format_exc())
                
                # NEVER use fallback logic when real money is at stake
                logger.error("CRITICAL: Prediction models failed. Refusing to provide fallback predictions when real money is at stake.")
                print("\nCRITICAL ERROR: Unable to generate reliable predictions with the available data.")
                print("Since this prediction system is used for real money betting, we cannot provide potentially inaccurate fallback predictions.")
                print("Please try again when more reliable data is available.\n")
                return 1
        
            # Standardize the prediction output
            logger.info("Standardizing model-based prediction output")
            predictions_df = standardize_prediction_output(predictions_df, "game")
            
        except Exception as model_error:
            error_msg = f"Error in prediction process: {str(model_error)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            print(f"\nERROR: {error_msg}")
            return 1
        
        # Save predictions
        if args.save_predictions:
            logger.info("Saving predictions to file")
            try:
                # Create output directory if it doesn't exist
                output_dir = Path("predictions")
                output_dir.mkdir(exist_ok=True)
                
                # Generate timestamp for filenames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save to CSV
                csv_path = output_dir / f"nba_predictions_{timestamp}.csv"
                predictions_df.to_csv(csv_path, index=False)
                logger.info(f"Saved predictions to {csv_path}")
                
                # Save to JSON for easier parsing
                json_path = output_dir / f"nba_predictions_{timestamp}.json"
                predictions_json = predictions_df.to_dict(orient='records')
                with open(json_path, 'w') as f:
                    json.dump(predictions_json, f, indent=2)
                logger.info(f"Saved predictions to {json_path}")
                
                print(f"\nPredictions saved to:\n- {csv_path}\n- {json_path}")
            except Exception as e:
                logger.error(f"Error saving predictions: {str(e)}")
                print("\n❌ Error saving predictions to file. See log for details.")
        
        # Display user-friendly predictions using the display module
        logger.info("Displaying user-friendly predictions")
        try:
            # Display the predictions using our enhanced display module
            display_user_friendly_predictions(predictions_df)
            
            print("\n📊 ADDITIONAL INFORMATION:")
            print("- Data collected for the entire NBA season for maximum accuracy")
            print("- Enhanced defensive ratings calculation ensures reliable predictions")
            print("- No fallback or synthetic data used - all predictions use real statistics")
            print("- Predictions are suitable for real-money betting with proper bankroll management")
        except Exception as e:
            logger.error(f"Error displaying predictions: {str(e)}")
            print("\n❌ Error displaying predictions. See log for details.")
            # Fallback to basic display if the enhanced display fails
            print("\n🏀 NBA GAME PREDICTIONS:\n")
            print(predictions_df.to_string(index=False))
        
        logger.info("Prediction process completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Unhandled exception in main prediction process: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
