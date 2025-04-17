import os
import sys
import logging
import argparse
import traceback
import json
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Import production readiness utilities
from ..utils.production_readiness import prepare_production_environment, system_health_check

# Add imports for our new systems
from ..models.model_registry import ModelRegistry
from ..utils.feature_evolution import FeatureEvolution
from ..utils.performance_tracker import PerformanceTracker

# Import API modules
from ..api.data_fetcher import (
    fetch_nba_games, fetch_team_stats, fetch_betting_odds, 
    fetch_historical_games, fetch_player_data, get_player_features
)

# Import model loader
from ..models.loader import load_ensemble_models

# Import prediction modules
from ..models.predictor import predict_todays_games
from ..models.player_props_loader import predict_player_props

# Import output modules
from ..output.display import display_predictions, display_player_predictions

# Import helper utilities
from ..utils.season_manager import get_season_manager
from ..utils.validation import check_api_keys
from ..utils.config import PredictionSettings

# Configure logging
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run NBA predictions with production-ready models')
    parser.add_argument('--date', type=str, help='Date for predictions (YYYY-MM-DD format)')
    parser.add_argument('--output-dir', type=str, default='predictions', help='Output directory for prediction results')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--include-players', action='store_true', help='Include player props predictions')
    parser.add_argument('--no-cache', action='store_true', help='Bypass all caching and fetch fresh data')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all cache before running')
    parser.add_argument('--clear-volatile-cache', action='store_true', help='Clear volatile cache before running')
    parser.add_argument("--history_days", type=int, default=30, help="Number of days of historical data to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--risk_level", type=str, choices=["conservative", "moderate", "aggressive"], help="Risk tolerance level")
    parser.add_argument("--bankroll", type=float, help="Starting bankroll amount")
    parser.add_argument("--track_clv", action="store_true", help="Track closing line value")
    parser.add_argument("--output_format", type=str, choices=["standard", "minimal", "detailed"], help="Output format")
    parser.add_argument("--auto_detect_season", action="store_true", default=True, help="Automatically detect the current NBA season")
    
    return parser.parse_args()


def load_production_models():
    """
    Load production models from the model registry with fallback to traditional loading
    
    Returns:
        Dict: Dictionary of loaded production models by type
    """
    try:
        logger.info("Loading production models from model registry")
        registry = ModelRegistry()
        
        # Get production models for each prediction type
        models = {
            "moneyline": registry.get_production_model("moneyline"),
            "spread": registry.get_production_model("spread"),
            "total": registry.get_production_model("total"),
            "player_props": registry.get_production_model("player_props")
        }
        
        # Check which models were successfully loaded
        loaded_models = [model_type for model_type, model in models.items() if model is not None]
        missing_models = [model_type for model_type, model in models.items() if model is None]
        
        if loaded_models:
            logger.info(f"Successfully loaded production models: {', '.join(loaded_models)}")
        
        if missing_models:
            logger.warning(f"Some production models not found in registry: {', '.join(missing_models)}")
            logger.info("Falling back to traditional model loading for missing models")
            
            # Try to load missing models using traditional method
            try:
                from ..models.loader import load_ensemble_models as load_ensemble_impl
                fallback_models = load_ensemble_impl()
                
                # Add any successfully loaded fallback models
                if fallback_models:
                    for model_type in missing_models:
                        if model_type in fallback_models:
                            models[model_type] = fallback_models[model_type]
                            logger.info(f"Loaded {model_type} model using fallback method")
            except Exception as e:
                logger.error(f"Error loading fallback models: {str(e)}")
        
        return models
        
    except Exception as e:
        logger.error(f"Error loading models from registry: {str(e)}")
        logger.error("Falling back to traditional model loading")
        
        # Fallback to traditional loading
        try:
            from ..models.loader import load_ensemble_models as load_ensemble_impl
            return load_ensemble_impl()
        except Exception as fallback_error:
            logger.error(f"Fallback loading also failed: {str(fallback_error)}")
            return {}


def get_optimized_features(prediction_type, data=None):
    """
    Get optimized features for the specified prediction type
    
    Args:
        prediction_type: Type of prediction (moneyline, spread, total, player_props)
        data: Optional DataFrame with training data
        
    Returns:
        List[str]: List of feature names to use
    """
    try:
        logger.info(f"Getting optimized features for {prediction_type} predictions")
        feature_evolution = FeatureEvolution()
        
        # If we have training data, we can try to select optimal features on the fly
        if data is not None and not data.empty:
            try:
                target_col = 'result'  # Assuming this is the standard target column name
                
                if target_col in data.columns:
                    is_classification = prediction_type in ["moneyline"]  # These are classification tasks
                    
                    # Select optimal features based on data
                    optimal_features = feature_evolution.select_optimal_features(
                        data=data.drop(columns=[target_col]),
                        target=data[target_col],
                        prediction_type=prediction_type,
                        is_classification=is_classification
                    )
                    
                    logger.info(f"Selected {len(optimal_features)} optimal features for {prediction_type}")
                    return optimal_features
                else:
                    logger.warning(f"Target column '{target_col}' not found in data")
            except Exception as e:
                logger.error(f"Error selecting optimal features: {str(e)}")
                logger.error("Falling back to production feature set")
        
        # If no data or error occurred, use the production feature set
        production_features = feature_evolution.get_production_feature_set(prediction_type)
        logger.info(f"Using {len(production_features)} production features for {prediction_type}")
        return production_features
        
    except Exception as e:
        logger.error(f"Error getting optimized features: {str(e)}")
        
        # Fall back to base features if everything else fails
        try:
            base_features = feature_evolution.base_features.get(prediction_type, [])
            logger.warning(f"Falling back to {len(base_features)} base features for {prediction_type}")
            return base_features
        except:
            logger.error("Could not retrieve even base features")
            return []  # Return empty list as last resort


def store_predictions_for_tracking(predictions, date_str=None, include_player_props=False):
    """
    Store predictions for later performance tracking
    
    Args:
        predictions: Dictionary of prediction results
        date_str: Date string in YYYY-MM-DD format, or None for today
        include_player_props: Whether player props are included
    """
    try:
        logger.info("Storing predictions for performance tracking")
        date_str = date_str or datetime.now().strftime("%Y-%m-%d")
        
        tracker = PerformanceTracker()
        
        # Store game predictions (moneyline, spread, total)
        if "games" in predictions:
            game_predictions = predictions["games"]
            
            # Store moneyline predictions
            if "moneyline" in game_predictions:
                tracker.store_predictions(
                    date=date_str,
                    model_name="moneyline_ensemble",
                    prediction_type="moneyline",
                    predictions=game_predictions["moneyline"]
                )
                logger.info(f"Stored {len(game_predictions['moneyline'])} moneyline predictions for tracking")
            
            # Store spread predictions
            if "spread" in game_predictions:
                tracker.store_predictions(
                    date=date_str,
                    model_name="spread_ensemble",
                    prediction_type="spread",
                    predictions=game_predictions["spread"]
                )
                logger.info(f"Stored {len(game_predictions['spread'])} spread predictions for tracking")
            
            # Store total predictions
            if "total" in game_predictions:
                tracker.store_predictions(
                    date=date_str,
                    model_name="total_ensemble",
                    prediction_type="total",
                    predictions=game_predictions["total"]
                )
                logger.info(f"Stored {len(game_predictions['total'])} total predictions for tracking")
        
        # Store player prop predictions
        if include_player_props and "player_props" in predictions:
            player_predictions = predictions["player_props"]
            
            tracker.store_predictions(
                date=date_str,
                model_name="player_props_ensemble",
                prediction_type="player_props",
                predictions=player_predictions
            )
            logger.info(f"Stored {len(player_predictions)} player prop predictions for tracking")
        
        logger.info("Successfully stored all predictions for performance tracking")
        return True
        
    except Exception as e:
        logger.error(f"Error storing predictions for tracking: {str(e)}")
        return False


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
        logger.warning("Empty predictions dataframe received. No standardization performed.")
        return predictions_df
    
    # Make a copy to avoid modifying the original
    df = predictions_df.copy()
    
    # Add tracking column for data quality issues
    df['data_quality_issues'] = ''
    
    # Define standard schema based on prediction type
    if prediction_type == "game":
        # Standard game prediction schema
        required_columns = [
            "game_id", "date", "home_team", "visitor_team", 
            "win_probability", "predicted_spread", "projected_total",
            "confidence_level", "confidence_score"
        ]
        
        # Check for missing required columns and add data quality flags
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns in prediction output: {', '.join(missing_columns)}")
            for col in missing_columns:
                df[col] = None  # Add column but with explicit None values
                # Update data quality issues tracking
                df['data_quality_issues'] += f"missing_{col},"
        
    elif prediction_type == "player":
        # Standard player prediction schema
        required_columns = [
            "player_id", "player_name", "team_id", "team_name", "position", "game_id",
            "points", "rebounds", "assists", "confidence_level", "confidence_score"
        ]
        
        # Check for missing required columns and add data quality flags
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns in player prediction output: {', '.join(missing_columns)}")
            for col in missing_columns:
                df[col] = None  # Add column but with explicit None values
                # Update data quality issues tracking
                df['data_quality_issues'] += f"missing_{col},"
    
    # Map column names to standardized names if needed
    column_mapping = {
        "home_win_probability": "win_probability",
        "predicted_total": "projected_total",
        "away_team": "visitor_team"
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Clean up data quality issues tracking
    df['data_quality_issues'] = df['data_quality_issues'].str.rstrip(',')  
    
    return df


def main():
    """
    Main entry point for NBA predictions
    """
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Prepare production environment
        # This sets up logging, performs system checks, and ensures the environment is ready
        logger.info("Preparing production environment...")
        env_ready = prepare_production_environment()
        if not env_ready:
            logger.error("Production environment preparation failed. Cannot proceed.")
            return 1
            
        # Configure logging based on arguments
        log_level = getattr(logging, args.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        
        # Initialize the season manager for automatic season detection
        season_manager = get_season_manager()
        current_season = season_manager.get_current_season_info()
        
        # Log season information
        logger.info(f"Current NBA season: {season_manager.get_current_season_display()} "  
                   f"({current_season['phase'].value})")
        
        # Handle cache management flags
        if args.clear_cache:
            from ..utils.cache_manager import clear_all_cache
            cleared_count = clear_all_cache()
            logger.info(f"Cleared {cleared_count} cache entries before running predictions")
        elif args.clear_volatile_cache:
            from ..utils.cache_manager import clear_cache, CacheTier
            cleared_count = clear_cache(tier=CacheTier.VOLATILE)
            logger.info(f"Cleared {cleared_count} volatile cache entries before running predictions")
            
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
        force_refresh = args.no_cache  # Use no-cache flag to force fresh data
        games = fetch_nba_games(prediction_date, force_refresh=force_refresh)
        
        if not games:
            logger.warning(f"No NBA games found for {prediction_date}")
            return 0
        
        logger.info(f"Found {len(games)} NBA games for {prediction_date}")
        
        # Load prediction models from model registry
        logger.info("Loading prediction models from model registry")
        game_models = load_production_models()  # Use our new function
        
        # Load player props models if requested
        player_models = None
        if args.include_players:
            logger.info("Loading player props models")
            if "player_props" in game_models:
                player_models = {"player_props": game_models["player_props"]}
                logger.info("Loaded player props model from registry")
            else:
                # Fallback to traditional loading
                from ..models.player_props_loader import load_player_props_models
                player_models = load_player_props_models()
                logger.info("Loaded player props model using fallback method")
        
        # Fetch team statistics with caching considerations
        logger.info("Fetching team statistics")
        team_stats = {}
        for game in games:
            # Always force fresh data for teams playing today when making predictions
            # This implements the hybrid approach - cached data for less critical contexts,
            # but always fresh data for actual game-day predictions
            home_team_id = game['home_team']['id']
            away_team_id = game['away_team']['id']
            
            # Fetch home team stats
            home_team_stats = fetch_team_stats(home_team_id, force_refresh=force_refresh)
            if home_team_stats:
                team_stats[home_team_id] = home_team_stats
            else:
                logger.warning(f"Could not fetch stats for home team {home_team_id}")
            
            # Fetch away team stats
            away_team_stats = fetch_team_stats(away_team_id, force_refresh=force_refresh)
            if away_team_stats:
                team_stats[away_team_id] = away_team_stats
            else:
                logger.warning(f"Could not fetch stats for away team {away_team_id}")
        
        # Get historical data for feature engineering
        historical_data = fetch_historical_games(args.history_days, force_refresh=force_refresh)
        
        # Build prediction dictionary
        predictions = {"games": {}, "metadata": {}}
        predictions["metadata"]["prediction_date"] = prediction_date
        predictions["metadata"]["generated_at"] = datetime.now().isoformat()
        predictions["metadata"]["season"] = season_manager.get_current_season_display()
        predictions["metadata"]["phase"] = current_season['phase'].value
        
        # Make game predictions
        if game_models:
            logger.info("Making game predictions")
            
            # Get optimized features for each prediction type
            moneyline_features = get_optimized_features("moneyline", historical_data) if "moneyline" in game_models else []
            spread_features = get_optimized_features("spread", historical_data) if "spread" in game_models else []
            total_features = get_optimized_features("total", historical_data) if "total" in game_models else []
            
            # Make predictions for each game
            moneyline_preds = []
            spread_preds = []
            total_preds = []
            
            for game in games:
                # Generate features for this game
                game_features = build_game_features(game, team_stats, historical_data, moneyline_features, spread_features, total_features)
                
                # Generate predictions if we have the models
                if "moneyline" in game_models and game_models["moneyline"] and moneyline_features:
                    ml_pred = predict_moneyline(game, game_features, game_models["moneyline"])
                    if ml_pred:
                        moneyline_preds.append(ml_pred)
                
                if "spread" in game_models and game_models["spread"] and spread_features:
                    spread_pred = predict_spread(game, game_features, game_models["spread"])
                    if spread_pred:
                        spread_preds.append(spread_pred)
                
                if "total" in game_models and game_models["total"] and total_features:
                    total_pred = predict_total(game, game_features, game_models["total"])
                    if total_pred:
                        total_preds.append(total_pred)
            
            # Add predictions to result dictionary
            if moneyline_preds:
                predictions["games"]["moneyline"] = moneyline_preds
                logger.info(f"Generated {len(moneyline_preds)} moneyline predictions")
            
            if spread_preds:
                predictions["games"]["spread"] = spread_preds
                logger.info(f"Generated {len(spread_preds)} spread predictions")
            
            if total_preds:
                predictions["games"]["total"] = total_preds
                logger.info(f"Generated {len(total_preds)} total predictions")
            
            # Display and save predictions
            if any([moneyline_preds, spread_preds, total_preds]):
                # Display predictions
                logger.info("Displaying game predictions")
                display_predictions(predictions)
                
                # Save predictions
                save_path = os.path.join(output_dir, f"nba_predictions_{prediction_date}.json")
                with open(save_path, 'w') as f:
                    json.dump(predictions, f, indent=2)
                logger.info(f"Saved game predictions to {save_path}")
                
                # Store for performance tracking
                store_predictions_for_tracking(predictions, prediction_date)
            else:
                logger.warning("No valid game predictions generated")
        else:
            logger.warning("No game models loaded, skipping game predictions")
        
        # Make player props predictions if requested
        if args.include_players and player_models:
            logger.info("Making player props predictions")
            
            # Fetch player data - always use fresh data for player props on game day (volatile)
            logger.info("Fetching player data")
            player_data = fetch_player_data(games, force_refresh=force_refresh)
            
            if player_data.empty:
                logger.warning("No player data available, skipping player props predictions")
            else:
                # Get optimized player prop features
                player_prop_features = get_optimized_features("player_props", player_data)
                
                # Build features for player props predictions
                player_features = get_player_features(games, team_stats, 
                                                  historical_data, 
                                                  force_refresh=force_refresh,
                                                  feature_list=player_prop_features)
                
                if player_features.empty:
                    logger.warning("Failed to build player features, skipping player props predictions")
                else:
                    # Make predictions
                    player_predictions = predict_player_props(player_features, player_models, feature_list=player_prop_features)
                    
                    if player_predictions:
                        # Add to predictions dictionary
                        predictions["player_props"] = player_predictions
                        
                        # Display player predictions
                        logger.info("Displaying player props predictions")
                        display_player_predictions(player_predictions)
                        
                        # Save player predictions
                        save_path = os.path.join(output_dir, f"nba_player_props_{prediction_date}.json")
                        with open(save_path, 'w') as f:
                            json.dump(player_predictions, f, indent=2)
                        logger.info(f"Saved player props predictions to {save_path}")
                        
                        # Update tracking to include player props
                        store_predictions_for_tracking(predictions, prediction_date, include_player_props=True)
                    else:
                        logger.warning("No valid player props predictions generated")
        
        # Check for performance issues
        # Initialize performance tracker for monitoring
        performance_tracker = PerformanceTracker()
        
        # Check for declining performance in prediction models
        model_types = ["moneyline", "spread", "total", "player_props"]
        for model_type in model_types:
            if performance_tracker.detect_performance_decline(model_type):
                logger.warning(f"⚠️ Detected declining performance in {model_type} predictions!")
                logger.warning(f"Consider retraining the {model_type} model with the latest data.")
        
        logger.info("Prediction process completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in prediction process: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
