#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-time NBA Game Prediction Script

This script loads trained models and generates predictions for today's NBA games.
It uses the real API data to fetch game information and team statistics.

Usage:
    python -m scripts.run_predictions
"""

import os
import sys
import json
import pickle
import logging
import traceback
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging at the module level to redirect to file instead of console
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"predictions_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Get the root logger and set it to use the file handler
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
        root_logger.removeHandler(handler)
        
# Define logger for this module
logger = logging.getLogger(__name__)

# Add the src directory to the path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import project modules
from src.api.theodds_client import TheOddsClient
from src.api.balldontlie_client import BallDontLieClient
from src.features.advanced_features import FeatureEngineer
from src.utils.robustness import validate_training_data


def load_model_from_file(model_path):
    """
    Load a model from a pickle file
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model or None if loading failed
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None


def load_models():
    """
    Load all available trained models from the models directory
    
    Returns:
        Dictionary of loaded models
    """
    models_dir = Path("models")
    models = {}
    
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return models
    
    # Get the most recently trained model of each type
    model_files = list(models_dir.glob("*.pkl"))
    
    # Group by model type
    model_types = {}
    for f in model_files:
        name_parts = f.stem.split('_')
        model_type = name_parts[0]
        timestamp = '_'.join(name_parts[1:-1]) if len(name_parts) > 2 else name_parts[1]
        
        if model_type not in model_types or timestamp > model_types[model_type][1]:
            model_types[model_type] = (f, timestamp)
    
    # Load the most recent model of each type
    for model_type, (model_file, _) in model_types.items():
        model = load_model_from_file(model_file)
        if model is not None:
            models[model_type] = model
    
    return models


def fetch_nba_games():
    """
    Fetch today's NBA games from the API
    
    Returns:
        List of game dictionaries
    """
    try:
        # Try to use the Odds API first
        odds_client = TheOddsClient()
        games = odds_client.get_nba_odds()
        
        if games:
            logger.info(f"Fetched {len(games)} games from The Odds API")
            return games
        else:
            logger.warning("No games found from The Odds API, trying BallDontLie")
    except Exception as e:
        logger.warning(f"Error fetching from The Odds API: {str(e)}")
    
    try:
        # Fall back to BallDontLie API
        balldontlie = BallDontLieClient()
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        response = balldontlie.get_games(start_date=today, end_date=tomorrow)
        games = []
        
        # Extract the actual games data from response
        if isinstance(response, dict) and 'data' in response:
            # API returned the expected format with 'data' key
            games = response['data']
            logger.info(f"Extracted {len(games)} games from BallDontLie API response 'data' field")
        elif isinstance(response, list):
            # API directly returned a list of games
            games = response
            logger.info(f"BallDontLie API directly returned {len(games)} games as a list")
        else:
            logger.warning(f"Unexpected response format from BallDontLie API: {type(response)}")
            logger.debug(f"Response sample: {str(response)[:200]}...")
        
        if games:
            logger.info(f"Fetched {len(games)} games from BallDontLie API")
            # Debug log to verify our game objects
            for i, game in enumerate(games[:2]):
                if isinstance(game, dict):
                    home = game.get('home_team', {}).get('name', 'Unknown') if isinstance(game.get('home_team'), dict) else 'Unknown'
                    away = game.get('visitor_team', {}).get('name', 'Unknown') if isinstance(game.get('visitor_team'), dict) else 'Unknown'
                    logger.info(f"Game {i+1}: {home} vs {away}")
                else:
                    logger.warning(f"Game {i+1} is not a dictionary: {type(game)}")
            return games
        else:
            logger.warning("No games found from either API")
            return []
    except Exception as e:
        logger.error(f"Error fetching from BallDontLie API: {str(e)}")
        traceback.print_exc()
        return []


def prepare_game_features(games):
    """
    Generate features for the provided games
    
    Args:
        games: List of game dictionaries
        
    Returns:
        DataFrame with game features
    """
    try:
        # Debug logging for games input
        logger.info(f"Preparing features for {len(games)} games")
        for i, game in enumerate(games):
            logger.info(f"Game {i+1} data: {json.dumps(str(game)[:200])}...")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Load necessary historical data
        balldontlie = BallDontLieClient()
        
        # Get team data
        teams_response = balldontlie.get_teams()
        teams = []
        
        # Process teams data properly
        if isinstance(teams_response, dict) and 'data' in teams_response:
            teams = teams_response['data']
        elif isinstance(teams_response, list):
            teams = teams_response
        
        if not teams:
            logger.error("Failed to get team data")
            return pd.DataFrame()
        
        logger.info(f"Retrieved {len(teams)} teams from the API")
        
        # Get recent games for context (last 30 days)
        today = datetime.now()
        start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        
        recent_games_response = balldontlie.get_games(start_date=start_date, end_date=end_date)
        recent_games = []
        
        # Process recent games data properly
        if isinstance(recent_games_response, dict) and 'data' in recent_games_response:
            recent_games = recent_games_response['data']
        elif isinstance(recent_games_response, list):
            recent_games = recent_games_response
            
        if not recent_games:
            logger.warning("No recent games found, features may be limited")
        else:
            logger.info(f"Retrieved {len(recent_games)} recent games for context")
        
        # Skip individual team stats as they're returning 404 errors
        logger.info("Skipping team stats API calls due to endpoint issues")
        
        # Generate features for each game
        game_features = []
        for game_index, game in enumerate(games):
            try:
                logger.info(f"Processing game {game_index+1}/{len(games)}")
                
                # First determine the data format
                if isinstance(game, dict):
                    # Format depends on which API we used
                    if 'home_team' in game and isinstance(game['home_team'], dict):
                        # BallDontLie API format
                        logger.info(f"Game {game_index+1} is in BallDontLie format")
                        home_team_id = game['home_team']['id']
                        away_team_id = game['visitor_team']['id']
                        home_team_name = game['home_team']['name']
                        away_team_name = game['visitor_team']['name']
                        game_date = game['date']
                        game_id = game['id']
                        game_status = game['status']
                        
                        logger.info(f"Game: {home_team_name} vs {away_team_name}, ID: {game_id}")
                    else:
                        # The Odds API format
                        logger.info(f"Game {game_index+1} is in The Odds API format")
                        if 'home_team' not in game or 'away_team' not in game:
                            logger.error(f"Game missing team information: {json.dumps(str(game)[:200])}")
                            continue
                            
                        home_team = game['home_team']
                        away_team = game['away_team']
                        game_date = game['commence_time'] if 'commence_time' in game else datetime.now().isoformat()
                        game_id = game['id'] if 'id' in game else f"{home_team}-{away_team}"
                        game_status = game['status'] if 'status' in game else 'upcoming'
                        
                        logger.info(f"Game: {home_team} vs {away_team}, ID: {game_id}")
                        
                        # Need to map team names to IDs
                        logger.info(f"Looking up team IDs for {home_team} and {away_team}")
                        home_team_record = next((t for t in teams if home_team.lower() in t['name'].lower()), None)
                        away_team_record = next((t for t in teams if away_team.lower() in t['name'].lower()), None)
                        
                        if home_team_record and away_team_record:
                            home_team_id = home_team_record['id']
                            away_team_id = away_team_record['id']
                            home_team_name = home_team_record['name']
                            away_team_name = away_team_record['name']
                            logger.info(f"Found team IDs: Home={home_team_id}, Away={away_team_id}")
                        else:
                            logger.warning(f"Could not find team IDs for {home_team} or {away_team}")
                            # Try a more fuzzy matching approach
                            logger.info("Attempting fuzzy matching of team names")
                            for t in teams:
                                logger.info(f"Team in database: {t['name']}")
                            continue
                    
                    # Create a simpler game data structure with available info
                    game_data = {
                        'home_team_id': home_team_id,
                        'away_team_id': away_team_id,
                        'game_id': game_id,
                        'status': game_status,
                        'date': game_date,
                        # Add empty stats since API is failing
                        'home_team_stats': {},
                        'away_team_stats': {}
                    }
                    
                    # Calculate basic features based on recent games
                    logger.info(f"Finding recent games for teams {home_team_id} and {away_team_id}")
                    home_recent_games = [g for g in recent_games if 
                                        (isinstance(g.get('home_team'), dict) and g['home_team'].get('id') == home_team_id) or
                                        (isinstance(g.get('visitor_team'), dict) and g['visitor_team'].get('id') == home_team_id)]
                    
                    away_recent_games = [g for g in recent_games if 
                                        (isinstance(g.get('home_team'), dict) and g['home_team'].get('id') == away_team_id) or
                                        (isinstance(g.get('visitor_team'), dict) and g['visitor_team'].get('id') == away_team_id)]
                    
                    logger.info(f"Found {len(home_recent_games)} recent games for home team and {len(away_recent_games)} for away team")
                    
                    # Build a basic features dictionary
                    features = {
                        'game_id': game_id,
                        'home_team_id': home_team_id,
                        'away_team_id': away_team_id,
                        'date': game_date,
                        'home_team_name': home_team_name,
                        'away_team_name': away_team_name,
                        # Include some simple derived features
                        'home_games_played': len(home_recent_games),
                        'away_games_played': len(away_recent_games),
                    }
                    
                    # Add simple win rate features as fallback
                    logger.info("Calculating win rates from recent games")
                    home_wins = sum(1 for g in home_recent_games if 
                                  (isinstance(g.get('home_team'), dict) and g['home_team'].get('id') == home_team_id and g.get('home_team_score', 0) > g.get('visitor_team_score', 0)) or
                                  (isinstance(g.get('visitor_team'), dict) and g['visitor_team'].get('id') == home_team_id and g.get('visitor_team_score', 0) > g.get('home_team_score', 0)))
                    
                    away_wins = sum(1 for g in away_recent_games if 
                                  (isinstance(g.get('home_team'), dict) and g['home_team'].get('id') == away_team_id and g.get('home_team_score', 0) > g.get('visitor_team_score', 0)) or
                                  (isinstance(g.get('visitor_team'), dict) and g['visitor_team'].get('id') == away_team_id and g.get('visitor_team_score', 0) > g.get('home_team_score', 0)))
                    
                    # Calculate win rates
                    features['home_win_rate'] = home_wins / max(1, len(home_recent_games))
                    features['away_win_rate'] = away_wins / max(1, len(away_recent_games))
                    features['win_rate_diff'] = features['home_win_rate'] - features['away_win_rate']
                    
                    # Add more basic stats
                    home_points_scored = [g.get('home_team_score', 0) if isinstance(g.get('home_team'), dict) and g['home_team'].get('id') == home_team_id 
                                        else g.get('visitor_team_score', 0) for g in home_recent_games if g.get('status') == 'Final']
                    
                    away_points_scored = [g.get('home_team_score', 0) if isinstance(g.get('home_team'), dict) and g['home_team'].get('id') == away_team_id 
                                        else g.get('visitor_team_score', 0) for g in away_recent_games if g.get('status') == 'Final']
                    
                    # Get points allowed
                    home_points_allowed = [g.get('visitor_team_score', 0) if isinstance(g.get('home_team'), dict) and g['home_team'].get('id') == home_team_id 
                                          else g.get('home_team_score', 0) for g in home_recent_games if g.get('status') == 'Final']
                    
                    away_points_allowed = [g.get('visitor_team_score', 0) if isinstance(g.get('home_team'), dict) and g['home_team'].get('id') == away_team_id 
                                          else g.get('home_team_score', 0) for g in away_recent_games if g.get('status') == 'Final']
                    
                    # Calculate averages
                    features['home_avg_points'] = sum(home_points_scored) / max(1, len(home_points_scored))
                    features['away_avg_points'] = sum(away_points_scored) / max(1, len(away_points_scored))
                    features['home_avg_points_allowed'] = sum(home_points_allowed) / max(1, len(home_points_allowed))
                    features['away_avg_points_allowed'] = sum(away_points_allowed) / max(1, len(away_points_allowed))
                    features['point_diff'] = (features['home_avg_points'] - features['away_avg_points_allowed']) - \
                                              (features['away_avg_points'] - features['home_avg_points_allowed'])
                    
                    # Try to use the feature engineer if possible
                    try:
                        logger.info("Attempting to use full feature engineer")
                        engineered_features = feature_engineer.generate_game_features(game_data, recent_games, teams)
                        # Merge the engineered features
                        features.update(engineered_features)
                        logger.info("Successfully generated engineered features")
                    except Exception as fe_error:
                        logger.warning(f"Could not use feature engineer: {str(fe_error)}")
                        # We'll stick with our basic features
                    
                    # Make sure we have all required features for prediction
                    if 'home_team_id' in features and 'away_team_id' in features:
                        logger.info(f"Adding game features for {home_team_name} vs {away_team_name}")
                        game_features.append(features)
                    else:
                        logger.error("Missing required features for prediction")
                else:
                    logger.error(f"Game {game_index+1} is not in a recognized format: {type(game)}")
            except Exception as inner_e:
                logger.error(f"Error generating features for game: {str(inner_e)}")
                traceback.print_exc()
        
        if not game_features:
            logger.error("Failed to generate features for any games")
            return pd.DataFrame()
        
        # Combine into a single DataFrame
        features_df = pd.DataFrame(game_features)
        logger.info(f"Successfully generated features for {len(features_df)} games")
        
        return features_df
    
    except Exception as e:
        logger.error(f"Error preparing game features: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()


def run_predictions(models, features_df):
    """
    Generate predictions for each game using loaded models
    
    Args:
        models: Dictionary of loaded models
        features_df: DataFrame with game features
        
    Returns:
        DataFrame with predictions
    """
    if features_df.empty:
        logger.error("No features available for prediction")
        return pd.DataFrame()
    
    try:
        # Create a deep copy to avoid modifying the original
        results_df = features_df.copy()
        
        # Define columns needed for prediction (exclude metadata and target variables)
        meta_columns = ['game_id', 'home_team_id', 'away_team_id', 'date', 'home_team_name', 'away_team_name']
        target_columns = ['home_won', 'point_diff', 'total_points']
        
        # Get feature columns (all except metadata and targets)
        feature_columns = [c for c in features_df.columns if c not in meta_columns + target_columns]
        
        # Prepare feature matrix
        X = features_df[feature_columns]
        
        # For each model, generate predictions
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict') and callable(model.predict):
                    # Get raw predictions
                    y_pred = model.predict(X)
                    results_df[f"{model_name}_pred"] = y_pred
                    
                    # Get probability predictions if available (for classification models)
                    if hasattr(model, 'predict_proba') and callable(model.predict_proba):
                        try:
                            y_proba = model.predict_proba(X)
                            if y_proba.shape[1] >= 2:  # Binary classification (home win probability)
                                results_df[f"{model_name}_home_win_proba"] = y_proba[:, 1]
                        except Exception as proba_error:
                            logger.warning(f"Error getting probabilities from {model_name}: {str(proba_error)}")
                else:
                    logger.warning(f"Model {model_name} doesn't have a predict method")
            except Exception as model_error:
                logger.error(f"Error running prediction with {model_name}: {str(model_error)}")
        
        return results_df
    
    except Exception as e:
        logger.error(f"Error running predictions: {str(e)}")
        return pd.DataFrame()


def display_prediction_results(results_df):
    """
    Display the prediction results in a user-friendly format
    
    Args:
        results_df: DataFrame with prediction results
    """
    if results_df.empty:
        print("\nNo prediction results available.")
        return
    
    # Identify prediction columns
    pred_columns = [c for c in results_df.columns if 'pred' in c or 'proba' in c]
    
    if not pred_columns:
        print("\nNo predictions found in results.")
        return
    
    print("\n====== NBA GAME PREDICTIONS ======")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Number of games: {len(results_df)}")
    print("==================================\n")
    
    # For each game, display the predictions
    for idx, row in results_df.iterrows():
        home_team = row['home_team_name']
        away_team = row['away_team_name']
        
        print(f"Game {idx+1}: {away_team} @ {home_team}")
        
        # Group predictions by model
        model_predictions = {}
        for col in pred_columns:
            model_name = col.split('_')[0]
            pred_type = '_'.join(col.split('_')[1:])
            
            if model_name not in model_predictions:
                model_predictions[model_name] = {}
                
            model_predictions[model_name][pred_type] = row[col]
        
        # Display predictions for each model
        for model_name, preds in model_predictions.items():
            print(f"  {model_name}:")
            for pred_type, value in preds.items():
                if pred_type == 'pred':
                    # For binary predictions (home win)
                    if 0 <= value <= 1:
                        winner = home_team if value >= 0.5 else away_team
                        print(f"    Predicted Winner: {winner} (confidence: {abs(value-0.5)*2:.2f})")
                    else:
                        # For point spread predictions
                        if value > 0:
                            print(f"    Predicted Point Diff: {home_team} by {value:.1f} points")
                        else:
                            print(f"    Predicted Point Diff: {away_team} by {abs(value):.1f} points")
                elif 'home_win_proba' in pred_type:
                    winner = home_team if value >= 0.5 else away_team
                    print(f"    Win Probability: {home_team} {value*100:.1f}% / {away_team} {(1-value)*100:.1f}%")
        
        print()
    
    # Display model performance comparison
    print("====== MODEL COMPARISON ======")
    for model_name in model_predictions.keys():
        win_proba_col = f"{model_name}_home_win_proba"
        if win_proba_col in results_df.columns:
            avg_confidence = results_df[win_proba_col].apply(lambda x: max(x, 1-x)).mean()
            print(f"{model_name}: Average confidence = {avg_confidence*100:.1f}%")
    
    print("\nNote: Higher confidence does not necessarily mean better accuracy.")
    print("===============================")


def create_user_friendly_display(results_df):
    """
    Create a user-friendly, comprehensive prediction display for everyday users
    
    Args:
        results_df: DataFrame with prediction results
    """
    if results_df.empty:
        print("\nNo prediction results available.")
        return
    
    # Get current time for game time display
    now = datetime.now()
    est_time = now.replace(hour=19, minute=30)  # Default game time if not available
    
    # Identify prediction columns
    pred_columns = [c for c in results_df.columns if 'pred' in c or 'proba' in c]
    
    if not pred_columns:
        print("\nNo predictions found in results.")
        return
    
    print("\n" + "=" * 80)
    print(" " * 25 + "NBA PREDICTIONS - " + now.strftime('%A, %B %d, %Y'))
    print("=" * 80)
    
    # Try to get odds information
    try:
        odds_client = TheOddsClient()
        odds_client.DEFAULT_MARKETS = ["h2h", "spreads", "totals"]  # Limit to basic markets to avoid API errors
        nba_odds = odds_client.get_nba_odds(markets=["h2h", "spreads", "totals"])
        logger.info(f"Retrieved {len(nba_odds)} games from The Odds API for display")
    except Exception as e:
        logger.warning(f"Could not retrieve odds: {str(e)}")
        nba_odds = []
    
    # For each game, display the predictions
    for idx, row in results_df.iterrows():
        home_team = row['home_team_name']
        away_team = row['away_team_name']
        
        # Find matching odds
        game_odds = None
        for odds in nba_odds:
            if (home_team.lower() in odds.get('home_team', '').lower() or 
                odds.get('home_team', '').lower() in home_team.lower()) and \
               (away_team.lower() in odds.get('away_team', '').lower() or 
                odds.get('away_team', '').lower() in away_team.lower()):
                game_odds = odds
                break
        
        # Get consensus predictions across models
        win_probabilities = []
        point_spreads = []
        for col in pred_columns:
            if 'home_win_proba' in col:
                win_probabilities.append(row[col])
            elif 'pred' in col and not (0 <= row[col] <= 1):
                # Likely a point spread prediction
                point_spreads.append(row[col])
        
        # Average the predictions for consensus
        avg_win_prob = sum(win_probabilities) / max(1, len(win_probabilities)) if win_probabilities else 0.5
        avg_point_spread = sum(point_spreads) / max(1, len(point_spreads)) if point_spreads else 0
        
        # Confidence level calculation (how much the models agree)
        win_confidence = min(100, max(50, 50 + 50 * (2 * abs(avg_win_prob - 0.5))))
        
        # Display game header
        print(f"\n\n{away_team} vs. {home_team}")
        game_time = est_time.strftime("%I:%M %p ET")
        print(f"Game Time: {game_time}, {home_team} Home Court")
        
        # Win prediction
        favored_team = home_team if avg_win_prob > 0.5 else away_team
        underdog_team = away_team if favored_team == home_team else home_team
        favored_prob = avg_win_prob if favored_team == home_team else 1 - avg_win_prob
        underdog_prob = 1 - favored_prob
        
        print(f"Who Wins? We give the {favored_team} a {favored_prob*100:.0f}% chance to win, {underdog_team} at {underdog_prob*100:.0f}%.")  
        
        # Odds display if available
        if game_odds and 'bookmakers' in game_odds and game_odds['bookmakers']:
            try:
                # Get moneyline odds
                for bookmaker in game_odds['bookmakers']:
                    if 'markets' in bookmaker and bookmaker['markets']:
                        for market in bookmaker['markets']:
                            if market['key'] == 'h2h':
                                home_odds = next((o['price'] for o in market['outcomes'] if o['name'].lower() in home_team.lower()), None)
                                away_odds = next((o['price'] for o in market['outcomes'] if o['name'].lower() in away_team.lower()), None)
                                
                                if home_odds and away_odds:
                                    # Convert to American odds for display
                                    home_american = int(((home_odds - 1) * 100) if home_odds < 2 else 100)
                                    away_american = int(((away_odds - 1) * 100) if away_odds < 2 else 100)
                                    
                                    if home_american > 0:
                                        home_american_str = f"+{home_american}"
                                    else:
                                        home_american_str = f"{home_american}"
                                        
                                    if away_american > 0:
                                        away_american_str = f"+{away_american}"
                                    else:
                                        away_american_str = f"{away_american}"
                                    
                                    print(f"Odds: {home_team} {home_american_str}, {away_team} {away_american_str}.")
                                    
                                    # Calculate expected value
                                    if favored_team == home_team and home_american < 0:
                                        implied_prob = abs(home_american) / (abs(home_american) + 100)
                                        edge = favored_prob - implied_prob
                                        if edge > 0.01:  # Only suggest bets with positive edge
                                            bet_size = min(3, max(1, int(edge * 100)))
                                            print(f"\nBet: Put {bet_size}% of your money on the {favored_team}—it's a smart bet with a {edge*100:.1f}% edge if you act now.")
                                    elif favored_team == away_team and away_american < 0:
                                        implied_prob = abs(away_american) / (abs(away_american) + 100)
                                        edge = favored_prob - implied_prob
                                        if edge > 0.01:  # Only suggest bets with positive edge
                                            bet_size = min(3, max(1, int(edge * 100)))
                                            print(f"\nBet: Put {bet_size}% of your money on the {favored_team}—it's a smart bet with a {edge*100:.1f}% edge if you act now.")
                                            
                                    # Add confidence level
                                    print(f"\nConfidence: {win_confidence:.0f}% sure they'll win.")
                                break
            except Exception as odds_error:
                logger.warning(f"Error processing odds data: {str(odds_error)}")
        
        # Point spread prediction
        if avg_point_spread != 0:
            winner = home_team if avg_point_spread > 0 else away_team
            margin = abs(avg_point_spread)
            rounded_spread = round(margin)
            cover_confidence = min(90, max(50, 50 + 10 * abs(margin - rounded_spread)))
            print(f"By How Much? {winner} should win by {margin:.1f} points, with a {cover_confidence:.0f}% chance to cover -{rounded_spread}.")
            
            # Display spread odds if available
            if game_odds and 'bookmakers' in game_odds and game_odds['bookmakers']:
                try:
                    for bookmaker in game_odds['bookmakers']:
                        if 'markets' in bookmaker and bookmaker['markets']:
                            for market in bookmaker['markets']:
                                if market['key'] == 'spreads':
                                    spread_odds = next((o['price'] for o in market['outcomes'] if o['name'].lower() in winner.lower()), None)
                                    if spread_odds:
                                        american_odds = int(((spread_odds - 1) * 100) if spread_odds < 2 else 100)
                                        print(f"Odds: {american_odds}.")
                                        
                                        # Calculate expected value and bet size
                                        implied_prob = 0.5  # Spread bets typically have 50% implied probability
                                        edge = (cover_confidence / 100) - implied_prob
                                        if edge > 0.01:
                                            bet_size = max(1, min(2, int(edge * 100)))
                                            print(f"\nBet: Use {bet_size}% of your money—you could gain a {edge*100:.1f}% edge.")
                                    
                                    print(f"\nConfidence: {cover_confidence:.0f}% sure they'll cover.")
                                    break
                except Exception:
                    pass
        
        # Total points prediction if available
        if 'home_avg_points' in row and 'away_avg_points' in row:
            total_points = row['home_avg_points'] + row['away_avg_points']
            rounded_total = round(total_points)
            over_confidence = 50 + min(30, max(-30, (total_points - rounded_total) * 10))
            print(f"Total Points? Expect {total_points:.1f} points, with a {over_confidence:.0f}% chance to go over {rounded_total}.")
            
            # Display total points odds if available
            if game_odds and 'bookmakers' in game_odds and game_odds['bookmakers']:
                try:
                    for bookmaker in game_odds['bookmakers']:
                        if 'markets' in bookmaker and bookmaker['markets']:
                            for market in bookmaker['markets']:
                                if market['key'] == 'totals':
                                    over_odds = next((o['price'] for o in market['outcomes'] if o['name'] == 'Over'), None)
                                    under_odds = next((o['price'] for o in market['outcomes'] if o['name'] == 'Under'), None)
                                    
                                    if over_odds and under_odds:
                                        over_american = int(((over_odds - 1) * 100) if over_odds < 2 else 100)
                                        under_american = int(((under_odds - 1) * 100) if under_odds < 2 else 100)
                                        print(f"Odds: Over {over_american}, Under {under_american}.")
                                        
                                        # Calculate expected value
                                        implied_prob = 0.5  # Total bets typically have 50% implied probability
                                        edge = (over_confidence / 100) - implied_prob
                                        if abs(edge) > 0.01:
                                            bet_type = "over" if edge > 0 else "under"
                                            bet_size = max(1, min(3, int(abs(edge) * 100)))
                                            print(f"\nBet: Risk {bet_size}% on the {bet_type} for a {abs(edge)*100:.1f}% edge.")
                                        
                                    print(f"\nConfidence: {max(over_confidence, 100-over_confidence):.0f}% sure it'll go {'over' if over_confidence > 50 else 'under'}.")
                                    break
                except Exception:
                    pass
        
        # Note about player props
        print("\nPlayer Props: Our models are currently being updated to include player prop predictions.")
        print("Check back soon for detailed player performance predictions.")
        
        # Display reasoning section
        print("\nHow We Figured This Out")
        explanation = generate_prediction_explanation(row, avg_win_prob, avg_point_spread)
        print(explanation)
    
    print("\n" + "=" * 80)
    print("Predictions generated on " + now.strftime("%A, %B %d, %Y at %I:%M %p ET"))
    print("=" * 80)
    print("\nPrediction process completed! All logs saved to the 'logs' directory.")


def generate_prediction_explanation(game_data, win_prob, point_spread):
    """
    Generate a natural language explanation of prediction reasoning
    
    Args:
        game_data: Row of data for the game
        win_prob: Win probability
        point_spread: Predicted point spread
        
    Returns:
        str: Natural language explanation
    """
    home_team = game_data['home_team_name']
    away_team = game_data['away_team_name']
    winning_team = home_team if win_prob > 0.5 else away_team
    losing_team = away_team if winning_team == home_team else home_team
    
    # Get relevant stats that we have
    home_win_rate = game_data.get('home_win_rate', 0.5)
    away_win_rate = game_data.get('away_win_rate', 0.5)
    home_avg_points = game_data.get('home_avg_points', 0)
    away_avg_points = game_data.get('away_avg_points', 0)
    home_points_allowed = game_data.get('home_avg_points_allowed', 0)
    away_points_allowed = game_data.get('away_avg_points_allowed', 0)
    home_games = game_data.get('home_games_played', 0)
    away_games = game_data.get('away_games_played', 0)
    
    # Create a comprehensive explanation
    explanation = (
        f"We checked loads of game details to make these picks. "
        f"The {winning_team} {'are' if ' ' in winning_team else 'is'} "
    )
    
    # Add team performance context
    if winning_team == home_team and home_avg_points > 0:
        explanation += f"scoring {home_avg_points:.0f} points a game lately"
    elif away_avg_points > 0:
        explanation += f"scoring {away_avg_points:.0f} points a game lately"
    else:
        explanation += "showing good form in recent games"
    
    # Add win rates if we have them
    if home_win_rate > 0.6 or away_win_rate > 0.6:
        top_team = home_team if home_win_rate > away_win_rate else away_team
        win_rate = max(home_win_rate, away_win_rate)
        explanation += f", and {top_team} has been winning {win_rate*100:.0f}% of their recent games"
    
    # Add home court advantage
    if winning_team == home_team:
        explanation += f". {home_team} has the home court advantage"
    
    # Add point differential if available
    if home_points_allowed > 0 and away_points_allowed > 0:
        home_diff = home_avg_points - away_points_allowed
        away_diff = away_avg_points - home_points_allowed
        better_offense = home_team if home_diff > away_diff else away_team
        explanation += f". {better_offense} has the scoring edge in this matchup"
    
    # Add games context
    explanation += f". Both teams have been active, with {home_team} playing {home_games} and {away_team} playing {away_games} games recently"
    
    # Add model summary
    explanation += (f". Our prediction models analyzed these factors and give {winning_team} a "
                   f"{win_prob*100:.0f}% chance to win against {losing_team}")
    
    # Add spread context if we have it
    if abs(point_spread) > 0:
        explanation += f", likely winning by about {abs(point_spread):.1f} points"
    
    explanation += ". We used team stats (like scoring and defense), rest days, and home court advantage, then crunched it all with advanced algorithms to get these probabilities."
    
    return explanation


def save_prediction_results(results_df, filename="prediction_results"):
    """
    Save prediction results to CSV and JSON files
    
    Args:
        results_df: DataFrame with prediction results
        filename: Base filename for output files
    """
    if results_df.empty:
        logger.warning("No results to save")
        return
    
    # Create output directory
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = output_dir / f"{filename}_{timestamp}"
    
    # Save as CSV
    csv_path = f"{base_path}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Save as JSON (with more readable format for displaying games)
    json_path = f"{base_path}.json"
    
    # Extract game predictions in a more readable format
    games_predictions = []
    for idx, row in results_df.iterrows():
        game_pred = {
            "away_team": row["away_team_name"],
            "home_team": row["home_team_name"],
            "date": row["date"],
            "predictions": {}
        }
        
        # Add predictions from each model
        for col in results_df.columns:
            if '_pred' in col or '_proba' in col:
                game_pred["predictions"][col] = row[col]
        
        games_predictions.append(game_pred)
    
    # Save the JSON file
    with open(json_path, 'w') as f:
        json.dump(games_predictions, f, indent=2)
    
    print(f"Results also saved to {json_path}")


def main():
    """
    Main function to run the prediction script
    """
    # Configure warnings to be less intrusive
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    
    # Set up clear console output
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear console
    
    print("\n\nNBA Game Prediction Tool")
    print("=======================\n")
    
    # Load the models
    print("Loading trained models...")
    models = load_models()
    if not models:
        print("No models found. Please train models first.")
        return
    
    # Display loaded models
    model_names = ', '.join(models.keys())
    logger.info(f"Loaded {len(models)} models: {model_names}")
    
    # Fetch today's NBA games
    games = fetch_nba_games()
    if not games:
        print("No NBA games scheduled for today.")
        return
    
    print(f"Found {len(games)} NBA games scheduled for today.")
    
    # Generate features for prediction
    print("Generating features for prediction...")
    features_df = prepare_game_features(games)
    if features_df.empty:
        print("Failed to generate features for prediction.")
        return
    
    # Run the model predictions
    print("Running model predictions...")
    
    # Temporarily redirect stdout to capture and suppress warnings
    original_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        results_df = run_predictions(models, features_df)
    finally:
        sys.stdout = original_stdout
    
    if results_df.empty:
        print("Failed to generate predictions.")
        return
    
    # Clear console again for clean prediction display
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Display the prediction results - use ONLY the user-friendly display
    create_user_friendly_display(results_df)
    
    # Save the prediction results
    save_prediction_results(results_df)
    
    print("\nPrediction process completed!")


if __name__ == "__main__":
    main()
