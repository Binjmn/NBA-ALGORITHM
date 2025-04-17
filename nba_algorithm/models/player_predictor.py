# -*- coding: utf-8 -*-
"""
Player Prediction Module

This module provides functionality for predicting player performance
using real player data and statistical models, with no synthetic data generation.
"""

import logging
import pandas as pd
import numpy as np
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any

from ..data.player_data import fetch_players_for_game, get_player_stats, fetch_player_injuries

logger = logging.getLogger(__name__)


def get_player_predictions(games, team_stats, historical_games=None):
    """
    Generate player-level predictions for all games
    
    This function uses real player data and statistical models to predict
    player performance metrics including points, rebounds, and assists.
    It does not use synthetic data generation, only real data from APIs.
    
    Args:
        games: List of games to predict
        team_stats: Dictionary of team statistics
        historical_games: Optional list of historical games for context
        
    Returns:
        pandas.DataFrame: Player predictions
    """
    try:
        logger.info("Generating player-level predictions")
        
        # Load player prediction models
        player_models = load_player_models()
        if not player_models:
            logger.error("Failed to load player prediction models")
            return pd.DataFrame()
        
        # Get player injuries for context
        try:
            injuries = fetch_player_injuries()
            logger.info(f"Fetched {len(injuries)} player injuries")
        except Exception as e:
            logger.warning(f"Failed to fetch player injuries: {str(e)}")
            injuries = []
        
        all_player_predictions = []
        
        # Generate predictions for each game
        for game in games:
            game_id = game.get("id")
            home_team = game.get("home_team", {})
            visitor_team = game.get("visitor_team", {})
            
            if not game_id or not home_team or not visitor_team:
                logger.warning(f"Skipping game with incomplete data: {game_id}")
                continue
                
            try:
                # Fetch players for this game
                logger.debug(f"Fetching players for game {game_id}")
                home_players = fetch_players_for_game(home_team.get("id"))
                visitor_players = fetch_players_for_game(visitor_team.get("id"))
                
                if not home_players or not visitor_players:
                    logger.warning(f"No players found for game {game_id}")
                    continue
                    
                # Combine all players
                all_players = home_players + visitor_players
                logger.debug(f"Found {len(all_players)} players for game {game_id}")
                
                # Get player stats
                player_stats = {}
                for player in all_players:
                    player_id = player.get("id")
                    if player_id:
                        try:
                            stats = get_player_stats(player_id)
                            if stats:
                                player_stats[str(player_id)] = stats
                        except Exception as e:
                            logger.debug(f"Failed to get stats for player {player_id}: {str(e)}")
                
                logger.debug(f"Fetched stats for {len(player_stats)} players")
                
                # Create feature matrix for player predictions
                player_features = prepare_player_features(
                    all_players, player_stats, team_stats, injuries, historical_games
                )
                
                if player_features.empty:
                    logger.warning(f"No valid player features for game {game_id}")
                    continue
                
                # Make predictions for each player
                player_prediction = predict_player_performance(player_features, player_models)
                
                if player_prediction.empty:
                    logger.warning(f"Failed to generate player predictions for game {game_id}")
                    continue
                
                # Add game context
                player_prediction["game_id"] = game_id
                player_prediction["home_team"] = home_team.get("full_name")
                player_prediction["visitor_team"] = visitor_team.get("full_name")
                
                all_player_predictions.append(player_prediction)
                
            except Exception as e:
                logger.warning(f"Error predicting player performance for game {game_id}: {str(e)}")
                logger.debug(traceback.format_exc())
                continue
        
        # Combine all player predictions
        if not all_player_predictions:
            logger.warning("No valid player predictions generated")
            return pd.DataFrame()
            
        player_predictions_df = pd.concat(all_player_predictions, ignore_index=True)
        logger.info(f"Generated predictions for {len(player_predictions_df)} players")
        
        return player_predictions_df
        
    except Exception as e:
        logger.error(f"Error generating player predictions: {str(e)}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def prepare_player_features(players, player_stats, team_stats, injuries=None, historical_games=None):
    """
    Prepare features for player performance prediction
    
    Args:
        players: List of players
        player_stats: Dictionary of player statistics
        team_stats: Dictionary of team statistics
        injuries: Optional list of player injuries
        historical_games: Optional list of historical games
        
    Returns:
        pandas.DataFrame: Player features for prediction
    """
    try:
        features_list = []
        
        for player in players:
            player_id = player.get("id")
            if not player_id:
                continue
                
            player_id_str = str(player_id)
            if player_id_str not in player_stats:
                logger.debug(f"No stats for player {player_id}")
                continue
            
            # Extract basic player info
            player_name = player.get("first_name", "") + " " + player.get("last_name", "").strip()
            team_id = player.get("team", {}).get("id")
            position = player.get("position", "")
            
            if not team_id or team_id not in (str(team_id) for team_id in team_stats):
                logger.debug(f"Invalid team ID for player {player_name}")
                continue
            
            team_id_str = str(team_id)
            
            # Get player stats
            stats = player_stats[player_id_str]
            
            # Create feature dictionary
            features = {
                "player_id": player_id,
                "player_name": player_name,
                "team_id": team_id,
                "team_name": player.get("team", {}).get("full_name", ""),
                "position": position,
            }
            
            # Add player stats as features
            for stat_name, stat_value in stats.items():
                if stat_name not in ["player_id", "player_name", "team_id", "team_name"]:
                    features[f"player_{stat_name}"] = stat_value
            
            # Add team stats as context
            if team_id_str in team_stats:
                team_data = team_stats[team_id_str]
                for stat_name, stat_value in team_data.items():
                    features[f"team_{stat_name}"] = stat_value
            
            # Check for injuries
            if injuries:
                is_injured = any(inj.get("player_id") == player_id for inj in injuries)
                features["is_injured"] = 1 if is_injured else 0
                
                # If player is injured, find injury details
                if is_injured:
                    injury = next(inj for inj in injuries if inj.get("player_id") == player_id)
                    features["injury_status"] = injury.get("status", "")
                    # Convert injury status to numeric value for model
                    if features["injury_status"].lower() in ["out", "doubtful"]:
                        features["injury_impact"] = 0.9  # High impact
                    elif features["injury_status"].lower() in ["questionable"]:
                        features["injury_impact"] = 0.5  # Medium impact
                    elif features["injury_status"].lower() in ["probable"]:
                        features["injury_impact"] = 0.2  # Low impact
                    else:
                        features["injury_impact"] = 0.0  # No impact
                else:
                    features["injury_status"] = "healthy"
                    features["injury_impact"] = 0.0
            
            # Add historical game context if available
            if historical_games:
                # Find recent games for this player's team
                team_games = [g for g in historical_games if 
                             g.get("home_team", {}).get("id") == team_id or 
                             g.get("visitor_team", {}).get("id") == team_id]
                
                if team_games:
                    # Calculate recent team performance
                    recent_games = sorted(team_games, key=lambda x: x.get("date", ""), reverse=True)[:5]
                    wins = sum(1 for g in recent_games if 
                              (g.get("home_team", {}).get("id") == team_id and g.get("home_team_score", 0) > g.get("visitor_team_score", 0)) or
                              (g.get("visitor_team", {}).get("id") == team_id and g.get("visitor_team_score", 0) > g.get("home_team_score", 0)))
                    
                    features["team_recent_win_pct"] = wins / len(recent_games) if recent_games else 0.5
            
            features_list.append(features)
        
        if not features_list:
            logger.warning("No valid player features created")
            return pd.DataFrame()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Fill missing values with reasonable defaults
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
        
        # Fill missing categorical values
        categorical_cols = features_df.select_dtypes(include=[object, 'category']).columns
        features_df[categorical_cols] = features_df[categorical_cols].fillna('')
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error preparing player features: {str(e)}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def predict_player_performance(player_features, player_models):
    """
    Predict player performance using machine learning models
    
    Args:
        player_features: DataFrame of player features
        player_models: Dictionary of player prediction models
        
    Returns:
        pandas.DataFrame: Predictions for each player
    """
    try:
        if player_features.empty or not player_models:
            logger.warning("No player features or models available")
            return pd.DataFrame()
        
        # Extract features for prediction
        feature_cols = [col for col in player_features.columns if col.startswith("player_") or 
                       col.startswith("team_") or col in ["is_injured", "injury_impact"]]
        
        # Check if we have the minimum required features
        required_features = ["player_pts_per_game", "player_reb_per_game", "player_ast_per_game"]
        missing_features = [f for f in required_features if f not in feature_cols]
        
        if missing_features:
            logger.warning(f"Missing required player features: {', '.join(missing_features)}")
            # Continue anyway, model will use whatever features are available
        
        # Make a copy of input features
        predictions_df = player_features.copy()
        
        # Predict points, rebounds, assists for each player
        if "points_model" in player_models:
            points_model = player_models["points_model"]
            try:
                X = predictions_df[feature_cols].fillna(0)
                predictions_df["predicted_points"] = points_model.predict(X)
            except Exception as e:
                logger.warning(f"Error predicting player points: {str(e)}")
                # Fall back to recent average from features
                predictions_df["predicted_points"] = predictions_df.get("player_pts_per_game", 10.0)
        
        if "rebounds_model" in player_models:
            rebounds_model = player_models["rebounds_model"]
            try:
                X = predictions_df[feature_cols].fillna(0)
                predictions_df["predicted_rebounds"] = rebounds_model.predict(X)
            except Exception as e:
                logger.warning(f"Error predicting player rebounds: {str(e)}")
                # Fall back to recent average from features
                predictions_df["predicted_rebounds"] = predictions_df.get("player_reb_per_game", 4.0)
        
        if "assists_model" in player_models:
            assists_model = player_models["assists_model"]
            try:
                X = predictions_df[feature_cols].fillna(0)
                predictions_df["predicted_assists"] = assists_model.predict(X)
            except Exception as e:
                logger.warning(f"Error predicting player assists: {str(e)}")
                # Fall back to recent average from features
                predictions_df["predicted_assists"] = predictions_df.get("player_ast_per_game", 2.0)
        
        # Calculate confidence score for predictions
        predictions_df["confidence_score"] = 0.7  # Default confidence
        
        # Adjust confidence based on data quality
        if "is_injured" in predictions_df.columns:
            # Lower confidence for injured players
            injured_mask = predictions_df["is_injured"] == 1
            predictions_df.loc[injured_mask, "confidence_score"] *= 0.7
        
        # Adjust confidence based on player consistency (if available)
        if "player_consistency" in predictions_df.columns:
            predictions_df["confidence_score"] *= (0.5 + 0.5 * predictions_df["player_consistency"])
        
        # Ensure confidence is between 0 and 1
        predictions_df["confidence_score"] = predictions_df["confidence_score"].clip(0, 1)
        
        # Map confidence score to descriptive level
        def map_confidence(score):
            if score >= 0.8:
                return "Very High"
            elif score >= 0.7:
                return "High"
            elif score >= 0.5:
                return "Medium"
            elif score >= 0.3:
                return "Low"
            else:
                return "Very Low"
        
        predictions_df["confidence_level"] = predictions_df["confidence_score"].apply(map_confidence)
        
        # Select and rename columns for output
        result_columns = [
            "player_id", "player_name", "team_id", "team_name", "position",
            "predicted_points", "predicted_rebounds", "predicted_assists",
            "confidence_score", "confidence_level"
        ]
        
        # Only include columns that exist
        result_columns = [col for col in result_columns if col in predictions_df.columns]
        
        return predictions_df[result_columns]
        
    except Exception as e:
        logger.error(f"Error predicting player performance: {str(e)}")
        logger.debug(traceback.format_exc())
        return pd.DataFrame()


def load_player_models():
    """
    Load player prediction models with detailed validation and error reporting
    
    Returns:
        Dict: Dictionary of loaded player models by name
        
    If models can't be loaded, this function will log detailed error messages
    explaining exactly why the models couldn't be loaded, instead of silently failing.
    """
    try:
        logger.info("Loading player prediction models")
        
        # Import the actual function from the module
        try:
            from ..models.loader import load_player_models as load_player_impl
            return load_player_impl()
        except ImportError as e:
            logger.error(f"Failed to import player model loader: {str(e)}")
            logger.error("This suggests the player prediction module may not be installed or properly configured")
            return None
        except AttributeError as e:
            logger.error(f"Failed to access player model loader: {str(e)}")
            logger.error("This suggests the player prediction module exists but doesn't have the expected function")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error loading player models: {str(e)}")
        logger.debug(traceback.format_exc())
        return None
