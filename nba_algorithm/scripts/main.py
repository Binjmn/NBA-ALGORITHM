#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Game Prediction System - Main Module

This is the main entry point for the NBA prediction system. It orchestrates the workflow
of fetching data, preparing features, making predictions, and displaying results.

Usage:
    python -m nba_algorithm.scripts.main

Author: Cascade
Date: April 2025
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Import from other modules
from ..utils.config import config_logging, LOG_FILE, DATA_DIR
from ..models.loader import load_enhanced_models
from ..data.game_data import fetch_nba_games
from ..data.team_data import fetch_team_stats
from ..data.player_data import fetch_player_data
from ..data.odds_data import fetch_betting_odds, fetch_player_props
from ..features.game_features import prepare_game_features
from ..models.predictor import predict_todays_games, run_predictions
from ..features.player_features import predict_props_for_player
from ..presentation.display import display_user_friendly_predictions, display_predictions
from ..utils.storage import save_prediction_results, clean_old_cache_files

# Configure logger
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command-line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="NBA Game Prediction System")
    parser.add_argument(
        "--date",
        type=str,
        help="Date to generate predictions for (YYYY-MM-DD format). Defaults to today."
    )
    parser.add_argument(
        "--props",
        action="store_true",
        help="Include player prop predictions"
    )
    parser.add_argument(
        "--props-only",
        action="store_true",
        help="Show only player prop predictions"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save prediction results to files"
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run NBA predictions
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize logging
    config_logging(LOG_FILE, args.verbose)
    logger.info("Starting NBA prediction system")
    
    try:
        # Clean old cache files
        clean_old_cache_files(DATA_DIR)
        
        # Load prediction models
        logger.info("Loading prediction models")
        models = load_enhanced_models()
        if not models:
            logger.error("Failed to load any prediction models. Exiting.")
            return 1
        
        # Fetch games for today or specified date
        logger.info("Fetching game data")
        games = fetch_nba_games(args.date)
        if not games:
            logger.error("No games found for the specified date. Exiting.")
            return 1
        
        # Add betting odds data if available
        try:
            logger.info("Fetching betting odds data")
            odds_data = fetch_betting_odds(games)
            
            if odds_data:
                # Add odds to the game data
                for game in games:
                    game_id = game.get('id')
                    if game_id in odds_data:
                        game['odds'] = odds_data[game_id]
        except Exception as e:
            logger.error(f"Error fetching betting odds: {str(e)}")
            logger.info("Continuing without betting odds data")
        
        # Fetch team statistics
        logger.info("Fetching team statistics")
        team_stats = fetch_team_stats()
        
        # Prepare features and make predictions
        logger.info("Preparing features and making predictions")
        features_df = prepare_game_features(games)
        results_df = run_predictions(models.get('game', {}), features_df)
        
        # Process player props if requested
        player_props = {}
        if args.props or args.props_only:
            logger.info("Processing player prop predictions")
            try:
                # Fetch player data
                player_data = fetch_player_data(games)
                
                if player_data:
                    # Make player prop predictions
                    player_props = predict_player_props(games, player_data, models.get('player_props', {}))
                else:
                    logger.warning("No player data available for prop predictions")
            except Exception as e:
                logger.error(f"Error processing player props: {str(e)}")
                logger.info("Continuing without player prop predictions")
        
        # Display predictions based on options
        if args.props_only:
            # Show only player props
            if player_props:
                predictions = {}
                for game in games:
                    game_id = game.get('id')
                    if game_id in player_props:
                        predictions[game_id] = {
                            'home_team': game.get('home_team', {}).get('name', 'Home Team'),
                            'away_team': game.get('away_team', {}).get('name', 'Away Team'),
                            'player_props': player_props.get(game_id, [])
                        }
                
                display_predictions(predictions, props_only=True)
            else:
                logger.error("No player prop predictions available")
                return 1
        else:
            # Show game predictions with or without props
            display_user_friendly_predictions(results_df, player_props if args.props else None)
        
        # Save prediction results
        if not args.no_save:
            logger.info("Saving prediction results")
            date_str = args.date if args.date else datetime.now().strftime("%Y%m%d")
            
            # Save game predictions
            save_prediction_results(results_df, f"game_predictions_{date_str}")
            
            # Save player props if available
            if player_props and (args.props or args.props_only):
                props_list = []
                for game_id, props in player_props.items():
                    for prop in props:
                        # Add game_id to each prop
                        prop_with_game = prop.copy()
                        prop_with_game['game_id'] = game_id
                        props_list.append(prop_with_game)
                
                save_prediction_results(props_list, f"player_props_{date_str}")
        
        logger.info("NBA prediction process completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Unexpected error in main function: {str(e)}")
        logger.exception("Stack trace:")
        return 1


def predict_player_props(games: List[Dict], player_data: List[Dict], prop_models: Dict) -> Dict[str, List[Dict[str, Any]]]:
    """
    Make player prop predictions using enhanced ML models
    
    Args:
        games: List of game dictionaries
        player_data: Player statistics data
        models: Dictionary of loaded prediction models
        
    Returns:
        Dictionary with player prop predictions
    """
    if not games or not player_data or not prop_models:
        logger.error("Missing required data for player prop predictions")
        return {}
    
    prop_predictions = {}
    prop_types = ['points', 'rebounds', 'assists']
    
    try:
        # Group players by game
        players_by_game = {}
        for player in player_data:
            game_id = player.get('game_id')
            if game_id not in players_by_game:
                players_by_game[game_id] = []
            players_by_game[game_id].append(player)
        
        # Process each game
        for game in games:
            game_id = game.get('id')
            home_team = game.get('home_team', {})
            away_team = game.get('away_team', {})
            
            if not game_id or not home_team or not away_team:
                logger.warning(f"Missing team information for game {game_id}. Skipping prop predictions.")
                continue
            
            home_team_name = home_team.get('name', 'Home Team')
            away_team_name = away_team.get('name', 'Away Team')
            
            # Get players for this game
            game_players = players_by_game.get(game_id, [])
            
            if not game_players:
                logger.warning(f"No players found for game {game_id} ({away_team_name} @ {home_team_name})")
                continue
            
            # Initialize props list for this game
            game_props = []
            
            # Make predictions for each player
            for player in game_players:
                is_home_team = player.get('is_home_team', False)
                team_name = home_team_name if is_home_team else away_team_name
                opponent_name = away_team_name if is_home_team else home_team_name
                
                # Predict props for this player
                from ..features.player_features import predict_props_for_player
                player_props = predict_props_for_player(
                    player, is_home_team, team_name, opponent_name, prop_models, prop_types
                )
                
                if player_props:
                    game_props.extend(player_props)
            
            # Store props for this game
            if game_props:
                prop_predictions[game_id] = game_props
                logger.info(f"Generated {len(game_props)} prop predictions for game {game_id}")
        
        return prop_predictions
    
    except Exception as e:
        logger.error(f"Error in predict_player_props: {str(e)}")
        return {}


if __name__ == "__main__":
    sys.exit(main())
