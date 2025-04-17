#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Display Module

This module handles the formatting and presentation of NBA game predictions and player props
in a user-friendly format.

Author: Cascade
Date: April 2025
"""

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)


def display_user_friendly_predictions(results_df: pd.DataFrame, player_props: Dict[str, List[Dict[str, Any]]] = None) -> None:
    """
    Create a user-friendly, comprehensive prediction display for everyday users
    
    Args:
        results_df: DataFrame with prediction results
        player_props: Dictionary mapping game IDs to lists of player prop predictions
    """
    if results_df.empty:
        logger.error("No prediction results to display")
        print("\n❌ No prediction results available.\n")
        return
    
    try:
        print("\n" + "="*80)
        print(f"🏀 NBA GAME PREDICTIONS - {datetime.now().strftime('%A, %B %d, %Y')}")
        print("="*80)
        
        # Group by date if multiple dates
        if 'game_date' in results_df.columns and results_df['game_date'].nunique() > 1:
            dates = sorted(results_df['game_date'].unique())
            for date in dates:
                date_df = results_df[results_df['game_date'] == date]
                _display_date_predictions(date_df, player_props, date)
        else:
            _display_date_predictions(results_df, player_props)
        
        print("\n" + "="*80)
        print("🔍 PREDICTION NOTES:")
        print("* Home team win probability incorporates home court advantage")
        print("* Predictions are based on team performance metrics and advanced analytics")
        print("* Higher confidence ratings indicate stronger prediction signals")
        
        if player_props:
            print("* Player prop predictions factor in matchups, rest, and recent performance")
            print("* 'Over/Under' recommendations reflect predicted value vs. season average")
        
        print("="*80 + "\n")
    
    except Exception as e:
        logger.error(f"Error displaying user-friendly predictions: {str(e)}")
        print("\n❌ Error displaying predictions. Please check the log for details.\n")


def _display_date_predictions(results_df: pd.DataFrame, player_props: Optional[Dict[str, List[Dict[str, Any]]]], date: Any = None) -> None:
    """
    Display predictions for a specific date
    
    Args:
        results_df: DataFrame with prediction results
        player_props: Dictionary mapping game IDs to lists of player prop predictions
        date: Date to display in the header (optional)
    """
    try:
        # Display date header if provided
        if date:
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime('%A, %B %d, %Y')
            else:
                date_str = str(date)
            print(f"\n📅 PREDICTIONS FOR {date_str}\n" + "-"*50)
        
        # Display each game
        for idx, row in results_df.iterrows():
            display_game_prediction(row, idx + 1)
            
            # Display player props for this game if available
            game_id = row.get('game_id')
            if player_props and game_id in player_props and player_props[game_id]:
                display_player_props(player_props[game_id])
            
            print("-"*80 + "\n")
    
    except Exception as e:
        logger.error(f"Error displaying date predictions: {str(e)}")
        print("\n❌ Error displaying date predictions. Please check the log for details.\n")


def display_game_prediction(game: pd.Series, game_num: int) -> None:
    """
    Display prediction for a single game in a user-friendly format
    
    Args:
        game: Series with game prediction data
        game_num: Game number for display purposes
    """
    try:
        # Extract relevant data
        home_team = game.get('home_team_name', 'Home Team')
        away_team = game.get('away_team_name', 'Away Team')
        spread = game.get('predicted_spread', 0.0)
        win_prob = game.get('home_win_probability', 0.5)
        total = game.get('predicted_total', 220.0)
        confidence = game.get('prediction_confidence', 0.7)
        
        # Format win probability as percentage
        win_pct = win_prob * 100
        
        # Determine favorite and spread direction
        if spread > 0:
            favorite = home_team
            underdog = away_team
            spread_value = abs(spread)
            favorite_is_home = True
        else:
            favorite = away_team
            underdog = home_team
            spread_value = abs(spread)
            favorite_is_home = False
        
        # Determine confidence level
        if confidence >= 0.8:
            confidence_text = "HIGH CONFIDENCE"
        elif confidence >= 0.6:
            confidence_text = "MEDIUM CONFIDENCE"
        else:
            confidence_text = "LOW CONFIDENCE"
        
        # Format the display
        print(f"🏀 GAME {game_num}: {away_team} @ {home_team}")
        
        # Win probability
        print(f"\n📊 WIN PROBABILITY: {home_team}: {win_pct:.1f}% | {away_team}: {(100-win_pct):.1f}%")
        
        # Point spread
        print(f"📏 PREDICTED SPREAD: {favorite} by {spread_value:.1f} points")
        
        # Over/under
        print(f"📈 PREDICTED TOTAL: {total:.1f} points")
        
        # Confidence
        conf_pct = confidence * 100
        print(f"🔒 PREDICTION CONFIDENCE: {conf_pct:.1f}% ({confidence_text})")
        
        # Betting insights (if available)
        if 'vegas_spread' in game and 'vegas_total' in game:
            vegas_spread = game.get('vegas_spread', 0.0)
            vegas_total = game.get('vegas_total', 220.0)
            
            # Calculate the differences
            spread_diff = abs(spread) - abs(vegas_spread)
            total_diff = total - vegas_total
            
            # Betting insights
            print("\n💰 BETTING INSIGHTS:")
            
            # Spread insight
            if abs(spread_diff) < 1.0:
                print(f"   Spread: Our model is in agreement with Vegas ({abs(vegas_spread):.1f})")
            elif (spread > 0 and vegas_spread < 0) or (spread < 0 and vegas_spread > 0):
                print(f"   Spread: Our model disagrees with Vegas on the FAVORITE")
            else:
                direction = "HIGHER" if abs(spread) > abs(vegas_spread) else "LOWER"
                print(f"   Spread: Our model predicts a {direction} spread than Vegas by {abs(spread_diff):.1f} points")
            
            # Total insight
            if abs(total_diff) < 5.0:
                print(f"   Total: Our model agrees with Vegas total ({vegas_total:.1f})")
            else:
                direction = "OVER" if total > vegas_total else "UNDER"
                print(f"   Total: Our model leans {direction} Vegas total by {abs(total_diff):.1f} points")
    
    except Exception as e:
        logger.error(f"Error displaying game prediction: {str(e)}")
        print(f"\n❌ Error displaying prediction for game {game_num}. Please check the log for details.\n")


def display_player_props(props: List[Dict[str, Any]]) -> None:
    """
    Display player prop predictions in a user-friendly format
    
    Args:
        props: List of player prop predictions
    """
    if not props:
        return
    
    try:
        print("\n⭐ PLAYER PROPS:")
        
        # Group props by player
        players = {}
        for prop in props:
            player_name = prop.get('player_name', 'Unknown Player')
            if player_name not in players:
                players[player_name] = []
            players[player_name].append(prop)
        
        # Display props for each player
        for player_name, player_props in players.items():
            team = player_props[0].get('team', 'Unknown Team')
            print(f"\n   {player_name} ({team}):")
            
            for prop in player_props:
                prop_type = prop.get('prop_type', 'unknown').title()
                prediction = prop.get('prediction', 0.0)
                season_avg = prop.get('season_average', 0.0)
                
                # Format the recommendation
                recommendation = prop.get('recommendation', {})
                direction = recommendation.get('direction', 'neutral').upper()
                strength = recommendation.get('strength', 'weak').title()
                
                # Pick an emoji based on recommendation strength
                if strength == 'Strong' and direction == 'OVER':
                    emoji = "🔥"
                elif strength == 'Strong' and direction == 'UNDER':
                    emoji = "❄️"
                elif strength == 'Moderate' and direction == 'OVER':
                    emoji = "📈"
                elif strength == 'Moderate' and direction == 'UNDER':
                    emoji = "📉"
                else:
                    emoji = "⚖️"
                
                # Calculate percentage difference from season average
                if season_avg > 0:
                    pct_diff = (prediction - season_avg) / season_avg * 100
                    diff_text = f"{pct_diff:+.1f}% vs. season avg"
                else:
                    diff_text = "no season average available"
                
                print(f"      {emoji} {prop_type}: {prediction:.1f} ({diff_text})")
                
                # Only print recommendation for medium or strong signals
                if strength != 'Weak':
                    print(f"         {strength} signal: Predicted to go {direction} season average")
    
    except Exception as e:
        logger.error(f"Error displaying player props: {str(e)}")
        print("\n❌ Error displaying player props. Please check the log for details.\n")


def display_predictions(predictions: Dict, include_props: bool = False, props_only: bool = False) -> None:
    """
    Display prediction results in a user-friendly format
    
    Args:
        predictions: Dictionary of predictions
        include_props: Whether to include player prop predictions
        props_only: Whether to show only player prop predictions
    """
    if not predictions:
        logger.error("No predictions to display")
        print("\n❌ No predictions available.\n")
        return
    
    try:
        # Extract game predictions and player props
        game_predictions = {}
        player_props = {}
        
        for game_id, prediction in predictions.items():
            if 'player_props' in prediction:
                player_props[game_id] = prediction['player_props']
                # Make a copy without player props for game predictions
                game_pred = prediction.copy()
                del game_pred['player_props']
                game_predictions[game_id] = game_pred
            else:
                game_predictions[game_id] = prediction
        
        # Display based on user preferences
        if props_only:
            print("\n" + "="*80)
            print(f"🏀 NBA PLAYER PROP PREDICTIONS - {datetime.now().strftime('%A, %B %d, %Y')}")
            print("="*80)
            
            for game_id, props in player_props.items():
                game = game_predictions.get(game_id, {})
                home_team = game.get('home_team', 'Home Team')
                away_team = game.get('away_team', 'Away Team')
                print(f"\n🏀 GAME: {away_team} @ {home_team}\n" + "-"*50)
                display_player_props(props)
                print("-"*80)
        elif not props_only:
            # Convert game predictions to DataFrame for display
            results_df = pd.DataFrame.from_dict(game_predictions, orient='index')
            
            # Display with or without props
            if include_props:
                display_user_friendly_predictions(results_df, player_props)
            else:
                display_user_friendly_predictions(results_df)
    
    except Exception as e:
        logger.error(f"Error in display_predictions: {str(e)}")
        print("\n❌ Error displaying predictions. Please check the log for details.\n")
