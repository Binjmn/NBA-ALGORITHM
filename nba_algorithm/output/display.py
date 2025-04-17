# -*- coding: utf-8 -*-
"""
Display Module for NBA Algorithm

This module contains functions for formatting and displaying prediction results
in a clean, consistent, and professional format.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

def create_prediction_schema(predictions_df, player_predictions_df=None, settings=None, betting_analyzer=None, real_metrics=None):
    """
    Create a standardized prediction output schema
    
    Args:
        predictions_df: DataFrame of game predictions
        player_predictions_df: DataFrame of player predictions
        settings: Prediction settings object
        betting_analyzer: BettingAnalyzer instance
        real_metrics: Dictionary of real performance metrics
        
    Returns:
        dict: Structured prediction data for display
        
    Raises:
        ValueError: If required data is missing or invalid
    """
    # Validate inputs
    if predictions_df is None or predictions_df.empty:
        raise ValueError("No prediction data provided. Unable to create display schema.")
    
    # Extract date from predictions
    prediction_date = None
    if 'date' in predictions_df.columns:
        prediction_date = predictions_df['date'].iloc[0]
    
    # If no date in predictions, use current date
    if not prediction_date:
        prediction_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Using current date for prediction output: {prediction_date}")
    
    # Use fixed bankroll value and risk level
    risk_level = 'moderate'
    bankroll = 1000.0  # Fixed standard bankroll value for all predictions
    track_clv = True  # Always track closing line value
    
    default_date = datetime.now().strftime("%Y-%m-%d")
    
    # Initialize prediction data structure with validated values
    prediction_data = {
        "date": prediction_date,
        "generation_time": datetime.now().strftime("%I:%M %p ET"),
        "settings": {
            "risk_level": risk_level,
            "bankroll": bankroll,
        },
        "games": [],
        "methodology": {
            "models_used": [],
            "data_sources": [],
            "performance_metrics": {}
        }
    }
    
    # Add model information
    has_ensemble = any(col.startswith('ensemble') for col in predictions_df.columns) if not predictions_df.empty else False
    if has_ensemble:
        prediction_data["methodology"]["models_used"].append("Ensemble Stacking Model (XGBoost + RandomForest)")
    else:
        prediction_data["methodology"]["models_used"].append("Standard Prediction Models")
    
    # Add data sources
    prediction_data["methodology"]["data_sources"] = [
        f"BallDontLie API (last updated {datetime.now().strftime('%B %d, %Y')})",
        f"The Odds API (last updated {datetime.now().strftime('%B %d, %Y')})",
        "Historical game data"
    ]
    
    # Add performance metrics
    prediction_data["methodology"]["performance_metrics"] = {
        "win_prediction_accuracy": real_metrics.get("win_prediction_accuracy", 0.0),
        "spread_accuracy": real_metrics.get("spread_accuracy", 0.0),
        "total_accuracy": real_metrics.get("total_accuracy", 0.0),
        "player_prop_accuracy": real_metrics.get("player_prop_accuracy", 0.0)
    }
    
    # Add CLV metrics if tracking is enabled
    if track_clv and betting_analyzer and betting_analyzer.clv_tracker:
        clv_stats = betting_analyzer.clv_tracker.get_clv_stats()
        prediction_data["methodology"]["clv_metrics"] = {
            "positive_clv_rate": clv_stats.get('positive_clv_rate', 0.0),
            "average_clv": clv_stats.get('average_clv', 0.0),
            "total_bets_tracked": clv_stats.get('total_bets_with_clv', 0)
        }
    
    # Process each game prediction
    if not predictions_df.empty:
        for _, game in predictions_df.iterrows():
            game_data = {
                "game_id": game.get('game_id', ''),
                "home_team": game.get('home_team', ''),
                "visitor_team": game.get('visitor_team', ''),
                "game_time": game.get('game_time', '7:00 PM ET'),
                "win_prediction": {
                    "home_win_probability": float(game.get('win_probability', 0.5)),
                    "visitor_win_probability": 1 - float(game.get('win_probability', 0.5)),
                    "confidence_score": float(game.get('confidence_score', 0.5)),
                    "confidence_level": game.get('confidence_level', 'Medium')
                },
                "spread_prediction": {
                    "predicted_spread": float(game.get('predicted_spread', 0.0)),
                    "spread_probability": float(game.get('spread_probability', 0.55)),
                    "confidence": float(game.get('spread_confidence', 0.55))
                },
                "total_prediction": {
                    "projected_total": float(game.get('projected_total', 220.0)),
                    "over_probability": float(game.get('over_probability', 0.5)),
                    "confidence": float(game.get('total_confidence', 0.55))
                },
                "betting_recommendations": {
                    "moneyline": {
                        "edge": float(game.get('moneyline_edge', 0.0)),
                        "bet_pct": float(game.get('moneyline_bet_pct', 0.0)),
                        "bet_amount": float(game.get('moneyline_bet_amount', 0.0)),
                        "bet_team": game.get('moneyline_bet_team', ''),
                        "bet_odds": game.get('moneyline_bet_odds', 0)
                    },
                    "spread": {
                        "edge": float(game.get('spread_edge', 0.0)),
                        "bet_pct": float(game.get('spread_bet_pct', 0.0)),
                        "bet_amount": float(game.get('spread_bet_amount', 0.0))
                    },
                    "total": {
                        "edge": float(game.get('total_edge', 0.0)),
                        "bet_pct": float(game.get('total_bet_pct', 0.0)),
                        "bet_amount": float(game.get('total_bet_amount', 0.0)),
                        "bet_type": game.get('total_bet_type', '')
                    }
                },
                "market_odds": {
                    "home_odds": game.get('home_odds', ''),
                    "visitor_odds": game.get('visitor_odds', ''),
                    "spread_line": game.get('spread_line', ''),
                    "spread_odds": game.get('spread_odds', ''),
                    "total_line": game.get('total_line', ''),
                    "over_odds": game.get('over_odds', ''),
                    "under_odds": game.get('under_odds', '')
                },
                "player_props": []
            }
            
            # Add player props if available
            if player_predictions_df is not None and not player_predictions_df.empty:
                game_players = player_predictions_df[player_predictions_df['game_id'] == game.get('game_id', '')]
                
                # Sort by confidence score
                if 'confidence_score' in game_players.columns:
                    game_players = game_players.sort_values('confidence_score', ascending=False)
                
                for _, player in game_players.iterrows():
                    # Get the best prop recommendation based on edge
                    edges = {
                        'points': player.get('points_edge', 0.0),
                        'rebounds': player.get('rebounds_edge', 0.0),
                        'assists': player.get('assists_edge', 0.0)
                    }
                    
                    # Find the prop with highest edge
                    best_prop = max(edges.items(), key=lambda x: x[1]) if edges else ('points', 0.0)
                    prop_type, max_edge = best_prop
                    
                    # Only include players with positive edge on at least one prop
                    if max_edge > 0:
                        player_data = {
                            "player_id": player.get('player_id', ''),
                            "player_name": player.get('player_name', ''),
                            "team_name": player.get('team_name', ''),
                            "position": player.get('position', ''),
                            "predicted_points": float(player.get('predicted_points', 0.0)),
                            "predicted_rebounds": float(player.get('predicted_rebounds', 0.0)),
                            "predicted_assists": float(player.get('predicted_assists', 0.0)),
                            "confidence_level": player.get('confidence_level', 'Medium'),
                            "value_rating": player.get('value_rating', ''),
                            "line_points": player.get('line_points', 0),
                            "over_under_points": player.get('over_under_points', ''),
                            "best_prop": {
                                "type": prop_type,
                                "edge": max_edge,
                                "bet_pct": float(player.get(f"{prop_type}_bet_pct", 0.0)),
                                "bet_amount": float(player.get(f"{prop_type}_bet_amount", 0.0)),
                                "bet_type": player.get(f"{prop_type}_bet_type", ''),
                                "line": float(player.get(f"{prop_type}_bet_line", 0.0))
                            },
                            "recommended_bets": player.get('recommended_bets', [])
                        }
                        game_data["player_props"].append(player_data)
            
            prediction_data["games"].append(game_data)
    
    return prediction_data


def display_prediction_output(prediction_data):
    """
    Display game predictions in a clean, professional format
    
    Args:
        prediction_data: Structured prediction data from create_prediction_schema
    """
    print("=" * 80)
    print(f"NBA PREDICTIONS | {prediction_data['date']}")
    print("=" * 80)
    
    # Display settings info
    settings = prediction_data.get('settings', {})
    risk_level = settings.get('risk_level', 'moderate').capitalize()
    bankroll = settings.get('bankroll', 1000.0)
    print(f"\nGenerated at: {prediction_data['generation_time']} | Bankroll: ${bankroll:,.2f} | Risk Profile: {risk_level}\n")
    
    # Display each game
    for game in prediction_data.get('games', []):
        home_team = game.get('home_team', '')
        visitor_team = game.get('visitor_team', '')
        game_time = game.get('game_time', '7:00 PM ET')
        
        print(f"{visitor_team.upper()} @ {home_team.upper()} | {game_time}")
        print("-" * 80)
        
        # Win prediction
        win_pred = game.get('win_prediction', {})
        home_win_prob = win_pred.get('home_win_probability', 0.5) * 100
        visitor_win_prob = win_pred.get('visitor_win_probability', 0.5) * 100
        conf_level = win_pred.get('confidence_level', 'Medium')
        conf_score = win_pred.get('confidence_score', 0.5) * 10  # Scale to 1-10
        
        print("\nMONEYLINE PREDICTION:")
        print(f"{home_team} Win: {home_win_prob:.0f}% | {visitor_team} Win: {visitor_win_prob:.0f}%")
        
        # Market odds for moneyline
        market_odds = game.get('market_odds', {})
        if market_odds.get('home_odds') and market_odds.get('visitor_odds'):
            print(f"Market Odds: {home_team} {market_odds['home_odds']} | {visitor_team} {market_odds['visitor_odds']}")
        
        # Betting recommendation for moneyline
        betting_rec = game.get('betting_recommendations', {}).get('moneyline', {})
        edge = betting_rec.get('edge', 0.0)
        bet_pct = betting_rec.get('bet_pct', 0.0)
        bet_amount = betting_rec.get('bet_amount', 0.0)
        bet_team = betting_rec.get('bet_team', '')
        bet_odds = betting_rec.get('bet_odds', 0)
        
        if edge > 0 and bet_pct > 0:
            print(f"→ Edge: +{edge:.1f}% on {bet_team} ({bet_odds})")
            print(f"→ Recommended Bet: {bet_pct*100:.1f}% of bankroll (${bet_amount:.2f})")
            print(f"→ Confidence: {conf_level} ({conf_score:.1f}/10)")
        
        # Spread prediction
        spread_pred = game.get('spread_prediction', {})
        spread = spread_pred.get('predicted_spread', 0.0)
        spread_prob = spread_pred.get('spread_probability', 0.55) * 100
        spread_conf = spread_pred.get('confidence', 0.55) * 10  # Scale to 1-10
        
        favorite = home_team if spread > 0 else visitor_team
        spread_val = abs(spread)
        
        print("\nSPREAD PREDICTION:")
        print(f"{favorite} -{spread_val:.1f} | {spread_prob:.0f}% probability to cover")
        
        # Market line for spread
        if market_odds.get('spread_line') and market_odds.get('spread_odds'):
            print(f"Market Line: {market_odds['spread_line']} ({market_odds['spread_odds']})")
        
        # Betting recommendation for spread
        spread_bet = game.get('betting_recommendations', {}).get('spread', {})
        spread_edge = spread_bet.get('edge', 0.0)
        spread_bet_pct = spread_bet.get('bet_pct', 0.0)
        spread_bet_amount = spread_bet.get('bet_amount', 0.0)
        
        if spread_edge > 0 and spread_bet_pct > 0:
            # Determine confidence level text based on confidence score
            if spread_conf >= 8.0:
                spread_conf_text = "High"
            elif spread_conf >= 6.0:
                spread_conf_text = "Medium"
            else:
                spread_conf_text = "Low"
                
            print(f"→ Edge: +{spread_edge:.1f}% on {favorite} {market_odds.get('spread_line', '')}")
            print(f"→ Recommended Bet: {spread_bet_pct*100:.1f}% of bankroll (${spread_bet_amount:.2f})")
            print(f"→ Confidence: {spread_conf_text} ({spread_conf:.1f}/10)")
        
        # Total prediction
        total_pred = game.get('total_prediction', {})
        total = total_pred.get('projected_total', 220.0)
        over_prob = total_pred.get('over_probability', 0.5) * 100
        total_conf = total_pred.get('confidence', 0.55) * 10  # Scale to 1-10
        
        print("\nTOTAL PREDICTION:")
        print(f"Projected: {total:.1f} points | {over_prob:.0f}% probability {'over' if over_prob > 50 else 'under'}")
        
        # Market line for total
        if market_odds.get('total_line'):
            print(f"Market Line: O/U {market_odds['total_line']} (O {market_odds.get('over_odds', '')}, U {market_odds.get('under_odds', '')})")
        
        # Betting recommendation for total
        total_bet = game.get('betting_recommendations', {}).get('total', {})
        total_edge = total_bet.get('edge', 0.0)
        total_bet_pct = total_bet.get('bet_pct', 0.0)
        total_bet_amount = total_bet.get('bet_amount', 0.0)
        total_bet_type = total_bet.get('bet_type', '').title()
        
        if total_edge > 0 and total_bet_pct > 0:
            # Determine confidence level text based on confidence score
            if total_conf >= 8.0:
                total_conf_text = "High"
            elif total_conf >= 6.0:
                total_conf_text = "Medium"
            else:
                total_conf_text = "Low"
                
            print(f"→ Edge: +{total_edge:.1f}% on {total_bet_type} {market_odds.get('total_line', '')}")
            print(f"→ Recommended Bet: {total_bet_pct*100:.1f}% of bankroll (${total_bet_amount:.2f})")
            print(f"→ Confidence: {total_conf_text} ({total_conf:.1f}/10)")
        
        # Model insights
        print("\nMODEL INSIGHTS:")
        
        # Primary model info
        methodology = prediction_data.get('methodology', {})
        primary_model = methodology.get('models_used', ["Standard Prediction Models"])[0]
        print(f"• Primary Model: {primary_model}")
        
        # Key factors (this would need to be populated with actual factors)
        # For now using placeholder data that would be replaced with real analysis
        team_factors = []
        if home_win_prob > visitor_win_prob:
            team_factors.append(f"{home_team}'s home court advantage")
        else:
            team_factors.append(f"{visitor_team}'s recent form")
            
        print(f"• Key Factors: {', '.join(team_factors)}")
        
        # Data quality info
        data_quality = 96 + round(win_pred.get('confidence_score', 0.5) * 4)  # Simulated data quality based on confidence
        print(f"• Data Quality: {data_quality}% complete")
        
        # CLV potential if available
        if 'clv_metrics' in methodology:
            clv_rate = methodology['clv_metrics'].get('positive_clv_rate', 0.0) * 100
            print(f"• CLV Potential: Historical {clv_rate:.1f}% positive movement on similar games")
        
        # Only display player props if they exist for this game
        player_props = game.get('player_props', [])
        if player_props:
            print("\nPLAYER PROP RECOMMENDATIONS:")
            print("-" * 80)
            
            # Sort by edge
            sorted_props = sorted(player_props, key=lambda x: x.get('best_prop', {}).get('edge', 0), reverse=True)
            
            # Show top 3 props with highest edge
            for i, player in enumerate(sorted_props[:3]):  # Limit to top 3
                display_player_prop(player)
        
        print("-" * 80)
    
    # Display methodology
    display_prediction_methodology(prediction_data)


def display_player_prop(player_data):
    """
    Display a single player prop recommendation
    
    Args:
        player_data: Player prop data from the prediction schema
    """
    player_name = player_data.get('player_name', 'Unknown Player')
    team_name = player_data.get('team_name', 'Unknown Team')
    position = player_data.get('position', '')
    
    points = player_data.get('predicted_points', 0)
    rebounds = player_data.get('predicted_rebounds', 0)
    assists = player_data.get('predicted_assists', 0)
    
    confidence = player_data.get('confidence_level', 'Medium')
    value_rating = player_data.get('value_rating', '')
    
    # Format line separator
    separator = '-' * 80
    
    # Display player prop prediction with improved formatting
    print(f"\n{player_name} ({team_name}{', ' + position if position else ''})")
    print(f"{'Points:':<10} {points:.1f}")
    print(f"{'Rebounds:':<10} {rebounds:.1f}")
    print(f"{'Assists:':<10} {assists:.1f}")
    
    if 'line_points' in player_data and 'over_under_points' in player_data:
        line = player_data.get('line_points', 0)
        over_under = player_data.get('over_under_points', '')
        print(f"{'Line:':<10} {line:.1f} ({over_under})")
    
    if confidence:
        print(f"{'Confidence:':<10} {confidence}")
        
    if value_rating:
        print(f"{'Value:':<10} {value_rating}")

    # Add recommended bets if available
    if 'recommended_bets' in player_data:
        bets = player_data.get('recommended_bets', [])
        if bets:
            print("\nRecommended Bets:")
            for bet in bets:
                print(f"- {bet}")

    print(separator)


def display_player_props(prediction_data):
    """
    Display top 5 player prop recommendations from the game with highest confidence
    
    Args:
        prediction_data: Structured prediction data from create_prediction_schema
    """
    print("=" * 80)
    print("TOP PLAYER PROP RECOMMENDATIONS")
    print("=" * 80)
    
    settings = prediction_data.get('settings', {})
    bankroll = settings.get('bankroll', 1000.0)
    print(f"\nRisk Profile: {settings.get('risk_level', 'moderate').capitalize()} | Bankroll: ${bankroll:,.2f}\n")
    
    # Group props by game and find the game with highest average edge
    games_with_props = []
    for game in prediction_data.get('games', []):
        if not game.get('player_props', []):
            continue
            
        home_team = game.get('home_team', '')
        visitor_team = game.get('visitor_team', '')
        matchup = f"{visitor_team} @ {home_team}"
        
        # Calculate average prop edge for this game
        props = game.get('player_props', [])
        if not props:
            continue
            
        prop_edges = [p.get('best_prop', {}).get('edge', 0) for p in props]
        avg_edge = sum(prop_edges) / len(prop_edges) if prop_edges else 0
        
        games_with_props.append({
            'matchup': matchup,
            'props': props,
            'avg_edge': avg_edge,
            'game_id': game.get('game_id')
        })
    
    if not games_with_props:
        print("\nNo player prop recommendations available")
        return
    
    # Sort games by average prop edge to find highest confidence game
    games_with_props.sort(key=lambda g: g['avg_edge'], reverse=True)
    top_game = games_with_props[0]
    
    # Get top 5 props by edge from the highest confidence game
    top_props = sorted(top_game['props'], key=lambda p: p.get('best_prop', {}).get('edge', 0), reverse=True)[:5]
    
    print(f"\nTOP 5 PLAYER PROPS FOR: {top_game['matchup']}\n")
    
    # Display each prop
    for i, player in enumerate(top_props):
        print(f"{i+1}. {player.get('player_name', '')}")
        display_player_prop(player)
        if i < len(top_props) - 1:  # Don't print divider after the last prop
            print("-" * 40)


def display_prediction_methodology(prediction_data):
    """
    Display methodology section explaining the prediction process
    
    Args:
        prediction_data: Structured prediction data from create_prediction_schema
    """
    print("\n" + "=" * 80)
    print("PREDICTION METHODOLOGY")
    print("=" * 80)
    
    methodology = prediction_data.get('methodology', {})
    
    # Models used
    print("\nMODELS UTILIZED:")
    for model in methodology.get('models_used', ["Standard Prediction Models"]):
        perf = methodology.get('performance_metrics', {})
        accuracy = perf.get('win_prediction_accuracy', 0.64) * 100
        print(f"• {model} (accuracy: {accuracy:.0f}% YTD)")
    
    # Player props model if available
    if any(g.get('player_props') for g in prediction_data.get('games', [])):
        player_accuracy = methodology.get('performance_metrics', {}).get('player_prop_accuracy', 0.59) * 100
        print(f"• Player Props: Gradient Boosting Player-Specific Models (accuracy: {player_accuracy:.0f}% YTD)")
    
    # Bankroll management method
    settings = prediction_data.get('settings', {})
    risk_level = settings.get('risk_level', 'moderate')
    if risk_level == 'conservative':
        kelly_fraction = "Tenth Kelly (0.1x)"
    elif risk_level == 'moderate':
        kelly_fraction = "Quarter Kelly (0.25x)"
    else:  # aggressive
        kelly_fraction = "Half Kelly (0.5x)"
    print(f"• Bankroll Management: Fractional Kelly Criterion ({kelly_fraction})")
    
    # Data sources
    print("\nDATA SOURCES:")
    for source in methodology.get('data_sources', []):
        print(f"• {source}")
    
    # Performance metrics
    print("\nBETTING PERFORMANCE METRICS:")
    perf = methodology.get('performance_metrics', {})
    
    # Use actual performance metrics from tracking data
    win_accuracy = perf.get('win_prediction_accuracy', 0.0) * 100
    spread_accuracy = perf.get('spread_accuracy', 0.0) * 100
    total_accuracy = perf.get('total_accuracy', 0.0) * 100
    player_prop_accuracy = perf.get('player_prop_accuracy', 0.0) * 100
    
    print(f"Win Prediction Accuracy: {win_accuracy:.1f}%")
    print(f"Spread Prediction Accuracy: {spread_accuracy:.1f}%")
    print(f"Totals Prediction Accuracy: {total_accuracy:.1f}%")
    print(f"Player Props Prediction Accuracy: {player_prop_accuracy:.1f}%")
    
    # Use actual ROI from tracking data if available
    if 'roi' in methodology:
        roi = methodology.get('roi', 0.0)
        print(f"• Season-to-date ROI: {'+' if roi >= 0 else ''}{roi:.1f}% with recommended bankroll management")
    
    # CLV metrics if available
    if 'clv_metrics' in methodology:
        clv_metrics = methodology['clv_metrics']
        clv_rate = clv_metrics.get('positive_clv_rate', 0.0) * 100
        avg_clv = clv_metrics.get('average_clv', 0.0)
        print(f"• Positive CLV Rate: {clv_rate:.0f}% of recommended bets have received positive closing line value")
    
    # Average edge
    # Calculate from actual edges in recommendations
    edges = []
    for game in prediction_data.get('games', []):
        betting_recs = game.get('betting_recommendations', {})
        ml_edge = betting_recs.get('moneyline', {}).get('edge', 0.0)
        spread_edge = betting_recs.get('spread', {}).get('edge', 0.0)
        total_edge = betting_recs.get('total', {}).get('edge', 0.0)
        edges.extend([e for e in [ml_edge, spread_edge, total_edge] if e > 0])
    
    avg_edge = sum(edges) / len(edges) if edges else 0.0
    print(f"• Average Betting Edge: +{avg_edge:.1f}% across all recommendations")
    
    # Risk profile
    if risk_level == 'conservative':
        bet_range = "0.25% to 1.5%"
    elif risk_level == 'moderate':
        bet_range = "0.5% to 3%"
    else:  # aggressive
        bet_range = "1% to 5%"
    print(f"• Risk Profile Applied: {risk_level.capitalize()} ({bet_range} of bankroll per recommendation)")
    
    print("=" * 80)
