#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Engineering Module

This module handles the preparation of game features for NBA prediction models.
It includes comprehensive feature engineering without fallbacks to placeholder data.

Author: Cascade
Date: April 2025
"""

import logging
import traceback
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)


def prepare_game_features(games: List[Dict], team_stats: Dict, odds: Dict, historical_games: Optional[List[Dict]] = None) -> pd.DataFrame:
    """
    Prepare features for game predictions with comprehensive error handling and no fallbacks
    
    Args:
        games: List of game dictionaries from the API
        team_stats: Dictionary of team statistics by team ID
        odds: Dictionary of betting odds by game ID
        historical_games: Optional list of historical games for team form calculation
        
    Returns:
        pandas.DataFrame: Game features for prediction
        
    Raises:
        ValueError: If there is insufficient data to prepare features
    """
    try:
        logger.info("Preparing game features for prediction")
        
        if not games:
            raise ValueError("No games provided for feature preparation")
        
        # Initialize features list
        features = []
        
        # Extract game features
        for game in games:
            try:
                # Extract team IDs
                home_team = game.get("home_team", {})
                visitor_team = game.get("visitor_team", {})
                
                if not home_team or not visitor_team:
                    raise ValueError(f"Missing team data for game {game.get('id')}")
                
                home_id = home_team.get("id")
                visitor_id = visitor_team.get("id")
                
                if not home_id or not visitor_id:
                    raise ValueError(f"Missing team IDs for game {game.get('id')}")
                
                # Extract team statistics
                if home_id not in team_stats:
                    raise ValueError(f"No statistics available for home team: {home_team.get('full_name')} (ID: {home_id})")
                    
                if visitor_id not in team_stats:
                    raise ValueError(f"No statistics available for visitor team: {visitor_team.get('full_name')} (ID: {visitor_id})")
                
                home_stats = team_stats[home_id].get("stats", {})
                visitor_stats = team_stats[visitor_id].get("stats", {})
                
                if not home_stats:
                    raise ValueError(f"Missing statistics for home team: {home_team.get('full_name')}")
                
                if not visitor_stats:
                    raise ValueError(f"Missing statistics for visitor team: {visitor_team.get('full_name')}")
                
                # Calculate team form if historical games are available
                home_form = calculate_team_form(home_id, historical_games) if historical_games else {}
                visitor_form = calculate_team_form(visitor_id, historical_games) if historical_games else {}
                
                # Get betting odds for the game
                game_odds = odds.get(game.get("id"), {})
                
                # Extract market-implied probabilities
                market_data = extract_market_data(game_odds, home_team.get("full_name"), visitor_team.get("full_name"))
                
                # Calculate defensive metrics
                home_defensive_metrics = calculate_defensive_metrics(home_id, team_stats)
                visitor_defensive_metrics = calculate_defensive_metrics(visitor_id, team_stats)
                
                # Only proceed if we have all required data
                if not home_defensive_metrics or not visitor_defensive_metrics:
                    logger.warning(f"Insufficient defensive metrics for game {game.get('id')}. Skipping.")
                    continue
                
                # Assemble feature dictionary
                game_features = {
                    # Game metadata
                    "game_id": game.get("id"),
                    "date": game.get("date"),
                    "home_team": home_team.get("full_name"),
                    "visitor_team": visitor_team.get("full_name"),
                    "season": game.get("season"),
                    
                    # Team offensive stats
                    "home_pts": float(home_stats.get("pts", 0)),
                    "visitor_pts": float(visitor_stats.get("pts", 0)),
                    "home_fg_pct": float(home_stats.get("fg_pct", 0)),
                    "visitor_fg_pct": float(visitor_stats.get("fg_pct", 0)),
                    "home_fg3_pct": float(home_stats.get("fg3_pct", 0)),
                    "visitor_fg3_pct": float(visitor_stats.get("fg3_pct", 0)),
                    "home_ft_pct": float(home_stats.get("ft_pct", 0)),
                    "visitor_ft_pct": float(visitor_stats.get("ft_pct", 0)),
                    "home_ast": float(home_stats.get("ast", 0)),
                    "visitor_ast": float(visitor_stats.get("ast", 0)),
                    
                    # Team defensive stats
                    "home_reb": float(home_stats.get("reb", 0)),
                    "visitor_reb": float(visitor_stats.get("reb", 0)),
                    "home_stl": float(home_stats.get("stl", 0)),
                    "visitor_stl": float(visitor_stats.get("stl", 0)),
                    "home_blk": float(home_stats.get("blk", 0)),
                    "visitor_blk": float(visitor_stats.get("blk", 0)),
                    "home_turnover": float(home_stats.get("turnover", 0)),
                    "visitor_turnover": float(visitor_stats.get("turnover", 0)),
                    "home_pf": float(home_stats.get("pf", 0)),
                    "visitor_pf": float(visitor_stats.get("pf", 0)),
                    
                    # Defensive metrics
                    "home_defensive_rating": home_defensive_metrics.get("defensive_rating"),
                    "visitor_defensive_rating": visitor_defensive_metrics.get("defensive_rating"),
                    "home_opp_pts_per_game": home_defensive_metrics.get("opp_pts_per_game"),
                    "visitor_opp_pts_per_game": visitor_defensive_metrics.get("opp_pts_per_game"),
                    
                    # Team records
                    "home_wins": float(home_stats.get("w", 0)),
                    "home_losses": float(home_stats.get("l", 0)),
                    "visitor_wins": float(visitor_stats.get("w", 0)),
                    "visitor_losses": float(visitor_stats.get("l", 0)),
                    
                    # Home court advantage
                    "is_home_court": 1  # Always 1 since home vs visitor format
                }
                
                # Add team form data if available
                if home_form:
                    game_features.update({
                        "home_form_win_pct": home_form.get("win_percentage", 0),
                        "home_recent_pts": home_form.get("avg_points_scored", 0),
                        "home_recent_pts_allowed": home_form.get("avg_points_allowed", 0),
                        "home_win_streak": home_form.get("win_streak", 0)
                    })
                
                if visitor_form:
                    game_features.update({
                        "visitor_form_win_pct": visitor_form.get("win_percentage", 0),
                        "visitor_recent_pts": visitor_form.get("avg_points_scored", 0),
                        "visitor_recent_pts_allowed": visitor_form.get("avg_points_allowed", 0),
                        "visitor_win_streak": visitor_form.get("win_streak", 0)
                    })
                
                # Add market data if available
                if market_data:
                    game_features.update({
                        "implied_home_win_prob": market_data.get("home_win_probability", 0.5),
                        "implied_total": market_data.get("total", 0),
                        "market_spread": market_data.get("spread", 0)
                    })
                
                # Add derived features
                # Win percentage
                home_games = game_features["home_wins"] + game_features["home_losses"]
                visitor_games = game_features["visitor_wins"] + game_features["visitor_losses"]
                
                if home_games > 0:
                    game_features["home_win_pct"] = game_features["home_wins"] / home_games
                else:
                    # If no games played, we can't calculate a win percentage
                    # Rather than using a placeholder, leave as missing to trigger ValueError
                    raise ValueError(f"Home team {home_team.get('full_name')} has no games played")
                
                if visitor_games > 0:
                    game_features["visitor_win_pct"] = game_features["visitor_wins"] / visitor_games
                else:
                    # If no games played, we can't calculate a win percentage
                    # Rather than using a placeholder, leave as missing to trigger ValueError
                    raise ValueError(f"Visitor team {visitor_team.get('full_name')} has no games played")
                
                # Append to features list
                features.append(game_features)
                
            except ValueError as e:
                # Log the error and skip this game
                logger.warning(f"Skipping game due to data issue: {str(e)}")
                continue
            except Exception as e:
                # Log unexpected errors and skip this game
                logger.warning(f"Unexpected error preparing features for game: {str(e)}")
                logger.debug(traceback.format_exc())
                continue
        
        # Create DataFrame from features
        if not features:
            raise ValueError("No valid games with sufficient features for prediction")
            
        features_df = pd.DataFrame(features)
        logger.info(f"Prepared features for {len(features_df)} games")
        
        return features_df
        
    except ValueError as e:
        # Re-raise ValueError for caller to handle
        raise
    except Exception as e:
        # Log unexpected errors and raise a more informative exception
        logger.error(f"Error in prepare_game_features: {str(e)}")
        logger.debug(traceback.format_exc())
        raise ValueError(f"Failed to prepare game features: {str(e)}")


def calculate_team_form(team_id: int, historical_games: List[Dict], n_games: int = 10) -> Dict:
    """
    Calculate team form based on recent performance
    
    Args:
        team_id: ID of the team to calculate form for
        historical_games: List of historical games to analyze
        n_games: Number of recent games to consider
        
    Returns:
        Dict: Dictionary containing team form metrics
        
    Raises:
        ValueError: If there is insufficient data to calculate form
    """
    # Sort games by date (most recent first)
    team_games = [g for g in historical_games if 
                (g.get("home_team", {}).get("id") == team_id or 
                 g.get("visitor_team", {}).get("id") == team_id) and
                 g.get("status") == "Final"]
    
    if not team_games:
        raise ValueError(f"No historical games found for team {team_id}")
    
    # Sort by date
    sorted_games = sorted(team_games, key=lambda g: g.get("date", ""), reverse=True)
    
    # Take only the N most recent games
    recent_games = sorted_games[:n_games]
    
    if len(recent_games) < 3:  # Need at least 3 games for meaningful form
        raise ValueError(f"Insufficient recent games ({len(recent_games)}) for team {team_id}")
    
    # Initialize metrics
    wins = 0
    losses = 0
    points_scored = []
    points_allowed = []
    win_streak = 0
    current_streak = 0
    
    # Analyze each game
    for game in recent_games:
        home_team = game.get("home_team", {})
        visitor_team = game.get("visitor_team", {})
        
        # Determine if team is home or away
        is_home = home_team.get("id") == team_id
        team_score = game.get("home_team_score" if is_home else "visitor_team_score")
        opponent_score = game.get("visitor_team_score" if is_home else "home_team_score")
        
        if team_score is None or opponent_score is None:
            logger.warning(f"Missing score data for game {game.get('id')}")
            continue
        
        # Record points
        points_scored.append(team_score)
        points_allowed.append(opponent_score)
        
        # Record win/loss
        if team_score > opponent_score:
            wins += 1
            current_streak = max(1, current_streak + 1)  # Increment win streak
        else:
            losses += 1
            current_streak = min(-1, current_streak - 1)  # Decrement (negative = loss streak)
        
        # Update win streak
        win_streak = max(win_streak, current_streak)
    
    # Calculate form metrics
    games_analyzed = len(points_scored)
    if games_analyzed == 0:
        raise ValueError(f"No valid games with scores for team {team_id}")
    
    win_percentage = wins / games_analyzed
    avg_points_scored = sum(points_scored) / games_analyzed
    avg_points_allowed = sum(points_allowed) / games_analyzed
    
    form_data = {
        "wins": wins,
        "losses": losses,
        "win_percentage": win_percentage,
        "avg_points_scored": avg_points_scored,
        "avg_points_allowed": avg_points_allowed,
        "win_streak": win_streak,
        "games_analyzed": games_analyzed
    }
    
    return form_data


def calculate_defensive_metrics(team_id: int, team_stats: Dict) -> Dict:
    """
    Calculate defensive metrics for a team
    
    Args:
        team_id: ID of the team to calculate metrics for
        team_stats: Dictionary of team statistics
        
    Returns:
        Dict: Dictionary of defensive metrics
        
    Raises:
        ValueError: If there is insufficient data to calculate metrics
    """
    if team_id not in team_stats:
        raise ValueError(f"No team stats found for team ID {team_id}")
    
    team_data = team_stats[team_id]
    team_stats_data = team_data.get("stats", {})
    
    if not team_stats_data:
        raise ValueError(f"No statistics available for team ID {team_id}")
    
    # Initialize metrics dictionary
    metrics = {}
    
    # If API provides defensive rating directly, use it
    if "defensive_rating" in team_stats_data:
        metrics["defensive_rating"] = float(team_stats_data["defensive_rating"])
    else:
        # We need to calculate a defensive rating
        # Check if we have the necessary data
        required_fields = ["opp_pts", "games_played"]
        missing_fields = [f for f in required_fields if f not in team_stats_data]
        
        if missing_fields:
            raise ValueError(f"Missing required fields for defensive calculation: {', '.join(missing_fields)}")
        
        # Calculate defensive rating as points allowed per game
        opp_pts = float(team_stats_data["opp_pts"])
        games = float(team_stats_data["games_played"])
        
        if games <= 0:
            raise ValueError(f"Invalid games_played value ({games}) for team ID {team_id}")
        
        metrics["defensive_rating"] = opp_pts / games
    
    # If API provides opponent points per game directly, use it
    if "opp_pts_per_game" in team_stats_data:
        metrics["opp_pts_per_game"] = float(team_stats_data["opp_pts_per_game"])
    else:
        # Calculate from available data
        if "opp_pts" in team_stats_data and "games_played" in team_stats_data:
            opp_pts = float(team_stats_data["opp_pts"])
            games = float(team_stats_data["games_played"])
            
            if games <= 0:
                raise ValueError(f"Invalid games_played value ({games}) for team ID {team_id}")
            
            metrics["opp_pts_per_game"] = opp_pts / games
        else:
            raise ValueError(f"Insufficient data to calculate opponent points per game for team ID {team_id}")
    
    return metrics


def extract_market_data(odds: Dict, home_team_name: str, visitor_team_name: str) -> Dict:
    """
    Extract market-implied data from betting odds
    
    Args:
        odds: Dictionary of betting odds for a game
        home_team_name: Name of the home team
        visitor_team_name: Name of the visitor team
        
    Returns:
        Dict: Dictionary with market-implied probabilities, spreads, and totals
    """
    if not odds or not odds.get("bookmakers"):
        logger.debug(f"No odds data available for {home_team_name} vs {visitor_team_name}")
        return {}
    
    # Initialize market data
    market_data = {
        "home_win_probability": 0.5,  # Default to even probability
        "spread": 0.0,  # Default to no spread
        "total": 0.0  # Default to no total
    }
    
    # Process bookmaker data
    for bookmaker in odds.get("bookmakers", []):
        # Get moneyline (h2h) market
        h2h_market = {}
        for market in bookmaker.get("markets", []):
            if market.get("key") == "h2h":
                h2h_market = market
                break
        
        if h2h_market and h2h_market.get("outcomes"):
            for outcome in h2h_market.get("outcomes", []):
                if outcome.get("name").lower() == home_team_name.lower():
                    # Convert American odds to probability
                    odds_value = outcome.get("price")
                    if odds_value:
                        if odds_value > 0:
                            probability = 100 / (odds_value + 100)
                        else:
                            probability = abs(odds_value) / (abs(odds_value) + 100)
                        market_data["home_win_probability"] = probability
        
        # Get spread market
        spread_market = {}
        for market in bookmaker.get("markets", []):
            if market.get("key") == "spreads":
                spread_market = market
                break
        
        if spread_market and spread_market.get("outcomes"):
            for outcome in spread_market.get("outcomes", []):
                if outcome.get("name").lower() == home_team_name.lower():
                    spread_value = outcome.get("point")
                    if spread_value is not None:
                        market_data["spread"] = float(spread_value)
        
        # Get totals market
        totals_market = {}
        for market in bookmaker.get("markets", []):
            if market.get("key") == "totals":
                totals_market = market
                break
        
        if totals_market and totals_market.get("outcomes"):
            for outcome in totals_market.get("outcomes", []):
                if outcome.get("name").lower() == "over":
                    total_value = outcome.get("point")
                    if total_value is not None:
                        market_data["total"] = float(total_value)
    
    return market_data
