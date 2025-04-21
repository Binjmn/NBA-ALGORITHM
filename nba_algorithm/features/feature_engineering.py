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

# Import modules from models directory
from ..models.injury_analysis import get_team_injury_data, compare_team_injuries
from ..models.advanced_metrics import (
    fetch_team_advanced_metrics, 
    fetch_player_advanced_metrics,
    get_team_efficiency_comparison
)

# Import advanced feature engineering
from .advanced_features import (
    create_momentum_features,
    create_matchup_features,
    integrate_advanced_features
)

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
        skipped_games = []
        
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
                
                # Our team_stats structure has stats directly in the team dict, not in a nested 'stats' key
                home_team_data = team_stats[home_id]
                visitor_team_data = team_stats[visitor_id]
                
                # Log the stats structure to help with debugging
                logger.debug(f"Home team stats keys: {list(home_team_data.keys())}")
                logger.debug(f"Visitor team stats keys: {list(visitor_team_data.keys())}")
                
                if not home_team_data:
                    raise ValueError(f"Missing statistics for home team: {home_team.get('full_name')}")
                
                if not visitor_team_data:
                    raise ValueError(f"Missing statistics for visitor team: {visitor_team.get('full_name')}")
                
                # Calculate team form if historical games are available
                home_form = None
                visitor_form = None
                
                if historical_games:
                    try:
                        home_form = calculate_team_form(home_id, historical_games) if historical_games else None
                        if not home_form:
                            logger.warning(f"Insufficient historical data for home team: {home_team.get('full_name')}")
                    except Exception as e:
                        logger.warning(f"Error calculating home team form: {str(e)}")
                    
                    try:
                        visitor_form = calculate_team_form(visitor_id, historical_games) if historical_games else None
                        if not visitor_form:
                            logger.warning(f"Insufficient historical data for visitor team: {visitor_team.get('full_name')}")
                    except Exception as e:
                        logger.warning(f"Error calculating visitor team form: {str(e)}")
                else:
                    logger.warning("No historical games available for team form calculation")
                
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
                    skipped_games.append(game)
                    continue
                
                # Get injury data for both teams - make this optional with try/except
                injury_comparison = {
                    'home_impact': {'overall_impact': 0.0, 'key_player_injuries': False},
                    'away_impact': {'overall_impact': 0.0, 'key_player_injuries': False},
                    'injury_advantage': 0.0
                }
                
                try:
                    injury_data = compare_team_injuries(home_id, visitor_id)
                    if injury_data and isinstance(injury_data, dict):
                        injury_comparison = injury_data
                        
                        # Log any key player injuries
                        if injury_comparison.get('home_key_players_injured'):
                            logger.info(f"Home team {home_team.get('full_name')} has key player injuries (impact: {injury_comparison['home_impact']['overall_impact']:.2f})")
                        
                        if injury_comparison.get('away_key_players_injured'):
                            logger.info(f"Away team {visitor_team.get('full_name')} has key player injuries (impact: {injury_comparison['away_impact']['overall_impact']:.2f})")
                except Exception as e:
                    logger.warning(f"Error fetching injury data: {str(e)}. Continuing without injury features.")
                
                # Initialize with default values for advanced metrics
                efficiency_comparison = {
                    'home_efficiency': 0.5,
                    'away_efficiency': 0.5,
                    'offensive_differential': 0.0,
                    'defensive_differential': 0.0,
                    'overall_differential': 0.0
                }
                
                # Get advanced metrics for both teams
                try:
                    metrics_data = get_team_efficiency_comparison(home_id, visitor_id)
                    if metrics_data and isinstance(metrics_data, dict):
                        efficiency_comparison = metrics_data
                    
                        # Log efficiency information
                        logger.info(f"Team efficiency comparison: {home_team.get('full_name')}: {efficiency_comparison['home_efficiency']:.2f}, "
                                    f"{visitor_team.get('full_name')}: {efficiency_comparison['away_efficiency']:.2f}, "
                                    f"Overall differential: {efficiency_comparison['overall_differential']:.2f}")
                except Exception as e:
                    logger.warning(f"Error fetching advanced metrics: {str(e)}. Continuing without advanced metrics features.")
                
                # Initialize with empty dictionaries for momentum features
                home_momentum_features = {}
                away_momentum_features = {}
                
                # Generate advanced momentum features for both teams
                try:
                    if historical_games and len(historical_games) > 5:  # Only attempt if we have sufficient historical data
                        home_data = create_momentum_features(
                            historical_games, home_id, decay_factor=0.85, window_size=10
                        )
                        if home_data:
                            home_momentum_features = home_data
                            if 'win_momentum' in home_momentum_features:
                                logger.info(f"Home team {home_team.get('full_name')} momentum: {home_momentum_features.get('win_momentum', 0):.2f}")
                        
                        away_data = create_momentum_features(
                            historical_games, visitor_id, decay_factor=0.85, window_size=10
                        )
                        if away_data:
                            away_momentum_features = away_data
                            if 'win_momentum' in away_momentum_features:
                                logger.info(f"Away team {visitor_team.get('full_name')} momentum: {away_momentum_features.get('win_momentum', 0):.2f}")
                except Exception as e:
                    logger.warning(f"Error calculating momentum features: {str(e)}. Continuing without momentum features.")
                
                # Initialize with empty dictionary for matchup features
                matchup_features = {}
                
                # Generate matchup-specific historical features
                try:
                    if historical_games and len(historical_games) > 0:  # Only attempt if we have historical data
                        matchup_data = create_matchup_features(
                            historical_games, home_id, visitor_id, max_years_back=4
                        )
                        
                        if matchup_data and isinstance(matchup_data, dict):
                            matchup_features = matchup_data
                            
                            if matchup_features.get('matchup_games_count', 0) > 0:
                                logger.info(f"Historical matchup: {home_team.get('full_name')} vs {visitor_team.get('full_name')} - "
                                           f"Home win rate: {matchup_features.get('home_win_pct', 0):.2f}, "
                                           f"Avg point diff: {matchup_features.get('avg_point_diff', 0):.1f}")
                except Exception as e:
                    logger.warning(f"Error calculating matchup features: {str(e)}. Continuing without matchup features.")
                
                # Assemble feature dictionary
                game_features = {
                    # Game metadata
                    "game_id": game.get("id"),
                    "date": game.get("date"),
                    "home_team": home_team.get("full_name"),
                    "visitor_team": visitor_team.get("full_name"),
                    "season": game.get("season"),
                    
                    # Team offensive stats
                    "home_pts": float(home_team_data.get("pts", 0)),
                    "visitor_pts": float(visitor_team_data.get("pts", 0)),
                    "home_fg_pct": float(home_team_data.get("fg_pct", 0)),
                    "visitor_fg_pct": float(visitor_team_data.get("fg_pct", 0)),
                    "home_fg3_pct": float(home_team_data.get("fg3_pct", 0)),
                    "visitor_fg3_pct": float(visitor_team_data.get("fg3_pct", 0)),
                    "home_ft_pct": float(home_team_data.get("ft_pct", 0)),
                    "visitor_ft_pct": float(visitor_team_data.get("ft_pct", 0)),
                    "home_ast": float(home_team_data.get("ast", 0)),
                    "visitor_ast": float(visitor_team_data.get("ast", 0)),
                    
                    # Team defensive stats
                    "home_reb": float(home_team_data.get("reb", 0)),
                    "visitor_reb": float(visitor_team_data.get("reb", 0)),
                    "home_stl": float(home_team_data.get("stl", 0)),
                    "visitor_stl": float(visitor_team_data.get("stl", 0)),
                    "home_blk": float(home_team_data.get("blk", 0)),
                    "visitor_blk": float(visitor_team_data.get("blk", 0)),
                    "home_turnover": float(home_team_data.get("turnover", 0)),
                    "visitor_turnover": float(visitor_team_data.get("turnover", 0)),
                    "home_pf": float(home_team_data.get("pf", 0)),
                    "visitor_pf": float(visitor_team_data.get("pf", 0)),
                    
                    # Defensive metrics
                    "home_defensive_rating": home_defensive_metrics.get("defensive_rating"),
                    "visitor_defensive_rating": visitor_defensive_metrics.get("defensive_rating"),
                    "home_opp_pts_per_game": home_defensive_metrics.get("opp_pts_per_game"),
                    "visitor_opp_pts_per_game": visitor_defensive_metrics.get("opp_pts_per_game"),
                    
                    # Team records
                    "home_wins": float(home_team_data.get("w", 0)),
                    "home_losses": float(home_team_data.get("l", 0)),
                    "visitor_wins": float(visitor_team_data.get("w", 0)),
                    "visitor_losses": float(visitor_team_data.get("l", 0)),
                    
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
                
                # Add injury data
                game_features.update({
                    "home_injury_impact": injury_comparison['home_impact']['overall_impact'],
                    "home_key_players_injured": injury_comparison['home_key_players_injured'],
                    "away_injury_impact": injury_comparison['away_impact']['overall_impact'],
                    "away_key_players_injured": injury_comparison['away_key_players_injured'],
                    "injury_advantage": injury_comparison['injury_advantage']
                })
                
                # Add advanced metrics
                game_features.update({
                    "home_efficiency": efficiency_comparison['home_efficiency'],
                    "away_efficiency": efficiency_comparison['away_efficiency'],
                    "offensive_differential": efficiency_comparison['offensive_differential'],
                    "defensive_differential": efficiency_comparison['defensive_differential'],
                    "overall_differential": efficiency_comparison['overall_differential']
                })
                
                # Add momentum features
                game_features.update({
                    "home_win_momentum": home_momentum_features.get("win_momentum", 0),
                    "away_win_momentum": away_momentum_features.get("win_momentum", 0)
                })
                
                # Add matchup features
                game_features.update({
                    "matchup_games_count": matchup_features.get("matchup_games_count", 0),
                    "home_matchup_win_pct": matchup_features.get("home_win_pct", 0),
                    "avg_matchup_point_diff": matchup_features.get("avg_point_diff", 0)
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
                skipped_games.append(game)
                continue
            except Exception as e:
                # Log unexpected errors and skip this game
                logger.warning(f"Unexpected error preparing features for game: {str(e)}")
                logger.debug(traceback.format_exc())
                skipped_games.append(game)
                continue
        
        # Check if we have any valid games after all the processing
        if not features:
            logger.error("No valid games with sufficient features for prediction")
            raise ValueError("No valid games with sufficient data for reliable prediction. Refusing to use neutral or random values.")
            
        # Convert features to DataFrame and check for missing values
        features_df = pd.DataFrame(features)
        missing_values = features_df.isnull().sum().sum()
        
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in features DataFrame")
            # Handle numeric and non-numeric columns separately
            numeric_cols = features_df.select_dtypes(include=['number']).columns
            non_numeric_cols = features_df.select_dtypes(exclude=['number']).columns
            
            # Fill numeric columns with mean or 0
            if len(numeric_cols) > 0:
                features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].mean()).fillna(0)
            
            # Fill non-numeric columns with empty string or most common value
            if len(non_numeric_cols) > 0:
                for col in non_numeric_cols:
                    if features_df[col].dtype == 'object':
                        # For object/string columns, use most common value if available, else empty string
                        most_common = features_df[col].mode()[0] if not features_df[col].mode().empty else ''
                        features_df[col] = features_df[col].fillna(most_common)
        
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
        Dict: Dictionary containing team form metrics or None if insufficient data
        
    Raises:
        ValueError: If there is insufficient data to calculate form and require_data=True
    """
    # Sort games by date (most recent first)
    team_games = [g for g in historical_games if 
                (g.get("home_team", {}).get("id") == team_id or 
                 g.get("visitor_team", {}).get("id") == team_id) and
                 g.get("status") == "Final"]
    
    if not team_games:
        logger.warning(f"No historical games found for team {team_id}")
        return None
    
    # Sort by date
    sorted_games = sorted(team_games, key=lambda g: g.get("date", ""), reverse=True)
    
    # Take only the N most recent games
    recent_games = sorted_games[:n_games]
    
    # Check if we have enough games for meaningful form analysis
    if len(recent_games) < 3:  # Need at least 3 games for meaningful form
        logger.warning(f"Insufficient recent games ({len(recent_games)}) for team {team_id}")
        return None
    
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
        logger.warning(f"No valid games with scores for team {team_id}")
        return None
    
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


def calculate_defensive_metrics(team_id: int, team_stats: Dict[int, Dict]) -> Dict[str, float]:
    """
    Calculate defensive metrics for a team
    
    Args:
        team_id: Team ID to calculate defensive metrics for
        team_stats: Dictionary of team statistics by team ID
        
    Returns:
        Dictionary of defensive metrics
    """
    if team_id not in team_stats:
        logger.warning(f"No team stats found for team ID {team_id}")
        return {}
    
    team_data = team_stats[team_id]
    
    if not team_data:
        logger.warning(f"No statistics available for team ID {team_id}")
        return {}
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Check if we have the defensive rating from the enhanced data collection
    if "defensive_rating" in team_data:
        metrics["defensive_rating"] = float(team_data["defensive_rating"])
    else:
        # No defensive rating available - do NOT use fallbacks or approximations
        logger.error(f"Cannot calculate defensive rating for team {team_id}. Missing required data.")
        return None  # Signal that we can't make predictions for this team
    
    # Handle opponent points per game - only use real data, no fallbacks
    if "opp_points_pg" in team_data:
        metrics["opp_points_pg"] = float(team_data["opp_points_pg"])
    elif "opp_pts" in team_data:
        metrics["opp_points_pg"] = float(team_data["opp_pts"])
    elif "total_opp_points" in team_data and "games_played" in team_data and float(team_data["games_played"]) > 0:
        # Calculate from totals
        metrics["opp_points_pg"] = float(team_data["total_opp_points"]) / float(team_data["games_played"])
    else:
        # No opponent points data available
        logger.error(f"Cannot calculate opponent points per game for team {team_id}. Missing required data.")
        return None  # Signal that we can't make predictions for this team
            
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
