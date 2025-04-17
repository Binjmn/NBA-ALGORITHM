# -*- coding: utf-8 -*-
"""
Betting Analyzer Module

This module provides analysis of betting opportunities based on prediction data,
calculating edges, optimal bet sizes, and tracking CLV.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

from .bankroll import BankrollManager, calculate_edge
from .clv import CLVTracker

logger = logging.getLogger(__name__)


class BettingAnalyzer:
    """
    Class to analyze betting opportunities and manage bankroll
    """
    
    def __init__(self, settings):
        """
        Initialize the BettingAnalyzer
        
        Args:
            settings: PredictionSettings object or dict with settings
        """
        # Handle either a settings object or a dict
        if hasattr(settings, 'risk_level'):
            self.settings = settings
        else:
            # Treat as dict if not a settings object
            self.settings = type('Settings', (), settings) if settings else type('Settings', (), {})
        
        # Default values if not provided
        risk_level = getattr(self.settings, 'risk_level', 'moderate')
        bankroll = getattr(self.settings, 'bankroll', 1000.0)
        track_clv = getattr(self.settings, 'track_clv', False)
        
        self.bankroll_manager = BankrollManager(
            initial_bankroll=bankroll,
            risk_level=risk_level
        )
        self.clv_tracker = CLVTracker() if track_clv else None
        
    def analyze_game_predictions(self, predictions_df, odds_data=None):
        """
        Analyze game predictions and provide betting recommendations
        
        Args:
            predictions_df: DataFrame of game predictions
            odds_data: Optional odds data
            
        Returns:
            DataFrame: Enhanced predictions with betting analysis
        """
        if predictions_df.empty:
            return predictions_df
            
        # Make a copy to avoid modifying the original
        df = predictions_df.copy()
        
        # Add betting analysis columns
        df['moneyline_edge'] = 0.0
        df['moneyline_bet_pct'] = 0.0
        df['moneyline_bet_amount'] = 0.0
        df['spread_edge'] = 0.0
        df['spread_bet_pct'] = 0.0
        df['spread_bet_amount'] = 0.0
        df['total_edge'] = 0.0
        df['total_bet_pct'] = 0.0
        df['total_bet_amount'] = 0.0
        
        for i, game in df.iterrows():
            # Only perform analysis if we have odds data
            if 'home_odds' in game and 'visitor_odds' in game:
                # Moneyline analysis
                win_prob = game.get('win_probability', 0.5)
                home_odds = game.get('home_odds', -110)
                visitor_odds = game.get('visitor_odds', -110)
                
                # Determine which side to bet based on edge
                home_edge = calculate_edge(win_prob, home_odds)
                visitor_edge = calculate_edge(1 - win_prob, visitor_odds)
                
                if home_edge > visitor_edge and home_edge > 0:
                    # Bet on home team
                    bet_info = self.bankroll_manager.calculate_bet_size(
                        win_prob, home_odds, game.get('confidence_score', 0.7)
                    )
                    df.at[i, 'moneyline_edge'] = home_edge
                    df.at[i, 'moneyline_bet_pct'] = bet_info['bet_pct']
                    df.at[i, 'moneyline_bet_amount'] = bet_info['bet_amount']
                    df.at[i, 'moneyline_bet_team'] = game.get('home_team', '')
                    df.at[i, 'moneyline_bet_odds'] = home_odds
                elif visitor_edge > 0:
                    # Bet on visitor team
                    bet_info = self.bankroll_manager.calculate_bet_size(
                        1 - win_prob, visitor_odds, game.get('confidence_score', 0.7)
                    )
                    df.at[i, 'moneyline_edge'] = visitor_edge
                    df.at[i, 'moneyline_bet_pct'] = bet_info['bet_pct']
                    df.at[i, 'moneyline_bet_amount'] = bet_info['bet_amount']
                    df.at[i, 'moneyline_bet_team'] = game.get('visitor_team', '')
                    df.at[i, 'moneyline_bet_odds'] = visitor_odds
                
                # Spread analysis if available
                if 'spread_probability' in game and 'spread_odds' in game:
                    spread_prob = game.get('spread_probability', 0.5)
                    spread_odds = game.get('spread_odds', -110)
                    spread_edge = calculate_edge(spread_prob, spread_odds)
                    
                    if spread_edge > 0:
                        bet_info = self.bankroll_manager.calculate_bet_size(
                            spread_prob, spread_odds, game.get('spread_confidence', 0.6)
                        )
                        df.at[i, 'spread_edge'] = spread_edge
                        df.at[i, 'spread_bet_pct'] = bet_info['bet_pct']
                        df.at[i, 'spread_bet_amount'] = bet_info['bet_amount']
                
                # Total points analysis if available
                if 'over_probability' in game and 'over_odds' in game and 'under_odds' in game:
                    over_prob = game.get('over_probability', 0.5)
                    under_prob = 1 - over_prob
                    over_odds = game.get('over_odds', -110)
                    under_odds = game.get('under_odds', -110)
                    
                    # Determine which side to bet based on edge
                    over_edge = calculate_edge(over_prob, over_odds)
                    under_edge = calculate_edge(under_prob, under_odds)
                    
                    if over_edge > under_edge and over_edge > 0:
                        bet_info = self.bankroll_manager.calculate_bet_size(
                            over_prob, over_odds, game.get('total_confidence', 0.55)
                        )
                        df.at[i, 'total_edge'] = over_edge
                        df.at[i, 'total_bet_pct'] = bet_info['bet_pct']
                        df.at[i, 'total_bet_amount'] = bet_info['bet_amount']
                        df.at[i, 'total_bet_type'] = 'over'
                        df.at[i, 'total_bet_odds'] = over_odds
                    elif under_edge > 0:
                        bet_info = self.bankroll_manager.calculate_bet_size(
                            under_prob, under_odds, game.get('total_confidence', 0.55)
                        )
                        df.at[i, 'total_edge'] = under_edge
                        df.at[i, 'total_bet_pct'] = bet_info['bet_pct']
                        df.at[i, 'total_bet_amount'] = bet_info['bet_amount']
                        df.at[i, 'total_bet_type'] = 'under'
                        df.at[i, 'total_bet_odds'] = under_odds
                        
                # Track CLV if enabled
                if self.clv_tracker and 'game_id' in game:
                    self.clv_tracker.track_odds_movement(
                        game['game_id'], 
                        'h2h'  # moneyline market
                    )
        
        return df
    
    def analyze_player_predictions(self, player_predictions_df, odds_data=None):
        """
        Analyze player predictions and provide prop betting recommendations
        
        Args:
            player_predictions_df: DataFrame of player predictions
            odds_data: Optional player prop odds data
            
        Returns:
            DataFrame: Enhanced player predictions with betting analysis
        """
        if player_predictions_df.empty:
            return player_predictions_df
            
        # Make a copy to avoid modifying the original
        df = player_predictions_df.copy()
        
        # Add betting analysis columns
        df['points_edge'] = 0.0
        df['points_bet_pct'] = 0.0
        df['points_bet_amount'] = 0.0
        df['rebounds_edge'] = 0.0
        df['rebounds_bet_pct'] = 0.0
        df['rebounds_bet_amount'] = 0.0
        df['assists_edge'] = 0.0
        df['assists_bet_pct'] = 0.0
        df['assists_bet_amount'] = 0.0
        
        # For player props, we'd ideally have odds data from a player props API
        # Since we might not have that, we can simulate reasonable prop lines
        for i, player in df.iterrows():
            # For each stat category, create a simulated prop line and odds
            # Points props
            if 'predicted_points' in player:
                points = player['predicted_points']
                # Create a realistic prop line (round to nearest 0.5)
                prop_line = round(points * 2) / 2
                # Adjust for typical vig (e.g., -110)
                prop_odds = -110
                
                # Determine over/under recommendation
                over_prob = 0.5 + ((points - prop_line) / 10)  # Simple linear approximation
                over_prob = max(0.1, min(0.9, over_prob))  # Cap between 10% and 90%
                under_prob = 1 - over_prob
                
                # Calculate edge for over and under
                over_edge = calculate_edge(over_prob, prop_odds)
                under_edge = calculate_edge(under_prob, prop_odds)
                
                # Store the better option
                if over_edge > under_edge and over_edge > 0:
                    bet_info = self.bankroll_manager.calculate_bet_size(
                        over_prob, prop_odds, player.get('confidence_score', 0.6)
                    )
                    df.at[i, 'points_edge'] = over_edge
                    df.at[i, 'points_bet_pct'] = bet_info['bet_pct']
                    df.at[i, 'points_bet_amount'] = bet_info['bet_amount']
                    df.at[i, 'points_bet_type'] = 'over'
                    df.at[i, 'points_bet_line'] = prop_line
                elif under_edge > 0:
                    bet_info = self.bankroll_manager.calculate_bet_size(
                        under_prob, prop_odds, player.get('confidence_score', 0.6)
                    )
                    df.at[i, 'points_edge'] = under_edge
                    df.at[i, 'points_bet_pct'] = bet_info['bet_pct']
                    df.at[i, 'points_bet_amount'] = bet_info['bet_amount']
                    df.at[i, 'points_bet_type'] = 'under'
                    df.at[i, 'points_bet_line'] = prop_line
                    
            # Similar analysis for rebounds
            if 'predicted_rebounds' in player:
                rebounds = player['predicted_rebounds']
                prop_line = round(rebounds * 2) / 2
                prop_odds = -110
                
                over_prob = 0.5 + ((rebounds - prop_line) / 5)  # Rebounds have smaller ranges
                over_prob = max(0.1, min(0.9, over_prob))
                under_prob = 1 - over_prob
                
                over_edge = calculate_edge(over_prob, prop_odds)
                under_edge = calculate_edge(under_prob, prop_odds)
                
                if over_edge > under_edge and over_edge > 0:
                    bet_info = self.bankroll_manager.calculate_bet_size(
                        over_prob, prop_odds, player.get('confidence_score', 0.6)
                    )
                    df.at[i, 'rebounds_edge'] = over_edge
                    df.at[i, 'rebounds_bet_pct'] = bet_info['bet_pct']
                    df.at[i, 'rebounds_bet_amount'] = bet_info['bet_amount']
                    df.at[i, 'rebounds_bet_type'] = 'over'
                    df.at[i, 'rebounds_bet_line'] = prop_line
                elif under_edge > 0:
                    bet_info = self.bankroll_manager.calculate_bet_size(
                        under_prob, prop_odds, player.get('confidence_score', 0.6)
                    )
                    df.at[i, 'rebounds_edge'] = under_edge
                    df.at[i, 'rebounds_bet_pct'] = bet_info['bet_pct']
                    df.at[i, 'rebounds_bet_amount'] = bet_info['bet_amount']
                    df.at[i, 'rebounds_bet_type'] = 'under'
                    df.at[i, 'rebounds_bet_line'] = prop_line
                    
            # Similar analysis for assists
            if 'predicted_assists' in player:
                assists = player['predicted_assists']
                prop_line = round(assists * 2) / 2
                prop_odds = -110
                
                over_prob = 0.5 + ((assists - prop_line) / 4)  # Assists have smaller ranges
                over_prob = max(0.1, min(0.9, over_prob))
                under_prob = 1 - over_prob
                
                over_edge = calculate_edge(over_prob, prop_odds)
                under_edge = calculate_edge(under_prob, prop_odds)
                
                if over_edge > under_edge and over_edge > 0:
                    bet_info = self.bankroll_manager.calculate_bet_size(
                        over_prob, prop_odds, player.get('confidence_score', 0.6)
                    )
                    df.at[i, 'assists_edge'] = over_edge
                    df.at[i, 'assists_bet_pct'] = bet_info['bet_pct']
                    df.at[i, 'assists_bet_amount'] = bet_info['bet_amount']
                    df.at[i, 'assists_bet_type'] = 'over'
                    df.at[i, 'assists_bet_line'] = prop_line
                elif under_edge > 0:
                    bet_info = self.bankroll_manager.calculate_bet_size(
                        under_prob, prop_odds, player.get('confidence_score', 0.6)
                    )
                    df.at[i, 'assists_edge'] = under_edge
                    df.at[i, 'assists_bet_pct'] = bet_info['bet_pct']
                    df.at[i, 'assists_bet_amount'] = bet_info['bet_amount']
                    df.at[i, 'assists_bet_type'] = 'under'
                    df.at[i, 'assists_bet_line'] = prop_line
                    
        return df
