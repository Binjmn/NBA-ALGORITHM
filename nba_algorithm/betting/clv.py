# -*- coding: utf-8 -*-
"""
Closing Line Value (CLV) Module

This module provides tools for tracking and analyzing Closing Line Value,
which is a key metric for evaluating long-term sports betting success.

CLV measures how a bettor's odds compare to the closing line, with positive CLV
indicating value obtained compared to the market's final assessment.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from ..data.odds_data import fetch_betting_odds

logger = logging.getLogger(__name__)


def american_to_decimal(american_odds: float) -> float:
    """
    Convert American odds to decimal format
    
    Args:
        american_odds: Odds in American format (e.g. -110, +120)
        
    Returns:
        float: Odds in decimal format
    """
    try:
        if american_odds > 0:
            return 1 + (american_odds / 100)
        else:
            return 1 + (100 / abs(american_odds))
    except (ValueError, ZeroDivisionError) as e:
        logger.error(f"Error converting American odds to decimal: {str(e)}")
        return 2.0  # Default to 2.0 (even odds) on error


def calculate_clv_percentage(bet_odds: float, closing_odds: float) -> float:
    """
    Calculate CLV as a percentage
    
    Args:
        bet_odds: Odds obtained when placing the bet (American)
        closing_odds: Closing line odds (American)
        
    Returns:
        float: CLV as a percentage (positive = good, negative = bad)
    """
    try:
        # Convert to decimal odds for easier calculation
        bet_decimal = american_to_decimal(bet_odds)
        closing_decimal = american_to_decimal(closing_odds)
        
        # Calculate CLV percentage
        if bet_odds > 0:  # If we bet the underdog
            clv_pct = ((bet_decimal - closing_decimal) / closing_decimal) * 100
        else:  # If we bet the favorite
            clv_pct = ((closing_decimal - bet_decimal) / bet_decimal) * 100
        
        return clv_pct
    except Exception as e:
        logger.error(f"Error calculating CLV percentage: {str(e)}")
        return 0.0


def calculate_clv_ev(bet_odds: float, closing_odds: float) -> float:
    """
    Calculate CLV as expected value
    
    Args:
        bet_odds: Odds obtained when placing the bet (American)
        closing_odds: Closing line odds (American)
        
    Returns:
        float: CLV as expected value percentage
    """
    try:
        # Convert odds to implied probabilities
        bet_prob = 100 / (bet_odds + 100) if bet_odds > 0 else abs(bet_odds) / (abs(bet_odds) + 100)
        closing_prob = 100 / (closing_odds + 100) if closing_odds > 0 else abs(closing_odds) / (abs(closing_odds) + 100)
        
        # Calculate EV difference
        ev_diff = bet_prob - closing_prob
        
        # Return as a percentage
        return ev_diff * 100
    except Exception as e:
        logger.error(f"Error calculating CLV expected value: {str(e)}")
        return 0.0


class CLVTracker:
    """
    Class to track and analyze Closing Line Value for sports bets
    """
    
    def __init__(self):
        """
        Initialize the CLV Tracker
        """
        self.odds_collector = fetch_betting_odds
        self.tracked_bets = []
        self.market_snapshots = {}
    
    def get_current_odds(self, game_id, market_type=None):
        """
        Get current odds for a game
        
        Args:
            game_id: ID of the game
            market_type: Optional market type to get odds for
            
        Returns:
            Dict: Current odds data with timestamp
        """
        try:
            # Create a minimal game dictionary that fetch_betting_odds can use
            mock_game = {"id": game_id, "home_team": {"id": ""}, "visitor_team": {"id": ""}}
            
            # Call fetch_betting_odds with a list containing our game
            odds_data_dict = self.odds_collector([mock_game])
            
            # Extract odds for our specific game
            odds_data = odds_data_dict.get(game_id, {})
            
            if not odds_data:
                logger.warning(f"No odds data available for game ID {game_id}")
                return {}
            
            # Add timestamp
            timestamp = datetime.now().isoformat()
            odds_data['timestamp'] = timestamp
            
            # Store in market snapshots
            if game_id not in self.market_snapshots:
                self.market_snapshots[game_id] = {}
                
            if market_type not in self.market_snapshots[game_id]:
                self.market_snapshots[game_id][market_type] = []
                
            self.market_snapshots[game_id][market_type].append(odds_data)
            
            return odds_data
        except Exception as e:
            logger.error(f"Error getting current odds: {str(e)}")
            return {}
    
    def track_odds_movement(self, game_id: str, market_type: str = "h2h") -> Dict:
        """
        Track and store current market odds for a game
        
        Args:
            game_id: Unique identifier for the game
            market_type: Type of market (h2h, spreads, totals)
            
        Returns:
            Dict: Current odds data with timestamp
        """
        # Delegate to get_current_odds for implementation
        return self.get_current_odds(game_id, market_type)
        
    def record_bet(self, game_id: str, bet_type: str, selection: str, 
                 odds: float, stake: float) -> Dict:
        """
        Record a bet for CLV tracking
        
        Args:
            game_id: Unique identifier for the game
            bet_type: Type of bet (moneyline, spread, total)
            selection: Selection made (team name or over/under)
            odds: Odds obtained for the bet (American)
            stake: Amount wagered
            
        Returns:
            Dict: Recorded bet information
        """
        bet_info = {
            'game_id': game_id,
            'bet_type': bet_type,
            'selection': selection,
            'odds': odds,
            'stake': stake,
            'bet_time': datetime.now().isoformat(),
            'closing_odds': None,
            'clv': None,
            'clv_ev': None,
            'result': None
        }
        
        self.tracked_bets.append(bet_info)
        return bet_info
    
    def update_with_closing_line(self, game_id: str, bet_type: str, 
                               selection: str, closing_odds: float) -> Optional[Dict]:
        """
        Update a tracked bet with closing line information
        
        Args:
            game_id: Unique identifier for the game
            bet_type: Type of bet (moneyline, spread, total)
            selection: Selection made (team name or over/under)
            closing_odds: Closing odds (American)
            
        Returns:
            Optional[Dict]: Updated bet information or None if bet not found
        """
        # Find the bet to update
        for bet in self.tracked_bets:
            if (bet['game_id'] == game_id and 
                bet['bet_type'] == bet_type and 
                bet['selection'] == selection and
                bet['closing_odds'] is None):  # Only update if closing odds not set
                
                # Update with closing line
                bet['closing_odds'] = closing_odds
                
                # Calculate CLV metrics
                bet['clv'] = calculate_clv_percentage(bet['odds'], closing_odds)
                bet['clv_ev'] = calculate_clv_ev(bet['odds'], closing_odds)
                
                logger.info(f"Updated bet with CLV: {bet['clv']:.2f}% on {game_id}, {bet_type}")
                return bet
        
        logger.warning(f"No matching bet found to update CLV for {game_id}, {bet_type}, {selection}")
        return None
    
    def update_bet_result(self, game_id: str, bet_type: str, 
                        selection: str, result: str) -> Optional[Dict]:
        """
        Update a tracked bet with the result
        
        Args:
            game_id: Unique identifier for the game
            bet_type: Type of bet (moneyline, spread, total)
            selection: Selection made (team name or over/under)
            result: Outcome ('win', 'loss', or 'push')
            
        Returns:
            Optional[Dict]: Updated bet information or None if bet not found
        """
        for bet in self.tracked_bets:
            if (bet['game_id'] == game_id and 
                bet['bet_type'] == bet_type and 
                bet['selection'] == selection):
                
                bet['result'] = result
                logger.info(f"Updated bet result to {result} for {game_id}, {bet_type}")
                return bet
        
        logger.warning(f"No matching bet found to update result for {game_id}, {bet_type}, {selection}")
        return None
    
    def get_clv_stats(self) -> Dict[str, float]:
        """
        Calculate overall CLV statistics
        
        Returns:
            Dict[str, float]: CLV statistics
        """
        if not self.tracked_bets or not any(bet.get('clv') is not None for bet in self.tracked_bets):
            return {
                'average_clv': 0.0,
                'average_clv_ev': 0.0,
                'positive_clv_rate': 0.0,
                'total_bets_with_clv': 0
            }
        
        # Filter bets that have CLV calculated
        bets_with_clv = [bet for bet in self.tracked_bets if bet.get('clv') is not None]
        total_bets = len(bets_with_clv)
        
        if total_bets == 0:
            return {
                'average_clv': 0.0,
                'average_clv_ev': 0.0,
                'positive_clv_rate': 0.0,
                'total_bets_with_clv': 0
            }
        
        # Calculate average CLV
        avg_clv = sum(bet['clv'] for bet in bets_with_clv) / total_bets
        
        # Calculate average CLV expected value
        avg_clv_ev = sum(bet['clv_ev'] for bet in bets_with_clv if bet.get('clv_ev') is not None) / total_bets
        
        # Calculate rate of positive CLV
        positive_clv_count = sum(1 for bet in bets_with_clv if bet['clv'] > 0)
        positive_clv_rate = positive_clv_count / total_bets
        
        return {
            'average_clv': avg_clv,
            'average_clv_ev': avg_clv_ev,
            'positive_clv_rate': positive_clv_rate,
            'total_bets_with_clv': total_bets
        }
    
    def get_clv_by_bet_type(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate CLV statistics broken down by bet type
        
        Returns:
            Dict[str, Dict[str, float]]: CLV statistics by bet type
        """
        if not self.tracked_bets:
            return {}
        
        # Group bets by type
        bet_types = {}
        for bet in self.tracked_bets:
            if bet.get('clv') is not None:
                bet_type = bet['bet_type']
                if bet_type not in bet_types:
                    bet_types[bet_type] = []
                bet_types[bet_type].append(bet)
        
        # Calculate stats for each bet type
        stats_by_type = {}
        for bet_type, bets in bet_types.items():
            total = len(bets)
            if total == 0:
                continue
                
            avg_clv = sum(bet['clv'] for bet in bets) / total
            avg_clv_ev = sum(bet['clv_ev'] for bet in bets if bet.get('clv_ev') is not None) / total
            positive_count = sum(1 for bet in bets if bet['clv'] > 0)
            positive_rate = positive_count / total
            
            stats_by_type[bet_type] = {
                'average_clv': avg_clv,
                'average_clv_ev': avg_clv_ev,
                'positive_clv_rate': positive_rate,
                'total_bets': total
            }
        
        return stats_by_type
    
    def get_clv_trend(self, window_size: int = 10) -> List[float]:
        """
        Get CLV trend over time (moving average)
        
        Args:
            window_size: Size of the moving average window
            
        Returns:
            List[float]: CLV moving average values
        """
        bets_with_clv = [bet for bet in self.tracked_bets if bet.get('clv') is not None]
        if not bets_with_clv:
            return []
        
        # Sort by bet time
        bets_with_clv.sort(key=lambda x: x.get('bet_time', ''))
        
        # Extract CLV values
        clv_values = [bet['clv'] for bet in bets_with_clv]
        
        # Calculate moving average
        if len(clv_values) < window_size:
            # Not enough data for moving average, return simple average
            return [sum(clv_values) / len(clv_values)]
        
        moving_avgs = []
        for i in range(len(clv_values) - window_size + 1):
            window = clv_values[i:i+window_size]
            moving_avgs.append(sum(window) / window_size)
        
        return moving_avgs
