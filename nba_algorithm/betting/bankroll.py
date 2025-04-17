# -*- coding: utf-8 -*-
"""
Bankroll Management Module

This module provides tools for intelligent bankroll management in sports betting,
implementing various bet sizing strategies including Kelly Criterion, Fractional Kelly,
and fixed-percentage approaches.

The module enables risk-adjusted bet sizing based on predicted edge and confidence levels.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple, Union, List

logger = logging.getLogger(__name__)


def calculate_implied_probability(american_odds: float) -> float:
    """
    Convert American odds to implied probability
    
    Args:
        american_odds: Odds in American format (e.g. -110, +120)
        
    Returns:
        float: Implied probability (0-1 range)
    """
    try:
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    except (ValueError, ZeroDivisionError) as e:
        logger.error(f"Error calculating implied probability: {str(e)}")
        return 0.5  # Default to 50% on error


def calculate_edge(predicted_prob: float, market_odds: float) -> float:
    """
    Calculate betting edge based on predicted probability and market odds
    
    Args:
        predicted_prob: Our predicted probability (0-1 range)
        market_odds: Market odds in American format
        
    Returns:
        float: Edge as a percentage (-100% to +100% range)
    """
    try:
        implied_prob = calculate_implied_probability(market_odds)
        edge = predicted_prob - implied_prob
        return edge * 100  # Convert to percentage
    except Exception as e:
        logger.error(f"Error calculating betting edge: {str(e)}")
        return 0.0


def kelly_criterion(predicted_prob: float, market_odds: float, 
                   bankroll: float = 1000.0) -> Tuple[float, float]:
    """
    Calculate optimal bet size using the Kelly Criterion
    
    Args:
        predicted_prob: Our predicted probability (0-1 range)
        market_odds: Market odds in American format
        bankroll: Current bankroll amount (default 1000)
        
    Returns:
        Tuple[float, float]: (bet size as percentage, bet amount)
    """
    try:
        implied_prob = calculate_implied_probability(market_odds)
        
        # Calculate decimal odds
        if market_odds > 0:
            decimal_odds = (market_odds / 100) + 1
        else:
            decimal_odds = (100 / abs(market_odds)) + 1
        
        # Kelly formula: f* = (bp - q) / b
        # where b = decimal odds - 1, p = predicted probability, q = 1 - p
        b = decimal_odds - 1
        p = predicted_prob
        q = 1 - p
        
        # Calculate Kelly percentage
        kelly_pct = max(0, (b * p - q) / b)  # Don't allow negative Kelly
        
        # Calculate bet amount
        bet_amount = kelly_pct * bankroll
        
        # Return both percentage and absolute amount
        return kelly_pct, bet_amount
    except Exception as e:
        logger.error(f"Error calculating Kelly criterion: {str(e)}")
        return 0.0, 0.0


def fractional_kelly(predicted_prob: float, market_odds: float, fraction: float = 0.25, 
                    bankroll: float = 1000.0) -> Tuple[float, float]:
    """
    Calculate a fractional Kelly bet (more conservative than full Kelly)
    
    Args:
        predicted_prob: Our predicted probability (0-1 range)
        market_odds: Market odds in American format
        fraction: Fraction of full Kelly to use (default 0.25 or quarter Kelly)
        bankroll: Current bankroll amount (default 1000)
        
    Returns:
        Tuple[float, float]: (bet size as percentage, bet amount)
    """
    try:
        kelly_pct, _ = kelly_criterion(predicted_prob, market_odds, bankroll)
        fractional_pct = kelly_pct * fraction
        bet_amount = fractional_pct * bankroll
        return fractional_pct, bet_amount
    except Exception as e:
        logger.error(f"Error calculating fractional Kelly: {str(e)}")
        return 0.0, 0.0


def fixed_percentage(edge: float, confidence: float, 
                    min_bet: float = 0.01, max_bet: float = 0.05,
                    bankroll: float = 1000.0) -> Tuple[float, float]:
    """
    Calculate bet size using a simpler fixed percentage approach based on edge and confidence
    
    Args:
        edge: Calculated edge percentage (-100 to 100)
        confidence: Confidence score (0-1)
        min_bet: Minimum bet percentage (default 1%)
        max_bet: Maximum bet percentage (default 5%)
        bankroll: Current bankroll amount (default 1000)
        
    Returns:
        Tuple[float, float]: (bet size as percentage, bet amount)
    """
    try:
        # Only bet when edge is positive
        if edge <= 0:
            return 0.0, 0.0
        
        # Scale bet size based on edge and confidence
        # Higher edge and higher confidence = larger bet
        edge_factor = min(edge / 10, 1.0)  # Cap at 10% edge
        
        # Calculate bet percentage, constrained between min and max
        bet_pct = min_bet + (max_bet - min_bet) * edge_factor * confidence
        bet_amount = bet_pct * bankroll
        
        return bet_pct, bet_amount
    except Exception as e:
        logger.error(f"Error calculating fixed percentage bet: {str(e)}")
        return 0.0, 0.0


class BankrollManager:
    """
    Class to manage sports betting bankroll and calculate optimal bet sizes
    """
    
    def __init__(self, initial_bankroll: float = 1000.0, risk_level: str = "moderate"):
        """
        Initialize the BankrollManager
        
        Args:
            initial_bankroll: Starting bankroll amount
            risk_level: Risk tolerance ('conservative', 'moderate', or 'aggressive')
        """
        self.bankroll = initial_bankroll
        self.starting_bankroll = initial_bankroll
        self.risk_level = risk_level
        self.bet_history = []
        
        # Set risk parameters based on risk level
        if risk_level == "conservative":
            self.kelly_fraction = 0.1  # 1/10 Kelly
            self.max_bet_pct = 0.02  # 2% max bet
            self.min_edge_required = 2.0  # Require 2% edge
        elif risk_level == "moderate":
            self.kelly_fraction = 0.25  # 1/4 Kelly
            self.max_bet_pct = 0.04  # 4% max bet
            self.min_edge_required = 1.0  # Require 1% edge
        elif risk_level == "aggressive":
            self.kelly_fraction = 0.5  # 1/2 Kelly
            self.max_bet_pct = 0.05  # 5% max bet
            self.min_edge_required = 0.5  # Require 0.5% edge
        else:
            logger.warning(f"Unknown risk level: {risk_level}, defaulting to moderate")
            self.kelly_fraction = 0.25
            self.max_bet_pct = 0.03
            self.min_edge_required = 1.0
    
    def calculate_bet_size(self, predicted_prob: float, market_odds: float, 
                          confidence: float = 0.7) -> Dict[str, Union[float, str, bool]]:
        """
        Calculate the optimal bet size based on risk tolerance
        
        Args:
            predicted_prob: Our predicted probability (0-1 range)
            market_odds: Market odds in American format
            confidence: Confidence score in our prediction (0-1)
            
        Returns:
            Dict: Bet recommendation details
        """
        # Calculate edge
        edge = calculate_edge(predicted_prob, market_odds)
        
        # Only recommend bets with positive expected value above our minimum
        if edge < self.min_edge_required:
            return {
                "recommendation": "No Bet",
                "edge": edge,
                "bet_pct": 0.0,
                "bet_amount": 0.0,
                "explanation": f"Edge of {edge:.2f}% is below minimum threshold of {self.min_edge_required:.2f}%"
            }
        
        # Calculate Kelly and fractional Kelly
        _, full_kelly_amount = kelly_criterion(predicted_prob, market_odds, self.bankroll)
        frac_pct, frac_amount = fractional_kelly(
            predicted_prob, market_odds, self.kelly_fraction, self.bankroll
        )
        
        # Calculate fixed percentage approach
        fixed_pct, fixed_amount = fixed_percentage(
            edge, confidence, 0.01, self.max_bet_pct, self.bankroll
        )
        
        # Choose the more conservative approach between fractional Kelly and fixed
        bet_pct = min(frac_pct, fixed_pct)
        
        # Apply maximum bet constraint
        if bet_pct > self.max_bet_pct:
            bet_pct = self.max_bet_pct
            bet_amount = bet_pct * self.bankroll
            explanation = f"Capped at maximum {self.max_bet_pct*100:.1f}% of bankroll"
        else:
            bet_amount = bet_pct * self.bankroll
            if bet_pct == frac_pct:
                explanation = f"Based on {self.kelly_fraction*100:.0f}% Kelly criterion"
            else:
                explanation = f"Based on fixed percentage with {edge:.2f}% edge"
        
        # Round to 2 decimal places
        bet_pct = round(bet_pct * 100) / 100
        bet_amount = round(bet_amount * 100) / 100
        
        return {
            "recommendation": "Bet" if bet_amount > 0 else "No Bet",
            "edge": edge,
            "bet_pct": bet_pct,
            "bet_amount": bet_amount,
            "explanation": explanation
        }
    
    def record_bet(self, bet_amount: float, odds: float, result: str) -> None:
        """
        Record a bet in history and update bankroll
        
        Args:
            bet_amount: Amount wagered
            odds: Market odds in American format
            result: Outcome ('win', 'loss', or 'push')
        """
        bet_record = {
            "amount": bet_amount,
            "odds": odds,
            "result": result
        }
        
        # Update bankroll based on result
        if result.lower() == "win":
            if odds > 0:
                profit = bet_amount * (odds / 100)
            else:
                profit = bet_amount * (100 / abs(odds))
            self.bankroll += profit
            bet_record["profit"] = profit
        elif result.lower() == "loss":
            self.bankroll -= bet_amount
            bet_record["profit"] = -bet_amount
        # For push, bankroll doesn't change
        
        # Add to history
        self.bet_history.append(bet_record)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics from bet history
        
        Returns:
            Dict: Performance metrics including ROI, win rate, etc.
        """
        if not self.bet_history:
            return {
                "roi": 0.0,
                "win_rate": 0.0,
                "profit": 0.0,
                "total_bets": 0
            }
        
        total_bets = len(self.bet_history)
        wins = sum(1 for bet in self.bet_history if bet["result"].lower() == "win")
        losses = sum(1 for bet in self.bet_history if bet["result"].lower() == "loss")
        
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        # Calculate profit
        profit = sum(bet.get("profit", 0) for bet in self.bet_history)
        
        # Calculate ROI
        total_wagered = sum(bet["amount"] for bet in self.bet_history)
        roi = (profit / total_wagered) * 100 if total_wagered > 0 else 0
        
        return {
            "roi": roi,
            "win_rate": win_rate,
            "profit": profit,
            "total_bets": total_bets,
            "current_bankroll": self.bankroll,
            "bankroll_growth": ((self.bankroll / self.starting_bankroll) - 1) * 100
        }
