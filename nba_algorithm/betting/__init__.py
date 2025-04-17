# -*- coding: utf-8 -*-
"""
Betting Package for NBA Algorithm

This package contains modules related to betting strategy, bankroll management,
and closing line value (CLV) tracking for sports betting applications.
"""

from .bankroll import (
    calculate_implied_probability,
    calculate_edge,
    kelly_criterion,
    fractional_kelly,
    fixed_percentage,
    BankrollManager
)

from .clv import (
    american_to_decimal,
    calculate_clv_percentage,
    calculate_clv_ev,
    CLVTracker
)

from .analyzer import BettingAnalyzer

__all__ = [
    # Bankroll management
    'calculate_implied_probability',
    'calculate_edge',
    'kelly_criterion',
    'fractional_kelly',
    'fixed_percentage',
    'BankrollManager',
    
    # CLV tracking
    'american_to_decimal',
    'calculate_clv_percentage',
    'calculate_clv_ev',
    'CLVTracker',
    
    # Betting analysis
    'BettingAnalyzer'
]
