# -*- coding: utf-8 -*-
"""
Settings Module

This module provides classes and utilities for managing application settings,
including prediction settings, user preferences, and command-line configuration.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class PredictionSettings:
    """
    Class to manage prediction system settings
    
    This class provides a centralized way to manage settings for the NBA prediction system,
    including risk tolerance, bankroll management, output formats, and feature toggles.
    """
    
    def __init__(self, risk_level="moderate", bankroll=1000.0, default_date=None):
        """
        Initialize prediction settings
        
        Args:
            risk_level: Risk tolerance level ('conservative', 'moderate', 'aggressive')
            bankroll: Starting bankroll amount
            default_date: Default prediction date (uses today's date if None)
        """
        self.risk_level = risk_level
        self.bankroll = bankroll
        self.default_date = default_date or datetime.now().strftime("%Y-%m-%d")
        self.track_clv = True
        self.include_players = False
        self.verbose = False
        self.output_format = "standard"  # standard, minimal, or detailed
        self.output_dir = "predictions"  # default output directory
        self.history_days = 30  # default days of historical data to use
        
    def update_from_args(self, args):
        """
        Update settings from command line arguments
        
        Args:
            args: Command line arguments (typically from argparse)
        """
        # Use hasattr to safely check for attributes that might not exist
        if hasattr(args, 'risk_level') and args.risk_level:
            self.risk_level = args.risk_level
            
        if hasattr(args, 'bankroll') and args.bankroll:
            self.bankroll = float(args.bankroll)
            
        if hasattr(args, 'date') and args.date:
            self.default_date = args.date
            
        if hasattr(args, 'track_clv') and args.track_clv is not None:
            self.track_clv = args.track_clv
            
        if hasattr(args, 'include_players') and args.include_players is not None:
            self.include_players = args.include_players
            
        if hasattr(args, 'verbose') and args.verbose is not None:
            self.verbose = args.verbose
            
        if hasattr(args, 'output_format') and args.output_format:
            self.output_format = args.output_format
            
        if hasattr(args, 'output_dir') and args.output_dir:
            self.output_dir = args.output_dir
            
        if hasattr(args, 'history_days') and args.history_days:
            self.history_days = args.history_days
            
        # Configure logging level based on verbosity
        if self.verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("Verbose mode enabled")
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary
        
        Returns:
            Dict: Dictionary representation of settings
        """
        return {
            'risk_level': self.risk_level,
            'bankroll': self.bankroll,
            'default_date': self.default_date,
            'track_clv': self.track_clv,
            'include_players': self.include_players,
            'verbose': self.verbose,
            'output_format': self.output_format,
            'output_dir': self.output_dir,
            'history_days': self.history_days
        }
    
    @classmethod
    def from_dict(cls, settings_dict: Dict[str, Any]) -> 'PredictionSettings':
        """
        Create settings object from dictionary
        
        Args:
            settings_dict: Dictionary with settings
            
        Returns:
            PredictionSettings: New settings object
        """
        settings = cls()
        
        for key, value in settings_dict.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
                
        return settings
    
    def __str__(self) -> str:
        """
        String representation of settings
        
        Returns:
            str: Human-readable settings summary
        """
        return (f"PredictionSettings(risk_level={self.risk_level}, "
                f"bankroll=${self.bankroll:.2f}, date={self.default_date}, "
                f"players={self.include_players}, format={self.output_format})")
