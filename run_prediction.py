#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBA Prediction System Entry Point

This script provides a simple entry point for running the NBA prediction system.
It uses the modular, season-aware prediction engine from the nba_algorithm package.

Usage:
    python run_prediction.py --date YYYY-MM-DD
    python run_prediction.py --include_players --risk_level moderate
"""

import sys
import logging

# Import the prediction engine from the organized package structure
from nba_algorithm.scripts.production_prediction import main

# Run the prediction system
if __name__ == "__main__":
    sys.exit(main())
