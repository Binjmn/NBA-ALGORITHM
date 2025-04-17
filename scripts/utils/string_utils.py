#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
String Utilities Module

This module provides string processing helpers for the NBA prediction system.

Author: Cascade
Date: April 2025
"""

import logging
import re
from typing import Optional
from difflib import SequenceMatcher
import traceback

# Configure logger
logger = logging.getLogger(__name__)


def similar_team_names(name1: str, name2: str) -> bool:
    """
    Check if two team names are similar (handles variations like 'Blazers' vs 'Trail Blazers')
    
    Args:
        name1: First team name
        name2: Second team name
        
    Returns:
        True if names are similar, False otherwise
    """
    try:
        if not name1 or not name2:
            return False
        
        # Convert to lowercase and strip whitespace
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()
        
        # Direct match
        if n1 == n2:
            return True
        
        # Check if one is a subset of the other
        if n1 in n2 or n2 in n1:
            return True
        
        # Handle common nickname variations
        nickname_map = {
            'sixers': 'philadelphia',
            'philly': 'philadelphia',
            'wolves': 'timberwolves',
            'blazers': 'trail blazers',
            'mavs': 'mavericks',
            'cavs': 'cavaliers',
            'knicks': 'new york',
            'nets': 'brooklyn',
            'pels': 'pelicans',
            'hornets': 'charlotte',
            'heat': 'miami',
            'suns': 'phoenix',
            'kings': 'sacramento',
            'wizards': 'washington',
            'celtics': 'boston',
            'lakers': 'los angeles l',
            'clippers': 'los angeles c',
            'warriors': 'golden state',
            'dubs': 'golden state',
            'bulls': 'chicago',
            'pistons': 'detroit',
            'bucks': 'milwaukee',
            'spurs': 'san antonio',
            'hawks': 'atlanta',
            'grizzlies': 'memphis',
            'grizz': 'memphis',
            'jazz': 'utah',
            'thunder': 'oklahoma city',
            'okc': 'oklahoma city',
            'rockets': 'houston',
            'nuggets': 'denver',
            'pacers': 'indiana',
            'raptors': 'toronto'
        }
        
        # Check for nickname variations
        for nickname, full_name in nickname_map.items():
            if (nickname in n1 and full_name in n2) or (nickname in n2 and full_name in n1):
                return True
        
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, n1, n2).ratio()
        return similarity > 0.8
    
    except Exception as e:
        logger.error(f"Error comparing team names: {str(e)}")
        return False


def position_to_numeric(position: str) -> float:
    """
    Convert player position to numeric value for model consumption
    
    Args:
        position: Player position (G, F, C, or combinations)
        
    Returns:
        Numeric representation of position
    """
    try:
        if not position:
            return 2.0  # Default to average (SF/PF range)
        
        position = position.upper().strip()
        
        # Simple positions
        if position == 'G':
            return 1.0
        elif position == 'F':
            return 3.0
        elif position == 'C':
            return 5.0
        
        # Combined positions
        if 'PG' in position:
            return 1.0
        elif 'SG' in position:
            return 2.0
        elif 'SF' in position:
            return 3.0
        elif 'PF' in position:
            return 4.0
        elif 'C' in position:
            return 5.0
        
        # Handle hyphenated positions (take average)
        if '-' in position:
            parts = position.split('-')
            values = [position_to_numeric(p) for p in parts]
            return sum(values) / len(values)
        
        # Handle G-F or F-C type designations
        if 'G' in position and 'F' in position:
            return 2.0  # Between guard and forward
        elif 'F' in position and 'C' in position:
            return 4.0  # Between forward and center
        
        # Default fallback
        return 2.0
    
    except Exception as e:
        logger.error(f"Error converting position to numeric: {str(e)}")
        return 2.0  # Default to average (SF/PF range)


def parse_height(height_str: str) -> float:
    """
    Convert height string (e.g., '6-8') to inches with proper error handling
    
    Args:
        height_str: Height string in feet-inches format
        
    Returns:
        Height in inches as float or raises ValueError if parsing fails
    """
    try:
        if not height_str:
            raise ValueError("Empty height string provided")
        
        # Try to match the pattern "feet-inches"
        feet_inches_pattern = re.compile(r'(\d+)[\-\'\s]+(\d+)')
        match = feet_inches_pattern.search(height_str)
        
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2))
            return (feet * 12) + inches
        
        # If the pattern didn't match, try to extract just a number (assuming inches)
        inches_pattern = re.compile(r'(\d+)')
        match = inches_pattern.search(height_str)
        
        if match:
            return float(match.group(1))
        
        # If we can't parse the height, raise a proper error
        raise ValueError(f"Unable to parse height from string: '{height_str}'")
    
    except Exception as e:
        logger.error(f"Error parsing height '{height_str}': {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to parse height '{height_str}': {str(e)}")
