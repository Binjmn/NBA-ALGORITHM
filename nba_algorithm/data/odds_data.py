#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Odds Data Module

This module handles fetching betting odds and player props data from external APIs.

Author: Cascade
Date: April 2025
"""

import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests

from ..utils.config import DATA_DIR, DEFAULT_API_TIMEOUT, MAX_RETRIES, RETRY_DELAY

# Configure logger
logger = logging.getLogger(__name__)

# Constants
ODDS_API_URL = "https://api.the-odds-api.com/v4"


def fetch_betting_odds(games: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch betting odds for the upcoming games with robust error handling
    
    Args:
        games: List of game dictionaries
        
    Returns:
        Dictionary of betting odds by game ID
    """
    if not games:
        logger.error("No games provided to fetch betting odds")
        return {}
    
    logger.info(f"Fetching betting odds for {len(games)} games")
    
    # Get API key from environment
    api_key = os.environ.get("ODDS_API_KEY")
    
    if not api_key:
        logger.error("No Odds API key found in environment variables. Cannot fetch odds data.")
        raise ValueError("ODDS_API_KEY environment variable is required to fetch odds data")
    
    # API endpoint for NBA odds
    odds_endpoint = f"{ODDS_API_URL}/sports/basketball_nba/odds"
    
    # Query parameters
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals,h2h",
        "oddsFormat": "american",
        "dateFormat": "iso"
    }
    
    odds_by_game = {}
    
    try:
        # Make API request with retries
        response = None
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            try:
                response = requests.get(
                    odds_endpoint, 
                    params=params,
                    timeout=DEFAULT_API_TIMEOUT
                )
                response.raise_for_status()  # Raise exception for HTTP errors
                break  # Exit the retry loop if successful
            except requests.RequestException as e:
                retry_count += 1
                logger.warning(f"API request failed for odds (attempt {retry_count}): {str(e)}")
                
                if retry_count >= MAX_RETRIES:
                    logger.error(f"Failed to fetch odds after {MAX_RETRIES} attempts")
                    raise
                
                # Wait before retrying
                import time
                time.sleep(RETRY_DELAY * retry_count)  # Exponential backoff
        
        # Parse the response
        if response and response.status_code == 200:
            odds_data = response.json()
            
            if not odds_data:
                logger.warning("No odds data found in API response")
                return {}
            
            logger.info(f"Successfully fetched odds data for {len(odds_data)} events")
            
            # Cache the full odds data
            cache_file = DATA_DIR / f"odds_data_{datetime.now().strftime('%Y%m%d')}.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump(odds_data, f)
                logger.info(f"Cached odds data to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache odds data: {str(e)}")
            
            # Match odds data with our games
            for game in games:
                home_team = game.get('home_team', {}).get('name', '').strip()
                away_team = game.get('away_team', {}).get('name', '').strip()
                game_id = game.get('id')
                
                if not home_team or not away_team or not game_id:
                    logger.warning(f"Missing team information for game {game_id}. Skipping odds matching.")
                    continue
                
                # Match the game with odds data
                matching_odds = None
                
                for odds_event in odds_data:
                    event_home = odds_event.get('home_team', '').strip()
                    event_away = odds_event.get('away_team', '').strip()
                    
                    # Check for team name matches (allowing for slight variations)
                    from ..utils.string_utils import similar_team_names
                    if (similar_team_names(home_team, event_home) and 
                        similar_team_names(away_team, event_away)):
                        matching_odds = odds_event
                        break
                
                if matching_odds:
                    # Extract the odds data we care about
                    bookmakers = matching_odds.get('bookmakers', [])
                    if not bookmakers:
                        logger.warning(f"No bookmakers found for {home_team} vs {away_team}")
                        continue
                    
                    # Use the first bookmaker for simplicity
                    # In a production setting, you might aggregate across multiple bookmakers
                    bookmaker = bookmakers[0]
                    markets = bookmaker.get('markets', {})
                    
                    # Extract spread, total, and moneyline odds
                    game_odds = {
                        'game_id': game_id,
                        'home_team': home_team,
                        'away_team': away_team,
                        'bookmaker': bookmaker.get('title', ''),
                        'last_update': bookmaker.get('last_update', ''),
                    }
                    
                    # Extract spread
                    spread_market = next((m for m in markets if m.get('key') == 'spreads'), None)
                    if spread_market:
                        outcomes = spread_market.get('outcomes', [])
                        home_spread = next((o.get('point') for o in outcomes if similar_team_names(o.get('name', ''), home_team)), None)
                        away_spread = next((o.get('point') for o in outcomes if similar_team_names(o.get('name', ''), away_team)), None)
                        
                        if home_spread is not None:
                            game_odds['spread'] = home_spread  # Positive favors home team
                        elif away_spread is not None:
                            game_odds['spread'] = -away_spread  # Convert to home team perspective
                    
                    # Extract total
                    total_market = next((m for m in markets if m.get('key') == 'totals'), None)
                    if total_market:
                        outcomes = total_market.get('outcomes', [])
                        over = next((o for o in outcomes if o.get('name') == 'Over'), None)
                        if over:
                            game_odds['total'] = over.get('point')
                    
                    # Extract moneyline
                    h2h_market = next((m for m in markets if m.get('key') == 'h2h'), None)
                    if h2h_market:
                        outcomes = h2h_market.get('outcomes', [])
                        home_ml = next((o.get('price') for o in outcomes if similar_team_names(o.get('name', ''), home_team)), None)
                        away_ml = next((o.get('price') for o in outcomes if similar_team_names(o.get('name', ''), away_team)), None)
                        
                        if home_ml is not None:
                            game_odds['home_moneyline'] = home_ml
                        
                        if away_ml is not None:
                            game_odds['away_moneyline'] = away_ml
                    
                    # Store the odds for this game
                    odds_by_game[game_id] = game_odds
                    logger.info(f"Matched odds data for {home_team} vs {away_team}")
                else:
                    logger.warning(f"No matching odds found for {home_team} vs {away_team}")
            
            logger.info(f"Successfully matched odds data for {len(odds_by_game)} of {len(games)} games")
            return odds_by_game
        else:
            logger.error(f"Failed to fetch odds: HTTP {response.status_code if response else 'No response'}")
            if response:
                logger.error(f"Response: {response.text}")
            
            # Try to load from cache as fallback
            cache_files = list(DATA_DIR.glob("odds_data_*.json"))
            if cache_files:
                # Use the most recent cache file
                most_recent = max(cache_files, key=lambda f: f.stat().st_mtime)
                try:
                    with open(most_recent, 'r') as f:
                        cached_odds = json.load(f)
                    logger.info(f"Loaded odds data from cache: {most_recent.name}")
                    
                    # Process the cached odds the same way
                    return process_odds_data(cached_odds, games)
                except Exception as e:
                    logger.error(f"Failed to load odds from cache: {str(e)}")
            
            return {}
    
    except Exception as e:
        logger.error(f"Error fetching betting odds: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try to load from cache as fallback
        cache_files = list(DATA_DIR.glob("odds_data_*.json"))
        if cache_files:
            # Use the most recent cache file
            most_recent = max(cache_files, key=lambda f: f.stat().st_mtime)
            try:
                with open(most_recent, 'r') as f:
                    cached_odds = json.load(f)
                logger.info(f"Loaded odds data from cache: {most_recent.name}")
                
                # Process the cached odds the same way
                return process_odds_data(cached_odds, games)
            except Exception as cache_e:
                logger.error(f"Failed to load odds from cache: {str(cache_e)}")
        
        return {}


def process_odds_data(odds_data: List[Dict[str, Any]], games: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Process odds data and match it with games
    
    Args:
        odds_data: List of odds events from the API
        games: List of game dictionaries
        
    Returns:
        Dictionary of betting odds by game ID
    """
    if not odds_data or not games:
        return {}
    
    odds_by_game = {}
    
    for game in games:
        home_team = game.get('home_team', {}).get('name', '').strip()
        away_team = game.get('away_team', {}).get('name', '').strip()
        game_id = game.get('id')
        
        if not home_team or not away_team or not game_id:
            continue
        
        # Match the game with odds data
        matching_odds = None
        
        for odds_event in odds_data:
            event_home = odds_event.get('home_team', '').strip()
            event_away = odds_event.get('away_team', '').strip()
            
            # Check for team name matches (allowing for slight variations)
            from ..utils.string_utils import similar_team_names
            if (similar_team_names(home_team, event_home) and 
                similar_team_names(away_team, event_away)):
                matching_odds = odds_event
                break
        
        if matching_odds:
            # Extract the odds data we care about
            bookmakers = matching_odds.get('bookmakers', [])
            if not bookmakers:
                continue
            
            # Use the first bookmaker for simplicity
            bookmaker = bookmakers[0]
            markets = bookmaker.get('markets', {})
            
            # Extract spread, total, and moneyline odds
            game_odds = {
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'bookmaker': bookmaker.get('title', ''),
                'last_update': bookmaker.get('last_update', ''),
            }
            
            # Extract spread
            spread_market = next((m for m in markets if m.get('key') == 'spreads'), None)
            if spread_market:
                outcomes = spread_market.get('outcomes', [])
                home_spread = next((o.get('point') for o in outcomes if similar_team_names(o.get('name', ''), home_team)), None)
                away_spread = next((o.get('point') for o in outcomes if similar_team_names(o.get('name', ''), away_team)), None)
                
                if home_spread is not None:
                    game_odds['spread'] = home_spread  # Positive favors home team
                elif away_spread is not None:
                    game_odds['spread'] = -away_spread  # Convert to home team perspective
            
            # Extract total
            total_market = next((m for m in markets if m.get('key') == 'totals'), None)
            if total_market:
                outcomes = total_market.get('outcomes', [])
                over = next((o for o in outcomes if o.get('name') == 'Over'), None)
                if over:
                    game_odds['total'] = over.get('point')
            
            # Extract moneyline
            h2h_market = next((m for m in markets if m.get('key') == 'h2h'), None)
            if h2h_market:
                outcomes = h2h_market.get('outcomes', [])
                home_ml = next((o.get('price') for o in outcomes if similar_team_names(o.get('name', ''), home_team)), None)
                away_ml = next((o.get('price') for o in outcomes if similar_team_names(o.get('name', ''), away_team)), None)
                
                if home_ml is not None:
                    game_odds['home_moneyline'] = home_ml
                
                if away_ml is not None:
                    game_odds['away_moneyline'] = away_ml
            
            # Store the odds for this game
            odds_by_game[game_id] = game_odds
    
    return odds_by_game


def fetch_player_props(game_id: str) -> List[Dict[str, Any]]:
    """
    Fetch player prop data for a specific game with robust error handling
    
    Args:
        game_id: ID of the game to fetch props for
        
    Returns:
        List of player prop dictionaries
    """
    logger.info(f"Fetching player props for game {game_id}")
    
    # Get API key from environment
    api_key = os.environ.get("ODDS_API_KEY")
    
    if not api_key:
        logger.error("No Odds API key found in environment variables. Cannot fetch player props.")
        raise ValueError("ODDS_API_KEY environment variable is required to fetch player props")
    
    # API endpoint for player props
    props_endpoint = f"{ODDS_API_URL}/sports/basketball_nba/events/{game_id}/odds"
    
    # Query parameters
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "player_points,player_rebounds,player_assists",
        "oddsFormat": "american",
        "dateFormat": "iso"
    }
    
    try:
        # Make API request with retries
        response = None
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            try:
                response = requests.get(
                    props_endpoint, 
                    params=params,
                    timeout=DEFAULT_API_TIMEOUT
                )
                response.raise_for_status()  # Raise exception for HTTP errors
                break  # Exit the retry loop if successful
            except requests.RequestException as e:
                retry_count += 1
                logger.warning(f"API request failed for player props (attempt {retry_count}): {str(e)}")
                
                if retry_count >= MAX_RETRIES:
                    logger.error(f"Failed to fetch player props for game {game_id} after {MAX_RETRIES} attempts")
                    raise
                
                # Wait before retrying
                import time
                time.sleep(RETRY_DELAY * retry_count)  # Exponential backoff
        
        # Parse the response
        if response and response.status_code == 200:
            props_data = response.json()
            
            if not props_data:
                logger.warning(f"No player props found for game {game_id}")
                return []
            
            # Extract player props
            player_props = []
            bookmakers = props_data.get('bookmakers', [])
            
            if not bookmakers:
                logger.warning(f"No bookmakers found for game {game_id}")
                return []
            
            # Use the first bookmaker for simplicity
            bookmaker = bookmakers[0]
            markets = bookmaker.get('markets', {})
            
            # Process player points props
            points_markets = [m for m in markets if m.get('key').startswith('player_points')]
            for market in points_markets:
                player_name = market.get('key').replace('player_points_', '')
                outcomes = market.get('outcomes', [])
                
                if len(outcomes) >= 2:  # We need both over and under
                    over = next((o for o in outcomes if o.get('name') == 'Over'), None)
                    under = next((o for o in outcomes if o.get('name') == 'Under'), None)
                    
                    if over and under and over.get('point') == under.get('point'):
                        player_props.append({
                            'player_name': player_name,
                            'prop_type': 'points',
                            'line': over.get('point'),
                            'over_odds': over.get('price'),
                            'under_odds': under.get('price'),
                        })
            
            # Process player rebounds props
            rebounds_markets = [m for m in markets if m.get('key').startswith('player_rebounds')]
            for market in rebounds_markets:
                player_name = market.get('key').replace('player_rebounds_', '')
                outcomes = market.get('outcomes', [])
                
                if len(outcomes) >= 2:  # We need both over and under
                    over = next((o for o in outcomes if o.get('name') == 'Over'), None)
                    under = next((o for o in outcomes if o.get('name') == 'Under'), None)
                    
                    if over and under and over.get('point') == under.get('point'):
                        player_props.append({
                            'player_name': player_name,
                            'prop_type': 'rebounds',
                            'line': over.get('point'),
                            'over_odds': over.get('price'),
                            'under_odds': under.get('price'),
                        })
            
            # Process player assists props
            assists_markets = [m for m in markets if m.get('key').startswith('player_assists')]
            for market in assists_markets:
                player_name = market.get('key').replace('player_assists_', '')
                outcomes = market.get('outcomes', [])
                
                if len(outcomes) >= 2:  # We need both over and under
                    over = next((o for o in outcomes if o.get('name') == 'Over'), None)
                    under = next((o for o in outcomes if o.get('name') == 'Under'), None)
                    
                    if over and under and over.get('point') == under.get('point'):
                        player_props.append({
                            'player_name': player_name,
                            'prop_type': 'assists',
                            'line': over.get('point'),
                            'over_odds': over.get('price'),
                            'under_odds': under.get('price'),
                        })
            
            # Cache the props data
            cache_file = DATA_DIR / f"player_props_{game_id}.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump(player_props, f)
                logger.info(f"Cached player props to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache player props: {str(e)}")
            
            logger.info(f"Successfully fetched {len(player_props)} player props for game {game_id}")
            return player_props
        else:
            logger.error(f"Failed to fetch player props for game {game_id}: HTTP {response.status_code if response else 'No response'}")
            if response:
                logger.error(f"Response: {response.text}")
            
            # Try to load from cache as fallback
            cache_file = DATA_DIR / f"player_props_{game_id}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        cached_props = json.load(f)
                    logger.info(f"Loaded {len(cached_props)} player props from cache for game {game_id}")
                    return cached_props
                except Exception as e:
                    logger.error(f"Failed to load player props from cache: {str(e)}")
            
            return []
    
    except Exception as e:
        logger.error(f"Error fetching player props for game {game_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Try to load from cache as fallback
        cache_file = DATA_DIR / f"player_props_{game_id}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_props = json.load(f)
                logger.info(f"Loaded {len(cached_props)} player props from cache for game {game_id}")
                return cached_props
            except Exception as cache_e:
                logger.error(f"Failed to load player props from cache: {str(cache_e)}")
        
        return []
