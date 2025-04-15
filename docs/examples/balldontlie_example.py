#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BallDontLie API Example
Purpose: Demonstrate usage of the BallDontLie API client and data processor.

This example script shows how to:
1. Initialize the API client and data processor
2. Fetch teams and players
3. Get today's games with odds
4. Save processed data
"""

import os
import sys
import logging
import json
from datetime import datetime
from pprint import pprint

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.api.balldontlie_client import BallDontLieClient
from src.utils.data_processor import NBADataProcessor
from config.api_keys import validate_api_keys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to demonstrate BallDontLie API usage
    """
    print("\n=== NBA Algorithm - BallDontLie API Example ===\n")
    
    # Validate API keys
    if not validate_api_keys():
        logger.error("API key validation failed. Please check your API keys.")
        return 1
    
    try:
        # Initialize the data processor (which contains the API client)
        print("Initializing NBA data processor...\n")
        processor = NBADataProcessor()
        
        # Get all teams
        print("Fetching all NBA teams...")
        teams = processor.get_teams()
        print(f"Found {len(teams)} teams")
        print("Example team:")
        if teams:
            sample_team_id = next(iter(teams))
            pprint(teams[sample_team_id])
        print()
        
        # Get active players
        print("Fetching active NBA players...")
        players = processor.get_players()
        print(f"Found {len(players)} active players")
        print("Example player:")
        if players:
            sample_player_id = next(iter(players))
            pprint(players[sample_player_id])
        print()
        
        # Get today's games with odds
        print("Fetching today's games with odds...")
        today = datetime.now().strftime('%Y-%m-%d')
        today_games = processor.get_todays_games_with_data()
        print(f"Found {len(today_games)} games scheduled for {today}")
        
        if today_games:
            print("Today's games:")
            for game in today_games:
                home_team = game['home_team'].get('full_name', 'Unknown')
                visitor_team = game['visitor_team'].get('full_name', 'Unknown')
                odds = game['odds']
                
                print(f"  {visitor_team} @ {home_team}")
                if odds:
                    spread = odds.get('spread', 'N/A')
                    total = odds.get('total', 'N/A')
                    home_ml = odds.get('home_team_moneyline', 'N/A')
                    visitor_ml = odds.get('visitor_team_moneyline', 'N/A')
                    print(f"    Spread: {spread}, Total: {total}")
                    print(f"    Moneyline: {visitor_team}: {visitor_ml}, {home_team}: {home_ml}")
            
            # Save processed data
            print("\nSaving today's games data to file...")
            saved_path = processor.save_processed_data(
                today_games, 
                f"games_{today}"
            )
            print(f"Saved to {saved_path}")
        else:
            print("No games scheduled for today")
        
    except Exception as e:
        logger.exception(f"Error in BallDontLie example: {e}")
        return 1
    finally:
        if 'processor' in locals():
            processor.close()
    
    print("\n=== Example completed successfully ===\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
