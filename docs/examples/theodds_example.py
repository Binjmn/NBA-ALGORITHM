"""
Example usage of The Odds API client

This script demonstrates how to use The Odds API client
to retrieve NBA betting odds and related data.

Usage:
    python -m src.examples.theodds_example
"""

import json
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'theodds_example.log'))
    ]
)
logger = logging.getLogger(__name__)

# Import our client
from src.api.theodds_client import TheOddsClient


def pretty_print(data: Any) -> None:
    """
    Pretty print JSON data
    
    Args:
        data: Data to print
    """
    print(json.dumps(data, indent=2, default=str))
    print("-" * 80)


def run_example():
    """Run the example"""
    logger.info("Starting The Odds API example")
    
    # Create the client
    try:
        client = TheOddsClient()
    except ValueError as e:
        logger.error(f"Failed to initialize client: {e}")
        return
    
    logger.info("Client initialized successfully")
    
    # 1. Check available sports
    logger.info("Getting available sports...")
    try:
        sports = client.get_sports()
        print(f"\nAvailable Sports ({len(sports)}):")
        for sport in sports:
            if sport['key'] == 'basketball_nba':
                print(f"- {sport['title']} (NBA): ✅ Active")
            else:
                print(f"- {sport['title']}")
        pretty_print(sports[:2])  # Show first two sports
    except Exception as e:
        logger.error(f"Error getting sports: {e}")
    
    # 2. Check if NBA is available
    logger.info("Checking if NBA is available...")
    try:
        nba_available = client.is_nba_available()
        print(f"\nNBA Available: {nba_available}")
    except Exception as e:
        logger.error(f"Error checking NBA availability: {e}")
    
    # 3. Get today's NBA odds
    logger.info("Getting today's NBA odds...")
    try:
        todays_odds = client.get_todays_odds()
        print(f"\nToday's NBA Odds ({len(todays_odds)} games):")
        for game in todays_odds:
            print(f"- {game['home_team']} vs {game['away_team']} at {game['commence_time']}")
        
        if todays_odds:
            # Show details for the first game
            print("\nDetailed odds for first game:")
            first_game = todays_odds[0]
            pretty_print(first_game)
            
            # Save this event ID for later use
            event_id = first_game['id']
            
            # 4. Get specific game odds
            logger.info(f"Getting odds for game {event_id}...")
            try:
                game_odds = client.get_game_odds(event_id)
                print(f"\nDetailed Odds for {game_odds['home_team']} vs {game_odds['away_team']}:")
                
                # Print bookmakers
                print("\nBookmakers:")
                for bookmaker in game_odds['bookmakers'][:3]:  # Show first three bookmakers
                    print(f"- {bookmaker['title']}")
                    for market in bookmaker['markets']:
                        print(f"  - {market['key']}")
                        for outcome in market['outcomes'][:2]:  # Show first two outcomes
                            print(f"    - {outcome['name']}: {outcome['price']}")
                
                pretty_print(game_odds['bookmakers'][0] if game_odds['bookmakers'] else {})
            except Exception as e:
                logger.error(f"Error getting game odds: {e}")
        else:
            print("No games found for today")
    except Exception as e:
        logger.error(f"Error getting today's odds: {e}")
    
    # 5. Get live scores
    logger.info("Getting live scores...")
    try:
        scores = client.get_live_scores()
        print(f"\nLive Scores ({len(scores)} games):")
        for score in scores:
            home_score = score.get('scores', [{}])[0].get('score', 'N/A')
            away_score = score.get('scores', [{}])[1].get('score', 'N/A')
            print(f"- {score['home_team']} ({home_score}) vs {score['away_team']} ({away_score})")
            print(f"  Status: {score.get('completed', False)}")
        
        if scores:
            pretty_print(scores[0])  # Show first score
    except Exception as e:
        logger.error(f"Error getting live scores: {e}")
    
    # 6. Get historical data (from yesterday)
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    logger.info(f"Getting historical data for {yesterday_str}...")
    
    try:
        historical_events = client.get_historical_events(yesterday_str)
        print(f"\nHistorical Events for {yesterday_str} ({len(historical_events)} games):")
        for event in historical_events:
            print(f"- {event['home_team']} vs {event['away_team']}")
        
        if historical_events:
            historical_event_id = historical_events[0]['id']
            
            # 7. Get historical odds for a specific game
            try:
                historical_odds = client.get_historical_game_odds(
                    historical_event_id, 
                    yesterday_str
                )
                print(f"\nHistorical Odds for {historical_odds['home_team']} vs {historical_odds['away_team']}:")
                
                # Print some bookmakers
                if historical_odds.get('bookmakers'):
                    for bookmaker in historical_odds['bookmakers'][:2]:
                        print(f"- {bookmaker['title']}")
                        for market in bookmaker['markets'][:2]:
                            print(f"  - {market['key']}")
                else:
                    print("No bookmaker data available")
            except Exception as e:
                logger.error(f"Error getting historical game odds: {e}")
    except Exception as e:
        logger.error(f"Error getting historical events: {e}")
    
    logger.info("Example completed")


if __name__ == "__main__":
    run_example()
