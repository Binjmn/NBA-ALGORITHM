"""
Combined BallDontLie and The Odds API Example

This script demonstrates how to use both the BallDontLie and The Odds API clients
together with the NBADataProcessor to combine data from both sources.

Usage:
    python -m src.examples.combined_api_example
"""

import json
import logging
import os
from datetime import datetime
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'combined_api_example.log'))
    ]
)
logger = logging.getLogger(__name__)

# Import our clients and data processor
from src.api.balldontlie_client import BallDontLieClient
from src.api.theodds_client import TheOddsClient
from src.utils.data_processor import NBADataProcessor


def pretty_print(data: any) -> None:
    """
    Pretty print JSON data
    
    Args:
        data: Data to print
    """
    print(json.dumps(data, indent=2, default=str))
    print("-" * 80)


def get_current_est_time() -> datetime:
    """
    Get current time in Eastern Standard Time
    
    Returns:
        datetime: Current time in EST
    """
    # Create a UTC datetime
    utc_now = datetime.now(pytz.UTC)
    # Convert to EST
    est = pytz.timezone('US/Eastern')
    return utc_now.astimezone(est)


def run_example():
    """Run the combined API example"""
    logger.info("Starting combined API example")
    
    # Display current time in EST (following the project rule for time standardization)
    est_now = get_current_est_time()
    print(f"Current time (EST): {est_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("-" * 80)
    
    # Create the clients
    try:
        balldontlie_client = BallDontLieClient()
        theodds_client = TheOddsClient()
    except ValueError as e:
        logger.error(f"Failed to initialize clients: {e}")
        return
    
    # Create the data processor
    data_processor = NBADataProcessor(
        balldontlie_client=balldontlie_client,
        theodds_client=theodds_client
    )
    
    logger.info("Clients and processor initialized successfully")
    
    # 1. Get today's games with odds
    logger.info("Getting today's games with odds...")
    try:
        games_with_odds = data_processor.get_todays_games_with_odds()
        print(f"\nToday's Games with Odds ({len(games_with_odds)} games):")
        
        # Check if we have any games
        if games_with_odds:
            for game in games_with_odds:
                home = game['home_team']['name']
                away = game['visitor_team']['name']
                status = game['status']
                
                # Check if we have odds
                if game['odds']:
                    has_odds = "✓"
                    # Get the first bookmaker's moneyline odds if available
                    if game['odds']['bookmakers']:
                        bookmaker = game['odds']['bookmakers'][0]
                        moneyline_market = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
                        if moneyline_market:
                            home_odds = next((o['price'] for o in moneyline_market['outcomes'] 
                                            if home.lower() in o['name'].lower()), "N/A")
                            away_odds = next((o['price'] for o in moneyline_market['outcomes']
                                            if away.lower() in o['name'].lower()), "N/A")
                            odds_text = f"({home_odds}/{away_odds})"
                        else:
                            odds_text = "(odds available)"
                    else:
                        odds_text = "(odds available)"
                else:
                    has_odds = "✗"
                    odds_text = ""
                
                print(f"- {home} vs {away} - {status} - Odds: {has_odds} {odds_text}")
            
            # Show detailed data for the first game
            first_game = games_with_odds[0]
            print("\nDetailed data for first game:")
            pretty_print(first_game)
            
            # If we found a game with odds, let's also get live data for it
            if first_game['odds']:
                # 2. Get live game data
                game_id = first_game['id']
                logger.info(f"Getting live data for game {game_id}...")
                try:
                    live_data = data_processor.get_live_game_data(game_id)
                    print(f"\nLive Data for {first_game['home_team']['name']} vs {first_game['visitor_team']['name']}:")
                    
                    # Show odds data if available
                    if 'odds_api_data' in live_data:
                        print("\nOdds API Live Data:")
                        if live_data['odds_api_data'].get('scores'):
                            for score in live_data['odds_api_data']['scores']:
                                print(f"- {score['name']}: {score.get('score', 'N/A')}")
                        
                        print(f"Game completed: {live_data['odds_api_data'].get('completed', False)}")
                    else:
                        print("No live odds data available")
                    
                    # Show some box score data
                    if 'data' in live_data and live_data['data']:
                        print("\nBox Score Data:")
                        home_score = live_data['data'].get('home_team_score', 'N/A')
                        away_score = live_data['data'].get('visitor_team_score', 'N/A')
                        print(f"Score: {first_game['home_team']['name']} {home_score} - {away_score} {first_game['visitor_team']['name']}")
                        print(f"Period: {live_data['data'].get('period', 'N/A')}")
                        print(f"Status: {live_data['data'].get('status', 'N/A')}")
                    
                    # Don't print the entire live data as it's too large
                    # pretty_print(live_data)
                except Exception as e:
                    logger.error(f"Error getting live game data: {e}")
        else:
            print("No games scheduled for today")
    except Exception as e:
        logger.error(f"Error getting today's games with odds: {e}")
    
    # 3. Get team data and combine with standings
    logger.info("Getting team data and standings...")
    try:
        teams = balldontlie_client.get_teams()
        standings = balldontlie_client.get_standings()
        
        # Create a dictionary of standings by team ID
        standings_by_team = {}
        for standing in standings.get('data', []):
            team_id = standing.get('team', {}).get('id')
            if team_id:
                standings_by_team[team_id] = standing
        
        print("\nTeams with Standings:")
        for team in teams.get('data', [])[:5]:  # Show first 5 teams only
            team_id = team['id']
            team_name = team['full_name']
            
            standing = standings_by_team.get(team_id, {})
            wins = standing.get('wins', 'N/A')
            losses = standing.get('losses', 'N/A')
            rank = standing.get('conference_rank', 'N/A')
            
            print(f"- {team_name}: {wins}-{losses} (Rank: {rank})")
    except Exception as e:
        logger.error(f"Error getting team data and standings: {e}")
    
    # 4. Check historical odds data
    logger.info("Getting historical odds data...")
    try:
        # Get yesterday's date in ISO format
        yesterday = (est_now.replace(hour=0, minute=0, second=0, microsecond=0) - 
                    datetime.timedelta(days=1)).isoformat()
        
        historical_events = theodds_client.get_historical_events(yesterday)
        print(f"\nHistorical Events for Yesterday ({len(historical_events)} games):")
        for event in historical_events[:3]:  # Show first 3 events
            home_team = event['home_team']
            away_team = event['away_team']
            print(f"- {home_team} vs {away_team}")
        
        if historical_events:
            print("\nNote: Historical odds data requires additional API credits")
            print("To avoid consuming credits, we won't retrieve the actual odds in this example")
    except Exception as e:
        logger.error(f"Error getting historical events: {e}")
    
    # Close clients
    logger.info("Closing clients and data processor")
    data_processor.close()
    
    logger.info("Example completed")


if __name__ == "__main__":
    run_example()
