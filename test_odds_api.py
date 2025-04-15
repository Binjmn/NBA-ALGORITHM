import sys
import logging
import traceback
from datetime import datetime, timedelta
from src.api.theodds_client import TheOddsClient

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def test_date_formats():
    print("Testing various date formats with The Odds API...\n")
    
    try:
        client = TheOddsClient()
        print("Client initialized successfully")
    except Exception as e:
        print(f"Error initializing TheOddsClient: {str(e)}")
        traceback.print_exc()
        return
    
    # Test various date formats
    test_dates = [
        # Try a date we know should have NBA games
        "2022-10-18",  # NBA regular season 2022-23 start date
        "2022-04-16",  # NBA playoffs 2022
        "2021-12-25",  # Christmas games 2021
        "2021-02-14",  # Valentine's Day 2021
        # Try more recent dates
        "2023-10-24",  # NBA regular season 2023-24 start date
        "2023-12-25",  # Christmas games 2023
        (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),  # 10 days ago
        (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),  # 30 days ago
    ]
    
    for date_str in test_dates:
        print(f"\nTesting date: {date_str}")
        try:
            # Try with only the basic markets
            odds = client.get_historical_odds(
                date=date_str,
                markets=["h2h"]
            )
            if odds and len(odds) > 0:
                print(f"SUCCESS! Found {len(odds)} games for {date_str}")
                # Print first game details
                if len(odds) > 0:
                    print(f"Sample game: {odds[0]['home_team']} vs {odds[0]['away_team']}")
                    break  # We found a working format, stop testing
            else:
                print(f"No odds data found for {date_str}")
        except Exception as e:
            print(f"Error getting odds for {date_str}: {str(e)}")

def test_current_odds():
    print("\nTesting current NBA odds...")
    
    try:
        client = TheOddsClient()
    except Exception as e:
        print(f"Error initializing TheOddsClient: {str(e)}")
        traceback.print_exc()
        return
    
    try:
        # Get current NBA odds
        current_odds = client.get_nba_odds(markets=["h2h", "spreads", "totals"])
        if current_odds and len(current_odds) > 0:
            print(f"SUCCESS! Found {len(current_odds)} current NBA games")
            # Print the first game details
            if len(current_odds) > 0:
                print(f"Sample game: {current_odds[0]['home_team']} vs {current_odds[0]['away_team']}")
        else:
            print("No current NBA odds found")
    except Exception as e:
        print(f"Error getting current odds: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    test_date_formats()
    test_current_odds()
