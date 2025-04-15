import sys
import logging
import traceback
from src.data.historical_collector import HistoricalDataCollector

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def run_diagnostics():
    print("Starting diagnostic tests for historical data collection...")
    
    # Initialize collector
    try:
        collector = HistoricalDataCollector()
        print("Collector initialized successfully")
    except Exception as e:
        print(f"ERROR initializing collector: {str(e)}")
        traceback.print_exc()
        return
    
    # Test teams collection
    try:
        print("\nTesting teams collection...")
        teams = collector.collect_teams()
        print(f"Teams collected: {len(teams)}")
    except Exception as e:
        print(f"ERROR collecting teams: {str(e)}")
        traceback.print_exc()
    
    # Test players collection
    try:
        print("\nTesting players collection...")
        players = collector.collect_players(limit=10)
        print(f"Players collected: {len(players)}")
    except Exception as e:
        print(f"ERROR collecting players: {str(e)}")
        traceback.print_exc()
    
    # Test games collection for a small date range
    try:
        print("\nTesting games collection...")
        start_date = "2024-10-01"
        end_date = "2024-10-15"
        games = collector.collect_games_for_date_range(start_date, end_date)
        print(f"Games collected: {len(games)}")
    except Exception as e:
        print(f"ERROR collecting games: {str(e)}")
        traceback.print_exc()
    
    # Test game stats collection if we have games
    if 'games' in locals() and games:
        try:
            print("\nTesting game stats collection...")
            game_id = str(games[0]['id']) if 'id' in games[0] else None
            if game_id:
                stats = collector.collect_game_stats(game_id)
                print(f"Stats collected for game {game_id}: {bool(stats)}")
            else:
                print("No valid game ID found for stats collection test")
        except Exception as e:
            print(f"ERROR collecting game stats: {str(e)}")
            traceback.print_exc()
    
    # Test historical odds collection
    try:
        print("\nTesting historical odds collection...")
        date = "2024-10-10"
        odds = collector.collect_historical_odds(date)
        print(f"Odds collected for {date}: {len(odds)}")
    except Exception as e:
        print(f"ERROR collecting odds: {str(e)}")
        traceback.print_exc()
    
    print("\nDiagnostic tests completed")

if __name__ == "__main__":
    run_diagnostics()
