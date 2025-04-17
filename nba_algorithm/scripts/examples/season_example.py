"""
NBA Season Manager Example

This script demonstrates how to use the SeasonManager to detect NBA seasons,
identify the current season phase, and handle season transitions.

Usage:
    python -m src.examples.season_example
"""

import logging
import os
from datetime import datetime, timedelta
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'season_example.log'))
    ]
)
logger = logging.getLogger(__name__)

# Import the season manager and configuration
from src.utils.season_manager import SeasonManager
from src.api.balldontlie_client import BallDontLieClient
from config.season_config import SeasonPhase, get_season_display_name, IN_SEASON_PHASES


def demonstrate_current_season(season_manager):
    """Demonstrate the current season information"""
    print("\n========== CURRENT SEASON INFO ==========")
    
    season_year = season_manager.get_current_season_year()
    season_display = season_manager.get_current_season_display()
    current_phase = season_manager.get_current_phase()
    is_in_season = season_manager.is_in_season()
    
    print(f"Current NBA Season: {season_display} ({season_year})")
    print(f"Current Season Phase: {current_phase.name}")
    print(f"In Active Season: {'Yes' if is_in_season else 'No'}")
    
    # Show days until next phases
    print("\nDays until upcoming phases:")
    for phase in SeasonPhase:
        if phase != current_phase and phase != SeasonPhase.UNKNOWN:
            days = season_manager.days_until_phase(phase)
            print(f"  {phase.name}: {days} days")


def simulate_season_transitions(season_manager):
    """Simulate season transitions"""
    print("\n========== SEASON TRANSITION SIMULATION ==========")
    
    # Get current date in EST
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Simulate a date in the next season
    next_year_date = now.replace(year=now.year + 1, month=1, day=15)
    
    print(f"Current date: {now.strftime('%Y-%m-%d')}")
    print(f"Simulated future date: {next_year_date.strftime('%Y-%m-%d')}")
    
    print("\nCurrent season before transition:")
    print(f"  Season: {season_manager.get_current_season_display()}")
    print(f"  Phase: {season_manager.get_current_phase().name}")
    
    # Update with simulated future date
    season_manager.update_current_season(next_year_date)
    
    print("\nCurrent season after transition:")
    print(f"  Season: {season_manager.get_current_season_display()}")
    print(f"  Phase: {season_manager.get_current_phase().name}")
    
    # Reset to current date
    season_manager.update_current_season()
    
    print("\nReset to current season:")
    print(f"  Season: {season_manager.get_current_season_display()}")
    print(f"  Phase: {season_manager.get_current_phase().name}")


def run_example():
    """Run the season manager example"""
    logger.info("Starting NBA Season Manager example")
    
    # Create API client for season validation
    try:
        api_client = BallDontLieClient()
    except ValueError as e:
        logger.warning(f"Could not initialize API client: {e}")
        api_client = None
    
    # Create the season manager
    season_manager = SeasonManager(
        api_client=api_client,
        data_dir='data',
        use_api_validation=api_client is not None
    )
    
    # Demonstrate current season information
    demonstrate_current_season(season_manager)
    
    # Simulate season transitions
    simulate_season_transitions(season_manager)
    
    logger.info("NBA Season Manager example completed")


if __name__ == "__main__":
    run_example()
