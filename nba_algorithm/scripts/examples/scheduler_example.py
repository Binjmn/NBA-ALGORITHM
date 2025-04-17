"""
NBA Prediction Scheduler Example

This script demonstrates how to use the PredictionScheduler to automate tasks for
the NBA Betting Prediction System, integrating with both the SeasonManager and
the data collection tasks.

Usage:
    python -m src.examples.scheduler_example
"""

import logging
import os
import signal
import time
from datetime import datetime, timedelta

import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'scheduler_example.log'))
    ]
)
logger = logging.getLogger(__name__)

# Import our components
from src.utils.scheduler import PredictionScheduler
from src.utils.season_manager import SeasonManager
from src.api.balldontlie_client import BallDontLieClient
from src.api.theodds_client import TheOddsClient


def signal_handler(sig, frame):
    """Handle interrupt signals"""
    logger.info("Interrupt received, shutting down")
    if 'scheduler' in globals() and scheduler:
        scheduler.shutdown()
    exit(0)


def demonstrate_scheduler_initialization():
    """Demonstrate how to initialize the scheduler"""
    print("\n========== SCHEDULER INITIALIZATION ==========")
    
    # Create season manager
    season_manager = SeasonManager()
    
    # Create scheduler
    scheduler = PredictionScheduler(season_manager=season_manager)
    
    # Initialize scheduler
    scheduler.initialize()
    
    print(f"Scheduler initialized with {len(scheduler.jobs)} jobs")
    return scheduler


def demonstrate_job_status(scheduler):
    """Demonstrate how to check job status"""
    print("\n========== JOB STATUS INFORMATION ==========")
    
    # Get all job statuses
    all_status = scheduler.get_all_job_status()
    
    print(f"Total jobs: {len(all_status)}")
    print("\nJob Status Summary:")
    
    for job_name, status in all_status.items():
        last_run = status['last_run'].strftime('%Y-%m-%d %H:%M:%S') if status['last_run'] else 'Never'
        success_rate = 0
        if status['run_count'] > 0:
            success_rate = (status['success_count'] / status['run_count']) * 100
            
        print(f"- {job_name}:")
        print(f"  Last run: {last_run}")
        print(f"  Success rate: {success_rate:.1f}% ({status['success_count']}/{status['run_count']})")
        print(f"  Currently running: {'Yes' if status['running'] else 'No'}")


def demonstrate_manual_job_execution(scheduler):
    """Demonstrate how to manually run a job"""
    print("\n========== MANUAL JOB EXECUTION ==========")
    
    # Run the data collection job manually
    print("Running daily data collection job manually...")
    success = scheduler.run_job_now('daily_data_collection')
    
    if success:
        print("Job triggered successfully")
        # Wait a bit for the job to finish
        time.sleep(2)
        
        # Get the job status
        status = scheduler.get_job_status('daily_data_collection')
        if status:
            if status['last_success'] and status['last_success'] > datetime.now(pytz.timezone('US/Eastern')) - timedelta(minutes=5):
                print("Job completed successfully!")
            else:
                print("Job may have failed or is still running")
    else:
        print("Failed to trigger job")


def run_scheduler_simulation():
    """Run a short simulation of the scheduler"""
    print("\n========== SCHEDULER SIMULATION ==========")
    print("Starting scheduler and running for 60 seconds")
    print("Press Ctrl+C to stop early")
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start the scheduler
        scheduler.start()
        
        # Let it run for a bit
        start_time = time.time()
        while time.time() - start_time < 60:  # Run for 60 seconds
            time.sleep(1)
            
            # Every 10 seconds, print a status update
            if int(time.time() - start_time) % 10 == 0 and int(time.time() - start_time) > 0:
                print(f"Scheduler running for {int(time.time() - start_time)} seconds...")
                
                # Check if any jobs are currently running
                running_jobs = []
                for job_name, status in scheduler.get_all_job_status().items():
                    if status['running']:
                        running_jobs.append(job_name)
                
                if running_jobs:
                    print(f"Currently running jobs: {', '.join(running_jobs)}")
                else:
                    print("No jobs currently running")
    finally:
        # Ensure we always shut down the scheduler
        scheduler.shutdown()
        print("Scheduler shut down")


def run_example():
    """Run the scheduler example"""
    logger.info("Starting NBA Prediction Scheduler example")
    
    print("====================================================")
    print("NBA PREDICTION SCHEDULER EXAMPLE")
    print("====================================================")
    print("This example demonstrates the scheduling system for")
    print("automating NBA prediction tasks.")
    print("====================================================")
    
    # Demonstrate scheduler initialization
    global scheduler
    scheduler = demonstrate_scheduler_initialization()
    
    # Let the user choose what to demonstrate
    print("\nChoose what to demonstrate:")
    print("1. View job status information")
    print("2. Run a job manually")
    print("3. Run scheduler simulation (60 seconds)")
    print("4. All of the above")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == '1' or choice == '4':
        demonstrate_job_status(scheduler)
        
    if choice == '2' or choice == '4':
        demonstrate_manual_job_execution(scheduler)
        
    if choice == '3' or choice == '4':
        run_scheduler_simulation()
    
    logger.info("NBA Prediction Scheduler example completed")


if __name__ == "__main__":
    run_example()
