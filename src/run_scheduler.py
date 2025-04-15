#!/usr/bin/env python
"""
NBA Prediction System Runner

This is the main entry point for the NBA Betting Prediction System when running in Docker.
It initializes the season manager, scheduler, and API clients, then starts the scheduler
to run continuously.

The system is designed to be non-coder friendly, with all operations automated and
logging detailed information for troubleshooting.
"""

import logging
import os
import signal
import sys
import time
from datetime import datetime
from flask import Flask, jsonify, request

import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'system.log'))
    ]
)
logger = logging.getLogger(__name__)

# Import our components - with error handling for missing components
try:
    from src.utils.scheduler import PredictionScheduler
    from src.utils.season_manager import SeasonManager
    from src.api.balldontlie_client import BallDontLieClient
    from src.api.theodds_client import TheOddsClient
except ImportError as e:
    logger.critical(f"Failed to import required modules: {e}")
    logger.critical("The system cannot start due to missing components")
    sys.exit(1)

# Create a simple API for health checks and status
app = Flask(__name__)

@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
        'system': {
            'scheduler_running': scheduler.is_active() if 'scheduler' in globals() else False,
            'current_season': season_manager.get_current_season_display() if 'season_manager' in globals() else 'Unknown',
            'current_phase': str(season_manager.get_current_phase()) if 'season_manager' in globals() else 'Unknown'
        }
    })

@app.route('/status')
def system_status():
    """Get detailed system status"""
    if 'scheduler' not in globals():
        return jsonify({'error': 'Scheduler not initialized'}), 500
    
    return jsonify({
        'timestamp': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
        'scheduler_running': scheduler.is_active(),
        'jobs': scheduler.get_all_job_status(),
        'season': {
            'year': season_manager.get_current_season_year(),
            'display': season_manager.get_current_season_display(),
            'phase': str(season_manager.get_current_phase()),
            'in_season': season_manager.is_in_season()
        }
    })

@app.route('/run_job', methods=['POST'])
def run_job():
    """Run a specific job on demand"""
    job_name = request.json.get('job_name')
    if not job_name:
        return jsonify({'error': 'No job_name provided'}), 400
    
    if 'scheduler' not in globals():
        return jsonify({'error': 'Scheduler not initialized'}), 500
    
    success = scheduler.run_job_now(job_name)
    if success:
        return jsonify({'message': f'Job {job_name} triggered successfully'})
    else:
        return jsonify({'error': f'Failed to trigger job {job_name}'}), 400

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Shutdown signal received, stopping scheduler...")
    if 'scheduler' in globals() and scheduler:
        scheduler.shutdown()
    logger.info("Scheduler stopped. Exiting...")
    sys.exit(0)

def main():
    """Main entry point for the NBA Prediction System"""
    logger.info("Starting NBA Betting Prediction System")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize API clients
    try:
        logger.info("Initializing API clients...")
        balldontlie_client = BallDontLieClient()
        theodds_client = TheOddsClient()
        logger.info("API clients initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize API clients: {e}")
        logger.critical("The system cannot start without API access")
        sys.exit(1)
    
    # Initialize season manager
    try:
        logger.info("Initializing season manager...")
        global season_manager
        season_manager = SeasonManager(api_client=balldontlie_client)
        logger.info(f"Season manager initialized. Current season: {season_manager.get_current_season_display()}")
    except Exception as e:
        logger.critical(f"Failed to initialize season manager: {e}")
        logger.critical("The system cannot start without season management")
        sys.exit(1)
    
    # Initialize scheduler
    try:
        logger.info("Initializing prediction scheduler...")
        global scheduler
        scheduler = PredictionScheduler(season_manager=season_manager)
        logger.info("Scheduler initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize scheduler: {e}")
        logger.critical("The system cannot start without the scheduler")
        sys.exit(1)
    
    # Start the scheduler
    try:
        logger.info("Starting the scheduler...")
        scheduler.start()
        logger.info("Scheduler started successfully")
    except Exception as e:
        logger.critical(f"Failed to start scheduler: {e}")
        logger.critical("The system cannot run without an active scheduler")
        sys.exit(1)
    
    # Start the API server in a separate thread
    logger.info("Starting API server for health checks and control...")
    from threading import Thread
    api_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000))
    api_thread.daemon = True
    api_thread.start()
    
    logger.info("NBA Betting Prediction System is now running")
    logger.info("Use the web interface or API endpoints to monitor and control the system")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(60)  # Sleep for 1 minute
            
            # Check if season has changed
            season_transition = season_manager.handle_season_transition()
            if season_transition:
                logger.info(f"Season transition detected! New season: {season_manager.get_current_season_display()}")
    except KeyboardInterrupt:
        # This will be caught by the signal handler
        pass

if __name__ == "__main__":
    main()
