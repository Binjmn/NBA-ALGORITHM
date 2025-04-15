#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automatic Retraining Scheduler for NBA Prediction System

This script sets up an automatic schedule to keep NBA prediction models up-to-date
with the latest game data, ensuring models are always trained on the most recent information.

Features:
- Configurable update schedule (daily, weekly, or after games)
- Automatic data collection from APIs
- Model retraining with performance validation
- Email notifications of retraining results
- Error handling and logging for unattended operation
- Configurable thresholds for model deployment
"""

import os
import sys
import logging
import argparse
import schedule
import time
from datetime import datetime, timedelta
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import json
import subprocess

# Add the src directory to the path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now import our modules
from src.database.connection import init_db, get_connection_pool
from src.database.models import ModelPerformance, ModelWeights, SystemLog
from src.api.theodds_client import OddsApiCollector
from src.api.balldontlie_client import HistoricalDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'auto_update_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutoUpdateScheduler:
    """Scheduler for automatic model retraining and updates"""
    
    def __init__(self, config=None):
        """
        Initialize the scheduler
        
        Args:
            config: Optional configuration dict or path to config file
        """
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config) if config else self._get_default_config()
        logger.info(f"Initialized AutoUpdateScheduler with config: {self.config}")
        
        # Initialize database connection
        if not init_db():
            logger.error("Failed to initialize database connection")
            raise ConnectionError("Failed to connect to database")
            
        # Email notification settings
        self.email_enabled = self.config.get('email_notifications', {}).get('enabled', False)
        if self.email_enabled:
            self.email_config = self.config.get('email_notifications', {})
            # Validate email config
            required_fields = ['smtp_server', 'smtp_port', 'sender_email', 'receiver_emails']
            if not all(field in self.email_config for field in required_fields):
                logger.warning("Incomplete email configuration, notifications will be disabled")
                self.email_enabled = False
        
        # Performance thresholds for model deployment
        self.performance_thresholds = self.config.get('performance_thresholds', {})
        
        # Last update tracking
        self.last_update = None

    def _load_config(self, config):
        """Load configuration from file or dict"""
        if isinstance(config, dict):
            return config
        elif isinstance(config, (str, Path)):
            try:
                with open(config, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config file: {str(e)}")
                return self._get_default_config()
        else:
            return self._get_default_config()

    def _get_default_config(self):
        """Get default configuration"""
        return {
            'update_schedule': {
                'frequency': 'daily',  # daily, weekly, after_games
                'time': '02:00',  # 24-hour format, for daily/weekly updates
                'day_of_week': 1,  # 0=Monday for weekly updates
                'check_for_games': True  # Only update if there were new games
            },
            'data_collection': {
                'historical_days': 30,  # Days of historical data to fetch
                'force_collection': False  # Force data collection even if no new games
            },
            'email_notifications': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'sender_email': '',
                'sender_password': '',
                'receiver_emails': []
            },
            'performance_thresholds': {
                'accuracy_min': 0.58,  # Minimum accuracy for deployment
                'rmse_max': 7.5,  # Maximum RMSE for deployment
                'auc_min': 0.65  # Minimum AUC for deployment
            },
            'retrain_options': {
                'models': ['RandomForestModel', 'GradientBoostingModel', 'BayesianModel',
                          'CombinedGradientBoostingModel', 'EnsembleModel', 'EnsembleStackingModel'],
                'skip_deployment': False
            }
        }

    def send_email_notification(self, subject, body):
        """Send email notification about update results"""
        if not self.email_enabled:
            logger.info("Email notifications disabled, skipping")
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['receiver_emails'])
            msg['Subject'] = f"NBA Prediction System: {subject}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['sender_email'], self.email_config['sender_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            return False

    def check_for_new_games(self, days=1):
        """Check if new games have been played since last update"""
        try:
            collector = HistoricalDataCollector()
            
            # Get most recent games
            today = datetime.now().date()
            start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")
            
            recent_games = collector.get_games_by_date_range(start_date, end_date)
            
            if recent_games.empty:
                logger.info(f"No games found between {start_date} and {end_date}")
                return False
            
            # If last update exists, check if games are newer
            if self.last_update:
                most_recent_game = pd.to_datetime(recent_games['date']).max()
                if most_recent_game <= self.last_update:
                    logger.info(f"No new games since last update ({self.last_update})")
                    return False
            
            logger.info(f"Found {len(recent_games)} games between {start_date} and {end_date}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking for new games: {str(e)}")
            return False

    def run_retraining(self):
        """Run model retraining process"""
        try:
            logger.info("Starting model retraining process")
            
            # Check if we need to update based on new games
            if self.config['update_schedule']['check_for_games'] and not self.check_for_new_games():
                if not self.config['data_collection']['force_collection']:
                    logger.info("No new games and force_collection=False, skipping update")
                    return False
                logger.info("No new games but force_collection=True, proceeding with update")
            
            # Build command to run training pipeline
            script_path = os.path.join('src', 'training_pipeline.py')
            cmd = [sys.executable, script_path]
            
            # Add force_collection flag if configured
            if self.config['data_collection']['force_collection']:
                cmd.append('--force-collection')
            
            # Add specific models if configured
            if self.config['retrain_options'].get('models'):
                cmd.append('--models')
                cmd.extend(self.config['retrain_options']['models'])
            
            # Add skip_deployment flag if configured
            if self.config['retrain_options'].get('skip_deployment', False):
                cmd.append('--skip-deployment')
            
            # Run the training pipeline as a subprocess
            logger.info(f"Running command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Capture output
            stdout, stderr = process.communicate()
            
            # Check if retraining was successful
            if process.returncode == 0:
                logger.info("Model retraining completed successfully")
                self.last_update = datetime.now()
                
                # Log successful update
                self._log_update_to_db(True, "Retraining completed successfully", stdout)
                
                # Send email notification if enabled
                if self.email_enabled:
                    self.send_email_notification(
                        "Model Retraining Successful",
                        f"NBA Prediction models were successfully retrained at {self.last_update}.\n\n" +
                        f"Output:\n{stdout[:1000]}..."
                    )
                
                return True
            else:
                logger.error(f"Model retraining failed with code {process.returncode}")
                logger.error(f"Error output: {stderr}")
                
                # Log failed update
                self._log_update_to_db(False, f"Retraining failed with code {process.returncode}", stderr)
                
                # Send email notification if enabled
                if self.email_enabled:
                    self.send_email_notification(
                        "Model Retraining FAILED",
                        f"NBA Prediction model retraining failed at {datetime.now()}.\n\n" +
                        f"Error output:\n{stderr}"
                    )
                
                return False
                
        except Exception as e:
            logger.error(f"Error running retraining process: {str(e)}")
            
            # Log failed update
            self._log_update_to_db(False, f"Retraining error: {str(e)}", str(e))
            
            # Send email notification if enabled
            if self.email_enabled:
                self.send_email_notification(
                    "Model Retraining ERROR",
                    f"NBA Prediction model retraining encountered an error at {datetime.now()}.\n\n" +
                    f"Error: {str(e)}"
                )
            
            return False

    def _log_update_to_db(self, success, message, output=""):
        """Log update attempt to database"""
        try:
            log_data = {
                "timestamp": datetime.now(),
                "success": success,
                "message": message,
                "config": self.config,
                "output": output[:2000]  # Truncate output to avoid too much data
            }
            
            # Convert to JSON for storage
            log_json = json.dumps(log_data)
            
            # Store in system_logs table
            with get_connection_pool().getconn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                    INSERT INTO system_logs (log_type, timestamp, data)
                    VALUES (%s, %s, %s)
                    """, ('auto_update', datetime.now(), log_json))
                    
                conn.commit()
                
            logger.info("Update logged to database")
            return True
            
        except Exception as e:
            logger.error(f"Error logging update to database: {str(e)}")
            return False

    def setup_schedule(self):
        """Set up the update schedule based on configuration"""
        frequency = self.config['update_schedule']['frequency']
        
        if frequency == 'daily':
            # Schedule daily update at specified time
            time_str = self.config['update_schedule'].get('time', '02:00')
            logger.info(f"Setting up daily schedule at {time_str}")
            schedule.every().day.at(time_str).do(self.run_retraining)
            
        elif frequency == 'weekly':
            # Schedule weekly update on specified day and time
            day_of_week = self.config['update_schedule'].get('day_of_week', 0)  # Default to Monday
            time_str = self.config['update_schedule'].get('time', '02:00')
            
            days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            day_name = days[day_of_week] if 0 <= day_of_week < 7 else 'monday'
            
            logger.info(f"Setting up weekly schedule on {day_name} at {time_str}")
            getattr(schedule.every(), day_name).at(time_str).do(self.run_retraining)
            
        elif frequency == 'after_games':
            # Schedule to check daily for new games
            time_str = self.config['update_schedule'].get('time', '09:00')  # Check in the morning
            logger.info(f"Setting up daily game check at {time_str}")
            schedule.every().day.at(time_str).do(self.run_retraining)
            
        else:
            logger.warning(f"Unknown frequency '{frequency}', defaulting to daily at 02:00")
            schedule.every().day.at('02:00').do(self.run_retraining)
            
        return schedule

    def run_scheduler(self):
        """Run the scheduler continuously"""
        logger.info("Starting the automatic update scheduler")
        
        # Set up the schedule
        self.setup_schedule()
        
        # Run immediately if requested
        if self.config.get('run_immediately', False):
            logger.info("Running initial update immediately")
            self.run_retraining()
        
        # Run the schedule loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")


def create_config_file(output_path="auto_update_config.json"):
    """Create a default configuration file"""
    scheduler = AutoUpdateScheduler()
    config = scheduler._get_default_config()
    
    try:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Created default configuration file at {output_path}")
        return True
    except Exception as e:
        print(f"Error creating configuration file: {str(e)}")
        return False


def main():
    """Main function to run the scheduler"""
    parser = argparse.ArgumentParser(description='NBA Prediction Model Auto-Update Scheduler')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--create-config', action='store_true', help='Create a default configuration file')
    parser.add_argument('--output', type=str, default='auto_update_config.json', help='Output path for configuration file')
    parser.add_argument('--run-now', action='store_true', help='Run update immediately then start scheduler')
    args = parser.parse_args()

    # Create default configuration file if requested
    if args.create_config:
        success = create_config_file(args.output)
        if not success:
            return

    # Load configuration if provided
    config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
            return
    
    # Set run_immediately flag if requested
    if config and args.run_now:
        config['run_immediately'] = True
    
    # Initialize scheduler
    try:
        scheduler = AutoUpdateScheduler(config)
        
        # Run the scheduler
        scheduler.run_scheduler()
    except Exception as e:
        logger.error(f"Error initializing or running scheduler: {str(e)}")


if __name__ == "__main__":
    main()
