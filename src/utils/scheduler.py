"""
NBA Prediction System Scheduler

This module provides a robust scheduling system for the NBA prediction pipeline,
coordinating data collection, processing, model training, and prediction generation
based on configurable schedules and the current NBA season phase.

Features:
- Season-aware scheduling (only runs relevant jobs based on current season phase)
- Job dependencies (ensures prerequisites are met before running a job)
- Error handling with automatic retries
- Logging of job execution and performance
- Dynamic job loading from configuration

Usage:
    from src.utils.scheduler import PredictionScheduler
    
    # Create scheduler
    scheduler = PredictionScheduler()
    
    # Start scheduler
    scheduler.start()
"""

import importlib
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor

from config.scheduler_config import (
    ACTIVE_JOBS,
    JOB_CONFIG_DEFAULTS,
    JobType,
    JobPriority
)
from src.utils.season_manager import SeasonManager

# Configure logging
logger = logging.getLogger(__name__)


class PredictionScheduler:
    """
    Scheduler for NBA prediction system jobs
    
    This class manages the scheduling and execution of all automated jobs
    in the NBA prediction system, ensuring they run at appropriate times
    and in the correct sequence based on the current NBA season.
    """
    
    def __init__(
        self,
        season_manager: Optional[SeasonManager] = None,
        job_configs: Optional[List[Dict[str, Any]]] = None,
        max_threads: int = 10,
        max_processes: int = 4
    ):
        """
        Initialize the NBA prediction scheduler
        
        Args:
            season_manager (Optional[SeasonManager]): Season manager instance
            job_configs (Optional[List[Dict[str, Any]]]): Job configuration overrides
            max_threads (int): Maximum number of threads for thread pool
            max_processes (int): Maximum number of processes for process pool
        """
        # Create a season manager if not provided
        if season_manager is None:
            self.season_manager = SeasonManager()
        else:
            self.season_manager = season_manager
            
        # Set up job configurations
        self.job_configs = job_configs or ACTIVE_JOBS
        
        # Initialize job tracking
        self.jobs = {}
        self.job_history = {}
        self.job_status = {}
        
        # Create APScheduler
        jobstores = {
            'default': MemoryJobStore()
        }
        executors = {
            'default': ThreadPoolExecutor(max_threads),
            'processpool': ProcessPoolExecutor(max_processes)
        }
        job_defaults = {
            'coalesce': True,
            'max_instances': 1,
            'misfire_grace_time': 60
        }
        
        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=pytz.timezone('US/Eastern')  # Use EST as per project requirements
        )
        
        # Initialize but don't start yet
        self._initialized = False
        logger.info("Initialized NBA prediction scheduler")
    
    def initialize(self) -> None:
        """Initialize the scheduler and load jobs"""
        if self._initialized:
            logger.warning("Scheduler is already initialized")
            return
        
        logger.info("Initializing scheduler...")
        
        # First check if season transition has occurred
        self.season_manager.handle_season_transition()
        
        # Load all jobs from configuration
        self._load_jobs()
        
        self._initialized = True
        logger.info("Scheduler initialization complete")
    
    def start(self) -> None:
        """Start the scheduler"""
        if not self._initialized:
            self.initialize()
        
        logger.info("Starting scheduler...")
        self.scheduler.start()
        logger.info("Scheduler started")
    
    def shutdown(self) -> None:
        """Shutdown the scheduler"""
        logger.info("Shutting down scheduler...")
        self.scheduler.shutdown()
        logger.info("Scheduler shut down")
    
    def _load_jobs(self) -> None:
        """Load all jobs from configuration"""
        logger.info(f"Loading {len(self.job_configs)} jobs...")
        
        # Get current season phase
        current_phase = self.season_manager.get_current_phase()
        
        for job_config in self.job_configs:
            # Apply defaults to job config
            for key, default_value in JOB_CONFIG_DEFAULTS.items():
                if key not in job_config:
                    job_config[key] = default_value
                    
            job_name = job_config['name']
            logger.debug(f"Processing job configuration: {job_name}")
            
            # Check if job should be enabled in current phase
            if current_phase not in job_config['active_phases']:
                logger.info(f"Job {job_name} not active in current phase {current_phase}")
                continue
                
            if not job_config['enabled']:
                logger.info(f"Job {job_name} is disabled")
                continue
            
            try:
                # Create a job wrapper function
                job_func = self._create_job_wrapper(job_config)
                
                # Create trigger based on schedule definition
                trigger = self._create_trigger(job_config['schedule'])
                
                if trigger:
                    # Add job to scheduler
                    self.scheduler.add_job(
                        func=job_func,
                        trigger=trigger,
                        id=job_name,
                        name=job_name,
                        max_instances=1,
                        coalesce=True,
                        misfire_grace_time=60
                    )
                    
                    # Update job tracking
                    self.jobs[job_name] = job_config
                    self.job_status[job_name] = {
                        'last_run': None,
                        'last_success': None,
                        'last_failure': None,
                        'run_count': 0,
                        'success_count': 0,
                        'failure_count': 0,
                        'running': False
                    }
                    
                    logger.info(f"Added job: {job_name}")
                else:
                    logger.warning(f"Skipped job {job_name}: No schedule defined")
            except Exception as e:
                logger.error(f"Failed to add job {job_name}: {e}")
    
    def _create_trigger(self, schedule: Union[Dict[str, Any], str, None]) -> Any:
        """
        Create a scheduler trigger from a schedule definition
        
        Args:
            schedule (Union[Dict[str, Any], str, None]): Schedule definition
            
        Returns:
            Any: APScheduler trigger object or None if schedule is None
        """
        if schedule is None:
            return None
            
        if isinstance(schedule, dict):
            # Check if it's an interval schedule (has keys like 'seconds', 'minutes', etc.)
            interval_keys = ['seconds', 'minutes', 'hours', 'days', 'weeks']
            if any(key in schedule for key in interval_keys):
                return IntervalTrigger(**schedule, timezone=pytz.timezone('US/Eastern'))
            
            # Otherwise assume it's a cron schedule
            return CronTrigger(**schedule, timezone=pytz.timezone('US/Eastern'))
        
        if isinstance(schedule, str):
            # Assume cron expression
            return CronTrigger.from_crontab(schedule, timezone=pytz.timezone('US/Eastern'))
            
        logger.warning(f"Unknown schedule type: {type(schedule)}")
        return None
    
    def _create_job_wrapper(self, job_config: Dict[str, Any]) -> Callable:
        """
        Create a wrapper function for a job
        
        Args:
            job_config (Dict[str, Any]): Job configuration
            
        Returns:
            Callable: Wrapped job function
        """
        job_name = job_config['name']
        module_name = job_config['module']
        function_name = job_config['function']
        args = job_config['args']
        kwargs = job_config['kwargs']
        timeout = job_config['timeout']
        retry_config = job_config['retry']
        depends_on = job_config['depends_on']
        
        def job_wrapper():
            """Wrapper function for the job"""
            # Update job status
            self.job_status[job_name]['last_run'] = datetime.now(pytz.timezone('US/Eastern'))
            self.job_status[job_name]['run_count'] += 1
            self.job_status[job_name]['running'] = True
            
            # Check season transition before running
            if self.season_manager.handle_season_transition():
                logger.info(f"Season transition detected before running {job_name}")
                
                # Check if job should still run in new phase
                current_phase = self.season_manager.get_current_phase()
                if current_phase not in job_config['active_phases']:
                    logger.info(f"Job {job_name} not active in new phase {current_phase}")
                    self.job_status[job_name]['running'] = False
                    return
            
            # Check dependencies
            for dependency in depends_on:
                if dependency not in self.job_status:
                    logger.warning(f"Job {job_name} depends on unknown job {dependency}")
                    continue
                    
                dep_status = self.job_status[dependency]
                if not dep_status['last_success']:
                    logger.warning(f"Job {job_name} dependency {dependency} has never succeeded")
                    self.job_status[job_name]['running'] = False
                    return
            
            # Load the function
            try:
                module = importlib.import_module(module_name)
                func = getattr(module, function_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to load function {module_name}.{function_name}: {e}")
                self.job_status[job_name]['last_failure'] = datetime.now(pytz.timezone('US/Eastern'))
                self.job_status[job_name]['failure_count'] += 1
                self.job_status[job_name]['running'] = False
                return
            
            # Run the job with retry logic
            for retry in range(retry_config['max_retries'] + 1):
                try:
                    logger.info(f"Running job {job_name} (attempt {retry + 1})")
                    
                    # Add season context to kwargs
                    job_kwargs = kwargs.copy()
                    job_kwargs['season_year'] = self.season_manager.get_current_season_year()
                    job_kwargs['season_phase'] = self.season_manager.get_current_phase()
                    
                    # Execute the function
                    result = func(*args, **job_kwargs)
                    
                    # Update job history
                    self.job_history.setdefault(job_name, []).append({
                        'timestamp': datetime.now(pytz.timezone('US/Eastern')),
                        'success': True,
                        'result': str(result)
                    })
                    
                    # Update job status
                    self.job_status[job_name]['last_success'] = datetime.now(pytz.timezone('US/Eastern'))
                    self.job_status[job_name]['success_count'] += 1
                    
                    logger.info(f"Job {job_name} completed successfully")
                    break
                except Exception as e:
                    logger.error(f"Job {job_name} failed (attempt {retry + 1}): {e}")
                    
                    if retry < retry_config['max_retries']:
                        delay = retry_config['delay']
                        logger.info(f"Retrying job {job_name} in {delay} seconds")
                        time.sleep(delay)
                    else:
                        # Update job history
                        self.job_history.setdefault(job_name, []).append({
                            'timestamp': datetime.now(pytz.timezone('US/Eastern')),
                            'success': False,
                            'error': str(e)
                        })
                        
                        # Update job status
                        self.job_status[job_name]['last_failure'] = datetime.now(pytz.timezone('US/Eastern'))
                        self.job_status[job_name]['failure_count'] += 1
            
            # Update running status
            self.job_status[job_name]['running'] = False
        
        return job_wrapper
    
    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """
        Get status of a specific job
        
        Args:
            job_name (str): Name of the job
            
        Returns:
            Dict[str, Any]: Job status information
        """
        if job_name not in self.job_status:
            logger.warning(f"Job {job_name} not found")
            return {}
            
        return self.job_status[job_name]
    
    def get_all_job_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all jobs
        
        Returns:
            Dict[str, Dict[str, Any]]: Status of all jobs
        """
        return self.job_status
    
    def run_job_now(self, job_name: str) -> bool:
        """
        Run a specific job immediately
        
        Args:
            job_name (str): Name of the job
            
        Returns:
            bool: True if job was triggered, False otherwise
        """
        if job_name not in self.jobs:
            logger.warning(f"Job {job_name} not found")
            return False
            
        try:
            self.scheduler.modify_job(job_id=job_name)
            self.scheduler.run_job(job_id=job_name)
            logger.info(f"Triggered immediate execution of job {job_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to run job {job_name} immediately: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
        
    def is_active(self) -> bool:
        """
        Check if the scheduler is active
        
        Returns:
            bool: True if scheduler is running
        """
        return self.scheduler.running
