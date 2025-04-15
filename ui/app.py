"""
NBA Prediction System Web Interface

This Flask application provides a user-friendly web interface for the NBA Prediction System,
allowing non-technical users to monitor and control the system without writing code.

Features:
- Dashboard for system status and prediction results
- Controls to start/stop jobs and view logs
- Configuration interface for API keys and settings
- Visualization of model performance
"""

import json
import logging
import os
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for

import pandas as pd
import plotly
import plotly.express as px
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'ui.log'))
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# API configuration
API_URL = os.environ.get('API_URL', 'http://localhost:5000')

# Helper functions
def get_system_status():
    """Get the current status of the prediction system"""
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get system status: {response.status_code}")
            return {"error": f"API returned status code {response.status_code}"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to API: {e}")
        return {"error": "Could not connect to the prediction system"}

def get_health_check():
    """Get basic health check information"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed health check: {response.status_code}")
            return {"status": "unhealthy", "error": f"Status code {response.status_code}"}
    except requests.exceptions.RequestException:
        return {"status": "unhealthy", "error": "Connection failed"}

def get_todays_predictions():
    """Get today's game predictions"""
    # This would normally fetch from our prediction API
    # For now we'll return placeholder data
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern)
    
    return {
        "date": today.strftime("%Y-%m-%d"),
        "games": [
            {
                "home_team": "Los Angeles Lakers",
                "away_team": "Boston Celtics",
                "start_time": (today + timedelta(hours=4)).strftime("%H:%M EST"),
                "prediction": {
                    "winner": "Los Angeles Lakers",
                    "confidence": 68.5,
                    "spread": -3.5,
                    "total": 218.5
                }
            },
            {
                "home_team": "Golden State Warriors",
                "away_team": "Phoenix Suns",
                "start_time": (today + timedelta(hours=6)).strftime("%H:%M EST"),
                "prediction": {
                    "winner": "Phoenix Suns",
                    "confidence": 52.1,
                    "spread": 1.5,
                    "total": 232.0
                }
            }
        ]
    }

def get_model_performance():
    """Get performance metrics for each prediction model"""
    # This would normally fetch from our database
    # For now we'll return placeholder data
    
    # Create date range for last 7 days
    eastern = pytz.timezone('US/Eastern')
    today = datetime.now(eastern)
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7, 0, -1)]
    
    # Simulated accuracy data for our 7 models
    model_data = []
    
    model_names = [
        "Random Forests",
        "XGBoost + LightGBM",
        "Bayesian",
        "Anomaly Detection",
        "Model Mixing",
        "Ensemble Stacking",
        "Hyperparameter Tuning"
    ]
    
    # Generate simulated data for each model
    for model in model_names:
        for date in dates:
            # Simulate different accuracy patterns for each model
            if model == "Random Forests":
                accuracy = 68 + (hash(date + model) % 10)
            elif model == "XGBoost + LightGBM":
                accuracy = 73 + (hash(date + model) % 8)
            elif model == "Bayesian":
                accuracy = 71 + (hash(date + model) % 7)
            elif model == "Anomaly Detection":
                accuracy = 65 + (hash(date + model) % 12)
            elif model == "Model Mixing":
                accuracy = 75 + (hash(date + model) % 6)
            elif model == "Ensemble Stacking":
                accuracy = 76 + (hash(date + model) % 5)
            else:  # Hyperparameter Tuning
                accuracy = 72 + (hash(date + model) % 9)
                
            model_data.append({
                "Date": date,
                "Model": model,
                "Accuracy": min(accuracy, 100)  # Cap at 100%
            })
    
    return pd.DataFrame(model_data)

# Routes
@app.route('/')
def index():
    """Dashboard home page"""
    system_status = get_system_status()
    health = get_health_check()
    
    # Get today's predictions
    predictions = get_todays_predictions()
    
    # Get system is healthy flag
    is_healthy = health.get("status") == "healthy"
    
    return render_template(
        'index.html',
        system_status=system_status,
        health=health,
        is_healthy=is_healthy,
        predictions=predictions
    )

@app.route('/models')
def models():
    """Model performance page"""
    # Get model performance data
    performance_df = get_model_performance()
    
    # Create plotly figure
    fig = px.line(
        performance_df, 
        x="Date", 
        y="Accuracy", 
        color="Model",
        title="7-Day Model Performance",
        labels={"Accuracy": "Accuracy (%)", "Date": "Date"},
        markers=True
    )
    
    # Convert to JSON for the template
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template(
        'models.html',
        graph_json=graph_json,
        system_status=get_system_status()
    )

@app.route('/configuration')
def configuration():
    """System configuration page"""
    # This would normally load from config files
    api_keys = {
        "balldontlie": "●●●●●●●●●●●●●●●●",
        "theodds": "●●●●●●●●●●●●●●●●"
    }
    
    # Get model settings (placeholder)
    model_settings = {
        "Random Forests": {
            "enabled": True,
            "weight": 15,
            "min_confidence": 60
        },
        "XGBoost + LightGBM": {
            "enabled": True,
            "weight": 20,
            "min_confidence": 65
        },
        "Bayesian": {
            "enabled": True,
            "weight": 15,
            "min_confidence": 70
        },
        "Anomaly Detection": {
            "enabled": False,
            "weight": 10,
            "min_confidence": 75
        },
        "Model Mixing": {
            "enabled": True,
            "weight": 15,
            "min_confidence": 60
        },
        "Ensemble Stacking": {
            "enabled": True,
            "weight": 20,
            "min_confidence": 65
        },
        "Hyperparameter Tuning": {
            "enabled": True,
            "weight": 5,
            "min_confidence": 60
        }
    }
    
    return render_template(
        'configuration.html',
        api_keys=api_keys,
        model_settings=model_settings,
        system_status=get_system_status()
    )

@app.route('/logs')
def logs():
    """System logs page"""
    # This would normally fetch from log files
    # For now, return placeholder logs
    sample_logs = [
        {"timestamp": "2025-04-14 06:00:01", "level": "INFO", "service": "scheduler", "message": "Starting daily data collection"},
        {"timestamp": "2025-04-14 06:00:23", "level": "INFO", "service": "data_collection", "message": "Collected data for 12 games"},
        {"timestamp": "2025-04-14 06:01:05", "level": "INFO", "service": "data_processor", "message": "Processing daily data"},
        {"timestamp": "2025-04-14 06:05:12", "level": "INFO", "service": "model_training", "message": "Running model evaluation"},
        {"timestamp": "2025-04-14 06:07:45", "level": "INFO", "service": "predictions", "message": "Generated predictions for today's games"},
        {"timestamp": "2025-04-14 10:00:01", "level": "INFO", "service": "scheduler", "message": "Starting odds update"},
        {"timestamp": "2025-04-14 10:00:18", "level": "INFO", "service": "data_collection", "message": "Updated odds for 12 games"},
        {"timestamp": "2025-04-14 14:00:01", "level": "INFO", "service": "scheduler", "message": "Starting odds update"},
        {"timestamp": "2025-04-14 14:00:20", "level": "INFO", "service": "data_collection", "message": "Updated odds for 12 games"},
    ]
    
    return render_template(
        'logs.html',
        logs=sample_logs,
        system_status=get_system_status()
    )

@app.route('/run-job', methods=['POST'])
def run_job():
    """Endpoint to run a job manually"""
    job_name = request.form.get('job_name')
    if not job_name:
        return jsonify({'error': 'No job specified'}), 400
    
    try:
        response = requests.post(
            f"{API_URL}/run_job",
            json={"job_name": job_name},
            timeout=5
        )
        
        if response.status_code == 200:
            return redirect(url_for('index', job_success=job_name))
        else:
            return redirect(url_for('index', job_error=f"Failed to run {job_name}: {response.text}"))
    except requests.exceptions.RequestException as e:
        return redirect(url_for('index', job_error=f"Connection error: {str(e)}"))

@app.route('/health')
def health():
    """Health check endpoint for the UI"""
    api_health = get_health_check()
    return jsonify({
        "ui_status": "healthy",
        "api_status": api_health.get("status", "unknown"),
        "timestamp": datetime.now(pytz.timezone('US/Eastern')).isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
