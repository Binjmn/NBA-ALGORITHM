"""
Simplified NBA Prediction System API Server

This module provides a basic Flask server for demonstrating the dashboard.
It uses mock data instead of actual database connections.
"""

import json
import logging
import os
import random
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import Flask, render_template, send_from_directory, jsonify, request
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(
    __name__,
    template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ui/templates')),
    static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ui/static'))
)
CORS(app)  # Enable CORS for all domains on all routes

# API configuration
API_TOKEN = os.environ.get("API_TOKEN", "dev_token_change_in_production")

# Mock data for demonstration
MOCK_MODELS = [
    {"model_name": "RandomForest", "model_type": "classifier", "prediction_target": "moneyline", "version": 1, "trained_at": "2025-04-10T12:00:00Z", "active": True, "needs_training": False},
    {"model_name": "XGBoost", "model_type": "classifier", "prediction_target": "moneyline", "version": 2, "trained_at": "2025-04-12T14:30:00Z", "active": True, "needs_training": False},
    {"model_name": "Bayesian", "model_type": "probabilistic", "prediction_target": "moneyline", "version": 1, "trained_at": "2025-04-08T09:15:00Z", "active": True, "needs_training": True},
    {"model_name": "AnomalyDetection", "model_type": "unsupervised", "prediction_target": "moneyline", "version": 1, "trained_at": "2025-04-05T10:45:00Z", "active": False, "needs_training": True},
    {"model_name": "ModelMixing", "model_type": "ensemble", "prediction_target": "moneyline", "version": 1, "trained_at": "2025-04-13T16:20:00Z", "active": True, "needs_training": False},
    {"model_name": "EnsembleStacking", "model_type": "ensemble", "prediction_target": "moneyline", "version": 1, "trained_at": "2025-04-14T08:30:00Z", "active": True, "needs_training": False},
    {"model_name": "RandomForest", "model_type": "classifier", "prediction_target": "spread", "version": 1, "trained_at": "2025-04-09T11:30:00Z", "active": True, "needs_training": False},
    {"model_name": "XGBoost", "model_type": "classifier", "prediction_target": "spread", "version": 1, "trained_at": "2025-04-11T13:45:00Z", "active": True, "needs_training": False},
    {"model_name": "RandomForest", "model_type": "classifier", "prediction_target": "total", "version": 1, "trained_at": "2025-04-10T14:15:00Z", "active": True, "needs_training": False},
    {"model_name": "XGBoost", "model_type": "classifier", "prediction_target": "total", "version": 1, "trained_at": "2025-04-12T15:00:00Z", "active": True, "needs_training": False},
]

# Generate mock performance data
def generate_mock_performance_data():
    performance_data = []
    
    for model in MOCK_MODELS:
        metrics_history = []
        base_accuracy = random.uniform(0.65, 0.85)  # Base accuracy between 65% and 85%
        
        # Generate 30 days of history
        for i in range(30):
            date = datetime.now() - timedelta(days=i)
            # Add some random variation to accuracy
            daily_accuracy = base_accuracy + random.uniform(-0.05, 0.05)
            daily_accuracy = max(0.5, min(0.95, daily_accuracy))  # Keep within reasonable bounds
            
            metrics_history.append({
                "metrics": {
                    "accuracy": daily_accuracy,
                    "precision": daily_accuracy - random.uniform(0, 0.1),
                    "recall": daily_accuracy - random.uniform(0, 0.1),
                    "f1_score": daily_accuracy - random.uniform(0, 0.05)
                },
                "created_at": date.strftime("%Y-%m-%dT%H:%M:%SZ")
            })
        
        performance_data.append({
            "model_name": model["model_name"],
            "prediction_target": model["prediction_target"],
            "metrics_history": metrics_history
        })
    
    return performance_data

# Generate mock drift data
def generate_mock_drift_data():
    drift_results = []
    
    for model in MOCK_MODELS:
        # Random drift detection
        has_drift = random.random() < 0.2  # 20% chance of drift
        
        drift_results.append({
            "model_name": model["model_name"],
            "prediction_target": model["prediction_target"],
            "drift_detected": has_drift,
            "current_metrics": {
                "accuracy": random.uniform(0.6, 0.9),
                "precision": random.uniform(0.6, 0.9),
                "recall": random.uniform(0.6, 0.9),
                "f1_score": random.uniform(0.6, 0.9)
            },
            "baseline_metrics": {
                "accuracy": random.uniform(0.7, 0.95),
                "precision": random.uniform(0.7, 0.95),
                "recall": random.uniform(0.7, 0.95),
                "f1_score": random.uniform(0.7, 0.95)
            }
        })
    
    return drift_results

# Generate mock predictions
def generate_mock_predictions():
    teams = ["Lakers", "Celtics", "Warriors", "Nets", "Bucks", "Heat", "Suns", "76ers", "Mavericks", "Nuggets"]
    games = []
    
    for i in range(random.randint(2, 5)):  # 2-5 games for today
        home_idx = random.randint(0, len(teams) - 1)
        away_idx = random.randint(0, len(teams) - 1)
        while away_idx == home_idx:
            away_idx = random.randint(0, len(teams) - 1)
            
        home_team = teams[home_idx]
        away_team = teams[away_idx]
        
        # Generate random predictions
        ml_confidence = random.uniform(0.55, 0.9)
        spread_confidence = random.uniform(0.55, 0.9)
        total_confidence = random.uniform(0.55, 0.9)
        
        games.append({
            "game_id": f"game_{i+1}",
            "home_team": home_team,
            "away_team": away_team,
            "start_time": (datetime.now() + timedelta(hours=random.randint(1, 8))).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "moneyline_prediction": {
                "prediction": home_team if random.random() > 0.4 else away_team,
                "confidence": ml_confidence
            },
            "spread_prediction": {
                "prediction": "home" if random.random() > 0.5 else "away",
                "predicted_value": random.randint(1, 12),
                "confidence": spread_confidence
            },
            "total_prediction": {
                "prediction": "over" if random.random() > 0.5 else "under",
                "predicted_value": random.randint(200, 240),
                "confidence": total_confidence
            }
        })
    
    return games


# Authentication decorator
def require_api_token(f):
    """Decorator to require API token for protected endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            token = token[7:]  # Remove 'Bearer ' prefix
            
        if not token or token != API_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function


# Route for serving the dashboard
@app.route('/')
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# Route for serving static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is operational"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.4"
    })


# Model training endpoints
@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get training status"""
    return jsonify({
        "is_training": False,  # Not training by default
        "queue_size": 0,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.route('/api/training/start', methods=['POST'])
@require_api_token
def start_training():
    """Start model training"""
    data = request.json or {}
    force = data.get('force', False)
    models = data.get('models', None)
    
    return jsonify({
        "status": "success",
        "message": "Training started",
        "force": force,
        "models": models
    })


@app.route('/api/training/cancel', methods=['POST'])
@require_api_token
def cancel_training():
    """Cancel ongoing training"""
    return jsonify({
        "status": "success",
        "message": "Training queue cleared, current model training will complete"
    })


# Model endpoints
@app.route('/api/models/list', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        "status": "success",
        "models": MOCK_MODELS
    })


@app.route('/api/models/<model_name>/details', methods=['GET'])
def get_model_details(model_name):
    """Get details for a specific model"""
    prediction_target = request.args.get('prediction_target', 'moneyline')
    
    # Filter models by name and prediction target
    models = [m for m in MOCK_MODELS if m["model_name"] == model_name and m["prediction_target"] == prediction_target]
    
    if not models:
        return jsonify({
            "status": "error",
            "message": "Model not found"
        }), 404
    
    model = models[0]  # Get the first matching model
    
    # Generate some mock performance history
    performance_history = []
    base_accuracy = random.uniform(0.65, 0.85)
    
    for i in range(10):
        date = datetime.now() - timedelta(days=i)
        daily_accuracy = base_accuracy + random.uniform(-0.05, 0.05)
        
        performance_history.append({
            "metrics": {
                "accuracy": daily_accuracy,
                "precision": daily_accuracy - random.uniform(0, 0.1),
                "recall": daily_accuracy - random.uniform(0, 0.1),
                "f1_score": daily_accuracy - random.uniform(0, 0.05)
            },
            "created_at": date.strftime("%Y-%m-%dT%H:%M:%SZ")
        })
    
    # Add the performance history to the model details
    model_details = dict(model)
    model_details["params"] = {
        "prediction_target": model["prediction_target"],
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    model_details["weights"] = "Binary model weights (not displayed)"
    model_details["performance_history"] = performance_history
    
    return jsonify({
        "status": "success",
        "model": model_details
    })


@app.route('/api/models/<model_name>/retrain', methods=['POST'])
@require_api_token
def retrain_model(model_name):
    """Retrain a specific model"""
    data = request.json or {}
    prediction_target = data.get('prediction_target')
    force = data.get('force', True)
    
    if not prediction_target:
        return jsonify({
            "status": "error",
            "message": "prediction_target is required"
        }), 400
    
    return jsonify({
        "status": "success",
        "message": f"Retraining of {model_name} ({prediction_target}) started"
    })


# Performance endpoints
@app.route('/api/models/performance', methods=['GET'])
def get_performance_metrics():
    """Get performance metrics for all models"""
    days = request.args.get('days', '7')
    try:
        days = int(days)
    except ValueError:
        days = 7
    
    # Generate mock performance data
    performance = generate_mock_performance_data()
    
    return jsonify({
        "status": "success",
        "days": days,
        "performance": performance
    })


@app.route('/api/models/drift', methods=['GET'])
def check_model_drift():
    """Check for model drift"""
    # Generate mock drift data
    drift_results = generate_mock_drift_data()
    
    return jsonify({
        "status": "success",
        "drift_results": drift_results
    })


# Prediction endpoints
@app.route('/api/predictions/today', methods=['GET'])
def get_todays_predictions():
    """Get predictions for today's games"""
    # Generate mock predictions
    predictions = generate_mock_predictions()
    
    return jsonify({
        "status": "success",
        "game_count": len(predictions),
        "predictions": predictions
    })


# Main entry point
def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the API server"""
    try:
        # Run the server
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error running API server: {str(e)}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "5000"))
    debug = os.environ.get("API_DEBUG", "False").lower() == "true"
    
    print(f"\n* Starting NBA Prediction System API Server (Demo Mode) *")
    print(f"* Dashboard will be available at: http://localhost:{port}/dashboard")
    print(f"* API endpoints will be available at: http://localhost:{port}/api/*")
    print(f"* Press Ctrl+C to stop the server *\n")
    
    run_server(host=host, port=port, debug=debug)
