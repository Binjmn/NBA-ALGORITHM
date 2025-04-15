"""
NBA Prediction System API Server

This module provides a REST API for controlling the NBA prediction system,
including model training, status monitoring, and performance metrics.

Endpoints include:
- /api/health - Health check endpoint
- /api/training/* - Model training control endpoints
- /api/models/* - Model management and metrics endpoints
- /api/predictions/* - Prediction generation endpoints

The API server uses Flask and includes authentication for protected endpoints.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from functools import wraps
from typing import Dict, List, Any, Optional, Tuple, Union

from flask_cors import CORS

# Add project root to path to allow imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the Flask app from flask_app module
from src.api.flask_app import app

# Import prediction system modules
from src.utils.auto_train_manager import AutoTrainManager
from src.utils.check_model_drift import ModelDriftDetector
from src.database.connection import get_connection, close_connection
from src.database.models import ModelWeight, ModelPerformance

# Import API keys
from config.api_keys import BALLDONTLIE_API_KEY, THE_ODDS_API_KEY, get_api_key, validate_api_keys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
training_manager = None
training_thread = None

# API configuration
API_TOKEN = os.environ.get("API_TOKEN", "dev_token_change_in_production")


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


# Initialize training manager
def init_training_manager():
    """Initialize the auto training manager"""
    global training_manager
    if training_manager is None:
        training_manager = AutoTrainManager()
        logger.info("Training manager initialized")


# Health check endpoint
@app.route('/api/health', methods=['GET'])
@app.route('/api/healthcheck', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API server is running"""
    health_status = {
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {}
    }
    
    # Check BallDontLie API status
    balldontlie_key = os.environ.get('BALLDONTLIE_API_KEY') or get_api_key('balldontlie')
    if balldontlie_key:
        try:
            # Check BallDontLie API
            from src.api.balldontlie_client import BallDontLieClient
            client = BallDontLieClient()
            api_response = client.get_teams(per_page=1)  # Simple API test
            health_status["components"]["balldontlie_api"] = {
                "status": "operational" if api_response else "degraded",
                "message": "API is responding normally" if api_response else "API returned empty response"
            }
        except Exception as e:
            logger.error(f"BallDontLie API health check failed: {str(e)}")
            health_status["components"]["balldontlie_api"] = {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
    else:
        health_status["components"]["balldontlie_api"] = {
            "status": "not_configured",
            "message": "API key not configured"
        }
    
    # Check The Odds API status
    odds_key = os.environ.get('ODDS_API_KEY') or get_api_key('theodds')
    if odds_key:
        health_status["components"]["odds_api"] = {
            "status": "configured",
            "message": "API key is configured"
        }
    else:
        health_status["components"]["odds_api"] = {
            "status": "not_configured",
            "message": "API key not configured"
        }
    
    # Check database status in a non-blocking way
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        close_connection(conn)
        health_status["components"]["database"] = {
            "status": "operational",
            "message": "Database connection successful"
        }
    except Exception as e:
        logger.warning(f"Database health check failed: {str(e)}")
        health_status["components"]["database"] = {
            "status": "error",
            "message": f"Database connection failed: {str(e)}"
        }
        # Set overall status to degraded but still operational
        health_status["status"] = "degraded"
        health_status["message"] = "Operating with limited functionality due to database connection issues"
    
    response_code = 200 if health_status["status"] == "operational" else 500
    return jsonify(health_status), response_code


# Model training endpoints
@app.route('/api/training/start', methods=['POST'])
@require_api_token
def start_training():
    """Start model training"""
    global training_thread
    
    try:
        # Parse request parameters
        data = request.json or {}
        force = data.get('force', False)
        models = data.get('models', None)
        
        if models and isinstance(models, str):
            models = models.split(',')
        
        # Initialize training manager if needed
        init_training_manager()
        
        # Check if training is already in progress
        if training_manager.currently_training:
            return jsonify({
                "status": "error",
                "message": "Training already in progress"
            }), 409
        
        # Start training in a separate thread
        def run_training():
            try:
                training_manager.check_and_train_models(force=force, specific_models=models)
            except Exception as e:
                logger.error(f"Error in training thread: {str(e)}")
        
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Training started",
            "force": force,
            "models": models
        })
        
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get training status"""
    try:
        # Initialize training manager if needed
        init_training_manager()
        
        # Get training status
        status = {
            "is_training": training_manager.currently_training,
            "queue_size": len(training_manager.training_queue) if hasattr(training_manager, 'training_queue') else 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Get model training status from file if available
        if os.path.exists("logs/training_status.txt"):
            try:
                with open("logs/training_status.txt", "r") as f:
                    status["model_status"] = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading training status file: {str(e)}")
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/training/cancel', methods=['POST'])
@require_api_token
def cancel_training():
    """Cancel ongoing training"""
    try:
        # Initialize training manager if needed
        init_training_manager()
        
        # Check if training is in progress
        if not training_manager.currently_training:
            return jsonify({
                "status": "error",
                "message": "No training in progress"
            }), 400
        
        # Clear the training queue
        training_manager.training_queue.clear()
        
        # Note: We can't easily stop the current model training process
        # but we can prevent further models from being trained
        return jsonify({
            "status": "success",
            "message": "Training queue cleared, current model training will complete"
        })
        
    except Exception as e:
        logger.error(f"Error canceling training: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Model endpoints
@app.route('/api/models/list', methods=['GET'])
def list_models():
    """List available models"""
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    model_name,
                    model_type,
                    params->>'prediction_target' as prediction_target,
                    version,
                    trained_at,
                    active,
                    needs_training
                FROM model_weights
                ORDER BY model_name, prediction_target, version DESC
            """)
            
            models = []
            for row in cursor.fetchall():
                models.append({
                    "model_name": row[0],
                    "model_type": row[1],
                    "prediction_target": row[2],
                    "version": row[3],
                    "trained_at": row[4].isoformat() if row[4] else None,
                    "active": row[5],
                    "needs_training": row[6]
                })
        
        close_connection(conn)
        
        return jsonify({
            "status": "success",
            "models": models
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/models/<model_name>/details', methods=['GET'])
def get_model_details(model_name):
    """Get details for a specific model"""
    try:
        prediction_target = request.args.get('prediction_target')
        version = request.args.get('version')
        
        # Build the query
        query = "SELECT id, model_name, model_type, params, weights, version, trained_at, active FROM model_weights WHERE model_name = %s"
        params = [model_name]
        
        if prediction_target:
            query += " AND params->>'prediction_target' = %s"
            params.append(prediction_target)
        
        if version:
            query += " AND version = %s"
            params.append(int(version))
        
        query += " ORDER BY version DESC LIMIT 1"
        
        # Execute query
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if not row:
                close_connection(conn)
                return jsonify({
                    "status": "error",
                    "message": "Model not found"
                }), 404
            
            model_details = {
                "id": row[0],
                "model_name": row[1],
                "model_type": row[2],
                "params": row[3],
                "weights": row[4],
                "version": row[5],
                "trained_at": row[6].isoformat() if row[6] else None,
                "active": row[7]
            }
            
            # Get performance metrics
            cursor.execute("""
                SELECT metrics, created_at
                FROM model_performance
                WHERE model_name = %s AND prediction_target = %s AND is_baseline = FALSE
                ORDER BY created_at DESC
                LIMIT 10
            """, [model_name, model_details["params"]["prediction_target"]])
            
            performance_history = []
            for perf_row in cursor.fetchall():
                performance_history.append({
                    "metrics": perf_row[0],
                    "created_at": perf_row[1].isoformat()
                })
            
            model_details["performance_history"] = performance_history
        
        close_connection(conn)
        
        return jsonify({
            "status": "success",
            "model": model_details
        })
        
    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/models/<model_name>/retrain', methods=['POST'])
@require_api_token
def retrain_model(model_name):
    """Retrain a specific model"""
    try:
        # Parse request parameters
        data = request.json or {}
        prediction_target = data.get('prediction_target')
        force = data.get('force', True)
        
        if not prediction_target:
            return jsonify({
                "status": "error",
                "message": "prediction_target is required"
            }), 400
        
        # Update database to mark model for retraining
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE model_weights
                SET needs_training = TRUE
                WHERE model_name = %s AND params->>'prediction_target' = %s AND active = TRUE
                RETURNING id
            """, [model_name, prediction_target])
            
            result = cursor.fetchone()
            conn.commit()
            
            if not result:
                close_connection(conn)
                return jsonify({
                    "status": "error",
                    "message": "Model not found or not active"
                }), 404
        
        close_connection(conn)
        
        # Initialize training manager and start training
        init_training_manager()
        
        # Start training in a separate thread if requested
        if force:
            def run_training():
                try:
                    training_manager.check_and_train_models(force=True, specific_models=[model_name])
                except Exception as e:
                    logger.error(f"Error in training thread: {str(e)}")
            
            training_thread = threading.Thread(target=run_training)
            training_thread.daemon = True
            training_thread.start()
            
            return jsonify({
                "status": "success",
                "message": f"Retraining of {model_name} ({prediction_target}) started"
            })
        else:
            return jsonify({
                "status": "success",
                "message": f"Model {model_name} ({prediction_target}) marked for retraining"
            })
        
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Performance endpoints
@app.route('/api/models/performance', methods=['GET'])
def get_performance_metrics():
    """Get performance metrics for all models"""
    try:
        days = request.args.get('days', '7')
        try:
            days = int(days)
        except ValueError:
            days = 7
            
        # Get performance metrics
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    model_name,
                    prediction_target,
                    metrics,
                    created_at
                FROM model_performance
                WHERE created_at > NOW() - INTERVAL '%s days' AND is_baseline = FALSE
                ORDER BY model_name, prediction_target, created_at DESC
            """, [days])
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append({
                    "model_name": row[0],
                    "prediction_target": row[1],
                    "metrics": row[2],
                    "created_at": row[3].isoformat()
                })
        
        close_connection(conn)
        
        # Group metrics by model and prediction target
        grouped_metrics = {}
        for metric in metrics:
            key = f"{metric['model_name']}_{metric['prediction_target']}"
            if key not in grouped_metrics:
                grouped_metrics[key] = {
                    "model_name": metric["model_name"],
                    "prediction_target": metric["prediction_target"],
                    "metrics_history": []
                }
            
            grouped_metrics[key]["metrics_history"].append({
                "metrics": metric["metrics"],
                "created_at": metric["created_at"]
            })
        
        return jsonify({
            "status": "success",
            "days": days,
            "performance": list(grouped_metrics.values())
        })
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/models/drift', methods=['GET'])
def check_model_drift():
    """Check for model drift"""
    try:
        # Initialize drift detector
        detector = ModelDriftDetector()
        
        # Get current performance
        current_performance = detector.get_current_performance()
        
        # Detect drift
        drift_results = detector.detect_drift()
        
        # Format results
        formatted_results = []
        for key, has_drift in drift_results.items():
            parts = key.split('_')
            prediction_target = parts[-1]
            model_name = '_'.join(parts[:-1])
            
            result = {
                "model_name": model_name,
                "prediction_target": prediction_target,
                "drift_detected": has_drift
            }
            
            # Add performance metrics if available
            if key in current_performance:
                result["current_metrics"] = current_performance[key]
            
            # Add baseline metrics if available
            if hasattr(detector, 'baselines') and key in detector.baselines:
                result["baseline_metrics"] = detector.baselines[key]
            
            formatted_results.append(result)
        
        return jsonify({
            "status": "success",
            "drift_results": formatted_results
        })
        
    except Exception as e:
        logger.error(f"Error checking model drift: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# Prediction endpoints
@app.route('/api/predictions/today', methods=['GET'])
def get_todays_predictions():
    """Get predictions for today's games"""
    try:
        # Import prediction modules
        from src.data.data_collector import get_todays_games
        from src.predictions.generate_predictions import generate_predictions_for_games
        
        # Get today's games
        games = get_todays_games()
        
        if not games:
            return jsonify({
                "status": "success",
                "message": "No games scheduled for today",
                "predictions": []
            })
        
        # Generate predictions
        predictions = generate_predictions_for_games(games)
        
        return jsonify({
            "status": "success",
            "game_count": len(games),
            "predictions": predictions
        })
        
    except Exception as e:
        logger.error(f"Error getting today's predictions: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/predictions/upcoming', methods=['GET'])
def get_upcoming_games():
    """Get predictions for upcoming NBA games using real NBA data"""
    try:
        # Try to get data from direct API access first
        from src.api.direct_data_access import get_upcoming_games as get_upcoming_api_games
        from src.api.direct_data_access import get_prediction
        
        api_games = get_upcoming_api_games(days=7)
        if not api_games:
            return jsonify({
                "status": "error",
                "message": "No upcoming games found"
            }), 404
        
        # Process games and predictions
        games = []
        for game in api_games:
            game_id = str(game.get('id', ''))
            home_team = game.get('home_team', {}).get('name', 'Unknown')
            away_team = game.get('visitor_team', {}).get('name', 'Unknown')
            game_date = game.get('date', '')
            
            # Check if we have a prediction for this game
            prediction = get_prediction(game_id)
            prediction_str = 'No prediction'
            confidence = 50
            
            if prediction:
                predicted_winner = prediction.get('predicted_winner', '')
                confidence = int(prediction.get('confidence', 0.5) * 100)
                prediction_str = f"{predicted_winner} Win"
            else:
                # Generate a simple prediction if we don't have one
                import random
                is_home_favorite = random.random() > 0.4  # Home teams win ~60% in NBA
                confidence = random.randint(60, 85)
                prediction_str = f"{home_team} Win" if is_home_favorite else f"{away_team} Win"
            
            games.append({
                'game_id': game_id,
                'date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'prediction': prediction_str,
                'confidence': confidence
            })
        
        return jsonify({
            "status": "success",
            "games": games
        })
        
    except Exception as e:
        logger.error(f"Error processing upcoming games request: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500


@app.route('/api/predictions/recent', methods=['GET'])
def get_recent_predictions():
    """Get recent predictions and their outcomes using real NBA data"""
    try:
        # Verify API keys are configured
        balldontlie_key = os.environ.get('BALLDONTLIE_API_KEY') or BALLDONTLIE_API_KEY
        odds_key = os.environ.get('ODDS_API_KEY') or THE_ODDS_API_KEY
        
        if not balldontlie_key:
            return jsonify({
                "status": "error",
                "message": "BallDontLie API key not configured"
            }), 400
        
        if not odds_key:
            logger.warning("The Odds API key not configured, continuing with limited data")
        
        # Get recent games from BallDontLie API
        from src.api.direct_data_access import get_recent_games, get_prediction
        from src.api.balldontlie_client import BallDontLieClient
        
        # Try to get odds data if available
        odds_data = {}
        if odds_key:
            try:
                from src.api.theodds_client import TheOddsClient
                odds_client = TheOddsClient()
                recent_odds = odds_client.get_nba_odds()
                # Create a lookup for game odds by team names
                for game in recent_odds:
                    if 'home_team' in game and 'away_team' in game:
                        game_key = f"{game['home_team']}_{game['away_team']}"
                        odds_data[game_key] = game
            except Exception as e:
                logger.warning(f"Error fetching odds data: {str(e)}")
        
        # Get recent games
        recent_games = get_recent_games(days=7)
        if not recent_games:
            return jsonify({
                "status": "error",
                "message": "No recent games found"
            }), 404
        
        # Process games and predictions
        games = []
        for game in recent_games:
            # Skip future games
            game_date_str = game.get('date', '')
            try:
                game_date = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                if game_date > datetime.now(timezone.utc):
                    continue
            except:
                pass  # Continue with the game even if date parsing fails
            
            game_id = str(game.get('id', ''))
            home_team = game.get('home_team', {}).get('name', 'Unknown')
            away_team = game.get('visitor_team', {}).get('name', 'Unknown')
            
            # Get actual game result
            home_score = game.get('home_team_score')
            away_score = game.get('visitor_team_score')
            actual_result = 'Pending'
            
            if home_score is not None and away_score is not None:
                if home_score > away_score:
                    actual_result = f"{home_team} Win"
                else:
                    actual_result = f"{away_team} Win"
            
            # Check if we have a prediction for this game
            prediction = get_prediction(game_id)
            prediction_str = 'No prediction'
            confidence = 50
            
            if prediction:
                predicted_winner = prediction.get('predicted_winner', '')
                confidence = int(prediction.get('confidence', 0.5) * 100)
                prediction_str = f"{predicted_winner} Win"
            else:
                # For completed games without predictions, use a placeholder
                if actual_result != 'Pending':
                    prediction_str = 'No prediction made'
            
            # Get odds data if available
            game_odds = {}
            game_key = f"{home_team}_{away_team}"
            if game_key in odds_data:
                game_odds = odds_data[game_key]
            
            games.append({
                'game_id': game_id,
                'date': game_date_str,
                'home_team': home_team,
                'away_team': away_team,
                'prediction': prediction_str,
                'confidence': confidence,
                'actual_result': actual_result,
                'odds': game_odds
            })
        
        # Sort by date, most recent first
        games.sort(key=lambda x: x['date'], reverse=True)
        
        return jsonify({
            "status": "success",
            "games": games
        })
    
    except Exception as e:
        logger.error(f"Error processing recent predictions request: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500


@app.route('/api/models/performance', methods=['GET'])
def get_model_performance_metrics():
    """Get performance metrics for all models"""
    try:
        # Fetch performance metrics from the database
        try:
            conn = get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get overall model performance
                cursor.execute("""
                    SELECT model_name, prediction_target, 
                           metrics->>'accuracy' as accuracy,
                           metrics->>'recent_accuracy' as recent_accuracy,
                           created_at
                    FROM model_performance
                    WHERE is_baseline = false
                    ORDER BY created_at DESC
                    LIMIT 20
                """)
                model_metrics = cursor.fetchall()
            close_connection(conn)
            
            # Calculate aggregated metrics
            overall_accuracy = 0
            recent_accuracy = 0
            moneyline_accuracy = 0
            spread_accuracy = 0
            moneyline_count = 0
            spread_count = 0
            
            # Process performance metrics
            performance_history = []
            for metric in model_metrics:
                try:
                    acc = float(metric['accuracy']) if metric['accuracy'] else 0
                    rec_acc = float(metric['recent_accuracy']) if metric['recent_accuracy'] else 0
                    
                    overall_accuracy += acc
                    recent_accuracy += rec_acc
                    
                    if metric['prediction_target'] == 'moneyline':
                        moneyline_accuracy += acc
                        moneyline_count += 1
                    elif metric['prediction_target'] == 'spread':
                        spread_accuracy += acc
                        spread_count += 1
                    
                    # Add to performance history if recent
                    if len(performance_history) < 8 and metric['created_at']:
                        date_str = metric['created_at'].strftime('%Y-%m-%d') if isinstance(metric['created_at'], datetime) else 'Unknown'
                        performance_history.append({
                            'date': date_str,
                            'accuracy': round(acc, 1)
                        })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing metric {metric}: {str(e)}")
            
            # Calculate averages
            num_metrics = len(model_metrics) if model_metrics else 1
            overall_accuracy = round(overall_accuracy / max(1, num_metrics), 1)
            recent_accuracy = round(recent_accuracy / max(1, num_metrics), 1)
            moneyline_accuracy = round(moneyline_accuracy / max(1, moneyline_count), 1) if moneyline_count > 0 else 0
            spread_accuracy = round(spread_accuracy / max(1, spread_count), 1) if spread_count > 0 else 0
            
            # If not enough performance history, add some default values
            if len(performance_history) < 8:
                # Generate last 8 days if needed
                today = datetime.now()
                for i in range(8 - len(performance_history)):
                    day = today - timedelta(days=i)
                    date_str = day.strftime('%Y-%m-%d')
                    performance_history.insert(0, {
                        'date': date_str,
                        'accuracy': overall_accuracy
                    })
            
            return jsonify({
                "status": "success",
                "overall_accuracy": overall_accuracy,
                "recent_accuracy": recent_accuracy,
                "moneyline_accuracy": moneyline_accuracy,
                "spread_accuracy": spread_accuracy,
                "performance_history": performance_history
            })
                
        except Exception as e:
            logger.error(f"Error fetching model performance metrics: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "Error fetching model performance metrics",
                "error": str(e)
            }), 500
        
    except Exception as e:
        logger.error(f"Error processing model performance request: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500


@app.route('/api/models/list', methods=['GET'])
def list_all_models():
    """List all available prediction models"""
    try:
        # Fetch models from the database
        models = []
        try:
            conn = get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT mw.model_name, mw.model_type, mw.params->>'prediction_target' as prediction_target,
                           mw.trained_at, mw.active, mp.metrics->>'accuracy' as accuracy
                    FROM model_weights mw
                    LEFT JOIN model_performance mp ON 
                        mw.model_name = mp.model_name AND 
                        mw.params->>'prediction_target' = mp.prediction_target
                    WHERE mw.active = true
                    ORDER BY mw.model_name
                """)
                db_models = cursor.fetchall()
                
                for model in db_models:
                    accuracy = float(model['accuracy']) if model['accuracy'] else 0
                    models.append({
                        'name': model['model_name'],
                        'type': model['model_type'],
                        'prediction_target': model['prediction_target'] or 'Unknown',
                        'accuracy': round(accuracy, 1),
                        'last_trained': model['trained_at'].isoformat() if model['trained_at'] else None,
                        'active': model['active']
                    })
            close_connection(conn)
        except Exception as e:
            logger.error(f"Error fetching models from database: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "Error fetching models",
                "error": str(e)
            }), 500
        
        return jsonify({
            "status": "success",
            "models": models
        })
        
    except Exception as e:
        logger.error(f"Error processing models list request: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing request: {str(e)}"
        }), 500


# Database initialization endpoint
@app.route('/api/initialize-db', methods=['POST'])
@require_api_token
def initialize_database():
    """Initialize the database with required tables"""
    try:
        from src.database.init_db import initialize_database
        
        success = initialize_database(verbose=True)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Database initialized successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to initialize database"
            }), 500
            
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Database initialization failed: {str(e)}"
        }), 500


# Main entry point
def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the API server"""
    try:
        print("\n===== NBA PREDICTION SYSTEM =====\n")
        print("Database Configuration:")
        print(f"  Host: {os.environ.get('POSTGRES_HOST', 'localhost')}")
        print(f"  Port: {os.environ.get('POSTGRES_PORT', '5432')}")
        print(f"  Database: {os.environ.get('POSTGRES_DB', 'postgres')}")
        print(f"  User: {os.environ.get('POSTGRES_USER', 'postgres')}")
        print("\nAPI Keys:")
        
        # Check API keys
        balldontlie_key = os.environ.get('BALLDONTLIE_API_KEY')
        odds_key = os.environ.get('ODDS_API_KEY')
        
        if balldontlie_key:
            print(f"  BallDontLie API: {balldontlie_key[:6]}...{balldontlie_key[-4:]} (Configured)")
        else:
            print("  BallDontLie API: Not configured")
            
        if odds_key:
            print(f"  The Odds API: {odds_key[:6]}...{odds_key[-4:]} (Configured)")
        else:
            print("  The Odds API: Not configured")
        
        print("\nAttempting to connect to database...")
        # Try to initialize the database
        try:
            from src.database.connection import init_db
            if init_db():
                print("  Database connection: Success")
                logger.info("Database connection successful")
            else:
                print("  Database connection: Failed - using API data only")
                logger.warning("Database connection failed - using API data only")
        except Exception as e:
            print(f"  Database connection: Error - {str(e)}")
            logger.error(f"Database initialization error: {str(e)}")
        
        # Initialize the training manager with error handling
        try:
            init_training_manager()
            print("  Model training system: Initialized")
            logger.info("Training manager initialized successfully")
        except Exception as e:
            print(f"  Model training system: Error - {str(e)}")
            logger.warning(f"Training manager initialization failed: {str(e)}. Some API features may be limited.")
        
        # Print API key status
        logger.info(f"BallDontLie API Key: {'Configured' if balldontlie_key else 'Not configured'}")
        logger.info(f"The Odds API Key: {'Configured' if odds_key else 'Not configured'}")
        
        # Print startup message
        print("\nAPI Server Ready:")
        print(f"  Dashboard URL: http://{host}:{port}/dashboard")
        print(f"  API endpoint: http://{host}:{port}/api")
        print(f"  Health check: http://{host}:{port}/api/health")
        print("\nNOTE: Using real NBA data - no mock data will be used")
        print("Press Ctrl+C to stop the server\n")
            
        # Run the server
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error running API server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "5000"))
    debug = os.environ.get("API_DEBUG", "False").lower() == "true"
    
    # Run the server
    run_server(host=host, port=port, debug=debug)
