"""
API Test Script

This script tests the NBA Prediction System API endpoints to verify they're working correctly.
Run this script after starting the API server to ensure all endpoints are functioning as expected.

Usage:
    python -m tests.test_api [--host localhost] [--port 5000] [--token your_api_token]
"""

import argparse
import json
import sys
import time
import requests
from datetime import datetime

# Add parent directory to path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_health_endpoint(base_url):
    """Test the health check endpoint"""
    print("\n1. Testing health check endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health")
        response.raise_for_status()
        data = response.json()
        print(f"  ✓ Health check successful: {data['status']}")
        print(f"  ✓ Server version: {data.get('version', 'unknown')}")
        return True
    except Exception as e:
        print(f"  ✗ Health check failed: {str(e)}")
        return False


def test_models_list(base_url):
    """Test the models list endpoint"""
    print("\n2. Testing models list endpoint...")
    try:
        response = requests.get(f"{base_url}/api/models/list")
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'success':
            models = data.get('models', [])
            print(f"  ✓ Found {len(models)} models")
            if models:
                print("  ✓ Model examples:")
                for i, model in enumerate(models[:3]):
                    print(f"    - {model['model_name']} ({model['prediction_target']}) v{model['version']}")
            return True
        else:
            print(f"  ✗ Error: {data.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"  ✗ Models list failed: {str(e)}")
        return False


def test_training_status(base_url):
    """Test the training status endpoint"""
    print("\n3. Testing training status endpoint...")
    try:
        response = requests.get(f"{base_url}/api/training/status")
        response.raise_for_status()
        data = response.json()
        print(f"  ✓ Training status: {'In progress' if data.get('is_training', False) else 'Not running'}")
        print(f"  ✓ Queue size: {data.get('queue_size', 'unknown')}")
        return True
    except Exception as e:
        print(f"  ✗ Training status failed: {str(e)}")
        return False


def test_model_performance(base_url):
    """Test the model performance endpoint"""
    print("\n4. Testing model performance endpoint...")
    try:
        response = requests.get(f"{base_url}/api/models/performance?days=7")
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'success':
            performance = data.get('performance', [])
            print(f"  ✓ Found performance data for {len(performance)} model variants")
            if performance:
                print("  ✓ Performance examples:")
                for i, perf in enumerate(performance[:2]):
                    model_name = perf.get('model_name')
                    target = perf.get('prediction_target')
                    history = perf.get('metrics_history', [])
                    print(f"    - {model_name} ({target}): {len(history)} data points")
                    if history:
                        metrics = history[0].get('metrics', {})
                        accuracy = metrics.get('accuracy', 'N/A')
                        print(f"      Last accuracy: {accuracy}")
            return True
        else:
            print(f"  ✗ Error: {data.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"  ✗ Model performance failed: {str(e)}")
        return False


def test_model_drift(base_url):
    """Test the model drift endpoint"""
    print("\n5. Testing model drift endpoint...")
    try:
        response = requests.get(f"{base_url}/api/models/drift")
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'success':
            drift_results = data.get('drift_results', [])
            print(f"  ✓ Checked drift for {len(drift_results)} model variants")
            drifting_models = [m for m in drift_results if m.get('drift_detected', False)]
            print(f"  ✓ Found {len(drifting_models)} models with drift detected")
            if drifting_models:
                print("  ✓ Models with drift:")
                for i, model in enumerate(drifting_models[:3]):
                    print(f"    - {model['model_name']} ({model['prediction_target']})")
            return True
        else:
            print(f"  ✗ Error: {data.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"  ✗ Model drift check failed: {str(e)}")
        return False


def test_predictions_today(base_url):
    """Test the today's predictions endpoint"""
    print("\n6. Testing today's predictions endpoint...")
    try:
        response = requests.get(f"{base_url}/api/predictions/today")
        response.raise_for_status()
        data = response.json()
        if data.get('status') == 'success':
            game_count = data.get('game_count', 0)
            predictions = data.get('predictions', [])
            print(f"  ✓ Found predictions for {game_count} games")
            if predictions:
                print("  ✓ Prediction examples:")
                for i, game in enumerate(predictions[:2]):
                    home_team = game.get('home_team', 'Unknown')
                    away_team = game.get('away_team', 'Unknown')
                    print(f"    - {away_team} @ {home_team}")
                    ml_pred = game.get('moneyline_prediction', {})
                    if ml_pred:
                        print(f"      Moneyline: {ml_pred.get('prediction', 'N/A')} ({ml_pred.get('confidence', 0):.2f})")
            return True
        else:
            print(f"  ✗ Error: {data.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"  ✗ Today's predictions failed: {str(e)}")
        return False


def test_authenticated_endpoint(base_url, token):
    """Test an authenticated endpoint (training start)"""
    if not token:
        print("\n7. Skipping authenticated endpoint test (no token provided)")
        return True
        
    print("\n7. Testing authenticated endpoint (training start)...")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        # Just testing authentication, not actually starting training (dry run)
        response = requests.post(
            f"{base_url}/api/training/start", 
            json={"force": False, "dry_run": True},
            headers=headers
        )
        
        if response.status_code == 401:
            print("  ✗ Authentication failed: Invalid token")
            return False
            
        response.raise_for_status()
        data = response.json()
        print(f"  ✓ Authentication successful")
        print(f"  ✓ Response: {data.get('message', 'Unknown')}")
        return True
    except Exception as e:
        print(f"  ✗ Authenticated endpoint test failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test the NBA Prediction System API')
    parser.add_argument('--host', default='localhost', help='API server host')
    parser.add_argument('--port', default=5000, type=int, help='API server port')
    parser.add_argument('--token', default=None, help='API token for protected endpoints')
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    print(f"\nTesting NBA Prediction System API at {base_url}")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    tests = [
        test_health_endpoint(base_url),
        test_models_list(base_url),
        test_training_status(base_url),
        test_model_performance(base_url),
        test_model_drift(base_url),
        test_predictions_today(base_url),
        test_authenticated_endpoint(base_url, args.token)
    ]
    
    # Print summary
    print("\n" + "-" * 50)
    print(f"Test Summary: {sum(tests)}/{len(tests)} tests passed")
    print("-" * 50)
    
    if all(tests):
        print("\n✅ All tests passed! The API is functioning correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the API server logs for errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
