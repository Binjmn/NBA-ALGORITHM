#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Health Check System for NBA Prediction Pipeline

Provides functionality to verify the operational status of all critical components
of the NBA prediction system, including:
- API connectivity
- Database connection
- Feature data availability
- Model integrity
- File system access

Use this module for both automated monitoring and manual troubleshooting.
"""

import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import project modules
from src.utils.logging_config import configure_logging
from src.api.balldontlie_client import BallDontLieClient
from src.api.theodds_client import TheOddsClient
from src.data.historical_collector import HistoricalDataCollector

# Configure logging
logger = configure_logging(__name__)


class HealthCheck:
    """System health check for the NBA prediction pipeline"""
    
    def __init__(self):
        """Initialize health check system"""
        self.results = {}
        self.status = "unknown"
    
    def check_api_connectivity(self) -> Dict[str, Any]:
        """Check connectivity to external APIs"""
        logger.info("Checking API connectivity")
        results = {}
        
        # Check BallDontLie API
        try:
            start_time = time.time()
            client = BallDontLieClient()
            response = client.get_teams()
            elapsed = time.time() - start_time
            
            if response and isinstance(response, dict) and 'data' in response:
                results["balldontlie"] = {
                    "status": "healthy",
                    "response_time": f"{elapsed:.2f}s",
                    "teams_returned": len(response.get('data', []))
                }
            else:
                results["balldontlie"] = {
                    "status": "degraded",
                    "message": "API returned unexpected data format",
                    "response_time": f"{elapsed:.2f}s"
                }
        except Exception as e:
            logger.error(f"BallDontLie API check failed: {str(e)}")
            results["balldontlie"] = {
                "status": "unhealthy",
                "message": str(e)
            }
        
        # Check The Odds API
        try:
            start_time = time.time()
            client = TheOddsClient()
            response = client.get_sports()
            elapsed = time.time() - start_time
            
            if response and isinstance(response, list):
                results["theodds"] = {
                    "status": "healthy",
                    "response_time": f"{elapsed:.2f}s",
                    "sports_returned": len(response)
                }
            else:
                results["theodds"] = {
                    "status": "degraded",
                    "message": "API returned unexpected data format",
                    "response_time": f"{elapsed:.2f}s"
                }
        except Exception as e:
            logger.error(f"The Odds API check failed: {str(e)}")
            results["theodds"] = {
                "status": "unhealthy",
                "message": str(e)
            }
        
        # Overall API connectivity status
        statuses = [r["status"] for r in results.values()]
        if all(s == "healthy" for s in statuses):
            results["overall"] = {"status": "healthy"}
        elif any(s == "unhealthy" for s in statuses):
            results["overall"] = {"status": "unhealthy"}
        else:
            results["overall"] = {"status": "degraded"}
        
        # Store results
        self.results["api_connectivity"] = results
        return results
    
    def check_database_connection(self) -> Dict[str, Any]:
        """Check database connection and integrity"""
        logger.info("Checking database connection")
        results = {}
        
        try:
            from src.database.connection import get_connection_pool
            
            # Get a connection from the pool
            start_time = time.time()
            pool = get_connection_pool()
            conn = pool.getconn()
            elapsed = time.time() - start_time
            
            # Check if we can execute a simple query
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                row = cursor.fetchone()
                cursor.close()
                
                if row and row[0] == 1:
                    results["connection"] = {
                        "status": "healthy",
                        "response_time": f"{elapsed:.2f}s"
                    }
                else:
                    results["connection"] = {
                        "status": "degraded",
                        "message": "Unexpected query result",
                        "response_time": f"{elapsed:.2f}s"
                    }
            except Exception as e:
                logger.error(f"Database query failed: {str(e)}")
                results["connection"] = {
                    "status": "unhealthy",
                    "message": f"Query failed: {str(e)}"
                }
            finally:
                # Return connection to the pool
                pool.putconn(conn)
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            results["connection"] = {
                "status": "unhealthy",
                "message": str(e)
            }
        
        # Overall database status
        if results.get("connection", {}).get("status") == "healthy":
            results["overall"] = {"status": "healthy"}
        elif results.get("connection", {}).get("status") == "degraded":
            results["overall"] = {"status": "degraded"}
        else:
            results["overall"] = {"status": "unhealthy"}
        
        # Store results
        self.results["database"] = results
        return results
    
    def check_data_availability(self) -> Dict[str, Any]:
        """Check data availability and integrity"""
        logger.info("Checking data availability")
        results = {}
        
        # Check historical data
        data_dir = Path("data/historical")
        feature_dir = Path("data/features")
        
        # Check historical data
        if data_dir.exists():
            games_dir = data_dir / "games"
            if games_dir.exists():
                game_files = list(games_dir.glob("*.json"))
                results["historical_games"] = {
                    "status": "healthy" if game_files else "degraded",
                    "file_count": len(game_files)
                }
            else:
                results["historical_games"] = {
                    "status": "unhealthy",
                    "message": "Games directory not found"
                }
        else:
            results["historical_games"] = {
                "status": "unhealthy",
                "message": "Historical data directory not found"
            }
        
        # Check feature data
        if feature_dir.exists():
            feature_file = feature_dir / "engineered_features.csv"
            if feature_file.exists():
                # Check file age
                file_age_days = (datetime.now() - datetime.fromtimestamp(feature_file.stat().st_mtime)).days
                results["features"] = {
                    "status": "healthy" if file_age_days < 7 else "degraded",
                    "file_age_days": file_age_days,
                    "file_size_kb": feature_file.stat().st_size / 1024
                }
            else:
                results["features"] = {
                    "status": "unhealthy",
                    "message": "Features file not found"
                }
        else:
            results["features"] = {
                "status": "unhealthy",
                "message": "Features directory not found"
            }
        
        # Overall data status
        statuses = [r["status"] for r in results.values()]
        if all(s == "healthy" for s in statuses):
            results["overall"] = {"status": "healthy"}
        elif any(s == "unhealthy" for s in statuses):
            results["overall"] = {"status": "unhealthy"}
        else:
            results["overall"] = {"status": "degraded"}
        
        # Store results
        self.results["data"] = results
        return results
    
    def check_model_integrity(self) -> Dict[str, Any]:
        """Check trained models integrity"""
        logger.info("Checking model integrity")
        results = {}
        
        # Model types to check
        model_types = [
            "RandomForestModel",
            "GradientBoostingModel",
            "BayesianModel",
            "CombinedGradientBoostingModel",
            "EnsembleModel",
            "EnsembleStackingModel"
        ]
        
        # Check each model type
        for model_type in model_types:
            try:
                # Import the model dynamically
                module_name = f"src.models.{model_type.lower()}"
                if model_type == "CombinedGradientBoostingModel":
                    module_name = "src.models.combined_gradient_boosting"
                elif model_type == "EnsembleStackingModel":
                    module_name = "src.models.ensemble_stacking"
                    
                # Try to load model and check if it has been trained
                exec(f"from {module_name} import {model_type}")
                exec(f"model = {model_type}(version=1)")
                model = locals()["model"]
                
                try:
                    loaded = model.load()
                    if loaded and hasattr(model, 'is_trained') and model.is_trained:
                        results[model_type] = {
                            "status": "healthy",
                            "trained_at": getattr(model, 'trained_at', 'unknown')
                        }
                    else:
                        results[model_type] = {
                            "status": "degraded",
                            "message": "Model not trained"
                        }
                except Exception as e:
                    logger.error(f"Failed to load {model_type}: {str(e)}")
                    results[model_type] = {
                        "status": "unhealthy",
                        "message": f"Load failed: {str(e)}"
                    }
                    
            except Exception as e:
                logger.error(f"Failed to import {model_type}: {str(e)}")
                results[model_type] = {
                    "status": "unhealthy",
                    "message": f"Import failed: {str(e)}"
                }
        
        # Overall model status
        statuses = [r["status"] for r in results.values()]
        if all(s == "healthy" for s in statuses):
            results["overall"] = {"status": "healthy"}
        elif any(s == "unhealthy" for s in statuses):
            results["overall"] = {"status": "unhealthy"}
        else:
            results["overall"] = {"status": "degraded"}
        
        # Store results
        self.results["models"] = results
        return results
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        logger.info("Running complete health check")
        
        # Run all checks
        api_status = self.check_api_connectivity()
        db_status = self.check_database_connection()
        data_status = self.check_data_availability()
        model_status = self.check_model_integrity()
        
        # Determine overall system status
        component_statuses = [
            api_status["overall"]["status"],
            db_status["overall"]["status"],
            data_status["overall"]["status"],
            model_status["overall"]["status"]
        ]
        
        if all(s == "healthy" for s in component_statuses):
            self.status = "healthy"
        elif any(s == "unhealthy" for s in component_statuses):
            self.status = "unhealthy"
        else:
            self.status = "degraded"
        
        self.results["overall_status"] = self.status
        self.results["timestamp"] = datetime.now().isoformat()
        
        return self.results
    
    def print_report(self) -> None:
        """Print a human-readable health check report"""
        if not self.results:
            print("No health check results available. Run run_all_checks() first.")
            return
        
        print("\n====== NBA PREDICTION SYSTEM HEALTH REPORT ======")
        print(f"Timestamp: {self.results.get('timestamp', datetime.now().isoformat())}")
        print(f"Overall Status: {self.results.get('overall_status', 'unknown').upper()}\n")
        
        # Print component results
        components = [
            ("API Connectivity", "api_connectivity"),
            ("Database", "database"),
            ("Data Availability", "data"),
            ("Model Integrity", "models")
        ]
        
        for component_name, component_key in components:
            component_results = self.results.get(component_key, {})
            component_status = component_results.get("overall", {}).get("status", "unknown").upper()
            print(f"{component_name}: {component_status}")
            
            # Print details for subcomponents
            for subkey, subresults in component_results.items():
                if subkey != "overall":
                    status = subresults.get("status", "unknown").upper()
                    print(f"  - {subkey}: {status}")
                    
                    # Print additional details
                    for detail_key, detail_value in subresults.items():
                        if detail_key != "status":
                            print(f"      {detail_key}: {detail_value}")
            print()


# Main function for testing
if __name__ == "__main__":
    # Run health check
    health_check = HealthCheck()
    health_check.run_all_checks()
    health_check.print_report()
