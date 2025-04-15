#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Model Trainer Module

This module provides advanced training capabilities for NBA prediction models using
real NBA data from the database. It implements feature engineering, model training,
evaluation, and persisting models to the database.
"""

import os
import sys
import logging
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import database connection utilities
from src.database.connection import get_connection, close_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/model_training.log')
    ]
)
logger = logging.getLogger(__name__)


class AdvancedModelTrainer:
    """Advanced model trainer for NBA prediction models"""
    
    def __init__(self, model_name: str, prediction_target: str):
        """
        Initialize the trainer
        
        Args:
            model_name: Name of the model to train
            prediction_target: Target to predict (moneyline, spread, total, etc.)
        """
        self.model_name = model_name
        self.prediction_target = prediction_target
        self.features = []
        self.model = None
        self.feature_importance = {}
        self.training_stats = {}
        self.validation_stats = {}
        self.model_params = {}
        self._log_startup_info()
    
    def _log_startup_info(self):
        """Log startup information"""
        logger.info(f"Initializing AdvancedModelTrainer for {self.model_name}")
        logger.info(f"Prediction target: {self.prediction_target}")
        try:
            # Log database connection status
            conn = get_connection()
            close_connection(conn)
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
    
    def _load_model_config(self) -> Dict:
        """Load model configuration from database"""
        logger.info(f"Loading configuration for model: {self.model_name}")
        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT model_type, params FROM model_weights 
                    WHERE model_name = %s AND params->>'prediction_target' = %s 
                    AND active = TRUE ORDER BY version DESC LIMIT 1""",
                    (self.model_name, self.prediction_target)
                )
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"No configuration found for {self.model_name}, using defaults")
                    return {'model_type': 'classification', 'params': {}}
                
                model_type, params = result
                logger.info(f"Loaded model config: {model_type}, params: {params}")
                return {'model_type': model_type, 'params': params}
        except Exception as e:
            logger.error(f"Error loading model config: {str(e)}")
            return {'model_type': 'classification', 'params': {}}
        finally:
            close_connection(conn)
    
    def fetch_training_data(self, days_back: int = 365, min_games: int = 100) -> pd.DataFrame:
        """Fetch training data from the database"""
        logger.info(f"Fetching training data for the past {days_back} days (min games: {min_games})")
        try:
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days_back)
            
            conn = get_connection()
            with conn.cursor() as cursor:
                # Query for completed games with features
                cursor.execute("""
                    SELECT g.game_id, g.date, g.season_year,
                           g.home_team_id, g.away_team_id, 
                           ht.name as home_team_name, at.name as away_team_name,
                           g.data->>'home_team_score' as home_score, 
                           g.data->>'visitor_team_score' as away_score,
                           g.features, g.odds
                    FROM games g
                    JOIN teams ht ON g.home_team_id = ht.team_id
                    JOIN teams at ON g.away_team_id = at.team_id
                    WHERE g.status = 'Final' 
                      AND g.date BETWEEN %s AND %s
                      AND g.features != '{}'
                    ORDER BY g.date DESC
                    LIMIT %s
                """, (start_date, end_date, max(1000, min_games * 2)))
                
                rows = cursor.fetchall()
                if not rows or len(rows) < min_games:
                    logger.warning(f"Insufficient training data: {len(rows) if rows else 0} games found")
                    if not rows:
                        return pd.DataFrame()
            
            # Process query results
            data = []
            for row in rows:
                game_id, date, season, home_id, away_id, home_name, away_name, \
                home_score, away_score, features, odds = row
                
                # Convert to proper types
                try:
                    home_score = int(home_score) if home_score else None
                    away_score = int(away_score) if away_score else None
                    features = json.loads(features) if features else {}
                    odds = json.loads(odds) if odds else {}
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parsing data for game {game_id}: {str(e)}")
                    continue
                
                # Skip games with missing scores
                if home_score is None or away_score is None:
                    continue
                
                # Determine actual outcome for different prediction targets
                outcomes = {
                    'moneyline': 'home' if home_score > away_score else 'away',
                    'spread': home_score - away_score,
                    'total': home_score + away_score,
                    'home_score': home_score,
                    'away_score': away_score
                }
                
                # Combine all data
                game_data = {
                    'game_id': game_id,
                    'date': date,
                    'season': season,
                    'home_team_id': home_id,
                    'away_team_id': away_id,
                    'home_team': home_name,
                    'away_team': away_name,
                    'home_score': home_score,
                    'away_score': away_score,
                    'outcome': outcomes.get(self.prediction_target, outcomes['moneyline'])
                }
                
                # Add features
                for key, value in features.items():
                    game_data[f'feature_{key}'] = value
                
                # Add odds information if available
                for key, value in odds.items():
                    game_data[f'odds_{key}'] = value
                
                data.append(game_data)
            
            df = pd.DataFrame(data)
            logger.info(f"Fetched {len(df)} games for training")
            
            # Store feature columns
            self.features = [col for col in df.columns if col.startswith('feature_') or col.startswith('odds_')]
            logger.info(f"Identified {len(self.features)} features for training")
            
            return df
        except Exception as e:
            logger.error(f"Error fetching training data: {str(e)}")
            return pd.DataFrame()
        finally:
            close_connection(conn)
    
    def train_model(self, df: pd.DataFrame) -> bool:
        """Train the model using the provided data"""
        if df.empty:
            logger.error("No training data available")
            return False
        
        logger.info(f"Training {self.model_name} model for {self.prediction_target}")
        try:
            # Load model configuration
            config = self._load_model_config()
            model_type = config['model_type']
            params = config['params']
            self.model_params = params
            
            # Prepare features and target
            X = df[self.features]
            
            # Handle different prediction targets
            if self.prediction_target == 'moneyline':
                y = df['outcome']  # Classification: 'home' or 'away'
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 10),
                    random_state=42
                )
            elif self.prediction_target in ['spread', 'total', 'home_score', 'away_score']:
                y = df[self.prediction_target] if self.prediction_target in df.columns else df['outcome']
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=params.get('n_estimators', 150),
                    learning_rate=params.get('learning_rate', 0.1),
                    random_state=42
                )
            else:
                logger.error(f"Unsupported prediction target: {self.prediction_target}")
                return False
            
            # Split data for training and validation
            from sklearn.model_selection import train_test_split
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            logger.info(f"Training with {len(X_train)} samples, validating with {len(X_valid)} samples")
            model.fit(X_train, y_train)
            self.model = model
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance = dict(zip(self.features, model.feature_importances_))
                
                # Log top features
                top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                logger.info(f"Top 10 important features: {top_features}")
            
            # Evaluate model
            self._evaluate_model(model, X_train, y_train, X_valid, y_valid)
            
            # Save model to database
            self._save_model(model)
            
            return True
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def _evaluate_model(self, model, X_train, y_train, X_valid, y_valid):
        """Evaluate the trained model"""
        logger.info("Evaluating model performance")
        try:
            # Training performance
            if self.prediction_target == 'moneyline':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                # Training metrics
                y_train_pred = model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                train_precision = precision_score(y_train, y_train_pred, pos_label='home', average='binary')
                train_recall = recall_score(y_train, y_train_pred, pos_label='home', average='binary')
                train_f1 = f1_score(y_train, y_train_pred, pos_label='home', average='binary')
                
                self.training_stats = {
                    'accuracy': train_accuracy,
                    'precision': train_precision,
                    'recall': train_recall,
                    'f1_score': train_f1,
                    'samples': len(X_train)
                }
                
                # Validation metrics
                y_valid_pred = model.predict(X_valid)
                valid_accuracy = accuracy_score(y_valid, y_valid_pred)
                valid_precision = precision_score(y_valid, y_valid_pred, pos_label='home', average='binary')
                valid_recall = recall_score(y_valid, y_valid_pred, pos_label='home', average='binary')
                valid_f1 = f1_score(y_valid, y_valid_pred, pos_label='home', average='binary')
                
                self.validation_stats = {
                    'accuracy': valid_accuracy,
                    'precision': valid_precision,
                    'recall': valid_recall,
                    'f1_score': valid_f1,
                    'samples': len(X_valid)
                }
                
                logger.info(f"Training accuracy: {train_accuracy:.4f}, Validation accuracy: {valid_accuracy:.4f}")
            else:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                # Training metrics
                y_train_pred = model.predict(X_train)
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                
                self.training_stats = {
                    'mse': train_mse,
                    'mae': train_mae,
                    'r2': train_r2,
                    'samples': len(X_train)
                }
                
                # Validation metrics
                y_valid_pred = model.predict(X_valid)
                valid_mse = mean_squared_error(y_valid, y_valid_pred)
                valid_mae = mean_absolute_error(y_valid, y_valid_pred)
                valid_r2 = r2_score(y_valid, y_valid_pred)
                
                self.validation_stats = {
                    'mse': valid_mse,
                    'mae': valid_mae,
                    'r2': valid_r2,
                    'samples': len(X_valid)
                }
                
                logger.info(f"Training MSE: {train_mse:.4f}, Validation MSE: {valid_mse:.4f}")
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
    
    def _save_model(self, model) -> bool:
        """Save trained model to database"""
        logger.info(f"Saving {self.model_name} model to database")
        try:
            # Serialize the model
            model_bytes = pickle.dumps(model)
            
            # Prepare metrics for storage
            metrics = {
                'training': self.training_stats,
                'validation': self.validation_stats,
                'feature_importance': dict(sorted(
                    self.feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:20]),  # Store top 20 features
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            conn = get_connection()
            with conn.cursor() as cursor:
                # Get current version
                cursor.execute(
                    """SELECT MAX(version) FROM model_weights 
                    WHERE model_name = %s AND params->>'prediction_target' = %s""",
                    (self.model_name, self.prediction_target)
                )
                max_version = cursor.fetchone()[0] or 0
                new_version = max_version + 1
                
                # Deactivate old versions
                cursor.execute(
                    """UPDATE model_weights SET active = FALSE 
                    WHERE model_name = %s AND params->>'prediction_target' = %s""",
                    (self.model_name, self.prediction_target)
                )
                
                # Update model parameters with metrics
                params = self.model_params.copy()
                params['metrics'] = metrics
                params['prediction_target'] = self.prediction_target
                
                # Insert new model version
                cursor.execute("""
                    INSERT INTO model_weights 
                    (model_name, model_type, params, weights, version, trained_at, active, needs_training)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.model_name,
                    'classification' if self.prediction_target == 'moneyline' else 'regression',
                    json.dumps(params),
                    model_bytes,
                    new_version,
                    datetime.now(timezone.utc),
                    True,  # Set as active
                    False  # No longer needs training
                ))
                
                # Update model performance
                cursor.execute("""
                    INSERT INTO model_performance 
                    (model_name, prediction_target, metrics, is_baseline)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO UPDATE SET
                        metrics = EXCLUDED.metrics,
                        is_baseline = EXCLUDED.is_baseline
                """, (
                    self.model_name,
                    self.prediction_target,
                    json.dumps(metrics),
                    False
                ))
                
                conn.commit()
                logger.info(f"Model saved successfully (version {new_version})")
                return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
        finally:
            close_connection(conn)


def train_all_models():
    """Train all models that need training"""
    logger.info("Starting training for all models that need training")
    
    try:
        # Get models that need training
        conn = get_connection()
        models_to_train = []
        
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT model_name, params->>'prediction_target' as prediction_target
                FROM model_weights
                WHERE needs_training = TRUE
                GROUP BY model_name, params->>'prediction_target'
            """)
            models_to_train = cursor.fetchall()
        
        if not models_to_train:
            logger.info("No models need training")
            return True
        
        logger.info(f"Found {len(models_to_train)} models that need training")
        
        # Train each model
        for model_name, prediction_target in models_to_train:
            try:
                logger.info(f"Training model: {model_name} (target: {prediction_target})")
                trainer = AdvancedModelTrainer(model_name, prediction_target)
                df = trainer.fetch_training_data()
                
                if df.empty:
                    logger.warning(f"No training data available for {model_name} ({prediction_target})")
                    continue
                
                success = trainer.train_model(df)
                if success:
                    logger.info(f"Successfully trained {model_name} ({prediction_target})")
                else:
                    logger.error(f"Failed to train {model_name} ({prediction_target})")
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error in train_all_models: {str(e)}")
        return False
    finally:
        close_connection(conn)


if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    print("\n===== NBA PREDICTION SYSTEM MODEL TRAINER =====\n")
    print("Training models using real NBA data...")
    
    success = train_all_models()
    
    if success:
        print("\n✅ Model training completed!")
        print("✅ Models have been saved to the database")
        print("\nYou can now use these models for making predictions.")
    else:
        print("\n❌ Model training failed!")
        print("Please check the logs for more information.")
