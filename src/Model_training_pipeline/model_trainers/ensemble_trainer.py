#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensemble Model Trainer

Responsibilities:
- Combine multiple base models (Random Forest, Gradient Boosting) into an ensemble
- Support stacking and voting ensemble methods
- Handle model validation
- Track training metrics
"""

import numpy as np
import pandas as pd
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from .base_trainer import BaseModelTrainer
from ..config import logger


class EnsembleTrainer(BaseModelTrainer):
    """
    Ensemble model trainer combining multiple base models
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Ensemble trainer
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.model_type = "ensemble"
        
        # Ensemble configuration
        self.ensemble_method = config.get('ensemble_method', 'stacking')  # 'stacking' or 'voting'
        self.weights = config.get('weights', None)  # For voting ensemble
        
        # Base estimator parameters
        self.rf_params = config.get('rf_params', {})
        self.gb_params = config.get('gb_params', {})
        
        # Final estimator parameters (for stacking)
        self.final_estimator_params = config.get('final_estimator_params', {})
        
        logger.info(f"Initialized EnsembleTrainer with method: {self.ensemble_method}")
    
    def train(self, X: np.ndarray, y: np.ndarray, prediction_type: str) -> Any:
        """
        Train an ensemble model combining multiple base models
        
        Args:
            X: Training features
            y: Training targets
            prediction_type: Type of prediction ('classification' or 'regression')
            
        Returns:
            Trained model or None if error
        """
        logger.info(f"Training {self.model_type} model for {prediction_type}")
        
        try:
            start_time = time.time()
            
            # Create base estimators
            base_estimators = self._create_base_estimators(prediction_type)
            
            # Create and train ensemble model
            if prediction_type == 'classification':
                if self.ensemble_method == 'stacking':
                    final_estimator = LogisticRegression(**self.final_estimator_params)
                    model = StackingClassifier(
                        estimators=base_estimators,
                        final_estimator=final_estimator,
                        cv=5,
                        stack_method='predict_proba'
                    )
                else:  # voting
                    model = VotingClassifier(
                        estimators=base_estimators,
                        voting='soft',
                        weights=self.weights
                    )
            else:  # regression
                if self.ensemble_method == 'stacking':
                    final_estimator = LinearRegression(**self.final_estimator_params)
                    model = StackingRegressor(
                        estimators=base_estimators,
                        final_estimator=final_estimator,
                        cv=5
                    )
                else:  # voting
                    model = VotingRegressor(
                        estimators=base_estimators,
                        weights=self.weights
                    )
            
            # Train the ensemble model
            model.fit(X, y)
            
            train_time = time.time() - start_time
            logger.info(f"Trained {self.model_type} model in {train_time:.2f} seconds")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training {self.model_type} model: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _create_base_estimators(self, prediction_type: str) -> List[Tuple[str, Any]]:
        """
        Create base estimators for the ensemble
        
        Args:
            prediction_type: Type of prediction ('classification' or 'regression')
            
        Returns:
            List of (name, estimator) tuples
        """
        # Set default parameters if not provided
        rf_defaults = {
            'n_estimators': 100,
            'max_depth': 20,
            'random_state': 42
        }
        gb_defaults = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        
        # Merge with user-provided parameters
        rf_params = {**rf_defaults, **self.rf_params}
        gb_params = {**gb_defaults, **self.gb_params}
        
        # Create base estimators based on prediction type
        if prediction_type == 'classification':
            rf = RandomForestClassifier(**rf_params)
            gb = GradientBoostingClassifier(**gb_params)
        else:  # regression
            rf = RandomForestRegressor(**rf_params)
            gb = GradientBoostingRegressor(**gb_params)
        
        # Return list of (name, estimator) tuples
        return [
            ('random_forest', rf),
            ('gradient_boosting', gb)
        ]
    
    def supports_prediction_type(self, prediction_type: str) -> bool:
        """
        Check if the trainer supports the given prediction type
        
        Args:
            prediction_type: Type of prediction ('classification' or 'regression')
            
        Returns:
            True if supported, False otherwise
        """
        return prediction_type in ['classification', 'regression']