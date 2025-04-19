#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradient Boosting Model Trainer

Responsibilities:
- Train Gradient Boosting models for classification and regression tasks
- Implement hyperparameter tuning
- Handle model validation
- Track training metrics
"""

import numpy as np
import pandas as pd
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .base_trainer import BaseModelTrainer
from ..config import logger


class GradientBoostingTrainer(BaseModelTrainer):
    """
    Gradient Boosting model trainer for classification and regression
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Gradient Boosting trainer
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        self.model_type = "gradient_boosting"
        
        # Default hyperparameters
        self.params = {
            'n_estimators': config.get('params', {}).get('n_estimators', 100),
            'learning_rate': config.get('params', {}).get('learning_rate', 0.1),
            'max_depth': config.get('params', {}).get('max_depth', 3),
            'subsample': config.get('params', {}).get('subsample', 1.0),
            'random_state': config.get('params', {}).get('random_state', 42)
        }
        
        # Hyperparameter tuning settings
        self.optimize_hyperparams = config.get('optimize_hyperparams', False)
        self.tuning_method = config.get('tuning_method', 'random')  # 'grid' or 'random'
        self.cv_folds = config.get('cv_folds', 5)
        self.n_iter = config.get('n_iter', 10)  # Number of iterations for RandomizedSearchCV
        
        logger.info(f"Initialized GradientBoostingTrainer with params: {self.params}")
        if self.optimize_hyperparams:
            logger.info(f"Hyperparameter tuning enabled using {self.tuning_method} search")
    
    def train(self, X: np.ndarray, y: np.ndarray, prediction_type: str) -> Any:
        """
        Train a Gradient Boosting model with optional hyperparameter tuning
        
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
            
            # Initialize appropriate model based on prediction type
            if prediction_type == 'classification':
                if self.optimize_hyperparams:
                    model = self._train_tuned_classifier(X, y)
                else:
                    model = GradientBoostingClassifier(**self.params)
                    model.fit(X, y)
            else:  # regression
                if self.optimize_hyperparams:
                    model = self._train_tuned_regressor(X, y)
                else:
                    model = GradientBoostingRegressor(**self.params)
                    model.fit(X, y)
            
            train_time = time.time() - start_time
            logger.info(f"Trained {self.model_type} model in {train_time:.2f} seconds")
            
            # Log key model parameters and feature importance
            if hasattr(model, 'feature_importances_'):
                top_features = np.argsort(model.feature_importances_)[::-1][:5]
                logger.info(f"Top feature indices: {top_features}")
                
            if hasattr(model, 'get_params'):
                params = model.get_params()
                logger.info(f"Model parameters: n_estimators={params.get('n_estimators')}, learning_rate={params.get('learning_rate')}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training {self.model_type} model: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _train_tuned_classifier(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingClassifier:
        """
        Train a classifier with hyperparameter tuning
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Tuned classifier
        """
        logger.info("Starting hyperparameter tuning for Gradient Boosting classifier")
        
        # Define parameter search space
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4, 5],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize base model
        base_model = GradientBoostingClassifier(random_state=self.params['random_state'])
        
        # Use appropriate search method
        if self.tuning_method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=self.cv_folds,
                n_jobs=-1,
                scoring='f1',
                verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                n_jobs=-1,
                scoring='f1',
                random_state=self.params['random_state'],
                verbose=1
            )
        
        # Perform search
        search.fit(X, y)
        
        # Log best parameters
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def _train_tuned_regressor(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
        """
        Train a regressor with hyperparameter tuning
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Tuned regressor
        """
        logger.info("Starting hyperparameter tuning for Gradient Boosting regressor")
        
        # Define parameter search space
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4, 5],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'loss': ['squared_error', 'absolute_error', 'huber']
        }
        
        # Initialize base model
        base_model = GradientBoostingRegressor(random_state=self.params['random_state'])
        
        # Use appropriate search method
        if self.tuning_method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=self.cv_folds,
                n_jobs=-1,
                scoring='neg_mean_squared_error',
                verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                n_jobs=-1,
                scoring='neg_mean_squared_error',
                random_state=self.params['random_state'],
                verbose=1
            )
        
        # Perform search
        search.fit(X, y)
        
        # Log best parameters
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def supports_prediction_type(self, prediction_type: str) -> bool:
        """
        Check if the trainer supports the given prediction type
        
        Args:
            prediction_type: Type of prediction ('classification' or 'regression')
            
        Returns:
            True if supported, False otherwise
        """
        return prediction_type in ['classification', 'regression']