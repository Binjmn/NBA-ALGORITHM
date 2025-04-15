"""
Hyperparameter Tuning for NBA Prediction System

This module provides functionality for automatically tuning the hyperparameters
of prediction models to optimize their performance. It uses Bayesian optimization
to efficiently search the hyperparameter space and find optimal configurations.

Features Used:
- Same features as the base models being tuned (team, player, context, and odds features)

Hyperparameter tuning is critical for maximizing model performance, as the optimal
parameter settings can vary depending on the specific prediction task and available data.
This module makes the tuning process automatic and efficient.
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from datetime import datetime, timezone

from src.models.base_model import BaseModel
from src.models.random_forest import RandomForestModel
from src.models.combined_gradient_boosting import CombinedGradientBoostingModel
from src.models.bayesian_model import BayesianModel

# Configure logging
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuning class for NBA prediction models
    
    This class is responsible for tuning the hyperparameters of various models
    using Bayesian optimization to find the best configuration for a given task.
    """
    
    def __init__(self, model_type: str, prediction_target: str):
        """
        Initialize the hyperparameter tuner
        
        Args:
            model_type: Type of model to tune ('RandomForest', 'CombinedGBM', 'Bayesian')
            prediction_target: What the model is predicting ('moneyline', 'spread', 'totals', 'player_points', etc.)
        """
        self.model_type = model_type
        self.prediction_target = prediction_target
        self.param_spaces = self._get_param_space(model_type, prediction_target)
        self.best_params = None
        self.best_score = None
        self.tuned_at = None
        self.cv_results = None
    
    def _get_param_space(self, model_type: str, prediction_target: str) -> Dict[str, Any]:
        """
        Get the hyperparameter search space for the specified model type
        
        Args:
            model_type: Type of model to tune
            prediction_target: What the model is predicting
            
        Returns:
            Dictionary mapping parameter names to search spaces
        """
        # Define model-specific parameter spaces
        if model_type == 'RandomForest':
            if prediction_target in ['moneyline']:  # Classification
                return {
                    'n_estimators': Integer(50, 300),
                    'max_depth': Integer(3, 20),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 10),
                    'max_features': Categorical(['sqrt', 'log2', None]),
                    'bootstrap': Categorical([True, False])
                }
            else:  # Regression
                return {
                    'n_estimators': Integer(50, 300),
                    'max_depth': Integer(3, 20),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 10),
                    'max_features': Categorical(['sqrt', 'log2', None]),
                    'bootstrap': Categorical([True, False])
                }
                
        elif model_type == 'CombinedGBM':
            # Separate spaces for XGBoost and LightGBM components
            xgb_space = {
                'xgb_params__n_estimators': Integer(50, 300),
                'xgb_params__max_depth': Integer(3, 10),
                'xgb_params__learning_rate': Real(0.01, 0.3, 'log-uniform'),
                'xgb_params__subsample': Real(0.5, 1.0),
                'xgb_params__colsample_bytree': Real(0.5, 1.0),
                'xgb_params__min_child_weight': Integer(1, 10),
                'xgb_params__reg_alpha': Real(0.0, 10.0, 'log-uniform'),
                'xgb_params__reg_lambda': Real(0.0, 10.0, 'log-uniform')
            }
            
            lgb_space = {
                'lgb_params__n_estimators': Integer(50, 300),
                'lgb_params__max_depth': Integer(3, 10),
                'lgb_params__learning_rate': Real(0.01, 0.3, 'log-uniform'),
                'lgb_params__subsample': Real(0.5, 1.0),
                'lgb_params__colsample_bytree': Real(0.5, 1.0),
                'lgb_params__min_child_samples': Integer(5, 50),
                'lgb_params__reg_alpha': Real(0.0, 10.0, 'log-uniform'),
                'lgb_params__reg_lambda': Real(0.0, 10.0, 'log-uniform')
            }
            
            mix_space = {
                'xgb_weight': Real(0.1, 0.9),
                'lgb_weight': Real(0.1, 0.9)
            }
            
            # Combine the spaces
            return {**xgb_space, **lgb_space, **mix_space}
            
        elif model_type == 'Bayesian':
            if prediction_target in ['moneyline']:  # Classification
                return {
                    'var_smoothing': Real(1e-12, 1e-6, 'log-uniform'),
                    'use_bagging': Categorical([True, False]),
                    'n_estimators': Integer(5, 20),
                    'max_samples': Real(0.5, 1.0)
                }
            else:  # Regression
                return {
                    'alpha_1': Real(1e-10, 1e-2, 'log-uniform'),
                    'alpha_2': Real(1e-10, 1e-2, 'log-uniform'),
                    'lambda_1': Real(1e-10, 1e-2, 'log-uniform'),
                    'lambda_2': Real(1e-10, 1e-2, 'log-uniform'),
                    'n_iter': Integer(100, 500),
                    'compute_score': Categorical([True, False]),
                    'fit_intercept': Categorical([True, False])
                }
        else:
            logger.warning(f"Unknown model type '{model_type}'. Using empty parameter space.")
            return {}
    
    def _create_model(self, **params) -> BaseModel:
        """
        Create a model instance with the specified parameters
        
        Args:
            **params: Model parameters
            
        Returns:
            BaseModel instance with the specified parameters
        """
        if self.model_type == 'RandomForest':
            model = RandomForestModel(prediction_target=self.prediction_target)
            model.params = params
            return model
            
        elif self.model_type == 'CombinedGBM':
            # Extract XGBoost and LightGBM parameters
            xgb_params = {k.replace('xgb_params__', ''): v for k, v in params.items() if k.startswith('xgb_params__')}
            lgb_params = {k.replace('lgb_params__', ''): v for k, v in params.items() if k.startswith('lgb_params__')}
            
            # Get mixing weights
            xgb_weight = params.get('xgb_weight', 0.5)
            lgb_weight = params.get('lgb_weight', 0.5)
            
            # Create model
            model = CombinedGradientBoostingModel(
                prediction_target=self.prediction_target,
                xgb_weight=xgb_weight,
                lgb_weight=lgb_weight
            )
            
            # Set parameters
            model.xgb_params = xgb_params
            model.lgb_params = lgb_params
            model.params = {
                'xgb_params': xgb_params,
                'lgb_params': lgb_params,
                'xgb_weight': xgb_weight,
                'lgb_weight': lgb_weight
            }
            
            return model
            
        elif self.model_type == 'Bayesian':
            model = BayesianModel(prediction_target=self.prediction_target)
            model.params = params
            return model
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def tune(self, X: pd.DataFrame, y: pd.Series, n_iter: int = 20, cv: int = 5, 
             n_jobs: int = -1, verbose: int = 1) -> Dict[str, Any]:
        """
        Tune hyperparameters using Bayesian optimization
        
        Args:
            X: Feature matrix
            y: Target variable
            n_iter: Number of iterations for Bayesian optimization
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all processors)
            verbose: Verbosity level
            
        Returns:
            Dictionary with the best parameters and scores
        """
        try:
            logger.info(f"Starting hyperparameter tuning for {self.model_type} ({self.prediction_target})")
            
            # Define the objective function for optimization
            def objective(params):
                # Create model with the parameters
                model = self._create_model(**params)
                
                # Perform cross-validation
                kf = KFold(n_splits=cv, shuffle=True, random_state=42)
                scores = []
                
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Train model
                    model.train(X_train, y_train)
                    
                    # Evaluate model
                    metrics = model.evaluate(X_val, y_val)
                    
                    # Get appropriate metric based on model type
                    if self.prediction_target in ['moneyline']:  # Classification
                        score = metrics.get('accuracy', 0.0)
                    else:  # Regression
                        # For regression, lower error is better but optimization maximizes
                        # So we use negative RMSE
                        score = -metrics.get('rmse', float('inf'))
                    
                    scores.append(score)
                
                # Return the mean score across folds
                return np.mean(scores)
            
            # Use Bayesian optimization to find the best parameters
            from skopt import gp_minimize
            from skopt.utils import use_named_args
            
            # Convert parameter space to a list for skopt
            space = []
            param_names = []
            for param_name, param_space in self.param_spaces.items():
                space.append(param_space)
                param_names.append(param_name)
            
            # Define the objective function with named parameters
            @use_named_args(space)
            def objective_named(**params):
                return -objective(params)  # Negative because we want to maximize
            
            # Run the optimization
            result = gp_minimize(
                objective_named,
                space,
                n_calls=n_iter,
                random_state=42,
                verbose=verbose
            )
            
            # Convert results to a dictionary
            best_params = {param_names[i]: result.x[i] for i in range(len(param_names))}
            best_score = -result.fun  # Convert back to positive score
            
            # Store the results
            self.best_params = best_params
            self.best_score = best_score
            self.tuned_at = datetime.now(timezone.utc)
            self.cv_results = result
            
            logger.info(f"Hyperparameter tuning completed with best score: {best_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'tuned_at': self.tuned_at,
                'model_type': self.model_type,
                'prediction_target': self.prediction_target
            }
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise
    
    def create_tuned_model(self) -> BaseModel:
        """
        Create a model with the tuned hyperparameters
        
        Returns:
            BaseModel instance with the tuned parameters
        """
        if not self.best_params:
            raise ValueError("No tuned parameters available. Run tune() first.")
            
        return self._create_model(**self.best_params)
    
    def save_results(self, save_dir: str = 'data/tuning') -> str:
        """
        Save the tuning results to disk
        
        Args:
            save_dir: Directory to save the results
            
        Returns:
            Path to the saved file
        """
        if not self.best_params or not self.tuned_at:
            raise ValueError("No tuning results available. Run tune() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a results dictionary
        results = {
            'model_type': self.model_type,
            'prediction_target': self.prediction_target,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'tuned_at': self.tuned_at.isoformat() if self.tuned_at else None
        }
        
        # Save to a JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.model_type}_{self.prediction_target}_tuning_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Tuning results saved to {filepath}")
        return filepath
    
    @staticmethod
    def load_results(filepath: str) -> Dict[str, Any]:
        """
        Load tuning results from a file
        
        Args:
            filepath: Path to the results file
            
        Returns:
            Dictionary with the tuning results
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        # Convert tuned_at back to datetime if present
        if 'tuned_at' in results and results['tuned_at']:
            results['tuned_at'] = datetime.fromisoformat(results['tuned_at'])
        
        return results
