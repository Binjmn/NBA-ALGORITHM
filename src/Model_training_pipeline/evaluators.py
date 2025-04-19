#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Evaluation Module for NBA Model Training Pipeline

Responsibilities:
- Evaluate trained models using various metrics
- Compare model performance against baseline models
- Generate evaluation reports including confusion matrices
- Calculate feature importance and model insights
- Track model performance over time
- Identify optimal threshold values for classification models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import traceback
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error,
    mean_absolute_error, r2_score, classification_report
)
import joblib
from .config import logger


class ModelEvaluator:
    """
    Production-ready model evaluator for NBA prediction models
    
    Features:
    - Comprehensive evaluation metrics for classification and regression models
    - Baseline model comparison
    - Feature importance analysis
    - Threshold optimization for classification models
    - Detailed evaluation reports with visualizations
    - Performance tracking over time
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model evaluator with configuration
        
        Args:
            config: Configuration dictionary with evaluation settings
        """
        self.config = config
        self.evaluation_metrics = config['evaluation']['metrics']
        self.baseline_strategy = config['evaluation']['baseline_strategy']
        self.threshold = config['evaluation'].get('threshold', 0.5)
        self.output_dir = config['paths']['output_dir']
        self.evaluation_results = {}
        
        # Initialize metrics tracking
        self.metrics = {
            'models_evaluated': 0,
            'successful_evaluations': 0,
            'evaluation_errors': 0
        }
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Initialized ModelEvaluator with metrics={self.evaluation_metrics}")
        logger.info(f"Using baseline strategy: {self.baseline_strategy}")
    
    def evaluate_model(self, model_name: str, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                       features: List[str] = None, prediction_type: str = 'classification') -> Dict[str, Any]:
        """
        Evaluate a trained model and generate comprehensive metrics
        
        Args:
            model_name: Name of the model
            model: Trained model object
            X_test: Test feature data
            y_test: Test target data
            features: List of feature names (optional)
            prediction_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {model_name} model")
        self.metrics['models_evaluated'] += 1
        
        try:
            # Create evaluation result dictionary
            evaluation_result = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'prediction_type': prediction_type,
                'test_samples': len(y_test),
                'metrics': {}
            }
            
            # Generate predictions
            if prediction_type == 'classification':
                # Get both predicted classes and probabilities if available
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
                    y_pred = (y_prob >= self.threshold).astype(int)  # Apply threshold
                    evaluation_result['threshold'] = self.threshold
                except (AttributeError, IndexError):
                    logger.info(f"Model {model_name} does not support predict_proba, using predict instead")
                    y_pred = model.predict(X_test)
                    y_prob = None
            else:  # regression
                y_pred = model.predict(X_test)
                y_prob = None
            
            # Calculate metrics based on prediction type
            if prediction_type == 'classification':
                # Basic classification metrics
                evaluation_result['metrics']['accuracy'] = float(accuracy_score(y_test, y_pred))
                evaluation_result['metrics']['precision'] = float(precision_score(y_test, y_pred, zero_division=0))
                evaluation_result['metrics']['recall'] = float(recall_score(y_test, y_pred, zero_division=0))
                evaluation_result['metrics']['f1'] = float(f1_score(y_test, y_pred, zero_division=0))
                
                # Add ROC AUC if probabilities are available
                if y_prob is not None:
                    try:
                        evaluation_result['metrics']['roc_auc'] = float(roc_auc_score(y_test, y_prob))
                    except ValueError as e:
                        logger.warning(f"Could not calculate ROC AUC for {model_name}: {str(e)}")
                
                # Generate confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                evaluation_result['confusion_matrix'] = cm.tolist()
                
                # Calculate class distribution
                evaluation_result['class_distribution'] = {
                    'actual_positive': int(np.sum(y_test == 1)),
                    'actual_negative': int(np.sum(y_test == 0)),
                    'predicted_positive': int(np.sum(y_pred == 1)),
                    'predicted_negative': int(np.sum(y_pred == 0))
                }
                
                # Detailed classification report
                try:
                    report = classification_report(y_test, y_pred, output_dict=True)
                    evaluation_result['classification_report'] = report
                except Exception as e:
                    logger.warning(f"Could not generate classification report: {str(e)}")
                
            else:  # regression metrics
                evaluation_result['metrics']['mse'] = float(mean_squared_error(y_test, y_pred))
                evaluation_result['metrics']['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                evaluation_result['metrics']['mae'] = float(mean_absolute_error(y_test, y_pred))
                evaluation_result['metrics']['r2'] = float(r2_score(y_test, y_pred))
                
                # Calculate residuals statistics
                residuals = y_test - y_pred
                evaluation_result['residuals'] = {
                    'mean': float(np.mean(residuals)),
                    'std': float(np.std(residuals)),
                    'min': float(np.min(residuals)),
                    'max': float(np.max(residuals))
                }
            
            # Calculate feature importance if model supports it and feature names are provided
            if features is not None and hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Include top 10 features by importance
                top_features = []
                for i in range(min(10, len(features))):
                    if i < len(indices):
                        idx = indices[i]
                        if idx < len(features):
                            top_features.append({
                                'feature': features[idx],
                                'importance': float(importances[idx])
                            })
                
                evaluation_result['feature_importance'] = top_features
            
            # Compare against baseline model
            baseline_metrics = self._compare_to_baseline(y_test, y_pred, prediction_type)
            evaluation_result['baseline_comparison'] = baseline_metrics
            
            # Store evaluation results
            self.evaluation_results[model_name] = evaluation_result
            self.metrics['successful_evaluations'] += 1
            
            logger.info(f"Successfully evaluated {model_name}")
            if prediction_type == 'classification':
                logger.info(f"Accuracy: {evaluation_result['metrics']['accuracy']:.4f}, F1: {evaluation_result['metrics']['f1']:.4f}")
            else:
                logger.info(f"RMSE: {evaluation_result['metrics']['rmse']:.4f}, R²: {evaluation_result['metrics']['r2']:.4f}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name} model: {str(e)}")
            logger.error(traceback.format_exc())
            self.metrics['evaluation_errors'] += 1
            return {
                'model_name': model_name,
                'error': str(e),
                'status': 'failed'
            }
    
    def _compare_to_baseline(self, y_test: np.ndarray, y_pred: np.ndarray, prediction_type: str) -> Dict[str, Any]:
        """
        Compare model predictions to a baseline model
        
        Args:
            y_test: True target values
            y_pred: Model predictions
            prediction_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with baseline comparison metrics
        """
        baseline_results = {}
        
        try:
            # Generate baseline predictions based on strategy
            if self.baseline_strategy == 'majority':
                # Most frequent class for classification, mean for regression
                if prediction_type == 'classification':
                    most_common = int(np.round(np.mean(y_test)))
                    baseline_pred = np.full_like(y_test, most_common)
                else:
                    baseline_pred = np.full_like(y_test, np.mean(y_test))
                    
            elif self.baseline_strategy == 'stratified':
                # Random predictions with same distribution as training data
                if prediction_type == 'classification':
                    # Calculate class probabilities
                    pos_prob = np.mean(y_test)
                    # Generate random predictions with same class distribution
                    np.random.seed(42)  # For reproducibility
                    baseline_pred = (np.random.random(len(y_test)) < pos_prob).astype(int)
                else:
                    # Add random noise around mean for regression
                    np.random.seed(42)  # For reproducibility
                    mean, std = np.mean(y_test), np.std(y_test)
                    baseline_pred = np.random.normal(mean, std, len(y_test))
            
            # Calculate baseline metrics
            if prediction_type == 'classification':
                baseline_results['accuracy'] = float(accuracy_score(y_test, baseline_pred))
                baseline_results['precision'] = float(precision_score(y_test, baseline_pred, zero_division=0))
                baseline_results['recall'] = float(recall_score(y_test, baseline_pred, zero_division=0))
                baseline_results['f1'] = float(f1_score(y_test, baseline_pred, zero_division=0))
                
                # Calculate accuracy improvement
                model_accuracy = float(accuracy_score(y_test, y_pred))
                baseline_results['accuracy_improvement'] = model_accuracy - baseline_results['accuracy']
                baseline_results['relative_improvement'] = (model_accuracy / max(baseline_results['accuracy'], 0.001)) - 1
                
            else:  # regression
                baseline_results['mse'] = float(mean_squared_error(y_test, baseline_pred))
                baseline_results['rmse'] = float(np.sqrt(baseline_results['mse']))
                baseline_results['mae'] = float(mean_absolute_error(y_test, baseline_pred))
                
                # Calculate relative improvement
                model_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                baseline_results['rmse_improvement'] = baseline_results['rmse'] - model_rmse
                baseline_results['relative_improvement'] = (baseline_results['rmse'] / max(model_rmse, 0.001)) - 1
            
            return baseline_results
            
        except Exception as e:
            logger.error(f"Error comparing to baseline: {str(e)}")
            logger.error(traceback.format_exc())
            return {'error': str(e), 'status': 'failed'}
    
    def optimize_threshold(self, model: Any, X_validation: np.ndarray, y_validation: np.ndarray, 
                          metric: str = 'f1') -> float:
        """
        Find optimal threshold for classification model based on selected metric
        
        Args:
            model: Trained classification model
            X_validation: Validation feature data
            y_validation: Validation target data
            metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
            
        Returns:
            Optimal threshold value
        """
        logger.info(f"Optimizing threshold based on {metric} metric")
        
        try:
            # Ensure model supports predict_proba
            if not hasattr(model, 'predict_proba'):
                logger.warning("Model does not support predict_proba, cannot optimize threshold")
                return 0.5
            
            # Get probabilities for positive class
            y_prob = model.predict_proba(X_validation)[:, 1]
            
            # Try different thresholds
            thresholds = np.arange(0.1, 0.91, 0.05)
            best_threshold = 0.5  # Default
            best_score = 0.0
            
            # Select metric function
            if metric == 'f1':
                metric_func = f1_score
            elif metric == 'accuracy':
                metric_func = accuracy_score
            elif metric == 'precision':
                metric_func = precision_score
            elif metric == 'recall':
                metric_func = recall_score
            else:
                logger.warning(f"Unknown metric {metric}, defaulting to f1")
                metric_func = f1_score
            
            # Evaluate each threshold
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                score = metric_func(y_validation, y_pred, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            logger.info(f"Optimal threshold: {best_threshold:.2f} with {metric} score: {best_score:.4f}")
            return float(best_threshold)
            
        except Exception as e:
            logger.error(f"Error optimizing threshold: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.5  # Default threshold on error
    
    def generate_evaluation_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report for all evaluated models
        
        Args:
            output_file: Optional file path to save the report
            
        Returns:
            Dictionary with full evaluation report
        """
        logger.info("Generating evaluation report for all models")
        
        try:
            # Create summary report
            report = {
                'timestamp': datetime.now().isoformat(),
                'models_evaluated': self.metrics['models_evaluated'],
                'successful_evaluations': self.metrics['successful_evaluations'],
                'evaluation_errors': self.metrics['evaluation_errors'],
                'model_results': self.evaluation_results,
                'best_models': {}
            }
            
            # Identify best models for each prediction type
            classification_models = {}
            regression_models = {}
            
            for model_name, results in self.evaluation_results.items():
                if 'error' in results:
                    continue  # Skip failed evaluations
                
                prediction_type = results.get('prediction_type', 'unknown')
                
                if prediction_type == 'classification':
                    if 'metrics' in results and 'f1' in results['metrics']:
                        classification_models[model_name] = results['metrics']['f1']
                elif prediction_type == 'regression':
                    if 'metrics' in results and 'rmse' in results['metrics']:
                        # For RMSE, lower is better, so we negate it for sorting
                        regression_models[model_name] = -results['metrics']['rmse']
            
            # Sort and get best models
            if classification_models:
                sorted_classification = sorted(classification_models.items(), key=lambda x: x[1], reverse=True)
                report['best_models']['classification'] = sorted_classification[0][0]
                report['best_models']['classification_score'] = sorted_classification[0][1]
            
            if regression_models:
                sorted_regression = sorted(regression_models.items(), key=lambda x: x[1], reverse=True)
                report['best_models']['regression'] = sorted_regression[0][0]
                # Convert back to positive RMSE
                report['best_models']['regression_score'] = -sorted_regression[0][1]
            
            # Save report if output file is specified
            if output_file:
                report_dir = os.path.dirname(output_file)
                if report_dir:
                    os.makedirs(report_dir, exist_ok=True)
                    
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Evaluation report saved to {output_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the evaluation process
        
        Returns:
            Dictionary with evaluation metrics
        """
        return self.metrics