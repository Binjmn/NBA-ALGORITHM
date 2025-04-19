#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the model trainers
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple

# Add src directory to path if needed for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.Model_training_pipeline.model_trainers.random_forest_trainer import RandomForestTrainer
from src.Model_training_pipeline.model_trainers.gradient_boosting_trainer import GradientBoostingTrainer
from src.Model_training_pipeline.model_trainers.ensemble_trainer import EnsembleTrainer
from src.Model_training_pipeline.config import get_default_config


class TestRandomForestTrainer(unittest.TestCase):
    """Test case for the RandomForestTrainer class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.config = get_default_config()
        self.model_config = {
            'params': {
                'n_estimators': 50,
                'max_depth': 10,
                'random_state': 42
            },
            'optimize_hyperparams': False  # Disable for testing speed
        }
        self.trainer = RandomForestTrainer(self.model_config)
        
        # Create sample training data
        np.random.seed(42)
        self.X = np.random.rand(100, 10)  # 100 samples, 10 features
        self.y_classification = np.random.randint(0, 2, 100)  # Binary classification
        self.y_regression = np.random.randn(100)  # Regression target
    
    def test_init(self):
        """Test initialization of RandomForestTrainer"""
        self.assertEqual(self.trainer.model_type, "random_forest")
        self.assertEqual(self.trainer.params['n_estimators'], 50)
        self.assertEqual(self.trainer.params['max_depth'], 10)
        
    def test_train_classification(self):
        """Test training a classification model"""
        model = self.trainer.train(self.X, self.y_classification, 'classification')
        
        # Verify the model is created and of the right type
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'RandomForestClassifier')
        
        # Test prediction
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)
        
    def test_train_regression(self):
        """Test training a regression model"""
        model = self.trainer.train(self.X, self.y_regression, 'regression')
        
        # Verify the model is created and of the right type
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'RandomForestRegressor')
        
        # Test prediction
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)
        
    def test_supports_prediction_type(self):
        """Test prediction type support check"""
        self.assertTrue(self.trainer.supports_prediction_type('classification'))
        self.assertTrue(self.trainer.supports_prediction_type('regression'))
        self.assertFalse(self.trainer.supports_prediction_type('unknown'))


class TestGradientBoostingTrainer(unittest.TestCase):
    """Test case for the GradientBoostingTrainer class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.config = get_default_config()
        self.model_config = {
            'params': {
                'n_estimators': 50,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            },
            'optimize_hyperparams': False  # Disable for testing speed
        }
        self.trainer = GradientBoostingTrainer(self.model_config)
        
        # Create sample training data
        np.random.seed(42)
        self.X = np.random.rand(100, 10)  # 100 samples, 10 features
        self.y_classification = np.random.randint(0, 2, 100)  # Binary classification
        self.y_regression = np.random.randn(100)  # Regression target
    
    def test_init(self):
        """Test initialization of GradientBoostingTrainer"""
        self.assertEqual(self.trainer.model_type, "gradient_boosting")
        self.assertEqual(self.trainer.params['n_estimators'], 50)
        self.assertEqual(self.trainer.params['learning_rate'], 0.1)
        
    def test_train_classification(self):
        """Test training a classification model"""
        model = self.trainer.train(self.X, self.y_classification, 'classification')
        
        # Verify the model is created and of the right type
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'GradientBoostingClassifier')
        
        # Test prediction
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)
        
    def test_train_regression(self):
        """Test training a regression model"""
        model = self.trainer.train(self.X, self.y_regression, 'regression')
        
        # Verify the model is created and of the right type
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'GradientBoostingRegressor')
        
        # Test prediction
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)


class TestEnsembleTrainer(unittest.TestCase):
    """Test case for the EnsembleTrainer class"""

    def setUp(self):
        """Set up test environment before each test"""
        self.config = get_default_config()
        self.model_config = {
            'ensemble_method': 'voting',  # Use voting for faster tests
            'rf_params': {
                'n_estimators': 10,  # Small value for testing
                'max_depth': 3
            },
            'gb_params': {
                'n_estimators': 10,  # Small value for testing
                'learning_rate': 0.1
            }
        }
        self.trainer = EnsembleTrainer(self.model_config)
        
        # Create sample training data
        np.random.seed(42)
        self.X = np.random.rand(100, 5)  # 100 samples, 5 features (smaller for speed)
        self.y_classification = np.random.randint(0, 2, 100)  # Binary classification
        self.y_regression = np.random.randn(100)  # Regression target
    
    def test_init(self):
        """Test initialization of EnsembleTrainer"""
        self.assertEqual(self.trainer.model_type, "ensemble")
        self.assertEqual(self.trainer.ensemble_method, "voting")
        
    def test_train_classification(self):
        """Test training a classification ensemble"""
        model = self.trainer.train(self.X, self.y_classification, 'classification')
        
        # Verify the model is created and of the right type
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'VotingClassifier')
        
        # Test prediction
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)
        
    def test_train_regression(self):
        """Test training a regression ensemble"""
        model = self.trainer.train(self.X, self.y_regression, 'regression')
        
        # Verify the model is created and of the right type
        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, 'VotingRegressor')
        
        # Test prediction
        preds = model.predict(self.X[:5])
        self.assertEqual(len(preds), 5)


if __name__ == '__main__':
    unittest.main()
