#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Model Trainer Module

Provides the abstract base class for all model trainers in the pipeline.
Defines required interfaces and common functionality for model training.
"""

import numpy as np
import pandas as pd
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod

from ..config import logger


class BaseModelTrainer(ABC):
    """
    Abstract base class for model trainers
    
    All model trainers must implement this interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base trainer
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model_type = "base"
        logger.info(f"Initialized BaseModelTrainer")
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, prediction_type: str) -> Any:
        """
        Train a model with the provided data
        
        Args:
            X: Training features
            y: Training targets
            prediction_type: Type of prediction ('classification' or 'regression')
            
        Returns:
            Trained model or None if error
        """
        pass
    
    def supports_prediction_type(self, prediction_type: str) -> bool:
        """
        Check if the trainer supports the given prediction type
        
        Args:
            prediction_type: Type of prediction ('classification' or 'regression')
            
        Returns:
            True if supported, False otherwise
        """
        return prediction_type in ['classification', 'regression']
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the hyperparameters for the model
        
        Returns:
            Dictionary of hyperparameters
        """
        return self.config.get('params', {})
    
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Set hyperparameters for the model
        
        Args:
            params: Dictionary of hyperparameters
        """
        if 'params' not in self.config:
            self.config['params'] = {}
            
        self.config['params'].update(params)
        logger.info(f"Updated {self.model_type} hyperparameters: {params}")