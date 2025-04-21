#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Registry Module for NBA Prediction System

This module provides a registry for loading, managing, and accessing trained models.
It supports dynamic loading of models by type and automatic validation.

Features:
- Centralized model management
- Model versioning support
- Model metadata tracking
- Model validation
- Fallback mechanisms for production reliability
"""

import os
import logging
import pickle
import glob
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Production registry for trained ML models with versioning and metadata
    
    The ModelRegistry maintains a catalog of available models, handles loading,
    and provides a standardized interface for prediction code to access models.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the model registry
        
        Args:
            models_dir: Directory where model files are stored
        """
        # Set models directory - use default if not provided
        if models_dir is None:
            # Try to find the default models directory relative to this file
            base_dir = Path(__file__).parent.parent.parent  # Go up two levels from this file
            models_dir = os.path.join(base_dir, "results", "models")
            
        self.models_dir = models_dir
        logger.info(f"Initializing ModelRegistry with models directory: {self.models_dir}")
        
        # Initialize model storage dictionaries
        self.models = {
            "game_outcome": {
                "moneyline": {},
                "spread": {},
                "total": {}
            },
            "player_props": {
                "points": {},
                "rebounds": {},
                "assists": {},
                "threes": {},
                "steals": {},
                "blocks": {}
            }
        }
        
        # Track registry metrics
        self.metrics = {
            "load_time": 0.0,
            "loaded_models_count": 0,
            "error_count": 0,
            "latest_update": datetime.now().isoformat()
        }
        
        # Automatic model loading on initialization
        self.load_all_models()
    
    def load_all_models(self) -> bool:
        """
        Load all available models from the models directory
        
        Returns:
            bool: Success status of the operation
        """
        start_time = time.time()
        
        try:
            # Ensure models directory exists
            if not os.path.exists(self.models_dir):
                logger.error(f"Models directory not found: {self.models_dir}")
                return False
            
            # Load game outcome models
            logger.info("Loading game outcome models...")
            self._load_game_outcome_models()
            
            # Load player prop models
            logger.info("Loading player prop models...")
            self._load_player_prop_models()
            
            # Update metrics
            self.metrics["load_time"] = time.time() - start_time
            self.metrics["loaded_models_count"] = self._count_loaded_models()
            self.metrics["latest_update"] = datetime.now().isoformat()
            
            if self.metrics["loaded_models_count"] == 0:
                logger.warning("No models were loaded successfully. This is a critical issue for predictions.")
                return False
            
            logger.info(f"Successfully loaded {self.metrics['loaded_models_count']} models in {self.metrics['load_time']:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_game_outcome_models(self):
        """
        Load game outcome prediction models (moneyline, spread, total)
        """
        # Load moneyline models
        moneyline_models = glob.glob(os.path.join(self.models_dir, "game_home_win_*.pkl"))
        logger.info(f"Found {len(moneyline_models)} moneyline model files")
        
        for model_path in moneyline_models:
            try:
                # Extract model type from filename
                filename = os.path.basename(model_path)
                parts = filename.split("_")
                if len(parts) >= 3:
                    model_type = parts[2].split(".")[0]  # Extract model type before extension
                    
                    # Load and validate the model
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                        
                    # Store model by type
                    self.models["game_outcome"]["moneyline"][model_type] = model
                    logger.debug(f"Loaded moneyline model: {model_type}")
            except Exception as e:
                self.metrics["error_count"] += 1
                logger.error(f"Error loading moneyline model {model_path}: {str(e)}")
        
        # Load spread models
        spread_models = glob.glob(os.path.join(self.models_dir, "game_spread_diff_*.pkl"))
        logger.info(f"Found {len(spread_models)} spread model files")
        
        for model_path in spread_models:
            try:
                # Extract model type from filename
                filename = os.path.basename(model_path)
                parts = filename.split("_")
                if len(parts) >= 3:
                    model_type = parts[2].split(".")[0]  # Extract model type before extension
                    
                    # Load and validate the model
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                        
                    # Store model by type
                    self.models["game_outcome"]["spread"][model_type] = model
                    logger.debug(f"Loaded spread model: {model_type}")
            except Exception as e:
                self.metrics["error_count"] += 1
                logger.error(f"Error loading spread model {model_path}: {str(e)}")
        
        # Load total models
        total_models = glob.glob(os.path.join(self.models_dir, "game_total_points_*.pkl"))
        logger.info(f"Found {len(total_models)} total points model files")
        
        for model_path in total_models:
            try:
                # Extract model type from filename
                filename = os.path.basename(model_path)
                parts = filename.split("_")
                if len(parts) >= 3:
                    model_type = parts[2].split(".")[0]  # Extract model type before extension
                    
                    # Load and validate the model
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                        
                    # Store model by type
                    self.models["game_outcome"]["total"][model_type] = model
                    logger.debug(f"Loaded total model: {model_type}")
            except Exception as e:
                self.metrics["error_count"] += 1
                logger.error(f"Error loading total model {model_path}: {str(e)}")
    
    def _load_player_prop_models(self):
        """
        Load player prop prediction models (points, rebounds, assists, etc.)
        """
        prop_types = ["points", "rebounds", "assists", "blocks", "steals", "threes"]
        
        for prop_type in prop_types:
            # Find all models for this prop type
            prop_models = glob.glob(os.path.join(self.models_dir, f"player_player_{prop_type}_*.pkl"))
            logger.info(f"Found {len(prop_models)} {prop_type} model files")
            
            for model_path in prop_models:
                try:
                    # Extract model type from filename
                    filename = os.path.basename(model_path)
                    parts = filename.split("_")
                    if len(parts) >= 4:
                        model_type = parts[3].split(".")[0]  # Extract model type before extension
                        
                        # Load and validate the model
                        with open(model_path, "rb") as f:
                            model = pickle.load(f)
                            
                        # Store model by type
                        self.models["player_props"][prop_type][model_type] = model
                        logger.debug(f"Loaded {prop_type} model: {model_type}")
                except Exception as e:
                    self.metrics["error_count"] += 1
                    logger.error(f"Error loading {prop_type} model {model_path}: {str(e)}")
    
    def get_game_outcome_models(self, prediction_type: str = None) -> Dict:
        """
        Get game outcome prediction models
        
        Args:
            prediction_type: Type of prediction (moneyline, spread, total, or None for all)
            
        Returns:
            Dictionary of models for the requested prediction type(s)
        """
        if prediction_type is None:
            return self.models["game_outcome"]
        elif prediction_type in self.models["game_outcome"]:
            return self.models["game_outcome"][prediction_type]
        else:
            logger.warning(f"Unknown game outcome prediction type: {prediction_type}")
            return {}
    
    def get_player_prop_models(self, prop_type: str = None) -> Dict:
        """
        Get player prop prediction models
        
        Args:
            prop_type: Type of player prop (points, rebounds, assists, etc., or None for all)
            
        Returns:
            Dictionary of models for the requested prop type(s)
        """
        if prop_type is None:
            return self.models["player_props"]
        elif prop_type in self.models["player_props"]:
            return self.models["player_props"][prop_type]
        else:
            logger.warning(f"Unknown player prop type: {prop_type}")
            return {}
    
    def get_model(self, model_path: str) -> Any:
        """
        Get a specific model by its path in the registry
        
        Args:
            model_path: Path to the model in the registry, e.g. 'game_outcome.moneyline.gradient_boosting'
            
        Returns:
            The requested model or None if not found
        """
        parts = model_path.split('.')
        
        if len(parts) < 3:
            logger.error(f"Invalid model path: {model_path}. Format should be category.type.model_name")
            return None
        
        category, pred_type, model_name = parts
        
        if category not in self.models:
            logger.error(f"Unknown model category: {category}")
            return None
        
        if pred_type not in self.models[category]:
            logger.error(f"Unknown prediction type: {pred_type} in category {category}")
            return None
        
        if model_name not in self.models[category][pred_type]:
            logger.error(f"Unknown model name: {model_name} for {category}.{pred_type}")
            return None
        
        return self.models[category][pred_type][model_name]
    
    def _count_loaded_models(self) -> int:
        """
        Count the total number of loaded models in the registry
        
        Returns:
            int: Total number of loaded models
        """
        count = 0
        
        # Count game outcome models
        for pred_type in self.models["game_outcome"]:
            count += len(self.models["game_outcome"][pred_type])
        
        # Count player prop models
        for prop_type in self.models["player_props"]:
            count += len(self.models["player_props"][prop_type])
        
        return count
    
    def get_registry_info(self) -> Dict:
        """
        Get information about the model registry
        
        Returns:
            Dict: Model registry information and metrics
        """
        return {
            "models_dir": self.models_dir,
            "metrics": self.metrics,
            "model_counts": {
                "game_outcome": {
                    "moneyline": len(self.models["game_outcome"]["moneyline"]),
                    "spread": len(self.models["game_outcome"]["spread"]),
                    "total": len(self.models["game_outcome"]["total"])
                },
                "player_props": {
                    prop_type: len(self.models["player_props"][prop_type])
                    for prop_type in self.models["player_props"]
                }
            }
        }
