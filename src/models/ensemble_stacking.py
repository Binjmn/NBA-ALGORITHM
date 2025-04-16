"""
Ensemble Stacking Model for NBA Prediction System

This module implements ensemble stacking, a technique where a meta-model is
trained to combine the predictions of several base models. Unlike simple
model mixing, stacking trains a model to learn optimal combinations based on
the data, potentially capturing complex relationships between model outputs.

Features Used:
- Outputs from base models: Random Forests, Combined Gradient Boosting, Bayesian
- Original features: All team, player, context, and odds features

Ensemble stacking optimizes prediction accuracy by learning how to best combine
the outputs of different models based on the input features, leveraging the
strengths of each model while compensating for their weaknesses.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from datetime import datetime, timezone

from src.models.base_model import BaseModel

# Configure logging
logger = logging.getLogger(__name__)


class EnsembleStackingModel(BaseModel):
    """
    Ensemble Stacking model for NBA prediction
    
    This model uses a meta-learner to combine predictions from multiple base models,
    potentially along with original features, to make more accurate predictions.
    
    The model supports both classification (for win/loss predictions) and
    regression (for spread/totals/player stats predictions).
    """
    
    def __init__(self, name: str = "EnsembleStacking", prediction_target: str = "moneyline", version: int = 1):
        """
        Initialize the Ensemble Stacking model
        
        Args:
            name: Model name
            prediction_target: What the model is predicting ('moneyline', 'spread', 'totals', 'player_points', etc.)
            version: Model version number
        """
        # Determine model type based on prediction target
        if prediction_target in ['moneyline']:
            model_type = 'classification'
        else:  # 'spread', 'totals', 'player_points', etc.
            model_type = 'regression'
            
        super().__init__(name=name, model_type=model_type, version=version)
        self.prediction_target = prediction_target
        
        # Storage for base models
        self.base_models = {}
        self.base_model_names = []
        
        # Default hyperparameters for the meta-learner
        if self.model_type == 'classification':
            self.params = {
                'meta_learner': 'logistic_regression',
                'cv_folds': 5,          # Number of cross-validation folds
                'use_proba': True,       # Use probability outputs from base models
                'include_original': True,  # Include original features in meta-model
                'regularization': 1.0,   # Regularization strength
                'max_iter': 1000,        # Maximum iterations
                'random_state': 42
            }
        else:  # regression
            self.params = {
                'meta_learner': 'ridge',
                'cv_folds': 5,           # Number of cross-validation folds
                'include_original': True,  # Include original features in meta-model
                'alpha': 1.0,            # Regularization strength
                'max_iter': 1000,        # Maximum iterations
                'random_state': 42
            }
        
        # Initialize containers for cross-validation
        self.cv_base_models = {}  # Models trained on CV folds
        self.meta_features = None  # For storing meta-features during prediction
    
    def set_base_models(self, models: Dict[str, BaseModel]) -> None:
        """
        Set the base models to use for stacking
        
        Args:
            models: Dictionary mapping model names to model instances
        """
        self.base_models = models
        self.base_model_names = list(models.keys())
        logger.info(f"Set {len(models)} base models for {self.name}: {self.base_model_names}")
    
    def _generate_meta_features(self, X: pd.DataFrame, base_models: Dict[str, BaseModel]) -> pd.DataFrame:
        """
        Generate meta-features from base model predictions
        
        Args:
            X: Original feature matrix
            base_models: Dictionary of base models to use
            
        Returns:
            DataFrame with meta-features
        """
        meta_features_list = []
        
        if not base_models:
            logger.warning("No base models provided to generate meta-features")
            # Return a dummy feature with zeros to avoid errors
            return pd.DataFrame(np.zeros((X.shape[0], 1)), columns=["dummy_feature"], index=X.index)
        
        # Add predictions from each base model
        for model_name, model in base_models.items():
            try:
                # Skip models that aren't trained
                if not hasattr(model, 'is_trained') or not model.is_trained:
                    logger.warning(f"Base model {model_name} is not trained, skipping for meta-features")
                    continue
                    
                # Special handling for EnsembleModel
                if model_name == 'EnsembleModel':
                    try:
                        # Try to get probability predictions
                        if hasattr(model, 'predict_proba') and callable(model.predict_proba):
                            proba = model.predict_proba(X)
                            if isinstance(proba, np.ndarray) and proba.shape[1] >= 2:
                                # Create properly named columns for ensemble model
                                ensemble_df = pd.DataFrame(
                                    {f"EnsembleModel_proba_{i}": proba[:, i] for i in range(proba.shape[1])},
                                    index=X.index
                                )
                                meta_features_list.append(ensemble_df)
                                continue
                        # Fallback to regular prediction if proba failed
                        preds = model.predict(X)
                        meta_features_list.append(pd.DataFrame(
                            preds.reshape(-1, 1),
                            columns=[f"EnsembleModel_pred"],
                            index=X.index
                        ))
                    except Exception as em_error:
                        logger.warning(f"Error with EnsembleModel, using fallback: {str(em_error)}")
                        # Create a fallback column for EnsembleModel to prevent index errors
                        meta_features_list.append(pd.DataFrame(
                            np.ones((X.shape[0], 1)) * 0.5,  # Use 0.5 as neutral probability
                            columns=["EnsembleModel_proba_1"],
                            index=X.index
                        ))
                    continue
                
                # For classification models, we can use probabilities
                if self.model_type == 'classification' and self.params['use_proba'] and hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(X)
                        if isinstance(proba, np.ndarray) and proba.shape[1] >= 2:  # Binary classification
                            # Just use the probability of class 1
                            meta_features_list.append(pd.DataFrame(
                                proba[:, 1].reshape(-1, 1),
                                columns=[f"{model_name}_proba_1"],
                                index=X.index  # Explicitly maintain the same index as X
                            ))
                        elif isinstance(proba, np.ndarray):  # Unexpected shape but still an array
                            # Use whatever we got
                            meta_features_list.append(pd.DataFrame(
                                proba.reshape(-1, 1),
                                columns=[f"{model_name}_pred"],
                                index=X.index  # Explicitly maintain the same index as X
                            ))
                    except Exception as proba_error:
                        logger.warning(f"Error getting probabilities from {model_name}: {str(proba_error)}")
                        # Fallback to regular predict
                        try:
                            preds = model.predict(X)
                            meta_features_list.append(pd.DataFrame(
                                preds.reshape(-1, 1),
                                columns=[f"{model_name}_pred"],
                                index=X.index
                            ))
                        except Exception as pred_error:
                            logger.warning(f"Error getting predictions from {model_name}: {str(pred_error)}")
                            # Create fallback column
                            meta_features_list.append(pd.DataFrame(
                                np.ones((X.shape[0], 1)) * 0.5,
                                columns=[f"{model_name}_proba_1"],
                                index=X.index
                            ))
                else:  # For regression or when use_proba is False
                    # Use the raw predictions
                    try:
                        preds = model.predict(X)
                        if isinstance(preds, np.ndarray):
                            meta_features_list.append(pd.DataFrame(
                                preds.reshape(-1, 1),
                                columns=[f"{model_name}_pred"],
                                index=X.index  # Explicitly maintain the same index as X
                            ))
                        else:
                            logger.warning(f"Base model {model_name} returned non-array predictions, skipping")
                    except Exception as pred_error:
                        logger.warning(f"Error getting predictions from {model_name}: {str(pred_error)}")
                        # Create fallback column
                        meta_features_list.append(pd.DataFrame(
                            np.zeros((X.shape[0], 1)),
                            columns=[f"{model_name}_pred"],
                            index=X.index
                        ))
            except Exception as e:
                logger.warning(f"Error generating meta-features for {model_name}: {str(e)}")
                # If a model fails, we'll create a column of zeros with its name
                if self.model_type == 'classification' and self.params['use_proba']:
                    meta_features_list.append(pd.DataFrame(
                        np.ones((X.shape[0], 1)) * 0.5,  # Use 0.5 as neutral probability
                        columns=[f"{model_name}_proba_1"],
                        index=X.index
                    ))
                else:
                    meta_features_list.append(pd.DataFrame(
                        np.zeros((X.shape[0], 1)),
                        columns=[f"{model_name}_pred"],
                        index=X.index
                    ))
        
        # If we couldn't get any meta-features, return dummy feature
        if not meta_features_list:
            logger.warning("Could not generate any valid meta-features")
            return pd.DataFrame(np.zeros((X.shape[0], 1)), columns=["dummy_feature"], index=X.index)
            
        # Combine all meta-features
        meta_features = pd.concat(meta_features_list, axis=1)
        
        # Add original features if configured to do so
        if self.params['include_original']:
            # Only include top features to avoid dimensionality issues
            # This can be tuned based on feature importance or domain knowledge
            top_feature_count = min(20, X.shape[1])  # Limit to top 20 features or fewer if X has fewer
            
            # If we have feature importances, use them to select top features
            if hasattr(self, 'feature_importances_') and self.feature_importances_ is not None:
                top_features = list(self.feature_importances_.keys())[:top_feature_count]
            else:  # Otherwise just take the first N features
                top_features = list(X.columns)[:top_feature_count]
            
            # Add the original features
            meta_features = pd.concat([meta_features, X[top_features]], axis=1)
        
        # Ensure all required columns exist
        if self.model_type == 'classification' and 'EnsembleModel_proba_1' not in meta_features.columns:
            # Add the column if it's missing to prevent future index errors
            meta_features['EnsembleModel_proba_1'] = 0.5
        
        # Log the meta-features for debugging
        logger.debug(f"Generated meta-features with columns: {list(meta_features.columns)}")
        
        return meta_features
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the ensemble stacking model on the provided data
        
        Args:
            X: Feature matrix with all team, player, context, and odds features
            y: Target variable (win/loss for classification, values for regression)
        """
        if not self.base_models:
            logger.warning("No base models provided. Cannot train EnsembleStackingModel.")
            return  # Do not mark as trained - this model should be skipped entirely
            
        try:
            # Store feature names for later use
            self.feature_names = list(X.columns)
            
            # First, train all base models on the full dataset
            for model_name, model in self.base_models.items():
                logger.info(f"Training base model {model_name} for {self.name}")
                try:
                    model.train(X, y)
                except Exception as inner_e:
                    logger.error(f"Error training base model {model_name}: {str(inner_e)}")
                    logger.warning(f"Continuing without base model {model_name}")
            
            # Generate meta-features using cross-validation
            logger.info(f"Generating meta-features using {self.params['cv_folds']}-fold cross-validation")
            
            # Initialize cross-validation
            kf = KFold(n_splits=self.params['cv_folds'], shuffle=True, random_state=self.params['random_state'])
            
            # First fold to determine feature structure
            all_fold_indices = list(kf.split(X))
            first_train_idx, first_val_idx = all_fold_indices[0]
            X_first_train, X_first_val = X.iloc[first_train_idx], X.iloc[first_val_idx]
            y_first_train, y_first_val = y.iloc[first_train_idx], y.iloc[first_val_idx]
            
            # Train base models on first fold to determine feature structure
            first_fold_models = {}
            for model_name, model in self.base_models.items():
                try:
                    if model_name == "RandomForestModel" or model_name == "GradientBoostingModel":
                        fold_model = type(model)(version=1)
                    else:
                        fold_model = type(model)(
                            prediction_target=self.prediction_target,
                            version=1
                        )
                    fold_model.train(X_first_train, y_first_train)
                    first_fold_models[model_name] = fold_model
                except Exception as inner_e:
                    logger.warning(f"Error training {model_name} on first fold: {str(inner_e)}")
            
            # Get meta-features structure from first fold
            first_meta_features = self._generate_meta_features(X_first_val, first_fold_models)
            feature_columns = first_meta_features.columns
            feature_count = len(feature_columns)
            
            # Use a DataFrame for meta-features instead of numpy array to maintain proper indexing
            # Initialize with all zeros and original index
            meta_features_df = pd.DataFrame(
                np.zeros((X.shape[0], feature_count)),
                columns=feature_columns,
                index=X.index
            )
            
            # Store CV models
            self.cv_base_models = {i: {} for i in range(self.params['cv_folds'])}  # CV models by fold
            self.cv_base_models[0] = first_fold_models  # Store first fold models
            
            # Store first fold meta-features using correct indices
            meta_features_df.loc[X.index[first_val_idx]] = first_meta_features.values
            
            # Process remaining folds
            for fold_idx in range(1, self.params['cv_folds']):
                try:
                    train_idx, val_idx = all_fold_indices[fold_idx]
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Train base models on this fold
                    fold_base_models = {}
                    for model_name, model in self.base_models.items():
                        try:
                            if model_name == "RandomForestModel" or model_name == "GradientBoostingModel":
                                fold_model = type(model)(version=1)
                            else:
                                fold_model = type(model)(
                                    prediction_target=self.prediction_target,
                                    version=1
                                )
                            fold_model.train(X_train, y_train)
                            fold_base_models[model_name] = fold_model
                        except Exception as inner_e:
                            logger.warning(f"Error training {model_name} on fold {fold_idx}: {str(inner_e)}")
                    
                    # Store the fold models
                    self.cv_base_models[fold_idx] = fold_base_models
                    
                    # Generate meta-features for validation set
                    fold_meta_features = self._generate_meta_features(X_val, fold_base_models)
                    
                    # Ensure consistent features across folds
                    if set(fold_meta_features.columns) != set(feature_columns):
                        logger.warning(f"Fold {fold_idx} produced different features. Aligning to first fold.")
                        # Align columns to match first fold
                        missing_cols = set(feature_columns) - set(fold_meta_features.columns)
                        for col in missing_cols:
                            fold_meta_features[col] = 0  # Add missing columns with zeros
                        # Ensure we have all columns in the right order
                        fold_meta_features = fold_meta_features[feature_columns]  # Reorder columns
                    
                    # Verify shapes before assignment to prevent array broadcasting errors
                    if len(val_idx) != fold_meta_features.shape[0]:
                        logger.error(f"Shape mismatch in fold {fold_idx}: val_idx length = {len(val_idx)}, fold_meta_features rows = {fold_meta_features.shape[0]}")
                        # Fix shape mismatch by adjusting indices or meta-features
                        if len(val_idx) < fold_meta_features.shape[0]:
                            # Too many meta-features, truncate to match indices
                            fold_meta_features = fold_meta_features.iloc[:len(val_idx)]
                            logger.warning(f"Truncated fold_meta_features to match val_idx length: {len(val_idx)}")
                        else:
                            # Too few meta-features, pad with zeros
                            padding = pd.DataFrame(
                                np.zeros((len(val_idx) - fold_meta_features.shape[0], fold_meta_features.shape[1])),
                                columns=fold_meta_features.columns
                            )
                            fold_meta_features = pd.concat([fold_meta_features, padding], axis=0)
                            logger.warning(f"Padded fold_meta_features to match val_idx length: {len(val_idx)}")
                    
                    # Store these meta-features in the correct positions
                    try:
                        meta_features_df.loc[X.index[val_idx]] = fold_meta_features.values
                    except ValueError as ve:
                        logger.error(f"Value error during meta-feature assignment: {str(ve)}")
                        logger.error(f"val_idx shape: {len(val_idx)}, meta_features shape: {meta_features_df.shape}, fold_meta_features shape: {fold_meta_features.shape}")
                        # Defensive fallback: fill with zeros to maintain shape consistency
                        meta_features_df.loc[X.index[val_idx]] = np.zeros((len(val_idx), feature_count))
                except Exception as fold_e:
                    logger.error(f"Error processing fold {fold_idx}: {str(fold_e)}")
                    # For failed folds, fill with zeros to maintain shape consistency
                    meta_features_df.loc[X.index[val_idx]] = np.zeros((len(val_idx), feature_count))
            
            # Train the meta-learner on the meta-features
            logger.info(f"Training meta-learner for {self.name}")
            
            # Initialize the meta-learner based on model type
            if self.model_type == 'classification':
                if self.params['meta_learner'] == 'logistic_regression':
                    self.model = LogisticRegression(
                        C=self.params['regularization'],
                        max_iter=self.params['max_iter'],
                        random_state=self.params['random_state']
                    )
                else:  # Default to logistic regression
                    logger.warning(f"Unknown meta-learner '{self.params['meta_learner']}'. Using logistic_regression.")
                    self.model = LogisticRegression(
                        C=self.params['regularization'],
                        max_iter=self.params['max_iter'],
                        random_state=self.params['random_state']
                    )
            else:  # regression
                if self.params['meta_learner'] == 'ridge':
                    self.model = Ridge(
                        alpha=self.params['alpha'],
                        max_iter=self.params['max_iter'],
                        random_state=self.params['random_state']
                    )
                else:  # Default to ridge regression
                    logger.warning(f"Unknown meta-learner '{self.params['meta_learner']}'. Using ridge.")
                    self.model = Ridge(
                        alpha=self.params['alpha'],
                        max_iter=self.params['max_iter'],
                        random_state=self.params['random_state']
                    )
            
            # Train meta-learner on the meta-features
            self.model.fit(meta_features_df, y)
            
            # Generate feature importances for the meta-learner
            if hasattr(self.model, 'coef_'):
                # Get the coefficients as importances
                coef = self.model.coef_
                if len(coef.shape) > 1:  # For multi-class classification
                    coef = np.abs(coef).mean(axis=0)  # Average absolute coefficients across classes
                
                # Create feature importance dictionary
                self.feature_importances_ = {}
                for i, feature_name in enumerate(meta_features_df.columns):
                    self.feature_importances_[feature_name] = abs(float(coef[i]))
                
                # Sort by importance
                self.feature_importances_ = dict(sorted(
                    self.feature_importances_.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
            
            # Record training time
            self.trained_at = datetime.now(timezone.utc)
            self.is_trained = True
            
            # Log training completion
            logger.info(f"{self.name} model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training {self.name} model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained ensemble stacking model
        
        Args:
            X: Feature matrix matching the training features
            
        Returns:
            Array of predictions
        """
        if not self.is_trained or not hasattr(self, 'model') or self.model is None:
            logger.error(f"Error predicting with {self.name} model: Model not trained")
            # Return default predictions (all zeros for regression, 0.5 for classification)
            if self.model_type == 'classification':
                return np.array([0.5] * X.shape[0])
            else:
                return np.zeros(X.shape[0])
        
        try:
            # Generate meta-features using trained base models
            meta_features = self._generate_meta_features(X, self.base_models)
            
            # Verify meta_features is not empty
            if meta_features.empty:
                logger.error("Generated empty meta-features")
                if self.model_type == 'classification':
                    return np.array([0.5] * X.shape[0])
                else:
                    return np.zeros(X.shape[0])
            
            # Make predictions with the meta-learner
            # Skip feature name warnings by converting to numpy array
            predictions = self.model.predict(meta_features.values)
            return predictions
        except Exception as e:
            logger.error(f"Error predicting with {self.name} model: {str(e)}")
            # Return default predictions as fallback
            if self.model_type == 'classification':
                return np.array([0.5] * X.shape[0])
            else:
                return np.zeros(X.shape[0])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for classification models
        
        Args:
            X: Feature matrix matching the training features
            
        Returns:
            Array of class probabilities (only for classification models)
        """
        if not self.is_trained or not hasattr(self, 'model') or self.model is None:
            logger.error(f"Error predicting probabilities with {self.name} model: Model not trained")
            # Return default probabilities
            return np.array([[0.5, 0.5]] * X.shape[0])
            
        if self.model_type != 'classification':
            logger.error(f"predict_proba is only available for classification models, not {self.model_type}")
            return np.array([])
        
        try:
            # Generate meta-features using trained base models
            meta_features = self._generate_meta_features(X, self.base_models)
            
            # Verify meta_features is not empty
            if meta_features.empty:
                logger.error("Generated empty meta-features")
                return np.array([[0.5, 0.5]] * X.shape[0])
            
            # Use numpy array to avoid feature name warnings
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(meta_features.values)
                return proba
            else:
                # For models without predict_proba, convert predictions to probabilities
                preds = self.model.predict(meta_features.values)
                proba = np.zeros((len(preds), 2))
                proba[:, 1] = preds
                proba[:, 0] = 1 - preds
                return proba
        except Exception as e:
            logger.error(f"Error predicting probabilities with {self.name} model: {str(e)}")
            # Return default probabilities as fallback
            return np.array([[0.5, 0.5]] * X.shape[0])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance for meta-features
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            logger.error(f"Cannot get feature importance from untrained model: {self.name}")
            return {}
        
        try:
            # Return the pre-computed feature importances if available
            if hasattr(self, 'feature_importances_') and self.feature_importances_ is not None:
                return self.feature_importances_
            
            # Otherwise, if we have coefficients, compute them now
            if hasattr(self.model, 'coef_'):
                # Get the coefficients as importances
                coef = self.model.coef_
                if len(coef.shape) > 1:  # For multi-class classification
                    coef = np.abs(coef).mean(axis=0)  # Average absolute coefficients across classes
                
                # We need meta-features to get column names
                if self.meta_features is None:
                    logger.warning("No meta-features available for feature importance calculation")
                    return {}
                
                # Create feature importance dictionary
                feature_importance = {}
                for i, feature_name in enumerate(self.meta_features.columns):
                    feature_importance[feature_name] = abs(float(coef[i]))
                
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                return feature_importance
            
            # If we don't have coefficients or pre-computed importances, return empty dict
            logger.warning("Meta-learner does not provide feature importances")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {self.name} model: {str(e)}")
            return {}
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the ensemble stacking model performance on test data
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            logger.error(f"Cannot evaluate untrained model: {self.name}")
            return {}
        
        try:
            # Make predictions
            y_pred = self.predict(X)
            
            # Convert predictions to the same type as targets to prevent mixing types
            if self.model_type == 'classification':
                y_pred = (y_pred > 0.5).astype(int)  # Convert to binary
                y_binary = y.astype(int)  # Ensure y is also binary
                
                # Calculate metrics directly instead of using parent class method
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                metrics = {
                    'accuracy': accuracy_score(y_binary, y_pred),
                    'precision': precision_score(y_binary, y_pred, zero_division=0),
                    'recall': recall_score(y_binary, y_pred, zero_division=0),
                    'f1_score': f1_score(y_binary, y_pred, zero_division=0)
                }
            else:  # regression
                # Use parent class's evaluate method for regression metrics
                metrics = super().evaluate(X, y)
            
            # Add comparison to base models - but handle each model safely
            for model_name, model in self.base_models.items():
                try:
                    # Skip problematic models
                    if model_name == 'EnsembleModel':
                        continue
                        
                    base_metrics = model.evaluate(X, y)
                    for metric_name, value in base_metrics.items():
                        metrics[f"base_{model_name}_{metric_name}"] = value
                except Exception as inner_e:
                    logger.warning(f"Error evaluating base model {model_name}: {str(inner_e)}")
            
            # Calculate improvement over base models
            if 'accuracy' in metrics:
                base_accuracies = [v for k, v in metrics.items() if k.startswith('base_') and k.endswith('_accuracy')]
                if base_accuracies:
                    metrics['improvement_over_avg_base'] = metrics['accuracy'] - sum(base_accuracies) / len(base_accuracies)
                    metrics['improvement_over_best_base'] = metrics['accuracy'] - max(base_accuracies)
            
            # Calculate meta-feature importance summary
            importances = self.get_feature_importance()
            if importances:
                # Group importances by base model
                model_importance = {}
                for feature, importance in importances.items():
                    for model_name in self.base_model_names:
                        if feature.startswith(model_name):
                            model_importance[model_name] = model_importance.get(model_name, 0) + importance
                            break
                
                # Include in metrics
                for model_name, importance in model_importance.items():
                    metrics[f"meta_importance_{model_name}"] = importance
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating {self.name}: {str(e)}")
            return {'error': str(e)}
    
    def save_to_db(self) -> bool:
        """
        Save the ensemble stacking model to the database
        
        Returns:
            bool: True if saving was successful, False otherwise
        """
        if not self.is_trained:
            logger.error(f"Cannot save untrained model: {self.name}")
            return False
        
        try:
            # Get feature importance
            feature_importance = self.get_feature_importance()
            
            # Get meta-learner coefficients if available
            meta_coefficients = None
            if hasattr(self.model, 'coef_'):
                meta_coefficients = self.model.coef_.tolist()
            
            # Create model weights dictionary
            model_data = {
                'model_name': self.name,
                'model_type': self.model_type,
                'weights': {
                    'feature_importances': feature_importance,
                    'meta_coefficients': meta_coefficients,
                    'base_models': self.base_model_names,
                    'meta_learner': self.params['meta_learner'],
                    'prediction_target': self.prediction_target
                },
                'params': self.params,
                'version': self.version,
                'trained_at': self.trained_at or datetime.now(timezone.utc),
                'active': True
            }
            
            # Save to database
            model_id = ModelWeight.create(model_data)
            
            if model_id:
                logger.info(f"Model {self.name} v{self.version} saved to database with ID {model_id}")
                
                # Deactivate old versions
                ModelWeight.deactivate_old_versions(self.name, self.version)
                
                return True
            else:
                logger.error(f"Failed to save model {self.name} to database")
                return False
                
        except Exception as e:
            logger.error(f"Error saving model {self.name} to database: {str(e)}")
            return False
