"""
Training loop implementation for the ChartRadar framework.

This module provides generic training loops that work with different ML frameworks.
"""

from typing import Any, Callable, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from chartradar.training.base import TrainingLoop
from chartradar.src.exceptions import TrainingError

logger = logging.getLogger(__name__)


class GenericTrainingLoop(TrainingLoop):
    """
    Generic training loop that works with scikit-learn, TensorFlow, PyTorch, etc.
    
    Provides a framework-agnostic interface for training models.
    """
    
    def __init__(
        self,
        framework: str = "auto",
        **kwargs: Any
    ):
        """
        Initialize the training loop.
        
        Args:
            framework: ML framework ('auto', 'sklearn', 'tensorflow', 'pytorch')
            **kwargs: Training parameters
        """
        self.framework = framework
        self.training_state = {
            "epoch": 0,
            "history": [],
            "best_loss": float('inf'),
            "best_metrics": {},
            "early_stopped": False
        }
        self.parameters = kwargs
    
    def train(
        self,
        model: Any,
        train_data: pd.DataFrame,
        train_labels: Optional[pd.Series] = None,
        validation_data: Optional[pd.DataFrame] = None,
        validation_labels: Optional[pd.Series] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Train a model using the appropriate framework method.
        
        Args:
            model: Model to train
            train_data: Training data
            train_labels: Training labels
            validation_data: Optional validation data
            validation_labels: Optional validation labels
            **kwargs: Training parameters (epochs, batch_size, etc.)
            
        Returns:
            Dictionary with training results
        """
        # Detect framework
        framework = self._detect_framework(model)
        
        # Merge parameters
        training_params = {**self.parameters, **kwargs}
        
        # Get early stopping configuration
        early_stopping = training_params.get('early_stopping', {})
        patience = early_stopping.get('patience', None)
        monitor = early_stopping.get('monitor', 'val_loss')
        min_delta = early_stopping.get('min_delta', 0.0)
        
        # Initialize training state
        self.training_state = {
            "epoch": 0,
            "history": [],
            "best_loss": float('inf'),
            "best_metrics": {},
            "early_stopped": False,
            "patience_counter": 0
        }
        
        try:
            if framework == "sklearn":
                return self._train_sklearn(
                    model, train_data, train_labels,
                    validation_data, validation_labels,
                    **training_params
                )
            elif framework in ["tensorflow", "keras"]:
                return self._train_tensorflow(
                    model, train_data, train_labels,
                    validation_data, validation_labels,
                    **training_params
                )
            elif framework == "pytorch":
                return self._train_pytorch(
                    model, train_data, train_labels,
                    validation_data, validation_labels,
                    **training_params
                )
            else:
                # Generic training (assume model has fit method)
                return self._train_generic(
                    model, train_data, train_labels,
                    validation_data, validation_labels,
                    **training_params
                )
        except Exception as e:
            raise TrainingError(
                f"Training failed: {str(e)}",
                details={"framework": framework, "error": str(e)}
            ) from e
    
    def _detect_framework(self, model: Any) -> str:
        """Detect the ML framework from the model."""
        model_type = type(model).__name__.lower()
        model_module = type(model).__module__.lower()
        
        if 'sklearn' in model_module or 'sklearn' in model_type:
            return "sklearn"
        elif 'tensorflow' in model_module or 'keras' in model_module:
            return "tensorflow"
        elif 'torch' in model_module or 'pytorch' in model_module:
            return "pytorch"
        else:
            return "generic"
    
    def _train_sklearn(
        self,
        model: Any,
        train_data: pd.DataFrame,
        train_labels: Optional[pd.Series],
        validation_data: Optional[pd.DataFrame],
        validation_labels: Optional[pd.Series],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Train a scikit-learn model."""
        # Scikit-learn models typically have a fit method
        if train_labels is not None:
            model.fit(train_data, train_labels)
        else:
            # Unsupervised learning
            model.fit(train_data)
        
        # Evaluate on validation set if provided
        val_metrics = {}
        if validation_data is not None:
            if validation_labels is not None:
                val_pred = model.predict(validation_data)
                # Calculate basic metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                val_metrics = {
                    "val_accuracy": accuracy_score(validation_labels, val_pred),
                    "val_precision": precision_score(validation_labels, val_pred, average='weighted', zero_division=0),
                    "val_recall": recall_score(validation_labels, val_pred, average='weighted', zero_division=0),
                    "val_f1": f1_score(validation_labels, val_pred, average='weighted', zero_division=0)
                }
        
        return {
            "model": model,
            "training_complete": True,
            "validation_metrics": val_metrics,
            "history": []
        }
    
    def _train_tensorflow(
        self,
        model: Any,
        train_data: pd.DataFrame,
        train_labels: Optional[pd.Series],
        validation_data: Optional[pd.DataFrame],
        validation_labels: Optional[pd.Series],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Train a TensorFlow/Keras model."""
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 32)
        verbose = kwargs.get('verbose', 1)
        
        # Prepare validation data
        validation_tuple = None
        if validation_data is not None and validation_labels is not None:
            validation_tuple = (validation_data, validation_labels)
        
        # Train model
        history = model.fit(
            train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_tuple,
            verbose=verbose,
            **{k: v for k, v in kwargs.items() if k not in ['epochs', 'batch_size', 'verbose']}
        )
        
        return {
            "model": model,
            "training_complete": True,
            "history": history.history if hasattr(history, 'history') else [],
            "validation_metrics": history.history.get('val_loss', []) if hasattr(history, 'history') else {}
        }
    
    def _train_pytorch(
        self,
        model: Any,
        train_data: pd.DataFrame,
        train_labels: Optional[pd.Series],
        validation_data: Optional[pd.DataFrame],
        validation_labels: Optional[pd.Series],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Train a PyTorch model."""
        # PyTorch training requires more setup (optimizer, loss function, etc.)
        # This is a placeholder - actual implementation would require
        # optimizer, criterion, and training loop
        raise TrainingError(
            "PyTorch training requires optimizer and loss function setup",
            details={"framework": "pytorch"}
        )
    
    def _train_generic(
        self,
        model: Any,
        train_data: pd.DataFrame,
        train_labels: Optional[pd.Series],
        validation_data: Optional[pd.DataFrame],
        validation_labels: Optional[pd.Series],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Train a model using generic fit method."""
        if hasattr(model, 'fit'):
            if train_labels is not None:
                model.fit(train_data, train_labels, **kwargs)
            else:
                model.fit(train_data, **kwargs)
            
            return {
                "model": model,
                "training_complete": True,
                "history": []
            }
        else:
            raise TrainingError(
                "Model does not have a 'fit' method",
                details={"model_type": type(model).__name__}
            )
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state."""
        return self.training_state.copy()

