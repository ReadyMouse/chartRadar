"""
Base training interface for the ChartRadar framework.

This module provides abstract interfaces for training and evaluation loops.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class TrainingLoop(ABC):
    """
    Abstract interface for training loops.
    
    Provides a standard interface for training ML models that can work
    with different frameworks (scikit-learn, TensorFlow, PyTorch).
    """
    
    @abstractmethod
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
        Train a model.
        
        Args:
            model: Model to train (framework-agnostic)
            train_data: Training data
            train_labels: Training labels (for supervised learning)
            validation_data: Optional validation data
            validation_labels: Optional validation labels
            **kwargs: Training parameters (epochs, batch_size, etc.)
            
        Returns:
            Dictionary with training results (loss, metrics, etc.)
        """
        pass
    
    @abstractmethod
    def get_training_state(self) -> Dict[str, Any]:
        """
        Get current training state.
        
        Returns:
            Dictionary with training state information
        """
        pass


class EvaluationLoop(ABC):
    """
    Abstract interface for evaluation loops.
    
    Provides a standard interface for evaluating ML models.
    """
    
    @abstractmethod
    def evaluate(
        self,
        model: Any,
        test_data: pd.DataFrame,
        test_labels: Optional[pd.Series] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Evaluate a model.
        
        Args:
            model: Model to evaluate
            test_data: Test data
            test_labels: Test labels (for supervised learning)
            **kwargs: Evaluation parameters
            
        Returns:
            Dictionary with evaluation results (metrics, predictions, etc.)
        """
        pass
    
    @abstractmethod
    def calculate_metrics(
        self,
        predictions: Any,
        labels: Any,
        **kwargs: Any
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions
            labels: True labels
            **kwargs: Metric calculation parameters
            
        Returns:
            Dictionary with metric names and values
        """
        pass

