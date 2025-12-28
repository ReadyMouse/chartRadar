"""
Evaluation loop implementation for the ChartRadar framework.

This module provides functionality to evaluate ML models and calculate metrics.
"""

from typing import Any, Callable, Dict, List, Optional
import pandas as pd
import numpy as np
import logging

from chartradar.training.base import EvaluationLoop
from chartradar.src.exceptions import EvaluationError

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class GenericEvaluationLoop(EvaluationLoop):
    """
    Generic evaluation loop for ML models.
    
    Supports various evaluation metrics and custom metric functions.
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        custom_metrics: Optional[Dict[str, Callable]] = None,
        **kwargs: Any
    ):
        """
        Initialize the evaluation loop.
        
        Args:
            metrics: List of metric names to calculate
            custom_metrics: Dictionary of custom metric functions
            **kwargs: Additional parameters
        """
        self.metrics = metrics or ["accuracy", "precision", "recall", "f1"]
        self.custom_metrics = custom_metrics or {}
        self.parameters = kwargs
    
    def evaluate(
        self,
        model: Any,
        test_data: pd.DataFrame,
        test_labels: Optional[pd.Series] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Evaluate a model on test data.
        
        Args:
            model: Model to evaluate
            test_data: Test data
            test_labels: Test labels (for supervised learning)
            **kwargs: Evaluation parameters
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Get predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(test_data)
            elif hasattr(model, '__call__'):
                predictions = model(test_data)
            else:
                raise EvaluationError(
                    "Model does not support prediction",
                    details={"model_type": type(model).__name__}
                )
            
            # Calculate metrics if labels provided
            metrics = {}
            if test_labels is not None:
                metrics = self.calculate_metrics(predictions, test_labels, **kwargs)
            
            return {
                "predictions": predictions,
                "metrics": metrics,
                "test_size": len(test_data),
                "model_type": type(model).__name__
            }
        except Exception as e:
            raise EvaluationError(
                f"Evaluation failed: {str(e)}",
                details={"error": str(e)}
            ) from e
    
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
        metrics = {}
        
        # Convert to numpy arrays if needed
        if isinstance(predictions, pd.Series):
            predictions = predictions.values
        if isinstance(labels, pd.Series):
            labels = labels.values
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Calculate standard metrics
        if SKLEARN_AVAILABLE:
            if "accuracy" in self.metrics:
                metrics["accuracy"] = float(accuracy_score(labels, predictions))
            
            if "precision" in self.metrics:
                try:
                    metrics["precision"] = float(precision_score(
                        labels, predictions, average='weighted', zero_division=0
                    ))
                except Exception:
                    metrics["precision"] = 0.0
            
            if "recall" in self.metrics:
                try:
                    metrics["recall"] = float(recall_score(
                        labels, predictions, average='weighted', zero_division=0
                    ))
                except Exception:
                    metrics["recall"] = 0.0
            
            if "f1" in self.metrics or "f1_score" in self.metrics:
                try:
                    metrics["f1"] = float(f1_score(
                        labels, predictions, average='weighted', zero_division=0
                    ))
                except Exception:
                    metrics["f1"] = 0.0
            
            if "confusion_matrix" in self.metrics:
                metrics["confusion_matrix"] = confusion_matrix(labels, predictions).tolist()
            
            # ROC AUC for binary classification
            if "roc_auc" in self.metrics:
                try:
                    if len(np.unique(labels)) == 2:
                        metrics["roc_auc"] = float(roc_auc_score(labels, predictions))
                except Exception:
                    pass
        else:
            # Fallback: basic accuracy calculation
            if "accuracy" in self.metrics:
                metrics["accuracy"] = float(np.mean(predictions == labels))
        
        # Calculate custom metrics
        for metric_name, metric_func in self.custom_metrics.items():
            try:
                metrics[metric_name] = float(metric_func(predictions, labels))
            except Exception as e:
                logger.warning(f"Failed to calculate custom metric '{metric_name}': {str(e)}")
        
        return metrics
    
    def generate_report(
        self,
        predictions: Any,
        labels: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            predictions: Model predictions
            labels: True labels
            **kwargs: Report generation parameters
            
        Returns:
            Dictionary with evaluation report
        """
        metrics = self.calculate_metrics(predictions, labels, **kwargs)
        
        report = {
            "metrics": metrics,
            "summary": {
                "total_samples": len(labels) if hasattr(labels, '__len__') else 0,
                "prediction_distribution": self._get_distribution(predictions),
                "label_distribution": self._get_distribution(labels)
            }
        }
        
        # Add classification report if sklearn available
        if SKLEARN_AVAILABLE and "classification_report" in self.metrics:
            try:
                report["classification_report"] = classification_report(
                    labels, predictions, output_dict=True, zero_division=0
                )
            except Exception:
                pass
        
        return report
    
    def _get_distribution(self, values: Any) -> Dict[str, int]:
        """Get value distribution."""
        if isinstance(values, pd.Series):
            values = values.values
        values = np.array(values)
        
        unique, counts = np.unique(values, return_counts=True)
        return dict(zip(unique.astype(str), counts.astype(int)))

