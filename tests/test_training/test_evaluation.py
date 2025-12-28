"""Tests for evaluation loop."""

import pytest
import pandas as pd
import numpy as np

from chartradar.training.evaluation import GenericEvaluationLoop
from chartradar.src.exceptions import EvaluationError


class TestGenericEvaluationLoop:
    """Tests for GenericEvaluationLoop class."""
    
    def test_evaluate_model(self):
        """Test evaluating a model."""
        loop = GenericEvaluationLoop()
        
        class MockModel:
            def predict(self, X):
                return np.array([0, 1, 0, 1])
        
        model = MockModel()
        test_data = pd.DataFrame({'feature1': [1, 2, 3, 4]})
        test_labels = pd.Series([0, 1, 0, 1])
        
        result = loop.evaluate(model, test_data, test_labels)
        
        assert "predictions" in result
        assert "metrics" in result
        assert "accuracy" in result["metrics"]
    
    def test_calculate_metrics(self):
        """Test calculating metrics."""
        loop = GenericEvaluationLoop(metrics=["accuracy", "precision", "recall", "f1"])
        
        predictions = np.array([0, 1, 0, 1])
        labels = np.array([0, 1, 0, 1])
        
        metrics = loop.calculate_metrics(predictions, labels)
        
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 1.0
    
    def test_evaluate_model_no_predict(self):
        """Test evaluating model without predict method."""
        loop = GenericEvaluationLoop()
        
        class MockModel:
            pass
        
        model = MockModel()
        test_data = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(EvaluationError) as exc_info:
            loop.evaluate(model, test_data)
        assert "does not support prediction" in str(exc_info.value)
    
    def test_generate_report(self):
        """Test generating evaluation report."""
        loop = GenericEvaluationLoop()
        
        predictions = np.array([0, 1, 0, 1])
        labels = np.array([0, 1, 0, 1])
        
        report = loop.generate_report(predictions, labels)
        
        assert "metrics" in report
        assert "summary" in report

