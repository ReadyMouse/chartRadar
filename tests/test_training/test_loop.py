"""Tests for training loop."""

import pytest
import pandas as pd
from unittest.mock import Mock

from chartradar.training.loop import GenericTrainingLoop
from chartradar.core.exceptions import TrainingError


class TestGenericTrainingLoop:
    """Tests for GenericTrainingLoop class."""
    
    def test_detect_framework_sklearn(self):
        """Test framework detection for sklearn."""
        loop = GenericTrainingLoop()
        
        # Mock sklearn model
        class SklearnModel:
            def fit(self, X, y):
                pass
        
        model = SklearnModel()
        framework = loop._detect_framework(model)
        assert framework == "sklearn"
    
    def test_train_sklearn_model(self):
        """Test training sklearn model."""
        loop = GenericTrainingLoop()
        
        # Mock sklearn model
        class MockSklearnModel:
            def fit(self, X, y):
                self.fitted = True
            def predict(self, X):
                return [0, 1, 0]
        
        model = MockSklearnModel()
        train_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        train_labels = pd.Series([0, 1, 0])
        
        result = loop.train(model, train_data, train_labels)
        
        assert result["training_complete"] is True
        assert model.fitted
    
    def test_train_generic_model(self):
        """Test training generic model with fit method."""
        loop = GenericTrainingLoop()
        
        class MockModel:
            def fit(self, X, y, **kwargs):
                self.fitted = True
        
        model = MockModel()
        train_data = pd.DataFrame({'feature1': [1, 2, 3]})
        train_labels = pd.Series([0, 1, 0])
        
        result = loop.train(model, train_data, train_labels)
        
        assert result["training_complete"] is True
        assert model.fitted
    
    def test_train_model_no_fit(self):
        """Test training model without fit method."""
        loop = GenericTrainingLoop()
        
        class MockModel:
            pass
        
        model = MockModel()
        train_data = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(TrainingError) as exc_info:
            loop.train(model, train_data)
        assert "does not have a 'fit' method" in str(exc_info.value)
    
    def test_get_training_state(self):
        """Test getting training state."""
        loop = GenericTrainingLoop()
        state = loop.get_training_state()
        
        assert "epoch" in state
        assert "history" in state

