"""Tests for algorithm base class."""

import pytest
import pandas as pd
from datetime import datetime

from chartradar.metrics.base import Algorithm
from chartradar.core.exceptions import AlgorithmError


class ConcreteAlgorithm(Algorithm):
    """Concrete algorithm implementation for testing."""
    
    def process(self, data: pd.DataFrame, **kwargs):
        """Process test data."""
        return {
            "algorithm_name": self.name,
            "results": [],
            "confidence_scores": [],
            "metadata": {},
            "timestamp": datetime.now()
        }
    
    def get_metadata(self):
        """Get test metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Test algorithm",
            "author": "Test",
            "parameters": {},
            "requirements": self.get_requirements()
        }
    
    def get_requirements(self):
        """Get test requirements."""
        return {
            "min_data_points": 10,
            "required_columns": ["open", "high", "low", "close"],
            "data_frequency": "any",
            "preprocessing": []
        }


class TestAlgorithm:
    """Tests for Algorithm base class."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        alg = ConcreteAlgorithm("test_alg", version="1.0.0", param1="value1")
        assert alg.name == "test_alg"
        assert alg.version == "1.0.0"
        assert alg.parameters["param1"] == "value1"
    
    def test_process(self):
        """Test processing data."""
        alg = ConcreteAlgorithm("test")
        data = pd.DataFrame({
            'open': range(10),
            'high': range(10, 20),
            'low': range(0, 10),
            'close': range(5, 15),
            'volume': range(100, 110)
        }, index=pd.date_range('2024-01-01', periods=10))
        
        result = alg.process(data)
        assert result["algorithm_name"] == "test"
        assert "timestamp" in result
    
    def test_validate_data_sufficient(self):
        """Test data validation with sufficient data."""
        alg = ConcreteAlgorithm("test")
        data = pd.DataFrame({
            'open': range(10),
            'high': range(10, 20),
            'low': range(0, 10),
            'close': range(5, 15)
        }, index=pd.date_range('2024-01-01', periods=10))
        
        assert alg.validate_data(data) is True
    
    def test_validate_data_insufficient(self):
        """Test data validation with insufficient data."""
        alg = ConcreteAlgorithm("test")
        data = pd.DataFrame({
            'open': range(5),
            'high': range(5, 10),
            'low': range(0, 5),
            'close': range(2, 7)
        }, index=pd.date_range('2024-01-01', periods=5))
        
        with pytest.raises(AlgorithmError) as exc_info:
            alg.validate_data(data)
        assert "requires at least" in str(exc_info.value).lower()
    
    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        alg = ConcreteAlgorithm("test")
        data = pd.DataFrame({
            'open': range(10),
            'high': range(10, 20)
        }, index=pd.date_range('2024-01-01', periods=10))
        
        with pytest.raises(AlgorithmError) as exc_info:
            alg.validate_data(data)
        assert "requires columns" in str(exc_info.value).lower()
    
    def test_process_to_result(self):
        """Test process_to_result method."""
        alg = ConcreteAlgorithm("test")
        data = pd.DataFrame({
            'open': range(10),
            'high': range(10, 20),
            'low': range(0, 10),
            'close': range(5, 15),
            'volume': range(100, 110)
        }, index=pd.date_range('2024-01-01', periods=10))
        
        result = alg.process_to_result(data)
        assert result.algorithm_name == "test"
        assert result.success is True
        assert result.processing_time_ms is not None

