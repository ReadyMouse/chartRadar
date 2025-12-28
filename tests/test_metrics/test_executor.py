"""Tests for algorithm executor."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from chartradar.metrics.executor import AlgorithmExecutor
from chartradar.metrics.registry import AlgorithmRegistry
from chartradar.metrics.base import Algorithm
from chartradar.src.types import AlgorithmResult


class MockAlgorithm(Algorithm):
    """Mock algorithm for testing executor."""
    
    def __init__(self, name="mock", version="1.0.0", should_fail=False, **kwargs):
        super().__init__(name, version, **kwargs)
        self.should_fail = should_fail
    
    def process(self, data, **kwargs):
        if self.should_fail:
            raise Exception("Mock algorithm failure")
        
        from datetime import datetime
        return {
            "algorithm_name": self.name,
            "results": [],
            "confidence_scores": [0.8],
            "metadata": {"test": True},
            "timestamp": datetime.now()
        }
    
    def get_metadata(self):
        return {
            "name": self.name,
            "version": self.version,
            "description": "Mock",
            "author": "Test",
            "parameters": {},
            "requirements": self.get_requirements()
        }
    
    def get_requirements(self):
        return {
            "min_data_points": 1,
            "required_columns": ["open", "high", "low", "close"],
            "data_frequency": "any",
            "preprocessing": []
        }


class TestAlgorithmExecutor:
    """Tests for AlgorithmExecutor class."""
    
    def test_execute_single_algorithm(self):
        """Test executing a single algorithm."""
        registry = AlgorithmRegistry()
        registry.register(MockAlgorithm, name="mock_alg")
        
        executor = AlgorithmExecutor(registry=registry)
        
        data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [99, 100],
            'close': [103, 104],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        configs = [{"name": "mock_alg", "enabled": True}]
        results = executor.execute(data, configs)
        
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].algorithm_name == "mock_alg"
    
    def test_execute_multiple_algorithms(self):
        """Test executing multiple algorithms."""
        registry = AlgorithmRegistry()
        registry.register(MockAlgorithm, name="alg1")
        registry.register(MockAlgorithm, name="alg2")
        
        executor = AlgorithmExecutor(registry=registry)
        
        data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [99, 100],
            'close': [103, 104],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        configs = [
            {"name": "alg1", "enabled": True},
            {"name": "alg2", "enabled": True}
        ]
        results = executor.execute(data, configs)
        
        assert len(results) == 2
        assert all(r.success for r in results)
    
    def test_execute_disabled_algorithm(self):
        """Test that disabled algorithms are skipped."""
        registry = AlgorithmRegistry()
        registry.register(MockAlgorithm, name="alg1")
        registry.register(MockAlgorithm, name="alg2")
        
        executor = AlgorithmExecutor(registry=registry)
        
        data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [99, 100],
            'close': [103, 104],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        configs = [
            {"name": "alg1", "enabled": True},
            {"name": "alg2", "enabled": False}
        ]
        results = executor.execute(data, configs)
        
        assert len(results) == 1
        assert results[0].algorithm_name == "alg1"
    
    def test_execute_with_failure(self):
        """Test handling algorithm failures."""
        registry = AlgorithmRegistry()
        registry.register(MockAlgorithm, name="failing_alg")
        
        executor = AlgorithmExecutor(registry=registry)
        
        data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [99, 100],
            'close': [103, 104],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        # Create algorithm that will fail
        with patch.object(registry, 'create_algorithm') as mock_create:
            mock_alg = MockAlgorithm(name="failing_alg", should_fail=True)
            mock_create.return_value = mock_alg
            
            configs = [{"name": "failing_alg", "enabled": True}]
            results = executor.execute(data, configs)
            
            assert len(results) == 1
            assert results[0].success is False
            assert results[0].error_message is not None
    
    def test_get_execution_summary(self):
        """Test getting execution summary."""
        executor = AlgorithmExecutor()
        
        results = [
            AlgorithmResult(
                algorithm_name="alg1",
                success=True,
                processing_time_ms=100.0,
                confidence_scores=[0.8, 0.9]
            ),
            AlgorithmResult(
                algorithm_name="alg2",
                success=True,
                processing_time_ms=150.0,
                confidence_scores=[0.7]
            )
        ]
        
        summary = executor.get_execution_summary(results)
        
        assert summary["total_algorithms"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0
        assert summary["total_processing_time_ms"] == 250.0

