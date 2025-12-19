"""Tests for algorithm registry."""

import pytest
from chartradar.metrics.registry import (
    AlgorithmRegistry,
    register_algorithm,
    get_algorithm,
    create_algorithm,
)
from chartradar.metrics.base import Algorithm
from chartradar.core.exceptions import AlgorithmNotFoundError


class TestAlgorithm(Algorithm):
    """Test algorithm for registry tests."""
    
    def process(self, data, **kwargs):
        return {
            "algorithm_name": self.name,
            "results": [],
            "confidence_scores": [],
            "metadata": {},
            "timestamp": None
        }
    
    def get_metadata(self):
        return {
            "name": self.name,
            "version": self.version,
            "description": "Test",
            "author": "Test",
            "parameters": {},
            "requirements": self.get_requirements()
        }
    
    def get_requirements(self):
        return {
            "min_data_points": 1,
            "required_columns": [],
            "data_frequency": "any",
            "preprocessing": []
        }


class TestAlgorithmRegistry:
    """Tests for AlgorithmRegistry class."""
    
    def test_register_algorithm(self):
        """Test registering an algorithm."""
        registry = AlgorithmRegistry()
        registry.register(TestAlgorithm, name="test_alg", version="1.0.0")
        
        assert "test_alg" in registry.list_algorithms()
    
    def test_get_algorithm(self):
        """Test getting an algorithm."""
        registry = AlgorithmRegistry()
        registry.register(TestAlgorithm, name="test_alg", version="1.0.0")
        
        alg_class = registry.get_algorithm("test_alg")
        assert alg_class == TestAlgorithm
    
    def test_get_algorithm_not_found(self):
        """Test getting non-existent algorithm."""
        registry = AlgorithmRegistry()
        
        with pytest.raises(AlgorithmNotFoundError):
            registry.get_algorithm("nonexistent")
    
    def test_get_latest_version(self):
        """Test getting latest version of algorithm."""
        registry = AlgorithmRegistry()
        registry.register(TestAlgorithm, name="test_alg", version="1.0.0")
        registry.register(TestAlgorithm, name="test_alg", version="2.0.0")
        
        alg_class = registry.get_algorithm("test_alg")  # Should get latest
        assert alg_class == TestAlgorithm
    
    def test_create_algorithm(self):
        """Test creating algorithm instance."""
        registry = AlgorithmRegistry()
        registry.register(TestAlgorithm, name="test_alg", version="1.0.0")
        
        alg = registry.create_algorithm("test_alg", param1="value1")
        assert isinstance(alg, TestAlgorithm)
        assert alg.parameters["param1"] == "value1"
    
    def test_list_algorithms(self):
        """Test listing algorithms."""
        registry = AlgorithmRegistry()
        registry.register(TestAlgorithm, name="alg1")
        registry.register(TestAlgorithm, name="alg2")
        
        algorithms = registry.list_algorithms()
        assert "alg1" in algorithms
        assert "alg2" in algorithms
    
    def test_list_versions(self):
        """Test listing algorithm versions."""
        registry = AlgorithmRegistry()
        registry.register(TestAlgorithm, name="test_alg", version="1.0.0")
        registry.register(TestAlgorithm, name="test_alg", version="2.0.0")
        
        versions = registry.list_versions("test_alg")
        assert "1.0.0" in versions
        assert "2.0.0" in versions


class TestRegisterDecorator:
    """Tests for register_algorithm decorator."""
    
    def test_register_decorator(self):
        """Test using register decorator."""
        registry = AlgorithmRegistry()
        
        @register_algorithm(name="decorated_alg", version="1.0.0", registry=registry)
        class DecoratedAlgorithm(Algorithm):
            def process(self, data, **kwargs):
                return {"algorithm_name": self.name, "results": [], "confidence_scores": [], "metadata": {}, "timestamp": None}
            
            def get_metadata(self):
                return {"name": self.name, "version": self.version, "description": "Test", "author": "Test", "parameters": {}, "requirements": self.get_requirements()}
            
            def get_requirements(self):
                return {"min_data_points": 1, "required_columns": [], "data_frequency": "any", "preprocessing": []}
        
        assert "decorated_alg" in registry.list_algorithms()


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_algorithm_function(self):
        """Test get_algorithm convenience function."""
        from chartradar.metrics.registry import _default_registry
        _default_registry.register(TestAlgorithm, name="convenience_alg")
        
        alg_class = get_algorithm("convenience_alg")
        assert alg_class == TestAlgorithm
    
    def test_create_algorithm_function(self):
        """Test create_algorithm convenience function."""
        from chartradar.metrics.registry import _default_registry
        _default_registry.register(TestAlgorithm, name="convenience_alg2")
        
        alg = create_algorithm("convenience_alg2")
        assert isinstance(alg, TestAlgorithm)

