"""Tests for fusion base class."""

import pytest
from datetime import datetime

from chartradar.fusion.base import FusionStrategy
from chartradar.src.types import AlgorithmResult, PatternDetection
from chartradar.src.exceptions import FusionError


class ConcreteFusionStrategy(FusionStrategy):
    """Concrete fusion strategy for testing."""
    
    def fuse(self, results_list, **kwargs):
        """Simple fusion that combines results."""
        self.validate_results(results_list)
        
        successful = [r for r in results_list if isinstance(r, dict) and r.get('success', True)]
        
        return {
            "fused_result": {"combined": True},
            "confidence_score": 0.8,
            "contributing_algorithms": [r.get('algorithm_name', 'unknown') for r in successful],
            "fusion_method": self.name,
            "metadata": {},
            "timestamp": datetime.now()
        }
    
    def get_metadata(self):
        """Get test metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Test fusion strategy",
            "parameters": {},
            "supported_result_types": ["test"]
        }


class TestFusionStrategy:
    """Tests for FusionStrategy base class."""
    
    def test_initialization(self):
        """Test fusion strategy initialization."""
        strategy = ConcreteFusionStrategy("test_strategy", version="1.0.0", param1="value1")
        assert strategy.name == "test_strategy"
        assert strategy.version == "1.0.0"
        assert strategy.parameters["param1"] == "value1"
    
    def test_fuse(self):
        """Test fusing results."""
        strategy = ConcreteFusionStrategy("test")
        results = [
            {"algorithm_name": "alg1", "success": True, "confidence_scores": [0.8]},
            {"algorithm_name": "alg2", "success": True, "confidence_scores": [0.9]}
        ]
        
        fused = strategy.fuse(results)
        assert fused["confidence_score"] == 0.8
        assert len(fused["contributing_algorithms"]) == 2
    
    def test_validate_results_empty(self):
        """Test validation with empty results."""
        strategy = ConcreteFusionStrategy("test")
        
        with pytest.raises(FusionError) as exc_info:
            strategy.validate_results([])
        assert "empty" in str(exc_info.value).lower()
    
    def test_validate_results_no_successful(self):
        """Test validation with no successful results."""
        strategy = ConcreteFusionStrategy("test")
        results = [
            {"algorithm_name": "alg1", "success": False}
        ]
        
        with pytest.raises(FusionError) as exc_info:
            strategy.validate_results(results)
        assert "no successful" in str(exc_info.value).lower()
    
    def test_fuse_to_result(self):
        """Test fuse_to_result method."""
        from chartradar.src.types import FusionResult
        strategy = ConcreteFusionStrategy("test")
        results = [
            AlgorithmResult(
                algorithm_name="alg1",
                success=True,
                confidence_scores=[0.8],
                results=[],
                timestamp=datetime.now()
            )
        ]
        
        fused_result = strategy.fuse_to_result(results)
        assert isinstance(fused_result, FusionResult)
        assert fused_result.fusion_method == "test"

