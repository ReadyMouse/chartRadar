"""Tests for fusion executor."""

import pytest
from datetime import datetime

from chartradar.fusion.executor import FusionExecutor
from chartradar.fusion.registry import FusionStrategyRegistry
from chartradar.core.types import AlgorithmResult
from chartradar.core.exceptions import FusionError


class TestFusionExecutor:
    """Tests for FusionExecutor class."""
    
    def test_execute_basic(self):
        """Test basic fusion execution."""
        executor = FusionExecutor()
        
        algorithm_results = [
            AlgorithmResult(
                algorithm_name="alg1",
                success=True,
                confidence_scores=[0.8],
                results=[],
                timestamp=datetime.now()
            ),
            AlgorithmResult(
                algorithm_name="alg2",
                success=True,
                confidence_scores=[0.9],
                results=[],
                timestamp=datetime.now()
            )
        ]
        
        fusion_config = {
            "strategy": "weighted_average",
            "enabled": True,
            "parameters": {}
        }
        
        result = executor.execute(algorithm_results, fusion_config)
        assert result.fusion_method == "weighted_average"
        assert result.confidence_score > 0
        assert len(result.contributing_algorithms) == 2
    
    def test_execute_disabled(self):
        """Test execution with disabled fusion."""
        executor = FusionExecutor()
        
        algorithm_results = [
            AlgorithmResult(
                algorithm_name="alg1",
                success=True,
                confidence_scores=[0.8],
                results=[],
                timestamp=datetime.now()
            )
        ]
        
        fusion_config = {
            "strategy": "weighted_average",
            "enabled": False
        }
        
        result = executor.execute(algorithm_results, fusion_config)
        assert result.fusion_method == "none"
        assert result.confidence_score == 0.0
    
    def test_execute_sequential(self):
        """Test sequential fusion pipeline."""
        executor = FusionExecutor()
        
        algorithm_results = [
            AlgorithmResult(
                algorithm_name="alg1",
                success=True,
                confidence_scores=[0.8],
                results=[],
                timestamp=datetime.now()
            )
        ]
        
        fusion_configs = [
            {
                "strategy": "weighted_average",
                "enabled": True,
                "parameters": {}
            },
            {
                "strategy": "voting",
                "enabled": True,
                "parameters": {}
            }
        ]
        
        result = executor.execute_sequential(algorithm_results, fusion_configs)
        assert result.fusion_method == "voting"  # Last strategy in pipeline
    
    def test_get_execution_summary(self):
        """Test getting execution summary."""
        executor = FusionExecutor()
        
        from chartradar.core.types import FusionResult
        result = FusionResult(
            fusion_method="weighted_average",
            timestamp=datetime.now(),
            fused_result={"patterns": []},
            confidence_score=0.85,
            contributing_algorithms=["alg1", "alg2"],
            individual_results=[],
            metadata={}
        )
        
        summary = executor.get_execution_summary(result)
        assert summary["fusion_method"] == "weighted_average"
        assert summary["confidence_score"] == 0.85
        assert summary["contributing_algorithms"] == 2

