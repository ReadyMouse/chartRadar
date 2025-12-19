"""Tests for fusion strategy implementations."""

import pytest
from datetime import datetime

from chartradar.fusion.strategies.weighted_average import WeightedAverageFusion
from chartradar.fusion.strategies.voting import VotingFusion
from chartradar.fusion.strategies.stacking import StackingFusion
from chartradar.core.types import PatternDetection


class TestWeightedAverageFusion:
    """Tests for WeightedAverageFusion."""
    
    def test_fuse_basic(self):
        """Test basic weighted average fusion."""
        strategy = WeightedAverageFusion()
        
        results = [
            {
                "algorithm_name": "alg1",
                "success": True,
                "confidence_scores": [0.8, 0.9],
                "results": []
            },
            {
                "algorithm_name": "alg2",
                "success": True,
                "confidence_scores": [0.7],
                "results": []
            }
        ]
        
        fused = strategy.fuse(results)
        assert "fused_result" in fused
        assert "confidence_score" in fused
        assert fused["confidence_score"] > 0
    
    def test_fuse_with_weights(self):
        """Test fusion with custom weights."""
        strategy = WeightedAverageFusion()
        
        results = [
            {
                "algorithm_name": "alg1",
                "success": True,
                "confidence_scores": [0.8],
                "results": []
            },
            {
                "algorithm_name": "alg2",
                "success": True,
                "confidence_scores": [0.9],
                "results": []
            }
        ]
        
        fused = strategy.fuse(results, weights={"alg1": 0.8, "alg2": 0.2})
        assert fused["confidence_score"] > 0
    
    def test_fuse_normalize_weights(self):
        """Test fusion with weight normalization."""
        strategy = WeightedAverageFusion(normalize_weights=True)
        
        results = [
            {
                "algorithm_name": "alg1",
                "success": True,
                "confidence_scores": [0.8],
                "results": []
            }
        ]
        
        fused = strategy.fuse(results, weights={"alg1": 2.0})
        # Weights should be normalized
        assert fused["metadata"]["weights_used"]["alg1"] <= 1.0


class TestVotingFusion:
    """Tests for VotingFusion."""
    
    def test_fuse_majority_voting(self):
        """Test majority voting fusion."""
        strategy = VotingFusion(use_weights=False)
        
        results = [
            {
                "algorithm_name": "alg1",
                "success": True,
                "results": [
                    {"pattern_type": "rising_wedge", "confidence": 0.8}
                ]
            },
            {
                "algorithm_name": "alg2",
                "success": True,
                "results": [
                    {"pattern_type": "rising_wedge", "confidence": 0.9}
                ]
            },
            {
                "algorithm_name": "alg3",
                "success": True,
                "results": [
                    {"pattern_type": "falling_wedge", "confidence": 0.7}
                ]
            }
        ]
        
        fused = strategy.fuse(results)
        assert "fused_result" in fused
        assert "prediction" in fused["fused_result"]
        # Should predict rising_wedge (majority)
        assert fused["fused_result"]["prediction"] == "rising_wedge"
    
    def test_fuse_weighted_voting(self):
        """Test weighted voting fusion."""
        strategy = VotingFusion(use_weights=True)
        
        results = [
            {
                "algorithm_name": "alg1",
                "success": True,
                "results": [
                    {"pattern_type": "pattern1", "confidence": 0.8}
                ]
            },
            {
                "algorithm_name": "alg2",
                "success": True,
                "results": [
                    {"pattern_type": "pattern2", "confidence": 0.9}
                ]
            }
        ]
        
        fused = strategy.fuse(results, weights={"alg1": 0.8, "alg2": 0.2})
        assert "prediction" in fused["fused_result"]


class TestStackingFusion:
    """Tests for StackingFusion."""
    
    def test_fuse_basic(self):
        """Test basic stacking fusion."""
        strategy = StackingFusion()
        
        results = [
            {
                "algorithm_name": "alg1",
                "success": True,
                "confidence_scores": [0.8],
                "results": []
            },
            {
                "algorithm_name": "alg2",
                "success": True,
                "confidence_scores": [0.9],
                "results": []
            }
        ]
        
        fused = strategy.fuse(results)
        assert "fused_result" in fused
        assert "confidence_score" in fused
        assert fused["confidence_score"] > 0
    
    def test_fuse_different_meta_learners(self):
        """Test different meta-learner types."""
        for meta_learner in ["average", "weighted_average", "max"]:
            strategy = StackingFusion(meta_learner=meta_learner)
            
            results = [
                {
                    "algorithm_name": "alg1",
                    "success": True,
                    "confidence_scores": [0.8],
                    "results": []
                }
            ]
            
            fused = strategy.fuse(results)
            assert fused["confidence_score"] >= 0

