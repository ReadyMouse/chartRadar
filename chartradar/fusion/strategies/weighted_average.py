"""
Weighted average fusion strategy.

This module implements a fusion strategy that combines algorithm outputs
using weighted averaging of confidence scores and predictions.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

from chartradar.fusion.base import FusionStrategy
from chartradar.fusion.registry import register_fusion_strategy
from chartradar.core.exceptions import FusionError


@register_fusion_strategy(name="weighted_average", version="1.0.0")
class WeightedAverageFusion(FusionStrategy):
    """
    Fusion strategy that combines algorithm results using weighted averaging.
    
    Weights can be specified per algorithm, and confidence scores are
    aggregated using weighted average.
    """
    
    def __init__(
        self,
        name: str = "weighted_average",
        version: str = "1.0.0",
        normalize_weights: bool = True,
        default_weight: float = 1.0,
        **kwargs: Any
    ):
        """
        Initialize the weighted average fusion strategy.
        
        Args:
            name: Strategy name
            version: Strategy version
            normalize_weights: Whether to normalize weights to sum to 1.0
            default_weight: Default weight for algorithms without specified weights
            **kwargs: Additional parameters
        """
        super().__init__(name, version, **kwargs)
        self.normalize_weights = normalize_weights
        self.default_weight = default_weight
    
    def fuse(self, results_list: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """
        Fuse algorithm results using weighted averaging.
        
        Args:
            results_list: List of algorithm result dictionaries
            **kwargs: Additional parameters (can include 'weights' dict)
            
        Returns:
            Fused result dictionary
        """
        self.validate_results(results_list)
        
        # Get parameters
        normalize_weights = kwargs.get('normalize_weights', self.normalize_weights)
        default_weight = kwargs.get('default_weight', self.default_weight)
        weights = kwargs.get('weights', {})
        
        # Filter successful results
        successful_results = [
            r for r in results_list
            if isinstance(r, dict) and r.get('success', True)
        ]
        
        if not successful_results:
            raise FusionError(
                "No successful results to fuse",
                details={"strategy": self.name, "total_results": len(results_list)}
            )
        
        # Get weights for each algorithm
        algorithm_weights = {}
        for result in successful_results:
            alg_name = result.get('algorithm_name', 'unknown')
            if alg_name in weights:
                algorithm_weights[alg_name] = weights[alg_name]
            else:
                # Try to get weight from result metadata
                alg_weight = result.get('metadata', {}).get('weight', default_weight)
                algorithm_weights[alg_name] = alg_weight
        
        # Normalize weights if requested
        if normalize_weights:
            total_weight = sum(algorithm_weights.values())
            if total_weight > 0:
                algorithm_weights = {
                    name: weight / total_weight
                    for name, weight in algorithm_weights.items()
                }
        
        # Aggregate confidence scores
        confidence_scores = []
        weighted_confidences = []
        
        for result in successful_results:
            alg_name = result.get('algorithm_name', 'unknown')
            weight = algorithm_weights.get(alg_name, default_weight)
            
            # Get confidence scores from result
            result_confidences = result.get('confidence_scores', [])
            if result_confidences:
                avg_confidence = np.mean(result_confidences)
                confidence_scores.append(avg_confidence)
                weighted_confidences.append(avg_confidence * weight)
            else:
                # If no confidence scores, use default
                confidence_scores.append(0.5)
                weighted_confidences.append(0.5 * weight)
        
        # Calculate weighted average confidence
        if weighted_confidences:
            fused_confidence = sum(weighted_confidences)
        else:
            fused_confidence = 0.0
        
        # Aggregate pattern detections (if present)
        all_patterns = []
        for result in successful_results:
            patterns = result.get('results', [])
            if patterns:
                all_patterns.extend(patterns)
        
        # Create fused result
        contributing_algorithms = [r.get('algorithm_name', 'unknown') for r in successful_results]
        
        return {
            "fused_result": {
                "patterns": all_patterns,
                "total_patterns": len(all_patterns),
                "weighted_confidence": fused_confidence,
                "algorithm_weights": algorithm_weights
            },
            "confidence_score": min(1.0, max(0.0, fused_confidence)),
            "contributing_algorithms": contributing_algorithms,
            "fusion_method": self.name,
            "metadata": {
                "normalize_weights": normalize_weights,
                "total_algorithms": len(successful_results),
                "weights_used": algorithm_weights
            },
            "timestamp": datetime.now()
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get strategy metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Combines algorithm results using weighted averaging of confidence scores",
            "parameters": {
                "normalize_weights": self.normalize_weights,
                "default_weight": self.default_weight
            },
            "supported_result_types": ["pattern_detection", "confidence_scores"]
        }

