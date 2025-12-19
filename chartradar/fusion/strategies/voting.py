"""
Voting fusion strategy.

This module implements a fusion strategy that combines algorithm outputs
using majority voting or weighted voting for categorical predictions.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import Counter
import numpy as np

from chartradar.fusion.base import FusionStrategy
from chartradar.fusion.registry import register_fusion_strategy
from chartradar.core.exceptions import FusionError


@register_fusion_strategy(name="voting", version="1.0.0")
class VotingFusion(FusionStrategy):
    """
    Fusion strategy that combines algorithm results using voting.
    
    Supports both simple majority voting and weighted voting.
    """
    
    def __init__(
        self,
        name: str = "voting",
        version: str = "1.0.0",
        use_weights: bool = False,
        threshold: float = 0.5,
        tie_breaker: str = "confidence",
        **kwargs: Any
    ):
        """
        Initialize the voting fusion strategy.
        
        Args:
            name: Strategy name
            version: Strategy version
            use_weights: Whether to use weighted voting
            threshold: Minimum vote threshold for acceptance
            tie_breaker: Method for breaking ties ('confidence', 'first', 'random')
            **kwargs: Additional parameters
        """
        super().__init__(name, version, **kwargs)
        self.use_weights = use_weights
        self.threshold = threshold
        self.tie_breaker = tie_breaker
    
    def fuse(self, results_list: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """
        Fuse algorithm results using voting.
        
        Args:
            results_list: List of algorithm result dictionaries
            **kwargs: Additional parameters (can include 'weights' dict)
            
        Returns:
            Fused result dictionary
        """
        self.validate_results(results_list)
        
        # Get parameters
        use_weights = kwargs.get('use_weights', self.use_weights)
        threshold = kwargs.get('threshold', self.threshold)
        tie_breaker = kwargs.get('tie_breaker', self.tie_breaker)
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
        
        # Collect votes (pattern types or predictions)
        votes = []
        vote_weights = []
        vote_confidences = []
        
        for result in successful_results:
            alg_name = result.get('algorithm_name', 'unknown')
            
            # Get weight for this algorithm
            if use_weights:
                if alg_name in weights:
                    weight = weights[alg_name]
                else:
                    weight = result.get('metadata', {}).get('weight', 1.0)
            else:
                weight = 1.0
            
            # Get predictions/patterns from result
            patterns = result.get('results', [])
            if patterns:
                # Vote for each pattern type
                for pattern in patterns:
                    if isinstance(pattern, dict):
                        pattern_type = pattern.get('pattern_type', 'unknown')
                        confidence = pattern.get('confidence', 0.5)
                    else:
                        # Assume it's a PatternDetection object
                        pattern_type = getattr(pattern, 'pattern_type', 'unknown')
                        confidence = getattr(pattern, 'confidence', 0.5)
                    
                    votes.append(pattern_type)
                    vote_weights.append(weight)
                    vote_confidences.append(confidence)
            else:
                # If no patterns, vote based on predicted direction or metadata
                predicted_direction = result.get('metadata', {}).get('predicted_direction')
                if predicted_direction:
                    votes.append(predicted_direction)
                    vote_weights.append(weight)
                    avg_confidence = np.mean(result.get('confidence_scores', [0.5]))
                    vote_confidences.append(avg_confidence)
        
        if not votes:
            # No votes to count
            return {
                "fused_result": {
                    "prediction": None,
                    "votes": {},
                    "total_votes": 0
                },
                "confidence_score": 0.0,
                "contributing_algorithms": [r.get('algorithm_name', 'unknown') for r in successful_results],
                "fusion_method": self.name,
                "metadata": {
                    "use_weights": use_weights,
                    "threshold": threshold
                },
                "timestamp": datetime.now()
            }
        
        # Count votes
        if use_weights:
            # Weighted voting
            vote_counts = {}
            for vote, weight in zip(votes, vote_weights):
                if vote not in vote_counts:
                    vote_counts[vote] = 0.0
                vote_counts[vote] += weight
        else:
            # Simple majority voting
            vote_counts = dict(Counter(votes))
        
        # Find winner
        if vote_counts:
            max_votes = max(vote_counts.values())
            winners = [vote for vote, count in vote_counts.items() if count == max_votes]
            
            # Handle ties
            if len(winners) > 1:
                if tie_breaker == "confidence":
                    # Break tie using confidence scores
                    winner_confidences = {}
                    for winner in winners:
                        confidences = [
                            conf for vote, conf in zip(votes, vote_confidences)
                            if vote == winner
                        ]
                        winner_confidences[winner] = np.mean(confidences) if confidences else 0.0
                    winner = max(winner_confidences, key=winner_confidences.get)
                elif tie_breaker == "first":
                    winner = winners[0]
                else:  # random
                    import random
                    winner = random.choice(winners)
            else:
                winner = winners[0]
            
            # Calculate confidence based on vote ratio
            total_votes = sum(vote_counts.values())
            winner_votes = vote_counts[winner]
            confidence = winner_votes / total_votes if total_votes > 0 else 0.0
            
            # Check threshold
            if confidence < threshold:
                winner = None
                confidence = 0.0
        else:
            winner = None
            confidence = 0.0
        
        contributing_algorithms = [r.get('algorithm_name', 'unknown') for r in successful_results]
        
        return {
            "fused_result": {
                "prediction": winner,
                "votes": vote_counts,
                "total_votes": sum(vote_counts.values()) if vote_counts else 0,
                "winner_votes": vote_counts.get(winner, 0) if winner else 0
            },
            "confidence_score": min(1.0, max(0.0, confidence)),
            "contributing_algorithms": contributing_algorithms,
            "fusion_method": self.name,
            "metadata": {
                "use_weights": use_weights,
                "threshold": threshold,
                "tie_breaker": tie_breaker,
                "vote_counts": vote_counts
            },
            "timestamp": datetime.now()
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get strategy metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Combines algorithm results using majority or weighted voting",
            "parameters": {
                "use_weights": self.use_weights,
                "threshold": self.threshold,
                "tie_breaker": self.tie_breaker
            },
            "supported_result_types": ["categorical_predictions", "pattern_types"]
        }

