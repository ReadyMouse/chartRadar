"""
Base fusion strategy interface for the ChartRadar framework.

This module provides the abstract base class for all fusion strategies
that combine results from multiple algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from datetime import datetime

from chartradar.core.interfaces import FusionStrategyBase
from chartradar.core.types import AlgorithmResult, FusionResult
from chartradar.core.exceptions import FusionError


class FusionStrategy(FusionStrategyBase):
    """
    Abstract base class for fusion strategies.
    
    All fusion strategy implementations must inherit from this class and
    implement the required methods.
    """
    
    def __init__(self, name: str, version: str = "1.0.0", **kwargs: Any):
        """
        Initialize the fusion strategy.
        
        Args:
            name: Fusion strategy name
            version: Strategy version
            **kwargs: Strategy-specific parameters
        """
        self.name = name
        self.version = version
        self.parameters = kwargs
    
    @abstractmethod
    def fuse(self, results_list: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """
        Fuse results from multiple algorithms.
        
        Args:
            results_list: List of algorithm result dictionaries
            **kwargs: Fusion strategy-specific parameters
            
        Returns:
            Dictionary containing:
            - fused_result: Combined/aggregated result
            - confidence_score: Aggregated confidence score
            - contributing_algorithms: List of algorithm names that contributed
            - fusion_method: Name of the fusion method used
            - metadata: Additional fusion-specific metadata
            - timestamp: Fusion timestamp
            
        Raises:
            FusionError: If fusion fails
        """
        pass
    
    def fuse_to_result(
        self,
        results_list: List[AlgorithmResult],
        **kwargs: Any
    ) -> FusionResult:
        """
        Fuse algorithm results and return standardized FusionResult.
        
        This is a convenience method that wraps fuse() and converts
        the output to FusionResult format.
        
        Args:
            results_list: List of AlgorithmResult objects
            **kwargs: Fusion strategy-specific parameters
            
        Returns:
            FusionResult object
        """
        # Convert AlgorithmResult objects to dictionaries
        results_dicts = []
        for result in results_list:
            if isinstance(result, AlgorithmResult):
                results_dicts.append({
                    "algorithm_name": result.algorithm_name,
                    "algorithm_version": result.algorithm_version,
                    "results": result.results,
                    "confidence_scores": result.confidence_scores,
                    "metadata": result.metadata,
                    "success": result.success,
                    "error_message": result.error_message
                })
            else:
                results_dicts.append(result)
        
        try:
            fused_dict = self.fuse(results_dicts, **kwargs)
            
            # Extract contributing algorithms
            contributing = fused_dict.get('contributing_algorithms', [])
            if not contributing:
                contributing = [r.algorithm_name for r in results_list if isinstance(r, AlgorithmResult) and r.success]
            
            return FusionResult(
                fusion_method=self.name,
                timestamp=fused_dict.get('timestamp', datetime.now()),
                fused_result=fused_dict.get('fused_result', {}),
                confidence_score=fused_dict.get('confidence_score', 0.0),
                contributing_algorithms=contributing,
                individual_results=results_list,
                metadata=fused_dict.get('metadata', {})
            )
        except Exception as e:
            raise FusionError(
                f"Fusion failed: {str(e)}",
                details={"strategy": self.name, "error": str(e)}
            ) from e
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this fusion strategy.
        
        Returns:
            Dictionary containing:
            - name: Fusion strategy name
            - version: Strategy version
            - description: Human-readable description
            - parameters: Available configuration parameters with defaults
            - supported_result_types: Types of results this strategy can fuse
        """
        pass
    
    def validate_results(self, results_list: List[Dict[str, Any]]) -> bool:
        """
        Validate that input results are suitable for fusion.
        
        Args:
            results_list: List of algorithm result dictionaries
            
        Returns:
            True if results are valid
            
        Raises:
            FusionError: If results are invalid
        """
        if not results_list:
            raise FusionError(
                "Cannot fuse empty results list",
                details={"strategy": self.name}
            )
        
        # Check that at least one result is successful
        successful_results = [
            r for r in results_list
            if isinstance(r, dict) and r.get('success', True)
        ]
        
        if not successful_results:
            raise FusionError(
                "No successful algorithm results to fuse",
                details={"strategy": self.name, "total_results": len(results_list)}
            )
        
        return True

