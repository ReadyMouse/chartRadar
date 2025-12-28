"""
Fusion execution engine for the ChartRadar framework.

This module provides functionality to execute fusion strategies on algorithm
results, supporting sequential fusion pipelines.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from chartradar.fusion.base import FusionStrategy
from chartradar.fusion.registry import FusionStrategyRegistry, _default_registry
from chartradar.src.types import AlgorithmResult, FusionResult
from chartradar.src.exceptions import FusionError

logger = logging.getLogger(__name__)


class FusionExecutor:
    """
    Executor for running fusion strategies on algorithm results.
    
    Provides functionality to load strategies from registry, apply them
    to algorithm results, and support sequential fusion pipelines.
    """
    
    def __init__(self, registry: Optional[FusionStrategyRegistry] = None):
        """
        Initialize the fusion executor.
        
        Args:
            registry: Fusion strategy registry to use (uses default global registry if None)
        """
        self.registry = registry or _default_registry
    
    def execute(
        self,
        algorithm_results: List[AlgorithmResult],
        fusion_config: Dict[str, Any],
        **kwargs: Any
    ) -> FusionResult:
        """
        Execute fusion strategy on algorithm results.
        
        Args:
            algorithm_results: List of AlgorithmResult objects
            fusion_config: Fusion configuration containing:
                - strategy: Strategy name
                - parameters: Strategy-specific parameters
                - enabled: Whether fusion is enabled
            **kwargs: Additional parameters
            
        Returns:
            FusionResult object
        """
        if not fusion_config.get('enabled', True):
            logger.warning("Fusion is disabled, returning empty result")
            return FusionResult(
                fusion_method="none",
                timestamp=datetime.now(),
                fused_result={},
                confidence_score=0.0,
                contributing_algorithms=[],
                individual_results=algorithm_results,
                metadata={"enabled": False}
            )
        
        strategy_name = fusion_config.get('strategy')
        if not strategy_name or strategy_name == 'none':
            logger.warning("No fusion strategy specified")
            return FusionResult(
                fusion_method="none",
                timestamp=datetime.now(),
                fused_result={},
                confidence_score=0.0,
                contributing_algorithms=[r.algorithm_name for r in algorithm_results if r.success],
                individual_results=algorithm_results,
                metadata={"strategy": "none"}
            )
        
        # Get strategy parameters
        strategy_params = fusion_config.get('parameters', {})
        merged_params = {**strategy_params, **kwargs}
        
        # Create strategy instance
        try:
            strategy = self.registry.create_strategy(
                name=strategy_name,
                version=fusion_config.get('version'),
                **merged_params
            )
        except Exception as e:
            raise FusionError(
                f"Failed to create fusion strategy '{strategy_name}': {str(e)}",
                details={"strategy": strategy_name, "error": str(e)}
            ) from e
        
        # Execute fusion
        try:
            result = strategy.fuse_to_result(algorithm_results, **merged_params)
            logger.info(
                f"Fusion '{strategy_name}' completed: "
                f"confidence={result.confidence_score:.3f}, "
                f"contributors={len(result.contributing_algorithms)}"
            )
            return result
        except Exception as e:
            raise FusionError(
                f"Fusion execution failed: {str(e)}",
                details={"strategy": strategy_name, "error": str(e)}
            ) from e
    
    def execute_sequential(
        self,
        algorithm_results: List[AlgorithmResult],
        fusion_configs: List[Dict[str, Any]],
        **kwargs: Any
    ) -> FusionResult:
        """
        Execute a sequence of fusion strategies in a pipeline.
        
        The output of each fusion becomes input to the next.
        
        Args:
            algorithm_results: Initial algorithm results
            fusion_configs: List of fusion configurations (executed in order)
            **kwargs: Additional parameters
            
        Returns:
            Final FusionResult from the pipeline
        """
        current_results = algorithm_results
        
        for i, fusion_config in enumerate(fusion_configs):
            logger.info(f"Executing fusion step {i+1}/{len(fusion_configs)}: {fusion_config.get('strategy')}")
            
            # Execute fusion
            fusion_result = self.execute(current_results, fusion_config, **kwargs)
            
            # Convert FusionResult back to AlgorithmResult for next step
            # (if needed - for now, we'll use the fusion result as-is)
            if i < len(fusion_configs) - 1:
                # Create a pseudo AlgorithmResult from the fusion result
                # for the next fusion step
                pseudo_result = AlgorithmResult(
                    algorithm_name=f"fusion_step_{i}",
                    algorithm_version="1.0.0",
                    timestamp=fusion_result.timestamp,
                    results=fusion_result.fused_result.get('patterns', []),
                    confidence_scores=[fusion_result.confidence_score],
                    metadata={
                        "fusion_method": fusion_result.fusion_method,
                        "step": i,
                        **fusion_result.metadata
                    },
                    processing_time_ms=0.0,
                    success=True
                )
                current_results = [pseudo_result]
            else:
                # Last step, return the fusion result
                return fusion_result
        
        # Should not reach here, but return last result if we do
        return fusion_result
    
    def get_execution_summary(self, result: FusionResult) -> Dict[str, Any]:
        """
        Generate execution summary from fusion result.
        
        Args:
            result: FusionResult object
            
        Returns:
            Dictionary with execution summary
        """
        return {
            "fusion_method": result.fusion_method,
            "confidence_score": result.confidence_score,
            "contributing_algorithms": len(result.contributing_algorithms),
            "total_patterns": len(result.fused_result.get('patterns', [])),
            "timestamp": result.timestamp.isoformat(),
            "metadata": result.metadata
        }

