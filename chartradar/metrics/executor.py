"""
Algorithm execution engine for the ChartRadar framework.

This module provides functionality to execute multiple algorithms on data,
handle errors gracefully, and collect standardized outputs.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import concurrent.futures
import logging
import time

import pandas as pd

from chartradar.metrics.base import Algorithm
from chartradar.metrics.registry import AlgorithmRegistry
from chartradar.src.types import AlgorithmResult
from chartradar.src.exceptions import AlgorithmError

logger = logging.getLogger(__name__)


class AlgorithmExecutor:
    """
    Executor for running multiple algorithms on data.
    
    Provides functionality to load algorithms from registry, execute them
    on data (with optional parallel execution), and collect results.
    """
    
    def __init__(
        self,
        registry: Optional[AlgorithmRegistry] = None,
        parallel: bool = False,
        max_workers: Optional[int] = None
    ):
        """
        Initialize the algorithm executor.
        
        Args:
            registry: Algorithm registry to use (creates new if None)
            parallel: Whether to execute algorithms in parallel
            max_workers: Maximum number of parallel workers
        """
        self.registry = registry or AlgorithmRegistry()
        self.parallel = parallel
        self.max_workers = max_workers
    
    def execute(
        self,
        data: pd.DataFrame,
        algorithm_configs: List[Dict[str, Any]],
        **kwargs: Any
    ) -> List[AlgorithmResult]:
        """
        Execute multiple algorithms on data.
        
        Args:
            data: DataFrame with OHLCV columns and datetime index
            algorithm_configs: List of algorithm configurations, each containing:
                - name: Algorithm name
                - version: Optional algorithm version
                - parameters: Algorithm-specific parameters
                - enabled: Whether algorithm is enabled (default: True)
            **kwargs: Additional parameters to pass to all algorithms
            
        Returns:
            List of AlgorithmResult objects
        """
        results: List[AlgorithmResult] = []
        
        # Filter enabled algorithms
        enabled_configs = [
            config for config in algorithm_configs
            if config.get('enabled', True)
        ]
        
        if not enabled_configs:
            logger.warning("No enabled algorithms to execute")
            return results
        
        if self.parallel and len(enabled_configs) > 1:
            results = self._execute_parallel(data, enabled_configs, **kwargs)
        else:
            results = self._execute_sequential(data, enabled_configs, **kwargs)
        
        return results
    
    def _execute_sequential(
        self,
        data: pd.DataFrame,
        algorithm_configs: List[Dict[str, Any]],
        **kwargs: Any
    ) -> List[AlgorithmResult]:
        """Execute algorithms sequentially."""
        results: List[AlgorithmResult] = []
        
        for config in algorithm_configs:
            try:
                result = self._execute_single(data, config, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Algorithm '{config.get('name')}' failed: {str(e)}",
                    exc_info=True
                )
                # Create error result
                error_result = AlgorithmResult(
                    algorithm_name=config.get('name', 'unknown'),
                    algorithm_version=config.get('version'),
                    timestamp=datetime.now(),
                    results=[],
                    confidence_scores=[],
                    metadata={},
                    processing_time_ms=0.0,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    def _execute_parallel(
        self,
        data: pd.DataFrame,
        algorithm_configs: List[Dict[str, Any]],
        **kwargs: Any
    ) -> List[AlgorithmResult]:
        """Execute algorithms in parallel."""
        results: List[AlgorithmResult] = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(self._execute_single, data, config, **kwargs): config
                for config in algorithm_configs
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Algorithm '{config.get('name')}' failed: {str(e)}",
                        exc_info=True
                    )
                    # Create error result
                    error_result = AlgorithmResult(
                        algorithm_name=config.get('name', 'unknown'),
                        algorithm_version=config.get('version'),
                        timestamp=datetime.now(),
                        results=[],
                        confidence_scores=[],
                        metadata={},
                        processing_time_ms=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    results.append(error_result)
        
        return results
    
    def _execute_single(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        **kwargs: Any
    ) -> AlgorithmResult:
        """
        Execute a single algorithm.
        
        Args:
            data: DataFrame with OHLCV columns
            config: Algorithm configuration
            **kwargs: Additional parameters
            
        Returns:
            AlgorithmResult object
        """
        algorithm_name = config.get('name')
        algorithm_version = config.get('version')
        algorithm_params = config.get('parameters', {})
        
        # Merge parameters
        merged_params = {**algorithm_params, **kwargs}
        
        # Get algorithm from registry
        try:
            algorithm = self.registry.create_algorithm(
                name=algorithm_name,
                version=algorithm_version,
                **merged_params
            )
        except Exception as e:
            raise AlgorithmError(
                f"Failed to create algorithm '{algorithm_name}': {str(e)}",
                details={"name": algorithm_name, "version": algorithm_version, "error": str(e)}
            ) from e
        
        # Validate data
        try:
            algorithm.validate_data(data)
        except AlgorithmError:
            raise
        except Exception as e:
            raise AlgorithmError(
                f"Data validation failed for algorithm '{algorithm_name}': {str(e)}",
                details={"name": algorithm_name, "error": str(e)}
            ) from e
        
        # Execute algorithm
        start_time = time.time()
        try:
            result = algorithm.process_to_result(data, **merged_params)
            execution_time = (time.time() - start_time) * 1000
            
            # Update processing time if not set
            if result.processing_time_ms is None:
                result.processing_time_ms = execution_time
            
            logger.info(
                f"Algorithm '{algorithm_name}' completed in {execution_time:.2f}ms "
                f"(success: {result.success}, patterns: {len(result.results)})"
            )
            
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                f"Algorithm '{algorithm_name}' failed after {execution_time:.2f}ms: {str(e)}",
                exc_info=True
            )
            raise AlgorithmError(
                f"Algorithm execution failed: {str(e)}",
                details={"name": algorithm_name, "error": str(e)}
            ) from e
    
    def get_execution_summary(self, results: List[AlgorithmResult]) -> Dict[str, Any]:
        """
        Generate execution summary from results.
        
        Args:
            results: List of AlgorithmResult objects
            
        Returns:
            Dictionary with execution summary
        """
        total_algorithms = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total_algorithms - successful
        
        total_patterns = sum(len(r.results) for r in results)
        total_time = sum(r.processing_time_ms or 0 for r in results)
        
        avg_confidence = 0.0
        confidence_scores = []
        for r in results:
            confidence_scores.extend(r.confidence_scores)
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        return {
            "total_algorithms": total_algorithms,
            "successful": successful,
            "failed": failed,
            "total_patterns_detected": total_patterns,
            "total_processing_time_ms": total_time,
            "average_processing_time_ms": total_time / total_algorithms if total_algorithms > 0 else 0,
            "average_confidence": avg_confidence,
            "timestamp": datetime.now().isoformat()
        }

