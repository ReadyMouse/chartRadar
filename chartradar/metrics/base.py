"""
Base algorithm interface for the ChartRadar framework.

This module provides the abstract base class for all pattern detection
and analysis algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
import pandas as pd

from chartradar.src.interfaces import AlgorithmBase
from chartradar.src.types import AlgorithmResult, PatternDetection
from chartradar.src.exceptions import AlgorithmError


class Algorithm(AlgorithmBase):
    """
    Abstract base class for pattern detection and analysis algorithms.
    
    All algorithm implementations must inherit from this class and
    implement the required methods.
    """
    
    def __init__(self, name: str, version: str = "1.0.0", **kwargs: Any):
        """
        Initialize the algorithm.
        
        Args:
            name: Algorithm name
            version: Algorithm version
            **kwargs: Algorithm-specific parameters
        """
        self.name = name
        self.version = version
        self.parameters = kwargs
    
    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """
        Process data and return analysis results.
        
        Args:
            data: DataFrame with OHLCV columns and datetime index
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Dictionary containing:
            - algorithm_name: Name of the algorithm
            - results: List of detected patterns/analysis results
            - confidence_scores: Confidence scores for each result
            - metadata: Additional algorithm-specific metadata
            - timestamp: Processing timestamp
            
        Raises:
            AlgorithmError: If processing fails
        """
        pass
    
    def process_to_result(self, data: pd.DataFrame, **kwargs: Any) -> AlgorithmResult:
        """
        Process data and return standardized AlgorithmResult.
        
        This is a convenience method that wraps process() and converts
        the output to AlgorithmResult format.
        
        Args:
            data: DataFrame with OHLCV columns and datetime index
            **kwargs: Algorithm-specific parameters
            
        Returns:
            AlgorithmResult object
        """
        start_time = datetime.now()
        
        try:
            result_dict = self.process(data, **kwargs)
            
            # Convert to AlgorithmResult
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Extract pattern detections if present
            pattern_detections = []
            if 'results' in result_dict:
                for result in result_dict['results']:
                    if isinstance(result, PatternDetection):
                        pattern_detections.append(result)
                    elif isinstance(result, dict):
                        # Convert dict to PatternDetection
                        pattern_detections.append(PatternDetection(**result))
            
            return AlgorithmResult(
                algorithm_name=self.name,
                algorithm_version=self.version,
                timestamp=result_dict.get('timestamp', datetime.now()),
                results=pattern_detections,
                confidence_scores=result_dict.get('confidence_scores', []),
                metadata=result_dict.get('metadata', {}),
                processing_time_ms=processing_time,
                success=True
            )
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return AlgorithmResult(
                algorithm_name=self.name,
                algorithm_version=self.version,
                timestamp=datetime.now(),
                results=[],
                confidence_scores=[],
                metadata={},
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e)
            )
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this algorithm.
        
        Returns:
            Dictionary containing:
            - name: Algorithm name
            - version: Algorithm version
            - description: Human-readable description
            - author: Algorithm author/contributor
            - parameters: Available configuration parameters with defaults
            - requirements: Data requirements (min_data_points, required_columns, etc.)
        """
        pass
    
    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """
        Get data and processing requirements for this algorithm.
        
        Returns:
            Dictionary containing:
            - min_data_points: Minimum number of data points required
            - required_columns: List of required DataFrame columns
            - data_frequency: Required data frequency (e.g., '1h', '1d')
            - preprocessing: Any required preprocessing steps
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that input data meets algorithm requirements.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid
            
        Raises:
            AlgorithmError: If data doesn't meet requirements
        """
        requirements = self.get_requirements()
        
        # Check minimum data points
        min_points = requirements.get('min_data_points', 0)
        if len(data) < min_points:
            raise AlgorithmError(
                f"Algorithm '{self.name}' requires at least {min_points} data points, got {len(data)}",
                details={
                    "algorithm": self.name,
                    "required": min_points,
                    "provided": len(data)
                }
            )
        
        # Check required columns
        required_columns = requirements.get('required_columns', [])
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise AlgorithmError(
                f"Algorithm '{self.name}' requires columns: {missing_columns}",
                details={
                    "algorithm": self.name,
                    "required": required_columns,
                    "missing": missing_columns,
                    "available": list(data.columns)
                }
            )
        
        return True

