"""
Core interfaces for the ChartRadar framework.

This module defines the abstract base classes that all components must implement:
- DataSource: For data ingestion from various sources
- Algorithm: For pattern detection and analysis algorithms
- FusionStrategy: For combining results from multiple algorithms
- Display: For visualization and export of results
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class DataSource(Protocol):
    """
    Protocol for data sources that can provide OHLCV data.
    
    Data sources can implement either batch loading, streaming, or both.
    """
    
    @abstractmethod
    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load historical data in batch mode.
        
        Args:
            start_date: Start date for data range (ISO format string or None)
            end_date: End date for data range (ISO format string or None)
            **kwargs: Additional source-specific parameters
            
        Returns:
            DataFrame with OHLCV columns (open, high, low, close, volume)
            and datetime index
            
        Raises:
            DataSourceError: If data cannot be loaded
        """
        ...
    
    @abstractmethod
    async def stream_data(
        self,
        callback: callable,
        **kwargs: Any
    ) -> None:
        """
        Stream real-time data.
        
        Args:
            callback: Async function to call with each new data point
            **kwargs: Additional source-specific parameters
            
        Raises:
            DataSourceError: If streaming cannot be started
        """
        ...
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this data source.
        
        Returns:
            Dictionary containing:
            - name: Source name
            - type: Source type (e.g., 'freqtrade', 'csv', 'exchange')
            - description: Human-readable description
            - capabilities: List of supported capabilities (e.g., ['batch', 'streaming'])
            - parameters: Available configuration parameters
        """
        ...


@runtime_checkable
class Algorithm(Protocol):
    """
    Protocol for pattern detection and analysis algorithms.
    
    Algorithms process OHLCV data and return standardized results.
    """
    
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
        ...
    
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
        ...
    
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
        ...


@runtime_checkable
class FusionStrategy(Protocol):
    """
    Protocol for fusion strategies that combine results from multiple algorithms.
    
    Fusion strategies take outputs from multiple algorithms and produce a unified result.
    """
    
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
        ...
    
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
        ...


@runtime_checkable
class Display(Protocol):
    """
    Protocol for display and export functionality.
    
    Display modules handle visualization and export of analysis results.
    """
    
    @abstractmethod
    def visualize(
        self,
        results: Dict[str, Any],
        data: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> Any:
        """
        Visualize analysis results.
        
        Args:
            results: Results dictionary from algorithms or fusion
            data: Optional original OHLCV data for context
            **kwargs: Visualization-specific parameters
            
        Returns:
            Visualization object (matplotlib figure, plotly figure, etc.)
            
        Raises:
            DisplayError: If visualization fails
        """
        ...
    
    @abstractmethod
    def export(
        self,
        results: Dict[str, Any],
        format: str,
        output_path: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Export results to a file.
        
        Args:
            results: Results dictionary to export
            format: Export format ('json', 'csv', 'png', 'svg', etc.)
            output_path: Optional path for output file (if None, returns data as string)
            **kwargs: Export-specific parameters
            
        Returns:
            Path to exported file, or exported data as string if output_path is None
            
        Raises:
            DisplayError: If export fails
        """
        ...


# Abstract base classes for implementations that need to inherit
class DataSourceBase(ABC):
    """Abstract base class for data source implementations."""
    
    @abstractmethod
    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """Load historical data in batch mode."""
        ...
    
    @abstractmethod
    async def stream_data(
        self,
        callback: callable,
        **kwargs: Any
    ) -> None:
        """Stream real-time data."""
        ...
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this data source."""
        ...


class AlgorithmBase(ABC):
    """Abstract base class for algorithm implementations."""
    
    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """Process data and return analysis results."""
        ...
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this algorithm."""
        ...
    
    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """Get data and processing requirements for this algorithm."""
        ...


class FusionStrategyBase(ABC):
    """Abstract base class for fusion strategy implementations."""
    
    @abstractmethod
    def fuse(self, results_list: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """Fuse results from multiple algorithms."""
        ...
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this fusion strategy."""
        ...


class DisplayBase(ABC):
    """Abstract base class for display implementations."""
    
    @abstractmethod
    def visualize(
        self,
        results: Dict[str, Any],
        data: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> Any:
        """Visualize analysis results."""
        ...
    
    @abstractmethod
    def export(
        self,
        results: Dict[str, Any],
        format: str,
        output_path: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Export results to a file."""
        ...

