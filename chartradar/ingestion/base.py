"""
Base data source interface for the ChartRadar framework.

This module provides the abstract base class for all data sources.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

from chartradar.core.interfaces import DataSourceBase
from chartradar.core.exceptions import DataSourceError


class DataSource(DataSourceBase):
    """
    Abstract base class for data sources.
    
    All data source implementations must inherit from this class and
    implement the required methods.
    """
    
    def __init__(self, name: str, **kwargs: Any):
        """
        Initialize the data source.
        
        Args:
            name: Name identifier for this data source
            **kwargs: Additional source-specific parameters
        """
        self.name = name
        self.parameters = kwargs
    
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
        pass
    
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
        pass
    
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
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has the required structure.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid
            
        Raises:
            DataSourceError: If data is invalid
        """
        # Check for empty data first
        if len(data) == 0:
            raise DataSourceError(
                "Data is empty",
                details={}
            )
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise DataSourceError(
                f"Data missing required columns: {missing_columns}",
                details={"required_columns": required_columns, "missing": missing_columns}
            )
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise DataSourceError(
                "Data index must be a DatetimeIndex",
                details={"index_type": type(data.index).__name__}
            )
        
        return True

