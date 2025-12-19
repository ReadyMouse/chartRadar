"""
Batch data source base implementation.

This module provides a base class for batch data sources with common
functionality like chunked loading, date range filtering, and pagination.
"""

from typing import Any, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta

from chartradar.ingestion.base import DataSource
from chartradar.core.exceptions import DataSourceError


class BatchDataSource(DataSource):
    """
    Base class for batch data sources.
    
    Provides common functionality for loading data in batches with
    date range filtering and pagination support.
    """
    
    def __init__(
        self,
        name: str,
        chunk_size: Optional[int] = None,
        **kwargs: Any
    ):
        """
        Initialize the batch data source.
        
        Args:
            name: Name identifier for this data source
            chunk_size: Optional chunk size for loading large datasets
            **kwargs: Additional source-specific parameters
        """
        super().__init__(name, **kwargs)
        self.chunk_size = chunk_size
    
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
            DataFrame with OHLCV columns and datetime index
        """
        # Parse date strings
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else None
        
        # Load data using the implementation-specific method
        data = self._load_raw_data(start_dt, end_dt, **kwargs)
        
        # Apply date range filtering if needed
        if start_dt is not None or end_dt is not None:
            data = self._filter_date_range(data, start_dt, end_dt)
        
        # Validate data
        self.validate_data(data)
        
        return data
    
    def load_data_chunked(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        chunk_size: Optional[int] = None,
        **kwargs: Any
    ):
        """
        Load data in chunks (generator).
        
        Args:
            start_date: Start date for data range
            end_date: End date for data range
            chunk_size: Size of each chunk (uses instance default if None)
            **kwargs: Additional parameters
            
        Yields:
            DataFrame chunks with OHLCV data
        """
        chunk_size = chunk_size or self.chunk_size
        
        if chunk_size is None:
            # If no chunk size specified, load all data at once
            yield self.load_data(start_date, end_date, **kwargs)
            return
        
        # Parse dates
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else None
        
        # Calculate total date range
        if start_dt and end_dt:
            total_days = (end_dt - start_dt).days
            chunk_days = max(1, total_days // max(1, (total_days // chunk_size) + 1))
        else:
            chunk_days = chunk_size  # Use chunk_size as days if no date range
        
        # Load data in chunks
        current_start = start_dt
        while True:
            if end_dt and current_start and current_start >= end_dt:
                break
            
            chunk_end = None
            if current_start:
                chunk_end = current_start + timedelta(days=chunk_days)
                if end_dt:
                    chunk_end = min(chunk_end, end_dt)
            
            chunk_start_str = current_start.isoformat() if current_start else None
            chunk_end_str = chunk_end.isoformat() if chunk_end else None
            
            try:
                chunk = self.load_data(chunk_start_str, chunk_end_str, **kwargs)
                if len(chunk) == 0:
                    break
                yield chunk
                
                if chunk_end:
                    current_start = chunk_end
                else:
                    break
            except DataSourceError:
                break
    
    def _load_raw_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load raw data from the source (implementation-specific).
        
        This method must be implemented by subclasses.
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            **kwargs: Additional parameters
            
        Returns:
            Raw DataFrame from source
        """
        raise NotImplementedError("Subclasses must implement _load_raw_data")
    
    def _filter_date_range(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Filter data by date range.
        
        Args:
            data: DataFrame to filter
            start_date: Start datetime (inclusive)
            end_date: End datetime (inclusive)
            
        Returns:
            Filtered DataFrame
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise DataSourceError(
                "Cannot filter by date range: index is not a DatetimeIndex",
                details={"index_type": type(data.index).__name__}
            )
        
        mask = pd.Series(True, index=data.index)
        
        if start_date:
            mask = mask & (data.index >= start_date)
        
        if end_date:
            mask = mask & (data.index <= end_date)
        
        return data[mask]
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this batch data source."""
        return {
            "name": self.name,
            "type": "batch",
            "description": f"Batch data source: {self.name}",
            "capabilities": ["batch"],
            "parameters": {
                "chunk_size": self.chunk_size,
                **self.parameters
            }
        }

