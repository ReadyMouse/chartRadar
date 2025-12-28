"""
CSV file data source implementation.

This module provides a data source for loading OHLCV data from CSV files.
"""

from typing import Any, Dict, Optional
import pandas as pd
from pathlib import Path

from chartradar.ingestion.batch import BatchDataSource
from chartradar.ingestion.normalizer import normalize_dataframe
from chartradar.src.exceptions import DataSourceError


class CSVDataSource(BatchDataSource):
    """
    Data source for loading OHLCV data from CSV files.
    
    Supports various CSV formats and handles large files with chunking.
    """
    
    def __init__(
        self,
        name: str,
        path: str,
        date_column: Optional[str] = None,
        column_mapping: Optional[Dict[str, Any]] = None,
        timezone: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize the CSV data source.
        
        Args:
            name: Data source name
            path: Path to CSV file
            date_column: Name of the date/timestamp column
            column_mapping: Custom column name mapping
            timezone: Timezone to apply
            **kwargs: Additional parameters (chunk_size, etc.)
        """
        super().__init__(name, **kwargs)
        self.path = Path(path)
        self.date_column = date_column
        self.column_mapping = column_mapping
        self.timezone = timezone
        
        # Check if file exists (but allow lazy loading for streaming scenarios)
        if not self.path.exists():
            raise DataSourceError(
                f"CSV file not found: {path}",
                details={"path": str(path), "absolute_path": str(self.path.absolute())}
            )
    
    def _load_raw_data(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load raw data from CSV file.
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            **kwargs: Additional parameters
            
        Returns:
            Raw DataFrame from CSV
        """
        try:
            # Try to load with chunking for large files
            chunk_size = kwargs.get('chunk_size', self.chunk_size)
            
            if chunk_size:
                # Load in chunks
                chunks = []
                for chunk in pd.read_csv(
                    self.path,
                    chunksize=chunk_size,
                    parse_dates=[self.date_column] if self.date_column else False,
                    **kwargs
                ):
                    chunks.append(chunk)
                
                if chunks:
                    data = pd.concat(chunks, ignore_index=True)
                else:
                    data = pd.DataFrame()
            else:
                # Load all at once
                data = pd.read_csv(
                    self.path,
                    parse_dates=[self.date_column] if self.date_column else False,
                    **kwargs
                )
            
            # Normalize the data
            data = normalize_dataframe(
                data,
                date_column=self.date_column,
                column_mapping=self.column_mapping,
                timezone=self.timezone
            )
            
            return data
            
        except Exception as e:
            raise DataSourceError(
                f"Failed to load CSV file: {str(e)}",
                details={"path": str(self.path), "error": str(e)}
            ) from e
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this CSV data source."""
        file_size = self.path.stat().st_size if self.path.exists() else 0
        
        return {
            "name": self.name,
            "type": "csv",
            "description": f"CSV file data source: {self.path.name}",
            "capabilities": ["batch"],
            "parameters": {
                "path": str(self.path),
                "date_column": self.date_column,
                "file_size_bytes": file_size,
                "chunk_size": self.chunk_size,
                **self.parameters
            }
        }

