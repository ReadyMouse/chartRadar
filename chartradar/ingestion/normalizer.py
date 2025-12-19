"""
Data normalization utilities for the ChartRadar framework.

This module provides functions to convert various data formats to
standard OHLCV DataFrame format with consistent column naming and types.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
from datetime import datetime
import pytz

from chartradar.core.exceptions import DataValidationError


# Common column name mappings
COLUMN_MAPPINGS = {
    # Standard names
    'open': ['open', 'Open', 'OPEN', 'o', 'O'],
    'high': ['high', 'High', 'HIGH', 'h', 'H'],
    'low': ['low', 'Low', 'LOW', 'l', 'L'],
    'close': ['close', 'Close', 'CLOSE', 'c', 'C'],
    'volume': ['volume', 'Volume', 'VOLUME', 'v', 'V', 'vol', 'Vol'],
    # Alternative names
    'timestamp': ['timestamp', 'Timestamp', 'TIMESTAMP', 'time', 'Time', 'TIME', 'date', 'Date', 'DATE', 'datetime', 'DateTime'],
}


def normalize_dataframe(
    df: pd.DataFrame,
    date_column: Optional[str] = None,
    column_mapping: Optional[Dict[str, Union[str, List[str]]]] = None,
    timezone: Optional[str] = None,
    ensure_timezone: bool = True
) -> pd.DataFrame:
    """
    Normalize a DataFrame to standard OHLCV format.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date/timestamp column (if not in index)
        column_mapping: Custom column name mapping
        timezone: Timezone to apply (e.g., 'UTC', 'America/New_York')
        ensure_timezone: Whether to ensure timezone-aware timestamps
        
    Returns:
        Normalized DataFrame with columns: open, high, low, close, volume
        and DatetimeIndex
        
    Raises:
        DataValidationError: If normalization fails
    """
    if df.empty:
        raise DataValidationError(
            "Cannot normalize empty DataFrame",
            details={}
        )
    
    # Work with a copy
    normalized = df.copy()
    
    # Handle date column
    if date_column:
        if date_column not in normalized.columns:
            raise DataValidationError(
                f"Date column '{date_column}' not found in DataFrame",
                details={"available_columns": list(normalized.columns)}
            )
        normalized = normalized.set_index(date_column)
    
    # Ensure index is datetime
    if not isinstance(normalized.index, pd.DatetimeIndex):
        try:
            normalized.index = pd.to_datetime(normalized.index)
        except Exception as e:
            raise DataValidationError(
                f"Could not convert index to datetime: {str(e)}",
                details={"index_type": type(normalized.index).__name__}
            ) from e
    
    # Apply timezone if specified
    if timezone:
        try:
            tz = pytz.timezone(timezone)
            if normalized.index.tz is None:
                normalized.index = normalized.index.tz_localize(tz)
            else:
                normalized.index = normalized.index.tz_convert(tz)
        except Exception as e:
            raise DataValidationError(
                f"Could not apply timezone '{timezone}': {str(e)}",
                details={"timezone": timezone}
            ) from e
    elif ensure_timezone and normalized.index.tz is None:
        # Default to UTC if timezone-aware is required
        normalized.index = normalized.index.tz_localize('UTC')
    
    # Map columns
    mapping = column_mapping or {}
    final_mapping = {}
    
    for target_col, source_names in COLUMN_MAPPINGS.items():
        if target_col in mapping:
            # Use custom mapping
            source_name = mapping[target_col]
            if isinstance(source_name, list):
                source_name = source_name[0]
            if source_name in normalized.columns:
                final_mapping[source_name] = target_col
        else:
            # Try to find matching column
            for source_name in source_names:
                if source_name in normalized.columns:
                    final_mapping[source_name] = target_col
                    break
    
    # Rename columns
    normalized = normalized.rename(columns=final_mapping)
    
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in normalized.columns]
    
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns after normalization: {missing_columns}",
            details={
                "required": required_columns,
                "missing": missing_columns,
                "available": list(normalized.columns)
            }
        )
    
    # Select only OHLCV columns
    normalized = normalized[required_columns]
    
    # Ensure numeric types
    for col in required_columns:
        try:
            normalized[col] = pd.to_numeric(normalized[col], errors='coerce')
        except Exception as e:
            raise DataValidationError(
                f"Could not convert column '{col}' to numeric: {str(e)}",
                details={"column": col}
            ) from e
    
    # Sort by index
    normalized = normalized.sort_index()
    
    # Remove any rows with NaN values in required columns
    normalized = normalized.dropna(subset=required_columns)
    
    return normalized


def normalize_timezone(
    df: pd.DataFrame,
    target_timezone: str = 'UTC',
    source_timezone: Optional[str] = None
) -> pd.DataFrame:
    """
    Normalize timezone of a DataFrame's index.
    
    Args:
        df: DataFrame with DatetimeIndex
        target_timezone: Target timezone (e.g., 'UTC', 'America/New_York')
        source_timezone: Source timezone (if index is timezone-naive)
        
    Returns:
        DataFrame with normalized timezone
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError(
            "DataFrame index must be a DatetimeIndex",
            details={"index_type": type(df.index).__name__}
        )
    
    normalized = df.copy()
    
    if normalized.index.tz is None:
        if source_timezone:
            normalized.index = normalized.index.tz_localize(source_timezone)
        else:
            normalized.index = normalized.index.tz_localize('UTC')
    
    if target_timezone:
        target_tz = pytz.timezone(target_timezone)
        normalized.index = normalized.index.tz_convert(target_tz)
    
    return normalized


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    normalized = df.copy()
    normalized.columns = [col.lower() for col in normalized.columns]
    return normalized

