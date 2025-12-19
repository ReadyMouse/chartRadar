"""
Data ingestion module for the ChartRadar framework.

This module provides data sources for loading OHLCV data from various sources
including CSV files, Freqtrade data, and exchange APIs.
"""

from chartradar.ingestion.base import DataSource
from chartradar.ingestion.batch import BatchDataSource
from chartradar.ingestion.streaming import StreamingDataSource
from chartradar.ingestion.normalizer import (
    normalize_dataframe,
    normalize_timezone,
    standardize_column_names,
)
from chartradar.ingestion.validator import DataValidator, validate_data
from chartradar.ingestion.cache import DataCache

__all__ = [
    # Base classes
    "DataSource",
    "BatchDataSource",
    "StreamingDataSource",
    # Normalization
    "normalize_dataframe",
    "normalize_timezone",
    "standardize_column_names",
    # Validation
    "DataValidator",
    "validate_data",
    # Caching
    "DataCache",
]

