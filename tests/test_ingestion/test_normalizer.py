"""Tests for data normalization."""

import pytest
import pandas as pd
from datetime import datetime

from chartradar.ingestion.normalizer import (
    normalize_dataframe,
    normalize_timezone,
    standardize_column_names,
)
from chartradar.src.exceptions import DataValidationError


class TestNormalizeDataframe:
    """Tests for normalize_dataframe function."""
    
    def test_normalize_standard_format(self):
        """Test normalizing already standard format."""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [99, 100],
            'close': [103, 104],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        result = normalize_dataframe(df)
        assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_normalize_with_date_column(self):
        """Test normalizing with date column."""
        df = pd.DataFrame({
            'timestamp': ['2024-01-01', '2024-01-02'],
            'open': [100, 101],
            'high': [105, 106],
            'low': [99, 100],
            'close': [103, 104],
            'volume': [1000, 1100]
        })
        
        result = normalize_dataframe(df, date_column='timestamp')
        assert isinstance(result.index, pd.DatetimeIndex)
        assert 'timestamp' not in result.columns
    
    def test_normalize_case_insensitive(self):
        """Test normalizing with different case column names."""
        df = pd.DataFrame({
            'Open': [100, 101],
            'HIGH': [105, 106],
            'low': [99, 100],
            'Close': [103, 104],
            'Volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        result = normalize_dataframe(df)
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_normalize_missing_columns(self):
        """Test normalizing with missing required columns."""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106]
        })
        
        with pytest.raises(DataValidationError) as exc_info:
            normalize_dataframe(df)
        assert "missing required columns" in str(exc_info.value).lower()
    
    def test_normalize_empty_dataframe(self):
        """Test normalizing empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(DataValidationError) as exc_info:
            normalize_dataframe(df)
        assert "empty" in str(exc_info.value).lower()
    
    def test_normalize_custom_mapping(self):
        """Test normalizing with custom column mapping."""
        df = pd.DataFrame({
            'price_open': [100, 101],
            'price_high': [105, 106],
            'price_low': [99, 100],
            'price_close': [103, 104],
            'vol': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        mapping = {
            'open': 'price_open',
            'high': 'price_high',
            'low': 'price_low',
            'close': 'price_close',
            'volume': 'vol'
        }
        
        result = normalize_dataframe(df, column_mapping=mapping)
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])


class TestNormalizeTimezone:
    """Tests for normalize_timezone function."""
    
    def test_normalize_timezone_naive_to_utc(self):
        """Test converting timezone-naive to UTC."""
        df = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [99],
            'close': [103],
            'volume': [1000]
        }, index=pd.date_range('2024-01-01', periods=1))
        
        result = normalize_timezone(df, target_timezone='UTC')
        assert result.index.tz is not None
    
    def test_normalize_timezone_conversion(self):
        """Test converting between timezones."""
        df = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [99],
            'close': [103],
            'volume': [1000]
        }, index=pd.date_range('2024-01-01', periods=1, tz='UTC'))
        
        result = normalize_timezone(df, target_timezone='America/New_York')
        assert str(result.index.tz) == 'America/New_York'


class TestStandardizeColumnNames:
    """Tests for standardize_column_names function."""
    
    def test_standardize_lowercase(self):
        """Test converting column names to lowercase."""
        df = pd.DataFrame({
            'Open': [100],
            'HIGH': [105],
            'Low': [99]
        })
        
        result = standardize_column_names(df)
        assert all(col.islower() for col in result.columns)

