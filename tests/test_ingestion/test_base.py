"""Tests for base data source classes."""

import pytest
import pandas as pd
from datetime import datetime

from chartradar.ingestion.base import DataSource
from chartradar.src.exceptions import DataSourceError


class ConcreteDataSource(DataSource):
    """Concrete implementation for testing."""
    
    def load_data(self, start_date=None, end_date=None, **kwargs):
        """Load test data."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'open': range(10),
            'high': range(10, 20),
            'low': range(0, 10),
            'close': range(5, 15),
            'volume': range(100, 110)
        }, index=dates)
    
    async def stream_data(self, callback, **kwargs):
        """Stream test data."""
        pass
    
    def get_metadata(self):
        """Get test metadata."""
        return {
            "name": self.name,
            "type": "test",
            "description": "Test source",
            "capabilities": ["batch"]
        }


class TestDataSource:
    """Tests for DataSource base class."""
    
    def test_initialization(self):
        """Test data source initialization."""
        source = ConcreteDataSource("test_source", param1="value1")
        assert source.name == "test_source"
        assert source.parameters["param1"] == "value1"
    
    def test_load_data(self):
        """Test loading data."""
        source = ConcreteDataSource("test")
        data = source.load_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        source = ConcreteDataSource("test")
        data = source.load_data()
        assert source.validate_data(data) is True
    
    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        source = ConcreteDataSource("test")
        data = pd.DataFrame({'open': [1, 2]})
        
        with pytest.raises(DataSourceError) as exc_info:
            source.validate_data(data)
        assert "missing required columns" in str(exc_info.value).lower()
    
    def test_validate_data_no_datetime_index(self):
        """Test data validation with non-datetime index."""
        source = ConcreteDataSource("test")
        data = pd.DataFrame({
            'open': [1], 'high': [2], 'low': [0.5], 'close': [1.5], 'volume': [100]
        }, index=[0])
        
        with pytest.raises(DataSourceError) as exc_info:
            source.validate_data(data)
        assert "must be a DatetimeIndex" in str(exc_info.value)
    
    def test_validate_data_empty(self):
        """Test data validation with empty data."""
        source = ConcreteDataSource("test")
        data = pd.DataFrame()
        
        with pytest.raises(DataSourceError) as exc_info:
            source.validate_data(data)
        assert "empty" in str(exc_info.value).lower()

