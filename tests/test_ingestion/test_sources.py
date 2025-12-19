"""Tests for data source implementations."""

import pytest
import pandas as pd
import json
from pathlib import Path
import tempfile

from chartradar.ingestion.sources.csv import CSVDataSource
from chartradar.ingestion.sources.freqtrade import FreqtradeDataSource
from chartradar.core.exceptions import DataSourceError


class TestCSVDataSource:
    """Tests for CSVDataSource."""
    
    def test_load_csv_file(self, tmp_path):
        """Test loading data from CSV file."""
        # Create test CSV
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("""timestamp,open,high,low,close,volume
2024-01-01,100,105,99,103,1000
2024-01-02,101,106,100,104,1100
""")
        
        source = CSVDataSource("test_csv", path=str(csv_file), date_column="timestamp")
        data = source.load_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert isinstance(data.index, pd.DatetimeIndex)
    
    def test_load_csv_file_no_date_column(self, tmp_path):
        """Test loading CSV with datetime index."""
        # Create test CSV with datetime index
        csv_file = tmp_path / "test2.csv"
        csv_file.write_text("""open,high,low,close,volume
100,105,99,103,1000
101,106,100,104,1100
""")
        
        # Create a CSV with proper structure
        import pandas as pd
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [99, 100],
            'close': [103, 104],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))
        df.to_csv(csv_file)
        
        source = CSVDataSource("test_csv2", path=str(csv_file))
        data = source.load_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
    
    def test_csv_file_not_found(self):
        """Test error when CSV file doesn't exist."""
        import tempfile
        import os
        # Create a path that definitely doesn't exist
        nonexistent = os.path.join(tempfile.gettempdir(), "nonexistent_chartradar_test_file_12345.csv")
        with pytest.raises(DataSourceError) as exc_info:
            CSVDataSource("test", path=nonexistent)
        assert "not found" in str(exc_info.value).lower() or "CSV file" in str(exc_info.value)
    
    def test_get_metadata(self, tmp_path):
        """Test getting metadata."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("timestamp,open,high,low,close,volume\n2024-01-01,100,105,99,103,1000\n")
        
        source = CSVDataSource("test_csv", path=str(csv_file))
        metadata = source.get_metadata()
        
        assert metadata["name"] == "test_csv"
        assert metadata["type"] == "csv"
        assert "path" in metadata["parameters"]


class TestFreqtradeDataSource:
    """Tests for FreqtradeDataSource."""
    
    def test_load_freqtrade_data(self, tmp_path):
        """Test loading Freqtrade JSON data."""
        # Create Freqtrade data structure
        data_dir = tmp_path / "data" / "kraken"
        data_dir.mkdir(parents=True)
        
        # Freqtrade format: [[timestamp_ms, open, high, low, close, volume], ...]
        freqtrade_data = [
            [1704067200000, 100.0, 105.0, 99.0, 103.0, 1000.0],
            [1704153600000, 101.0, 106.0, 100.0, 104.0, 1100.0],
        ]
        
        data_file = data_dir / "BTC-USDT-1h.json"
        with open(data_file, 'w') as f:
            json.dump(freqtrade_data, f)
        
        source = FreqtradeDataSource(
            "test_freqtrade",
            data_dir=str(tmp_path / "data"),
            exchange="kraken",
            pair="BTC/USDT",
            timeframe="1h"
        )
        
        data = source.load_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert isinstance(data.index, pd.DatetimeIndex)
    
    def test_freqtrade_data_dir_not_found(self):
        """Test error when data directory doesn't exist."""
        with pytest.raises(DataSourceError) as exc_info:
            FreqtradeDataSource("test", data_dir="/nonexistent/dir")
        assert "not found" in str(exc_info.value).lower()
    
    def test_freqtrade_data_file_not_found(self, tmp_path):
        """Test error when data file doesn't exist."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        with pytest.raises(DataSourceError) as exc_info:
            source = FreqtradeDataSource(
                "test",
                data_dir=str(data_dir),
                exchange="kraken",
                pair="BTC/USDT"
            )
            source.load_data()
        assert "not found" in str(exc_info.value).lower()
    
    def test_get_metadata(self, tmp_path):
        """Test getting metadata."""
        data_dir = tmp_path / "data" / "kraken"
        data_dir.mkdir(parents=True)
        
        source = FreqtradeDataSource(
            "test_freqtrade",
            data_dir=str(tmp_path / "data"),
            exchange="kraken",
            pair="BTC/USDT"
        )
        
        metadata = source.get_metadata()
        assert metadata["name"] == "test_freqtrade"
        assert metadata["type"] == "freqtrade"
        assert metadata["parameters"]["exchange"] == "kraken"

