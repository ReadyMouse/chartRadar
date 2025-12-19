"""
Freqtrade data source implementation.

This module provides a data source for loading OHLCV data from Freqtrade
data storage, supporting Kraken exchange data format.
"""

from typing import Any, Dict, Optional
import pandas as pd
from pathlib import Path
import json

from chartradar.ingestion.batch import BatchDataSource
from chartradar.ingestion.normalizer import normalize_dataframe
from chartradar.core.exceptions import DataSourceError


class FreqtradeDataSource(BatchDataSource):
    """
    Data source for loading OHLCV data from Freqtrade data storage.
    
    Supports both batch and streaming modes for Freqtrade data.
    """
    
    def __init__(
        self,
        name: str,
        data_dir: str,
        exchange: str = "kraken",
        pair: str = "BTC/USDT",
        timeframe: str = "1h",
        **kwargs: Any
    ):
        """
        Initialize the Freqtrade data source.
        
        Args:
            name: Data source name
            data_dir: Freqtrade data directory path
            exchange: Exchange name (e.g., 'kraken')
            pair: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '1d', '5m')
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        self.data_dir = Path(data_dir)
        self.exchange = exchange
        self.pair = pair
        self.timeframe = timeframe
        
        if not self.data_dir.exists():
            raise DataSourceError(
                f"Freqtrade data directory not found: {data_dir}",
                details={"data_dir": str(data_dir)}
            )
        
        # Construct file path
        # Freqtrade format: user_data/data/exchange/pair-timeframe.json
        pair_clean = pair.replace('/', '-')
        self.data_file = self.data_dir / exchange / f"{pair_clean}-{timeframe}.json"
    
    def _load_raw_data(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load raw data from Freqtrade JSON file.
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            **kwargs: Additional parameters
            
        Returns:
            Raw DataFrame from Freqtrade data
        """
        if not self.data_file.exists():
            raise DataSourceError(
                f"Freqtrade data file not found: {self.data_file}",
                details={
                    "data_file": str(self.data_file),
                    "exchange": self.exchange,
                    "pair": self.pair,
                    "timeframe": self.timeframe
                }
            )
        
        try:
            # Load JSON data
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            # Convert to DataFrame
            # Freqtrade format: list of [timestamp, open, high, low, close, volume]
            df = pd.DataFrame(
                data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Normalize the data
            df = normalize_dataframe(
                df,
                date_column=None,  # Already in index
                timezone='UTC'  # Freqtrade data is typically in UTC
            )
            
            return df
            
        except json.JSONDecodeError as e:
            raise DataSourceError(
                f"Failed to parse Freqtrade JSON file: {str(e)}",
                details={"data_file": str(self.data_file), "error": str(e)}
            ) from e
        except Exception as e:
            raise DataSourceError(
                f"Failed to load Freqtrade data: {str(e)}",
                details={"data_file": str(self.data_file), "error": str(e)}
            ) from e
    
    async def stream_data(
        self,
        callback: callable,
        **kwargs: Any
    ) -> None:
        """
        Stream real-time data (not typically supported by file-based sources).
        
        For Freqtrade, this would require integration with live exchange APIs.
        This is a placeholder for future implementation.
        """
        raise DataSourceError(
            "Streaming not yet implemented for Freqtrade data source. "
            "Use exchange data source for real-time streaming.",
            details={"source": self.name, "type": "freqtrade"}
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this Freqtrade data source."""
        file_size = self.data_file.stat().st_size if self.data_file.exists() else 0
        
        return {
            "name": self.name,
            "type": "freqtrade",
            "description": f"Freqtrade data source: {self.exchange}/{self.pair}",
            "capabilities": ["batch"],
            "parameters": {
                "data_dir": str(self.data_dir),
                "exchange": self.exchange,
                "pair": self.pair,
                "timeframe": self.timeframe,
                "data_file": str(self.data_file),
                "file_size_bytes": file_size,
                **self.parameters
            }
        }

