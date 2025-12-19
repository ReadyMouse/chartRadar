"""
Exchange API data source implementation using ccxt.

This module provides a data source for loading OHLCV data from cryptocurrency
exchanges via the ccxt library, supporting multiple exchanges and streaming.
"""

from typing import Any, Dict, Optional
import pandas as pd
import asyncio

from chartradar.ingestion.base import DataSource
from chartradar.ingestion.normalizer import normalize_dataframe
from chartradar.core.exceptions import DataSourceError

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


class ExchangeDataSource(DataSource):
    """
    Data source for loading OHLCV data from cryptocurrency exchanges.
    
    Uses the ccxt library to connect to multiple exchanges and supports
    both batch loading and streaming via WebSocket or polling.
    """
    
    def __init__(
        self,
        name: str,
        exchange: str,
        symbol: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        sandbox: bool = False,
        rate_limit: bool = True,
        **kwargs: Any
    ):
        """
        Initialize the exchange data source.
        
        Args:
            name: Data source name
            exchange: Exchange name (e.g., 'binance', 'kraken', 'coinbase')
            symbol: Trading symbol (e.g., 'BTC/USDT')
            api_key: API key for authenticated requests
            api_secret: API secret for authenticated requests
            sandbox: Whether to use sandbox/testnet
            rate_limit: Whether to respect rate limits
            **kwargs: Additional parameters
        """
        super().__init__(name, **kwargs)
        
        if not CCXT_AVAILABLE:
            raise DataSourceError(
                "ccxt library is not installed. Install it with: pip install ccxt",
                details={"package": "ccxt"}
            )
        
        self.exchange_name = exchange
        self.symbol = symbol
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox
        self.rate_limit = rate_limit
        
        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, exchange)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'sandbox': sandbox,
                'enableRateLimit': rate_limit,
            })
        except AttributeError:
            raise DataSourceError(
                f"Exchange '{exchange}' not supported by ccxt",
                details={"exchange": exchange, "available": dir(ccxt)}
            ) from None
        except Exception as e:
            raise DataSourceError(
                f"Failed to initialize exchange: {str(e)}",
                details={"exchange": exchange, "error": str(e)}
            ) from e
    
    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        timeframe: str = '1h',
        limit: Optional[int] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load historical OHLCV data from exchange.
        
        Args:
            start_date: Start date (ISO format string)
            end_date: End date (ISO format string)
            timeframe: Timeframe (e.g., '1h', '1d', '5m')
            limit: Maximum number of candles to fetch
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert dates to timestamps
            since = None
            if start_date:
                since = int(pd.to_datetime(start_date).timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                timeframe,
                since=since,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Filter by end date if provided
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
            
            # Normalize the data
            df = normalize_dataframe(
                df,
                date_column=None,  # Already in index
                timezone='UTC'  # Exchange data is typically in UTC
            )
            
            return df
            
        except ccxt.NetworkError as e:
            raise DataSourceError(
                f"Network error while fetching data: {str(e)}",
                details={"exchange": self.exchange_name, "error": str(e)}
            ) from e
        except ccxt.ExchangeError as e:
            raise DataSourceError(
                f"Exchange error while fetching data: {str(e)}",
                details={"exchange": self.exchange_name, "error": str(e)}
            ) from e
        except Exception as e:
            raise DataSourceError(
                f"Failed to load exchange data: {str(e)}",
                details={"exchange": self.exchange_name, "error": str(e)}
            ) from e
    
    async def stream_data(
        self,
        callback: callable,
        timeframe: str = '1h',
        **kwargs: Any
    ) -> None:
        """
        Stream real-time data from exchange.
        
        Uses polling for exchanges that don't support WebSocket.
        
        Args:
            callback: Async function to call with each new data point
            timeframe: Timeframe for candles
            **kwargs: Additional parameters
        """
        if not hasattr(self.exchange, 'watch_ohlcv'):
            # Fall back to polling if WebSocket not available
            await self._stream_polling(callback, timeframe, **kwargs)
        else:
            # Use WebSocket if available
            await self._stream_websocket(callback, timeframe, **kwargs)
    
    async def _stream_polling(
        self,
        callback: callable,
        timeframe: str,
        **kwargs: Any
    ) -> None:
        """Stream data using polling."""
        last_timestamp = None
        
        while True:
            try:
                # Fetch latest OHLCV
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe,
                    limit=1
                )
                
                if ohlcv and len(ohlcv) > 0:
                    latest = ohlcv[-1]
                    timestamp = latest[0]
                    
                    # Only process if new data
                    if last_timestamp is None or timestamp > last_timestamp:
                        df = pd.DataFrame(
                            [latest],
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df = df.set_index('timestamp')
                        df = normalize_dataframe(df, date_column=None, timezone='UTC')
                        
                        await callback(df)
                        last_timestamp = timestamp
                
                # Wait before next poll
                await asyncio.sleep(60)  # Poll every minute
                
            except Exception as e:
                raise DataSourceError(
                    f"Error during streaming: {str(e)}",
                    details={"exchange": self.exchange_name, "error": str(e)}
                ) from e
    
    async def _stream_websocket(
        self,
        callback: callable,
        timeframe: str,
        **kwargs: Any
    ) -> None:
        """Stream data using WebSocket (if supported)."""
        # This is a placeholder - actual WebSocket implementation would
        # depend on the specific exchange's WebSocket API
        raise DataSourceError(
            "WebSocket streaming not yet fully implemented",
            details={"exchange": self.exchange_name}
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this exchange data source."""
        return {
            "name": self.name,
            "type": "exchange",
            "description": f"Exchange data source: {self.exchange_name}/{self.symbol}",
            "capabilities": ["batch", "streaming"],
            "parameters": {
                "exchange": self.exchange_name,
                "symbol": self.symbol,
                "sandbox": self.sandbox,
                "rate_limit": self.rate_limit,
                **self.parameters
            }
        }

