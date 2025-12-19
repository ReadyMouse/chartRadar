"""
Streaming data source base implementation.

This module provides a base class for streaming data sources with
async/await pattern, callback-based delivery, and connection management.
"""

import asyncio
from typing import Any, Callable, Dict, Optional
import pandas as pd
from datetime import datetime

from chartradar.ingestion.base import DataSource
from chartradar.core.exceptions import DataSourceError


class StreamingDataSource(DataSource):
    """
    Base class for streaming data sources.
    
    Provides common functionality for streaming real-time data with
    async/await pattern and connection management.
    """
    
    def __init__(
        self,
        name: str,
        reconnect_delay: float = 5.0,
        max_reconnect_attempts: Optional[int] = None,
        **kwargs: Any
    ):
        """
        Initialize the streaming data source.
        
        Args:
            name: Name identifier for this data source
            reconnect_delay: Delay in seconds between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts (None for unlimited)
            **kwargs: Additional source-specific parameters
        """
        super().__init__(name, **kwargs)
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self._is_streaming = False
        self._connection = None
        self._reconnect_count = 0
    
    async def stream_data(
        self,
        callback: Callable,
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
        if self._is_streaming:
            raise DataSourceError(
                "Streaming is already active",
                details={"source": self.name}
            )
        
        self._is_streaming = True
        self._reconnect_count = 0
        
        try:
            await self._start_streaming(callback, **kwargs)
        except Exception as e:
            self._is_streaming = False
            raise DataSourceError(
                f"Streaming failed: {str(e)}",
                details={"source": self.name, "error": str(e)}
            ) from e
        finally:
            self._is_streaming = False
            await self._cleanup()
    
    async def _start_streaming(
        self,
        callback: Callable,
        **kwargs: Any
    ) -> None:
        """
        Start streaming (implementation-specific).
        
        This method must be implemented by subclasses.
        
        Args:
            callback: Async function to call with each new data point
            **kwargs: Additional parameters
        """
        raise NotImplementedError("Subclasses must implement _start_streaming")
    
    async def _handle_reconnect(
        self,
        callback: Callable,
        **kwargs: Any
    ) -> None:
        """
        Handle reconnection logic.
        
        Args:
            callback: Async function to call with each new data point
            **kwargs: Additional parameters
        """
        if self.max_reconnect_attempts is not None:
            if self._reconnect_count >= self.max_reconnect_attempts:
                raise DataSourceError(
                    f"Maximum reconnection attempts ({self.max_reconnect_attempts}) reached",
                    details={"source": self.name, "attempts": self._reconnect_count}
                )
        
        self._reconnect_count += 1
        
        await asyncio.sleep(self.reconnect_delay)
        
        # Try to reconnect
        await self._start_streaming(callback, **kwargs)
    
    async def _cleanup(self) -> None:
        """
        Clean up resources (close connections, etc.).
        
        Can be overridden by subclasses for custom cleanup.
        """
        if self._connection:
            try:
                await self._close_connection()
            except Exception:
                pass
            finally:
                self._connection = None
    
    async def _close_connection(self) -> None:
        """
        Close the connection (implementation-specific).
        
        Can be overridden by subclasses.
        """
        pass
    
    def stop_streaming(self) -> None:
        """
        Stop streaming (synchronous method for external control).
        
        Sets the streaming flag to False, which should cause
        the streaming loop to exit gracefully.
        """
        self._is_streaming = False
    
    def is_streaming(self) -> bool:
        """Check if streaming is currently active."""
        return self._is_streaming
    
    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Load historical data (not typically supported by streaming sources).
        
        Raises:
            DataSourceError: If batch loading is not supported
        """
        raise DataSourceError(
            "Batch loading is not supported by streaming data sources",
            details={"source": self.name, "type": "streaming"}
        )
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this streaming data source."""
        return {
            "name": self.name,
            "type": "streaming",
            "description": f"Streaming data source: {self.name}",
            "capabilities": ["streaming"],
            "parameters": {
                "reconnect_delay": self.reconnect_delay,
                "max_reconnect_attempts": self.max_reconnect_attempts,
                **self.parameters
            }
        }

