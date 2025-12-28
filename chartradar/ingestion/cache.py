"""
Data caching implementation for the ChartRadar framework.

This module provides file-based caching for batch data with cache key
generation and invalidation strategies.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta

from chartradar.src.exceptions import DataSourceError


class DataCache:
    """
    File-based cache for OHLCV data.
    
    Provides caching functionality with configurable expiration and
    cache key generation based on data source and parameters.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        default_ttl: Optional[timedelta] = None,
        use_compression: bool = True
    ):
        """
        Initialize the data cache.
        
        Args:
            cache_dir: Directory for cache files (default: ./cache)
            default_ttl: Default time-to-live for cache entries
            use_compression: Whether to use compression for cache files
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl or timedelta(days=1)
        self.use_compression = use_compression
    
    def get_cache_key(
        self,
        source_name: str,
        source_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **params: Any
    ) -> str:
        """
        Generate a cache key for the given parameters.
        
        Args:
            source_name: Data source name
            source_type: Data source type
            start_date: Start date
            end_date: End date
            **params: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create a dictionary of all parameters
        key_data = {
            "source_name": source_name,
            "source_type": source_type,
            "start_date": start_date,
            "end_date": end_date,
            **params
        }
        
        # Sort keys for consistent hashing
        key_str = json.dumps(key_data, sort_keys=True)
        
        # Generate hash
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        
        return f"{source_type}_{source_name}_{key_hash[:16]}"
    
    def get(
        self,
        cache_key: str,
        check_expiry: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get data from cache.
        
        Args:
            cache_key: Cache key
            check_expiry: Whether to check if cache entry has expired
            
        Returns:
            Cached DataFrame or None if not found/expired
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / f"{cache_key}.meta"
        
        if not cache_file.exists():
            return None
        
        # Check expiry if requested
        if check_expiry and metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                expiry_time = datetime.fromisoformat(metadata.get('expires_at', ''))
                if datetime.now() > expiry_time:
                    # Cache expired, remove files
                    cache_file.unlink(missing_ok=True)
                    metadata_file.unlink(missing_ok=True)
                    return None
            except Exception:
                # If metadata is invalid, try to load anyway
                pass
        
        # Load cached data
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            # If loading fails, remove corrupted cache
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            raise DataSourceError(
                f"Failed to load cache: {str(e)}",
                details={"cache_key": cache_key, "error": str(e)}
            ) from e
    
    def set(
        self,
        cache_key: str,
        data: pd.DataFrame,
        ttl: Optional[timedelta] = None
    ) -> None:
        """
        Store data in cache.
        
        Args:
            cache_key: Cache key
            data: DataFrame to cache
            ttl: Time-to-live for this entry (uses default if None)
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / f"{cache_key}.meta"
        
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + ttl
        
        try:
            # Save data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Save metadata
            metadata = {
                "cache_key": cache_key,
                "created_at": datetime.now().isoformat(),
                "expires_at": expires_at.isoformat(),
                "ttl_seconds": ttl.total_seconds(),
                "data_shape": list(data.shape),
                "data_columns": list(data.columns.tolist())
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            # Clean up on failure
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            raise DataSourceError(
                f"Failed to save cache: {str(e)}",
                details={"cache_key": cache_key, "error": str(e)}
            ) from e
    
    def invalidate(self, cache_key: Optional[str] = None) -> None:
        """
        Invalidate cache entry or all entries.
        
        Args:
            cache_key: Specific cache key to invalidate (None for all)
        """
        if cache_key:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            metadata_file = self.cache_dir / f"{cache_key}.meta"
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
        else:
            # Remove all cache files
            for file in self.cache_dir.glob("*.pkl"):
                file.unlink(missing_ok=True)
            for file in self.cache_dir.glob("*.meta"):
                file.unlink(missing_ok=True)
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries.
        
        Returns:
            Number of entries cleared
        """
        cleared = 0
        now = datetime.now()
        
        for metadata_file in self.cache_dir.glob("*.meta"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                expiry_time = datetime.fromisoformat(metadata.get('expires_at', ''))
                if now > expiry_time:
                    cache_key = metadata.get('cache_key', metadata_file.stem)
                    self.invalidate(cache_key)
                    cleared += 1
            except Exception:
                # Skip invalid metadata files
                continue
        
        return cleared
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cache_dir": str(self.cache_dir),
            "total_entries": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }

