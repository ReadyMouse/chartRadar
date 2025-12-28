"""Tests for data caching."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
from datetime import timedelta

from chartradar.ingestion.cache import DataCache
from chartradar.src.exceptions import DataSourceError


class TestDataCache:
    """Tests for DataCache class."""
    
    def test_cache_key_generation(self, tmp_path):
        """Test cache key generation."""
        cache = DataCache(cache_dir=str(tmp_path))
        
        key1 = cache.get_cache_key("source1", "csv", start_date="2024-01-01")
        key2 = cache.get_cache_key("source1", "csv", start_date="2024-01-01")
        key3 = cache.get_cache_key("source1", "csv", start_date="2024-01-02")
        
        assert key1 == key2  # Same parameters = same key
        assert key1 != key3  # Different parameters = different key
    
    def test_set_and_get(self, tmp_path):
        """Test setting and getting cache."""
        cache = DataCache(cache_dir=str(tmp_path))
        
        data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [99, 100],
            'close': [103, 104],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        cache_key = cache.get_cache_key("test", "csv")
        cache.set(cache_key, data)
        
        cached_data = cache.get(cache_key)
        assert cached_data is not None
        pd.testing.assert_frame_equal(cached_data, data)
    
    def test_get_nonexistent(self, tmp_path):
        """Test getting non-existent cache entry."""
        cache = DataCache(cache_dir=str(tmp_path))
        cached_data = cache.get("nonexistent_key")
        assert cached_data is None
    
    def test_invalidate(self, tmp_path):
        """Test cache invalidation."""
        cache = DataCache(cache_dir=str(tmp_path))
        
        data = pd.DataFrame({'open': [100]}, index=pd.date_range('2024-01-01', periods=1))
        cache_key = cache.get_cache_key("test", "csv")
        cache.set(cache_key, data)
        
        cache.invalidate(cache_key)
        assert cache.get(cache_key) is None
    
    def test_invalidate_all(self, tmp_path):
        """Test invalidating all cache entries."""
        cache = DataCache(cache_dir=str(tmp_path))
        
        data = pd.DataFrame({'open': [100]}, index=pd.date_range('2024-01-01', periods=1))
        cache.set("key1", data)
        cache.set("key2", data)
        
        cache.invalidate()
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_expired_cache(self, tmp_path):
        """Test that expired cache entries are not returned."""
        cache = DataCache(cache_dir=str(tmp_path), default_ttl=timedelta(seconds=-1))  # Already expired
        
        data = pd.DataFrame({'open': [100]}, index=pd.date_range('2024-01-01', periods=1))
        cache_key = cache.get_cache_key("test", "csv")
        cache.set(cache_key, data)
        
        cached_data = cache.get(cache_key, check_expiry=True)
        assert cached_data is None
    
    def test_get_cache_info(self, tmp_path):
        """Test getting cache information."""
        cache = DataCache(cache_dir=str(tmp_path))
        
        data = pd.DataFrame({'open': [100]}, index=pd.date_range('2024-01-01', periods=1))
        cache.set("key1", data)
        
        info = cache.get_cache_info()
        assert info["total_entries"] == 1
        assert info["total_size_bytes"] > 0

