"""Tests for data validation."""

import pytest
import pandas as pd
import numpy as np

from chartradar.ingestion.validator import DataValidator, validate_data
from chartradar.src.exceptions import DataValidationError


class TestDataValidator:
    """Tests for DataValidator class."""
    
    def test_validate_valid_data(self):
        """Test validating valid OHLCV data."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        validator = DataValidator()
        result = validator.validate(data)
        
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_ohlc_integrity_high_low(self):
        """Test OHLC integrity check for high < low."""
        data = pd.DataFrame({
            'open': [100],
            'high': [99],  # Invalid: high < low
            'low': [100],
            'close': [101],
            'volume': [1000]
        }, index=pd.date_range('2024-01-01', periods=1))
        
        validator = DataValidator()
        result = validator.validate(data, raise_on_error=False)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert "high < low" in str(result["errors"][0]).lower()
    
    def test_validate_missing_data(self):
        """Test validation with missing data."""
        data = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [105, 106, 107],
            'low': [99, 100, 101],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range('2024-01-01', periods=3))
        
        validator = DataValidator()
        result = validator.validate(data)
        
        assert result["valid"] is True  # Missing data is a warning, not error
        assert len(result["warnings"]) > 0
    
    def test_validate_outliers(self):
        """Test outlier detection."""
        data = pd.DataFrame({
            'open': [100] * 10 + [1000],  # Outlier
            'high': [105] * 10 + [1050],
            'low': [99] * 10 + [990],
            'close': [103] * 10 + [1030],
            'volume': [1000] * 11
        }, index=pd.date_range('2024-01-01', periods=11))
        
        validator = DataValidator(outlier_threshold=2.0)
        result = validator.validate(data)
        
        assert result["valid"] is True
        assert result["report"]["outliers"]["count"] > 0
    
    def test_validate_empty_data(self):
        """Test validation with empty data."""
        data = pd.DataFrame()
        
        validator = DataValidator()
        result = validator.validate(data, raise_on_error=False)
        
        assert result["valid"] is False
        assert "empty" in str(result["errors"][0]).lower()
    
    def test_validate_raise_on_error(self):
        """Test that validation raises error when requested."""
        data = pd.DataFrame({
            'open': [100],
            'high': [99],  # Invalid
            'low': [100],
            'close': [101],
            'volume': [1000]
        }, index=pd.date_range('2024-01-01', periods=1))
        
        validator = DataValidator()
        
        with pytest.raises(DataValidationError):
            validator.validate(data, raise_on_error=True)


class TestValidateData:
    """Tests for validate_data convenience function."""
    
    def test_validate_data_function(self):
        """Test the validate_data convenience function."""
        data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [99, 100],
            'close': [103, 104],
            'volume': [1000, 1100]
        }, index=pd.date_range('2024-01-01', periods=2))
        
        result = validate_data(data)
        assert result["valid"] is True

