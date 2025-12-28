"""Tests for label validator."""

import pytest

from chartradar.labeling.validator import LabelValidator
from chartradar.src.exceptions import LabelValidationError


class TestLabelValidator:
    """Tests for LabelValidator class."""
    
    def test_validate_valid_labels(self):
        """Test validating valid labels."""
        validator = LabelValidator()
        
        labels = [
            {
                "pattern_type": "rising_wedge",
                "start_index": 0,
                "end_index": 10,
                "confidence": 0.8
            }
        ]
        
        result = validator.validate(labels)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_missing_fields(self):
        """Test validation with missing required fields."""
        validator = LabelValidator()
        
        labels = [
            {
                "pattern_type": "rising_wedge"
                # Missing start_index and end_index
            }
        ]
        
        result = validator.validate(labels, raise_on_error=False)
        assert result["valid"] is False
        assert len(result["errors"]) > 0
    
    def test_validate_invalid_indices(self):
        """Test validation with invalid indices."""
        validator = LabelValidator()
        
        labels = [
            {
                "pattern_type": "rising_wedge",
                "start_index": 10,
                "end_index": 5  # Invalid: end < start
            }
        ]
        
        result = validator.validate(labels, raise_on_error=False)
        assert result["valid"] is False
    
    def test_validate_consistency(self):
        """Test consistency validation with data range."""
        validator = LabelValidator()
        
        labels = [
            {
                "pattern_type": "rising_wedge",
                "start_index": 0,
                "end_index": 10
            }
        ]
        
        data_range = {"start_index": 0, "end_index": 100}
        result = validator.validate(labels, data_range=data_range)
        assert result["valid"] is True
        
        # Test with out-of-range label
        labels_out_of_range = [
            {
                "pattern_type": "rising_wedge",
                "start_index": 0,
                "end_index": 150  # Beyond data range
            }
        ]
        
        result = validator.validate(labels_out_of_range, data_range=data_range, raise_on_error=False)
        assert result["valid"] is False
    
    def test_detect_duplicates(self):
        """Test duplicate detection."""
        validator = LabelValidator()
        
        labels = [
            {
                "pattern_type": "rising_wedge",
                "start_index": 0,
                "end_index": 10
            },
            {
                "pattern_type": "rising_wedge",
                "start_index": 0,
                "end_index": 10  # Duplicate
            }
        ]
        
        result = validator.validate(labels, raise_on_error=False)
        assert len(result["warnings"]) > 0
    
    def test_generate_validation_report(self):
        """Test generating validation report."""
        validator = LabelValidator()
        
        labels = [
            {
                "pattern_type": "rising_wedge",
                "start_index": 0,
                "end_index": 10,
                "confidence": 0.8
            }
        ]
        
        report = validator.generate_validation_report(labels)
        assert "statistics" in report
        assert "pattern_type_distribution" in report["statistics"]

