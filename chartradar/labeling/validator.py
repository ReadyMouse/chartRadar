"""
Label validation for the ChartRadar framework.

This module provides functionality to validate label format, consistency,
and detect duplicates or conflicts.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from chartradar.src.exceptions import LabelValidationError
from chartradar.src.types import PatternDetection


class LabelValidator:
    """
    Validator for pattern labels.
    
    Provides validation of label format, consistency, and conflict detection.
    """
    
    def __init__(self, **kwargs: Any):
        """
        Initialize the label validator.
        
        Args:
            **kwargs: Validator-specific parameters
        """
        self.parameters = kwargs
    
    def validate(
        self,
        labels: List[Dict[str, Any]],
        data_range: Optional[Dict[str, int]] = None,
        raise_on_error: bool = False
    ) -> Dict[str, Any]:
        """
        Validate labels.
        
        Args:
            labels: List of label dictionaries
            data_range: Optional data range dict with 'start_index' and 'end_index'
            raise_on_error: Whether to raise exception on validation errors
            
        Returns:
            Dictionary with validation results
            
        Raises:
            LabelValidationError: If raise_on_error is True and validation fails
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        # Validate format and structure
        format_errors = self._validate_format(labels)
        errors.extend(format_errors)
        
        # Validate consistency
        if data_range:
            consistency_errors = self._validate_consistency(labels, data_range)
            errors.extend(consistency_errors)
        
        # Detect duplicates and conflicts
        duplicate_warnings, conflict_errors = self._detect_duplicates_and_conflicts(labels)
        warnings.extend(duplicate_warnings)
        errors.extend(conflict_errors)
        
        valid = len(errors) == 0
        
        if errors and raise_on_error:
            error_msg = "Label validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise LabelValidationError(
                error_msg,
                details={"errors": errors, "warnings": warnings}
            )
        
        return {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "total_labels": len(labels),
            "valid_labels": len(labels) - len([e for e in errors if "label" in e.lower()])
        }
    
    def _validate_format(self, labels: List[Dict[str, Any]]) -> List[str]:
        """Validate label format and structure."""
        errors = []
        required_fields = ["pattern_type", "start_index", "end_index"]
        
        for i, label in enumerate(labels):
            # Check required fields
            missing_fields = [field for field in required_fields if field not in label]
            if missing_fields:
                errors.append(
                    f"Label {i}: Missing required fields: {missing_fields}"
                )
            
            # Validate field types and values
            if "start_index" in label and "end_index" in label:
                try:
                    start_idx = int(label["start_index"])
                    end_idx = int(label["end_index"])
                    
                    if start_idx < 0 or end_idx < 0:
                        errors.append(f"Label {i}: Indices must be non-negative")
                    
                    if end_idx < start_idx:
                        errors.append(f"Label {i}: end_index must be >= start_index")
                except (ValueError, TypeError):
                    errors.append(f"Label {i}: start_index and end_index must be integers")
            
            # Validate confidence if present
            if "confidence" in label:
                try:
                    confidence = float(label["confidence"])
                    if not (0.0 <= confidence <= 1.0):
                        errors.append(f"Label {i}: confidence must be between 0.0 and 1.0")
                except (ValueError, TypeError):
                    errors.append(f"Label {i}: confidence must be a float")
        
        return errors
    
    def _validate_consistency(
        self,
        labels: List[Dict[str, Any]],
        data_range: Dict[str, int]
    ) -> List[str]:
        """Validate label consistency with data range."""
        errors = []
        start_idx = data_range.get("start_index", 0)
        end_idx = data_range.get("end_index", 0)
        
        for i, label in enumerate(labels):
            label_start = label.get("start_index")
            label_end = label.get("end_index")
            
            if label_start is not None and label_end is not None:
                if label_start < start_idx or label_end > end_idx:
                    errors.append(
                        f"Label {i}: Pattern boundaries ({label_start}-{label_end}) "
                        f"outside data range ({start_idx}-{end_idx})"
                    )
        
        return errors
    
    def _detect_duplicates_and_conflicts(
        self,
        labels: List[Dict[str, Any]]
    ) -> tuple[List[str], List[str]]:
        """Detect duplicate or conflicting labels."""
        warnings = []
        errors = []
        
        # Group labels by pattern boundaries
        pattern_groups = {}
        for i, label in enumerate(labels):
            key = (
                label.get("pattern_type"),
                label.get("start_index"),
                label.get("end_index")
            )
            if key not in pattern_groups:
                pattern_groups[key] = []
            pattern_groups[key].append((i, label))
        
        # Check for duplicates
        for key, group in pattern_groups.items():
            if len(group) > 1:
                indices = [idx for idx, _ in group]
                warnings.append(
                    f"Duplicate labels found at indices {indices}: "
                    f"pattern_type={key[0]}, range={key[1]}-{key[2]}"
                )
        
        # Check for overlapping patterns with different types
        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels[i+1:], start=i+1):
                if self._labels_overlap(label1, label2):
                    if label1.get("pattern_type") != label2.get("pattern_type"):
                        errors.append(
                            f"Conflicting labels at indices {i} and {j}: "
                            f"Overlapping patterns with different types "
                            f"({label1.get('pattern_type')} vs {label2.get('pattern_type')})"
                        )
        
        return warnings, errors
    
    def _labels_overlap(
        self,
        label1: Dict[str, Any],
        label2: Dict[str, Any]
    ) -> bool:
        """Check if two labels overlap."""
        start1 = label1.get("start_index", 0)
        end1 = label1.get("end_index", 0)
        start2 = label2.get("start_index", 0)
        end2 = label2.get("end_index", 0)
        
        return not (end1 < start2 or end2 < start1)
    
    def generate_validation_report(
        self,
        labels: List[Dict[str, Any]],
        data_range: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Args:
            labels: List of labels to validate
            data_range: Optional data range
            
        Returns:
            Dictionary with validation report
        """
        validation_result = self.validate(labels, data_range, raise_on_error=False)
        
        # Additional statistics
        pattern_types = {}
        for label in labels:
            pattern_type = label.get("pattern_type", "unknown")
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
        
        return {
            **validation_result,
            "statistics": {
                "pattern_type_distribution": pattern_types,
                "total_labels": len(labels),
                "average_confidence": (
                    sum(label.get("confidence", 0.0) for label in labels) / len(labels)
                    if labels else 0.0
                )
            }
        }

