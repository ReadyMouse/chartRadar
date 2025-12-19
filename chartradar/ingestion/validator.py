"""
Data quality validation for the ChartRadar framework.

This module provides functions to validate OHLCV data integrity,
check for missing data, detect outliers, and generate validation reports.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

from chartradar.core.exceptions import DataValidationError


class DataValidator:
    """
    Validator for OHLCV data quality.
    
    Performs various checks on data integrity, completeness, and anomalies.
    """
    
    def __init__(
        self,
        check_ohlc_integrity: bool = True,
        check_missing_data: bool = True,
        check_outliers: bool = True,
        outlier_threshold: float = 3.0
    ):
        """
        Initialize the validator.
        
        Args:
            check_ohlc_integrity: Whether to check OHLC relationships
            check_missing_data: Whether to check for missing data
            check_outliers: Whether to detect outliers
            outlier_threshold: Standard deviations for outlier detection
        """
        self.check_ohlc_integrity = check_ohlc_integrity
        self.check_missing_data = check_missing_data
        self.check_outliers = check_outliers
        self.outlier_threshold = outlier_threshold
    
    def validate(
        self,
        data: pd.DataFrame,
        raise_on_error: bool = False
    ) -> Dict[str, Any]:
        """
        Validate OHLCV data.
        
        Args:
            data: DataFrame with OHLCV columns and datetime index
            raise_on_error: Whether to raise exception on validation errors
            
        Returns:
            Dictionary with validation results:
            - valid: bool
            - errors: List of error messages
            - warnings: List of warning messages
            - report: Detailed validation report
            
        Raises:
            DataValidationError: If raise_on_error is True and validation fails
        """
        errors: List[str] = []
        warnings: List[str] = []
        report: Dict[str, Any] = {
            "total_rows": len(data),
            "date_range": None,
            "ohlc_integrity": {},
            "missing_data": {},
            "outliers": {},
            "statistics": {}
        }
        
        # Basic structure checks
        self._check_structure(data, errors)
        
        if errors and raise_on_error:
            raise DataValidationError(
                f"Data validation failed: {errors[0]}",
                details={"errors": errors}
            )
        
        # Get date range
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
            report["date_range"] = {
                "start": data.index.min().isoformat(),
                "end": data.index.max().isoformat(),
                "duration_days": (data.index.max() - data.index.min()).days
            }
        
        # OHLC integrity checks
        if self.check_ohlc_integrity:
            ohlc_errors = self._check_ohlc_integrity(data)
            report["ohlc_integrity"] = ohlc_errors
            if ohlc_errors.get("errors"):
                errors.extend(ohlc_errors["errors"])
                warnings.extend(ohlc_errors.get("warnings", []))
        
        # Missing data checks
        if self.check_missing_data:
            missing_info = self._check_missing_data(data)
            report["missing_data"] = missing_info
            if missing_info.get("has_missing"):
                warnings.append(f"Found missing data: {missing_info['summary']}")
        
        # Outlier detection
        if self.check_outliers:
            outlier_info = self._detect_outliers(data)
            report["outliers"] = outlier_info
            if outlier_info.get("count", 0) > 0:
                warnings.append(
                    f"Detected {outlier_info['count']} potential outliers "
                    f"(threshold: {self.outlier_threshold}σ)"
                )
        
        # Statistics
        report["statistics"] = self._calculate_statistics(data)
        
        # Determine overall validity
        valid = len(errors) == 0
        
        if errors and raise_on_error:
            raise DataValidationError(
                f"Data validation failed with {len(errors)} error(s)",
                details={"errors": errors, "warnings": warnings, "report": report}
            )
        
        return {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "report": report
        }
    
    def _check_structure(self, data: pd.DataFrame, errors: List[str]) -> None:
        """Check basic data structure."""
        if data.empty:
            errors.append("Data is empty")
            return
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append(f"Index must be DatetimeIndex, got {type(data.index).__name__}")
    
    def _check_ohlc_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check OHLC data integrity.
        
        Validates:
        - high >= low
        - high >= open
        - high >= close
        - low <= open
        - low <= close
        """
        errors: List[str] = []
        warnings: List[str] = []
        invalid_rows: List[int] = []
        
        # Check high >= low
        invalid_high_low = data[data['high'] < data['low']]
        if len(invalid_high_low) > 0:
            errors.append(f"Found {len(invalid_high_low)} rows where high < low")
            invalid_rows.extend(invalid_high_low.index.tolist())
        
        # Check high >= open
        invalid_high_open = data[data['high'] < data['open']]
        if len(invalid_high_open) > 0:
            errors.append(f"Found {len(invalid_high_open)} rows where high < open")
            invalid_rows.extend(invalid_high_open.index.tolist())
        
        # Check high >= close
        invalid_high_close = data[data['high'] < data['close']]
        if len(invalid_high_close) > 0:
            errors.append(f"Found {len(invalid_high_close)} rows where high < close")
            invalid_rows.extend(invalid_high_close.index.tolist())
        
        # Check low <= open
        invalid_low_open = data[data['low'] > data['open']]
        if len(invalid_low_open) > 0:
            errors.append(f"Found {len(invalid_low_open)} rows where low > open")
            invalid_rows.extend(invalid_low_open.index.tolist())
        
        # Check low <= close
        invalid_low_close = data[data['low'] > data['close']]
        if len(invalid_low_close) > 0:
            errors.append(f"Found {len(invalid_low_close)} rows where low > close")
            invalid_rows.extend(invalid_low_close.index.tolist())
        
        # Check for zero or negative prices
        negative_prices = data[(data[['open', 'high', 'low', 'close']] <= 0).any(axis=1)]
        if len(negative_prices) > 0:
            errors.append(f"Found {len(negative_prices)} rows with zero or negative prices")
            invalid_rows.extend(negative_prices.index.tolist())
        
        # Check for negative volume
        negative_volume = data[data['volume'] < 0]
        if len(negative_volume) > 0:
            warnings.append(f"Found {len(negative_volume)} rows with negative volume")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "invalid_row_count": len(set(invalid_rows)),
            "invalid_rows": list(set(invalid_rows))[:10]  # Limit to first 10
        }
    
    def _check_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing data."""
        missing_counts = data.isnull().sum()
        has_missing = missing_counts.sum() > 0
        
        missing_info = {
            "has_missing": has_missing,
            "by_column": missing_counts.to_dict(),
            "total_missing": int(missing_counts.sum()),
            "missing_percentage": float((missing_counts.sum() / (len(data) * len(data.columns))) * 100)
        }
        
        if has_missing:
            missing_info["summary"] = ", ".join(
                f"{col}: {count}" for col, count in missing_counts.items() if count > 0
            )
        
        return missing_info
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        outliers_by_column: Dict[str, List[int]] = {}
        total_outliers = set()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in data.columns:
                continue
            
            # Calculate Z-scores
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outlier_mask = z_scores > self.outlier_threshold
            outlier_indices = data[outlier_mask].index.tolist()
            
            if outlier_indices:
                outliers_by_column[col] = outlier_indices
                total_outliers.update(outlier_indices)
        
        return {
            "count": len(total_outliers),
            "by_column": {col: len(indices) for col, indices in outliers_by_column.items()},
            "threshold": self.outlier_threshold
        }
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics."""
        stats = {}
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                stats[col] = {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "median": float(data[col].median())
                }
        
        return stats


def validate_data(
    data: pd.DataFrame,
    raise_on_error: bool = False,
    **validator_kwargs: Any
) -> Dict[str, Any]:
    """
    Validate OHLCV data (convenience function).
    
    Args:
        data: DataFrame to validate
        raise_on_error: Whether to raise exception on errors
        **validator_kwargs: Additional arguments for DataValidator
        
    Returns:
        Validation results dictionary
    """
    validator = DataValidator(**validator_kwargs)
    return validator.validate(data, raise_on_error=raise_on_error)

