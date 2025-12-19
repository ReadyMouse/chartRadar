"""
Custom exception classes for the ChartRadar framework.

This module defines all custom exceptions used throughout the framework
to provide clear error messages and enable proper error handling.
"""


class ChartRadarError(Exception):
    """Base exception class for all ChartRadar framework errors."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(ChartRadarError):
    """Raised when there is an error in configuration loading or validation."""
    pass


class DataSourceError(ChartRadarError):
    """Raised when there is an error with a data source."""
    pass


class DataValidationError(ChartRadarError):
    """Raised when data validation fails."""
    pass


class AlgorithmError(ChartRadarError):
    """Raised when there is an error in algorithm execution."""
    pass


class AlgorithmNotFoundError(AlgorithmError):
    """Raised when a requested algorithm is not found in the registry."""
    pass


class AlgorithmExecutionError(AlgorithmError):
    """Raised when an algorithm fails during execution."""
    pass


class FusionError(ChartRadarError):
    """Raised when there is an error in data fusion."""
    pass


class FusionStrategyNotFoundError(FusionError):
    """Raised when a requested fusion strategy is not found."""
    pass


class DisplayError(ChartRadarError):
    """Raised when there is an error in display or export operations."""
    pass


class TrainingError(ChartRadarError):
    """Raised when there is an error in training operations."""
    pass


class EvaluationError(ChartRadarError):
    """Raised when there is an error in evaluation operations."""
    pass


class LabelingError(ChartRadarError):
    """Raised when there is an error in labeling operations."""
    pass


class LabelValidationError(LabelingError):
    """Raised when label validation fails."""
    pass


class PipelineError(ChartRadarError):
    """Raised when there is an error in the main pipeline execution."""
    pass

