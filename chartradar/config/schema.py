"""
YAML configuration schema definition for the ChartRadar framework.

This module provides Pydantic models for validating YAML configuration files.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class DataSourceConfig(BaseModel):
    """
    Configuration for a data source.
    
    Defines how to connect to and load data from a specific source.
    """
    name: str = Field(..., description="Name identifier for this data source")
    type: str = Field(..., description="Data source type (e.g., 'freqtrade', 'csv', 'exchange')")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Source-specific parameters")
    enabled: bool = Field(True, description="Whether this data source is enabled")
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate data source type."""
        valid_types = ['freqtrade', 'csv', 'exchange', 'database']
        if v not in valid_types:
            raise ValueError(f"Invalid data source type: {v}. Must be one of {valid_types}")
        return v


class AlgorithmConfig(BaseModel):
    """
    Configuration for an algorithm.
    
    Defines which algorithm to use and its parameters.
    """
    name: str = Field(..., description="Algorithm name (must exist in registry)")
    version: Optional[str] = Field(None, description="Algorithm version (optional)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific parameters")
    enabled: bool = Field(True, description="Whether this algorithm is enabled")
    weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Weight for fusion (0.0-1.0)")


class FusionConfig(BaseModel):
    """
    Configuration for data fusion.
    
    Defines how to combine results from multiple algorithms.
    """
    strategy: str = Field(..., description="Fusion strategy name (e.g., 'weighted_average', 'voting')")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Fusion strategy-specific parameters")
    enabled: bool = Field(True, description="Whether fusion is enabled")
    
    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate fusion strategy name."""
        valid_strategies = ['weighted_average', 'voting', 'stacking', 'none']
        if v not in valid_strategies:
            raise ValueError(f"Invalid fusion strategy: {v}. Must be one of {valid_strategies}")
        return v


class DisplayConfig(BaseModel):
    """
    Configuration for display and export.
    
    Defines how to visualize and export results.
    """
    visualization: Dict[str, Any] = Field(default_factory=dict, description="Visualization settings")
    export: Dict[str, Any] = Field(default_factory=dict, description="Export settings")
    enabled: bool = Field(True, description="Whether display is enabled")
    
    @field_validator('visualization')
    @classmethod
    def validate_visualization(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate visualization settings."""
        if 'backend' in v:
            valid_backends = ['matplotlib', 'plotly', 'none']
            if v['backend'] not in valid_backends:
                raise ValueError(f"Invalid visualization backend: {v['backend']}. Must be one of {valid_backends}")
        return v


class TrainingConfig(BaseModel):
    """
    Configuration for training and evaluation.
    
    Defines training parameters for ML algorithms.
    """
    enabled: bool = Field(False, description="Whether training is enabled")
    data_split: Dict[str, Any] = Field(default_factory=dict, description="Data splitting configuration")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    checkpoint: Dict[str, Any] = Field(default_factory=dict, description="Checkpointing configuration")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Metrics logging configuration")
    tracking: Dict[str, Any] = Field(default_factory=dict, description="Experiment tracking configuration")


class LabelingConfig(BaseModel):
    """
    Configuration for data labeling.
    
    Defines labeling tool and storage settings.
    """
    enabled: bool = Field(False, description="Whether labeling is enabled")
    storage: Dict[str, Any] = Field(default_factory=dict, description="Label storage configuration")
    tool: Dict[str, Any] = Field(default_factory=dict, description="Labeling tool configuration")


class FrameworkConfig(BaseModel):
    """
    Root configuration model for the ChartRadar framework.
    
    Combines all configuration sections into a single validated structure.
    """
    version: str = Field("1.0", description="Configuration schema version")
    name: Optional[str] = Field(None, description="Configuration name/description")
    
    data_sources: List[DataSourceConfig] = Field(
        default_factory=list,
        description="List of data source configurations"
    )
    
    algorithms: List[AlgorithmConfig] = Field(
        default_factory=list,
        description="List of algorithm configurations"
    )
    
    fusion: Optional[FusionConfig] = Field(
        None,
        description="Fusion configuration (optional)"
    )
    
    display: Optional[DisplayConfig] = Field(
        None,
        description="Display configuration (optional)"
    )
    
    training: Optional[TrainingConfig] = Field(
        None,
        description="Training configuration (optional)"
    )
    
    labeling: Optional[LabelingConfig] = Field(
        None,
        description="Labeling configuration (optional)"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this configuration"
    )
    
    @field_validator('data_sources')
    @classmethod
    def validate_data_sources(cls, v: List[DataSourceConfig]) -> List[DataSourceConfig]:
        """Validate that at least one data source is enabled."""
        if not v:
            raise ValueError("At least one data source must be configured")
        enabled_sources = [ds for ds in v if ds.enabled]
        if not enabled_sources:
            raise ValueError("At least one data source must be enabled")
        return v
    
    @field_validator('algorithms')
    @classmethod
    def validate_algorithms(cls, v: List[AlgorithmConfig]) -> List[AlgorithmConfig]:
        """Validate that at least one algorithm is enabled."""
        if not v:
            raise ValueError("At least one algorithm must be configured")
        enabled_algorithms = [alg for alg in v if alg.enabled]
        if not enabled_algorithms:
            raise ValueError("At least one algorithm must be enabled")
        return v
    
    class Config:
        """Pydantic model configuration."""
        extra = "forbid"  # Reject extra fields not in schema
        validate_assignment = True  # Validate on assignment

