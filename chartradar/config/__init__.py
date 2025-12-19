"""
Configuration module for the ChartRadar framework.

This module provides YAML configuration loading, validation, and schema definitions.
"""

from chartradar.config.schema import (
    DataSourceConfig,
    AlgorithmConfig,
    FusionConfig,
    DisplayConfig,
    TrainingConfig,
    LabelingConfig,
    FrameworkConfig,
)
from chartradar.config.loader import (
    load_config,
    load_yaml_file,
    merge_configs,
    substitute_env_vars,
)
from chartradar.config.validator import (
    ConfigValidator,
    validate_config,
)

__all__ = [
    # Schema models
    "DataSourceConfig",
    "AlgorithmConfig",
    "FusionConfig",
    "DisplayConfig",
    "TrainingConfig",
    "LabelingConfig",
    "FrameworkConfig",
    # Loader functions
    "load_config",
    "load_yaml_file",
    "merge_configs",
    "substitute_env_vars",
    # Validator
    "ConfigValidator",
    "validate_config",
]

