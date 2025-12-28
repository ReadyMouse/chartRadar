"""
YAML configuration loader for the ChartRadar framework.

This module provides functionality to load and parse YAML configuration files
with support for environment variable substitution and multiple file merging.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from chartradar.src.exceptions import ConfigurationError
from chartradar.config.schema import FrameworkConfig


def substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in configuration values.
    
    Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.
    
    Args:
        value: Configuration value (can be dict, list, str, or other types)
        
    Returns:
        Value with environment variables substituted
    """
    if isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    elif isinstance(value, str):
        # Pattern: ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
        
        def replace_match(match):
            var_name = match.group(1)
            default = match.group(2) if match.group(2) else None
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            else:
                raise ConfigurationError(
                    f"Environment variable '{var_name}' not set and no default provided",
                    details={"variable": var_name, "context": "environment_substitution"}
                )
        
        return re.sub(pattern, replace_match, value)
    else:
        return value


def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary containing the YAML contents
        
    Raises:
        ConfigurationError: If the file cannot be read or parsed
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {file_path}",
            details={"file_path": str(file_path)}
        )
    
    if not file_path.is_file():
        raise ConfigurationError(
            f"Path is not a file: {file_path}",
            details={"file_path": str(file_path)}
        )
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
            if content is None:
                content = {}
            return content
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse YAML file: {file_path}",
            details={"file_path": str(file_path), "error": str(e)}
        ) from e
    except Exception as e:
        raise ConfigurationError(
            f"Failed to read configuration file: {file_path}",
            details={"file_path": str(file_path), "error": str(e)}
        ) from e


def merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Later configurations override earlier ones. For lists, later configs append
    unless the key is prefixed with '!' to indicate replacement.
    
    Args:
        configs: List of configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    if not configs:
        return {}
    
    merged = configs[0].copy()
    
    for config in configs[1:]:
        merged = _deep_merge(merged, config)
    
    return merged


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary to merge on top of base
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        # Handle list replacement: if key starts with '!', replace the list
        if key.startswith('!'):
            actual_key = key[1:]
            result[actual_key] = value
        elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            # For lists, append by default (can be overridden with !key)
            result[key] = result[key] + value
        else:
            result[key] = value
    
    return result


def load_config(
    config_path: Union[str, Path, List[Union[str, Path]]],
    substitute_env: bool = True,
    validate: bool = True
) -> FrameworkConfig:
    """
    Load and validate a YAML configuration file.
    
    Args:
        config_path: Path to configuration file(s). Can be a single path or list of paths.
        substitute_env: Whether to substitute environment variables
        validate: Whether to validate the configuration against the schema
        
    Returns:
        Validated FrameworkConfig object
        
    Raises:
        ConfigurationError: If loading or validation fails
    """
    # Handle single path or list of paths
    if isinstance(config_path, (str, Path)):
        config_paths = [config_path]
    else:
        config_paths = list(config_path)
    
    if not config_paths:
        raise ConfigurationError(
            "No configuration file paths provided",
            details={"config_path": config_path}
        )
    
    # Load all configuration files
    configs = []
    for path in config_paths:
        config = load_yaml_file(path)
        configs.append(config)
    
    # Merge configurations
    merged_config = merge_configs(configs)
    
    # Substitute environment variables if requested
    if substitute_env:
        merged_config = substitute_env_vars(merged_config)
    
        # Validate against schema if requested
        if validate:
            try:
                return FrameworkConfig(**merged_config)
            except Exception as e:
                raise ConfigurationError(
                    f"Configuration validation failed: {str(e)}",
                    details={"config_path": config_path, "error": str(e)}
                ) from e
        else:
            # Return unvalidated config (for testing or advanced use cases)
            # Use model_construct to skip validation
            return FrameworkConfig.model_construct(**merged_config)

