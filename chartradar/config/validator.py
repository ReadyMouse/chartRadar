"""
Configuration validation logic for the ChartRadar framework.

This module provides additional validation beyond Pydantic schema validation,
including checks for algorithm registry, data source configurations, etc.
"""

from typing import Any, Dict, List, Optional, Set
from chartradar.src.exceptions import ConfigurationError
from chartradar.config.schema import FrameworkConfig, DataSourceConfig, AlgorithmConfig


class ConfigValidator:
    """
    Validator for ChartRadar framework configurations.
    
    Performs additional validation beyond Pydantic schema validation,
    such as checking algorithm registry, data source availability, etc.
    """
    
    def __init__(
        self,
        algorithm_registry: Optional[Any] = None,
        fusion_registry: Optional[Any] = None
    ):
        """
        Initialize the validator.
        
        Args:
            algorithm_registry: Optional algorithm registry to validate against
            fusion_registry: Optional fusion strategy registry to validate against
        """
        self.algorithm_registry = algorithm_registry
        self.fusion_registry = fusion_registry
    
    def validate(self, config: FrameworkConfig) -> Dict[str, Any]:
        """
        Validate a configuration object.
        
        Args:
            config: FrameworkConfig object to validate
            
        Returns:
            Dictionary with validation results and warnings
            
        Raises:
            ConfigurationError: If validation fails
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        # Validate data sources
        self._validate_data_sources(config.data_sources, errors, warnings)
        
        # Validate algorithms
        self._validate_algorithms(config.algorithms, errors, warnings)
        
        # Validate fusion
        if config.fusion and config.fusion.enabled:
            self._validate_fusion(config.fusion, errors, warnings)
        
        # Validate algorithm weights if fusion is enabled
        if config.fusion and config.fusion.enabled:
            self._validate_algorithm_weights(config.algorithms, config.fusion, errors, warnings)
        
        # Validate display configuration
        if config.display:
            self._validate_display(config.display, errors, warnings)
        
        # Validate training configuration
        if config.training and config.training.enabled:
            self._validate_training(config.training, errors, warnings)
        
        # Validate labeling configuration
        if config.labeling and config.labeling.enabled:
            self._validate_labeling(config.labeling, errors, warnings)
        
        # Raise error if there are validation errors
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ConfigurationError(
                error_msg,
                details={"errors": errors, "warnings": warnings}
            )
        
        return {
            "valid": True,
            "errors": [],
            "warnings": warnings
        }
    
    def _validate_data_sources(
        self,
        data_sources: List[DataSourceConfig],
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Validate data source configurations."""
        enabled_sources = [ds for ds in data_sources if ds.enabled]
        
        if not enabled_sources:
            errors.append("At least one data source must be enabled")
            return
        
        # Check for duplicate names
        names = [ds.name for ds in enabled_sources]
        duplicates = [name for name in names if names.count(name) > 1]
        if duplicates:
            errors.append(f"Duplicate data source names: {', '.join(set(duplicates))}")
        
        # Validate source-specific parameters
        for ds in enabled_sources:
            if ds.type == 'csv':
                if 'path' not in ds.parameters:
                    errors.append(f"Data source '{ds.name}': CSV source requires 'path' parameter")
            elif ds.type == 'freqtrade':
                if 'data_dir' not in ds.parameters:
                    warnings.append(f"Data source '{ds.name}': Freqtrade source should specify 'data_dir'")
            elif ds.type == 'exchange':
                if 'exchange' not in ds.parameters:
                    errors.append(f"Data source '{ds.name}': Exchange source requires 'exchange' parameter")
                if 'symbol' not in ds.parameters:
                    errors.append(f"Data source '{ds.name}': Exchange source requires 'symbol' parameter")
    
    def _validate_algorithms(
        self,
        algorithms: List[AlgorithmConfig],
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Validate algorithm configurations."""
        enabled_algorithms = [alg for alg in algorithms if alg.enabled]
        
        if not enabled_algorithms:
            errors.append("At least one algorithm must be enabled")
            return
        
        # Check for duplicate names
        names = [alg.name for alg in enabled_algorithms]
        duplicates = [name for name in names if names.count(name) > 1]
        if duplicates:
            warnings.append(f"Duplicate algorithm names (will use last configuration): {', '.join(set(duplicates))}")
        
        # Validate against registry if available
        if self.algorithm_registry:
            for alg in enabled_algorithms:
                try:
                    # Try to get algorithm from registry
                    if hasattr(self.algorithm_registry, 'get_algorithm'):
                        self.algorithm_registry.get_algorithm(alg.name)
                    elif hasattr(self.algorithm_registry, 'get'):
                        self.algorithm_registry.get(alg.name)
                except (KeyError, AttributeError, Exception) as e:
                    warnings.append(
                        f"Algorithm '{alg.name}' not found in registry. "
                        f"It may need to be registered before use. Error: {str(e)}"
                    )
    
    def _validate_fusion(
        self,
        fusion_config: Any,
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Validate fusion configuration."""
        if fusion_config.strategy == 'none':
            warnings.append("Fusion is enabled but strategy is 'none'")
            return
        
        # Validate against registry if available
        if self.fusion_registry:
            try:
                if hasattr(self.fusion_registry, 'get_strategy'):
                    self.fusion_registry.get_strategy(fusion_config.strategy)
                elif hasattr(self.fusion_registry, 'get'):
                    self.fusion_registry.get(fusion_config.strategy)
            except (KeyError, AttributeError, Exception) as e:
                warnings.append(
                    f"Fusion strategy '{fusion_config.strategy}' not found in registry. "
                    f"It may need to be registered before use. Error: {str(e)}"
                )
        
        # Validate strategy-specific parameters
        if fusion_config.strategy == 'weighted_average':
            if 'normalize_weights' in fusion_config.parameters:
                normalize = fusion_config.parameters['normalize_weights']
                if not isinstance(normalize, bool):
                    errors.append("Fusion 'weighted_average': 'normalize_weights' must be a boolean")
        elif fusion_config.strategy == 'voting':
            if 'threshold' in fusion_config.parameters:
                threshold = fusion_config.parameters['threshold']
                if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                    errors.append("Fusion 'voting': 'threshold' must be a number between 0 and 1")
    
    def _validate_algorithm_weights(
        self,
        algorithms: List[AlgorithmConfig],
        fusion_config: Any,
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Validate algorithm weights for fusion."""
        if fusion_config.strategy in ['weighted_average', 'voting']:
            enabled_algorithms = [alg for alg in algorithms if alg.enabled]
            algorithms_with_weights = [alg for alg in enabled_algorithms if alg.weight is not None]
            
            if algorithms_with_weights:
                total_weight = sum(alg.weight for alg in algorithms_with_weights)
                if total_weight <= 0:
                    errors.append("Sum of algorithm weights must be greater than 0")
                
                # Warn if some algorithms have weights and others don't
                algorithms_without_weights = [
                    alg for alg in enabled_algorithms if alg.weight is None
                ]
                if algorithms_without_weights:
                    warnings.append(
                        f"Some algorithms have weights and others don't. "
                        f"Algorithms without weights: {[alg.name for alg in algorithms_without_weights]}"
                    )
    
    def _validate_display(
        self,
        display_config: Any,
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Validate display configuration."""
        if display_config.visualization:
            backend = display_config.visualization.get('backend')
            if backend and backend not in ['matplotlib', 'plotly', 'none']:
                errors.append(f"Invalid visualization backend: {backend}")
        
        if display_config.export:
            formats = display_config.export.get('formats', [])
            if formats:
                valid_formats = ['json', 'csv', 'png', 'svg', 'pdf']
                invalid_formats = [f for f in formats if f not in valid_formats]
                if invalid_formats:
                    warnings.append(f"Unknown export formats (may be custom): {', '.join(invalid_formats)}")
    
    def _validate_training(
        self,
        training_config: Any,
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Validate training configuration."""
        if training_config.data_split:
            split_ratios = training_config.data_split.get('ratios', {})
            if split_ratios:
                total = sum(split_ratios.values())
                if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                    errors.append(f"Training data split ratios must sum to 1.0, got {total}")
        
        if training_config.checkpoint:
            save_dir = training_config.checkpoint.get('save_dir')
            if save_dir and not isinstance(save_dir, str):
                errors.append("Training checkpoint 'save_dir' must be a string")
    
    def _validate_labeling(
        self,
        labeling_config: Any,
        errors: List[str],
        warnings: List[str]
    ) -> None:
        """Validate labeling configuration."""
        if labeling_config.storage:
            storage_type = labeling_config.storage.get('type')
            if storage_type and storage_type not in ['file', 'database']:
                warnings.append(f"Unknown storage type: {storage_type}")


def validate_config(
    config: FrameworkConfig,
    algorithm_registry: Optional[Any] = None,
    fusion_registry: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Validate a configuration object.
    
    Convenience function that creates a validator and validates the config.
    
    Args:
        config: FrameworkConfig object to validate
        algorithm_registry: Optional algorithm registry to validate against
        fusion_registry: Optional fusion strategy registry to validate against
        
    Returns:
        Dictionary with validation results
        
    Raises:
        ConfigurationError: If validation fails
    """
    validator = ConfigValidator(algorithm_registry, fusion_registry)
    return validator.validate(config)

