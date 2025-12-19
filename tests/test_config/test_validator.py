"""Tests for configuration validator."""

import pytest
from unittest.mock import Mock

from chartradar.core.exceptions import ConfigurationError
from chartradar.config.validator import ConfigValidator, validate_config
from chartradar.config.schema import (
    FrameworkConfig,
    DataSourceConfig,
    AlgorithmConfig,
    FusionConfig,
    DisplayConfig,
    TrainingConfig,
)


class TestConfigValidator:
    """Tests for ConfigValidator class."""
    
    def test_validate_minimal_config(self):
        """Test validating a minimal valid configuration."""
        config = FrameworkConfig(
            data_sources=[
                DataSourceConfig(name="source1", type="csv", enabled=True)
            ],
            algorithms=[
                AlgorithmConfig(name="alg1", enabled=True)
            ]
        )
        
        validator = ConfigValidator()
        result = validator.validate(config)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_duplicate_data_source_names(self):
        """Test that duplicate data source names are detected."""
        config = FrameworkConfig(
            data_sources=[
                DataSourceConfig(name="source1", type="csv", enabled=True),
                DataSourceConfig(name="source1", type="csv", enabled=True)
            ],
            algorithms=[
                AlgorithmConfig(name="alg1", enabled=True)
            ]
        )
        
        validator = ConfigValidator()
        with pytest.raises(ConfigurationError) as exc_info:
            validator.validate(config)
        assert "Duplicate data source names" in str(exc_info.value)
    
    def test_validate_csv_missing_path(self):
        """Test that CSV source without path parameter raises error."""
        config = FrameworkConfig(
            data_sources=[
                DataSourceConfig(
                    name="csv_source",
                    type="csv",
                    enabled=True,
                    parameters={}
                )
            ],
            algorithms=[
                AlgorithmConfig(name="alg1", enabled=True)
            ]
        )
        
        validator = ConfigValidator()
        with pytest.raises(ConfigurationError) as exc_info:
            validator.validate(config)
        assert "requires 'path' parameter" in str(exc_info.value)
    
    def test_validate_exchange_missing_params(self):
        """Test that exchange source without required parameters raises error."""
        config = FrameworkConfig(
            data_sources=[
                DataSourceConfig(
                    name="exchange_source",
                    type="exchange",
                    enabled=True,
                    parameters={}
                )
            ],
            algorithms=[
                AlgorithmConfig(name="alg1", enabled=True)
            ]
        )
        
        validator = ConfigValidator()
        with pytest.raises(ConfigurationError) as exc_info:
            validator.validate(config)
        assert "requires 'exchange' parameter" in str(exc_info.value)
    
    def test_validate_algorithm_registry(self):
        """Test validation against algorithm registry."""
        # Mock registry
        mock_registry = Mock()
        mock_registry.get_algorithm = Mock(side_effect=KeyError("Algorithm not found"))
        
        config = FrameworkConfig(
            data_sources=[
                DataSourceConfig(name="source1", type="csv", enabled=True, parameters={"path": "/tmp/data.csv"})
            ],
            algorithms=[
                AlgorithmConfig(name="unknown_alg", enabled=True)
            ]
        )
        
        validator = ConfigValidator(algorithm_registry=mock_registry)
        result = validator.validate(config)
        # Should not raise error, but should have warnings
        assert result["valid"] is True
        assert len(result["warnings"]) > 0
    
    def test_validate_fusion_strategy_registry(self):
        """Test validation against fusion strategy registry."""
        # Mock registry
        mock_registry = Mock()
        mock_registry.get_strategy = Mock(side_effect=KeyError("Strategy not found"))
        
        config = FrameworkConfig(
            data_sources=[
                DataSourceConfig(name="source1", type="csv", enabled=True, parameters={"path": "/tmp/data.csv"})
            ],
            algorithms=[
                AlgorithmConfig(name="alg1", enabled=True)
            ],
            fusion=FusionConfig(strategy="unknown_strategy", enabled=True)
        )
        
        validator = ConfigValidator(fusion_registry=mock_registry)
        result = validator.validate(config)
        # Should not raise error, but should have warnings
        assert result["valid"] is True
        assert len(result["warnings"]) > 0
    
    def test_validate_algorithm_weights(self):
        """Test validation of algorithm weights for fusion."""
        config = FrameworkConfig(
            data_sources=[
                DataSourceConfig(name="source1", type="csv", enabled=True, parameters={"path": "/tmp/data.csv"})
            ],
            algorithms=[
                AlgorithmConfig(name="alg1", enabled=True, weight=0.5),
                AlgorithmConfig(name="alg2", enabled=True, weight=0.3)
            ],
            fusion=FusionConfig(strategy="weighted_average", enabled=True)
        )
        
        validator = ConfigValidator()
        result = validator.validate(config)
        assert result["valid"] is True
    
    def test_validate_zero_total_weight(self):
        """Test that zero total weight raises error."""
        config = FrameworkConfig(
            data_sources=[
                DataSourceConfig(name="source1", type="csv", enabled=True, parameters={"path": "/tmp/data.csv"})
            ],
            algorithms=[
                AlgorithmConfig(name="alg1", enabled=True, weight=0.0),
                AlgorithmConfig(name="alg2", enabled=True, weight=0.0)
            ],
            fusion=FusionConfig(strategy="weighted_average", enabled=True)
        )
        
        validator = ConfigValidator()
        with pytest.raises(ConfigurationError) as exc_info:
            validator.validate(config)
        assert "Sum of algorithm weights" in str(exc_info.value)
    
    def test_validate_training_data_split(self):
        """Test validation of training data split ratios."""
        config = FrameworkConfig(
            data_sources=[
                DataSourceConfig(name="source1", type="csv", enabled=True, parameters={"path": "/tmp/data.csv"})
            ],
            algorithms=[
                AlgorithmConfig(name="alg1", enabled=True)
            ],
            training=TrainingConfig(
                enabled=True,
                data_split={"ratios": {"train": 0.5, "validation": 0.3, "test": 0.3}}
            )
        )
        
        validator = ConfigValidator()
        with pytest.raises(ConfigurationError) as exc_info:
            validator.validate(config)
        assert "must sum to 1.0" in str(exc_info.value)
    
    def test_validate_display_export_formats(self):
        """Test validation of display export formats."""
        config = FrameworkConfig(
            data_sources=[
                DataSourceConfig(name="source1", type="csv", enabled=True, parameters={"path": "/tmp/data.csv"})
            ],
            algorithms=[
                AlgorithmConfig(name="alg1", enabled=True)
            ],
            display=DisplayConfig(
                export={"formats": ["json", "csv", "unknown_format"]}
            )
        )
        
        validator = ConfigValidator()
        result = validator.validate(config)
        # Should have warnings about unknown formats
        assert result["valid"] is True
        assert len(result["warnings"]) > 0


class TestValidateConfig:
    """Tests for validate_config convenience function."""
    
    def test_validate_config_function(self):
        """Test the validate_config convenience function."""
        config = FrameworkConfig(
            data_sources=[
                DataSourceConfig(name="source1", type="csv", enabled=True, parameters={"path": "/tmp/data.csv"})
            ],
            algorithms=[
                AlgorithmConfig(name="alg1", enabled=True)
            ]
        )
        
        result = validate_config(config)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

