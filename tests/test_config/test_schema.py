"""Tests for configuration schema models."""

import pytest
from pydantic import ValidationError

from chartradar.config.schema import (
    DataSourceConfig,
    AlgorithmConfig,
    FusionConfig,
    DisplayConfig,
    TrainingConfig,
    LabelingConfig,
    FrameworkConfig,
)


class TestDataSourceConfig:
    """Tests for DataSourceConfig model."""
    
    def test_valid_config(self):
        """Test creating a valid data source config."""
        config = DataSourceConfig(
            name="test_source",
            type="csv",
            parameters={"path": "/tmp/data.csv"}
        )
        assert config.name == "test_source"
        assert config.type == "csv"
        assert config.enabled is True
        assert config.parameters["path"] == "/tmp/data.csv"
    
    def test_invalid_type(self):
        """Test that invalid data source type raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DataSourceConfig(name="test", type="invalid_type")
        assert "Invalid data source type" in str(exc_info.value)
    
    def test_disabled_source(self):
        """Test creating a disabled data source."""
        config = DataSourceConfig(
            name="test",
            type="csv",
            enabled=False
        )
        assert config.enabled is False


class TestAlgorithmConfig:
    """Tests for AlgorithmConfig model."""
    
    def test_valid_config(self):
        """Test creating a valid algorithm config."""
        config = AlgorithmConfig(
            name="test_algorithm",
            parameters={"param1": "value1"}
        )
        assert config.name == "test_algorithm"
        assert config.enabled is True
        assert config.parameters["param1"] == "value1"
    
    def test_weight_validation(self):
        """Test that weight must be between 0 and 1."""
        # Valid weight
        config = AlgorithmConfig(name="test", weight=0.5)
        assert config.weight == 0.5
        
        # Invalid weight (too high)
        with pytest.raises(ValidationError):
            AlgorithmConfig(name="test", weight=1.5)
        
        # Invalid weight (negative)
        with pytest.raises(ValidationError):
            AlgorithmConfig(name="test", weight=-0.1)


class TestFusionConfig:
    """Tests for FusionConfig model."""
    
    def test_valid_config(self):
        """Test creating a valid fusion config."""
        config = FusionConfig(strategy="weighted_average")
        assert config.strategy == "weighted_average"
        assert config.enabled is True
    
    def test_invalid_strategy(self):
        """Test that invalid fusion strategy raises error."""
        with pytest.raises(ValidationError) as exc_info:
            FusionConfig(strategy="invalid_strategy")
        assert "Invalid fusion strategy" in str(exc_info.value)


class TestDisplayConfig:
    """Tests for DisplayConfig model."""
    
    def test_valid_config(self):
        """Test creating a valid display config."""
        config = DisplayConfig(
            visualization={"backend": "matplotlib"},
            export={"formats": ["json", "csv"]}
        )
        assert config.visualization["backend"] == "matplotlib"
        assert "json" in config.export["formats"]
    
    def test_invalid_backend(self):
        """Test that invalid visualization backend raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DisplayConfig(visualization={"backend": "invalid"})
        assert "Invalid visualization backend" in str(exc_info.value)


class TestFrameworkConfig:
    """Tests for FrameworkConfig model."""
    
    def test_minimal_valid_config(self):
        """Test creating a minimal valid framework config."""
        config = FrameworkConfig(
            data_sources=[
                DataSourceConfig(name="source1", type="csv")
            ],
            algorithms=[
                AlgorithmConfig(name="alg1")
            ]
        )
        assert len(config.data_sources) == 1
        assert len(config.algorithms) == 1
    
    def test_no_data_sources(self):
        """Test that config without data sources raises error."""
        with pytest.raises(ValidationError) as exc_info:
            FrameworkConfig(
                data_sources=[],
                algorithms=[AlgorithmConfig(name="alg1")]
            )
        assert "At least one data source must be configured" in str(exc_info.value)
    
    def test_no_enabled_data_sources(self):
        """Test that config with no enabled data sources raises error."""
        with pytest.raises(ValidationError) as exc_info:
            FrameworkConfig(
                data_sources=[
                    DataSourceConfig(name="source1", type="csv", enabled=False)
                ],
                algorithms=[AlgorithmConfig(name="alg1")]
            )
        assert "At least one data source must be enabled" in str(exc_info.value)
    
    def test_no_algorithms(self):
        """Test that config without algorithms raises error."""
        with pytest.raises(ValidationError) as exc_info:
            FrameworkConfig(
                data_sources=[
                    DataSourceConfig(name="source1", type="csv")
                ],
                algorithms=[]
            )
        assert "At least one algorithm must be configured" in str(exc_info.value)
    
    def test_no_enabled_algorithms(self):
        """Test that config with no enabled algorithms raises error."""
        with pytest.raises(ValidationError) as exc_info:
            FrameworkConfig(
                data_sources=[
                    DataSourceConfig(name="source1", type="csv")
                ],
                algorithms=[
                    AlgorithmConfig(name="alg1", enabled=False)
                ]
            )
        assert "At least one algorithm must be enabled" in str(exc_info.value)
    
    def test_full_config(self):
        """Test creating a full configuration with all sections."""
        config = FrameworkConfig(
            version="1.0",
            name="Test Config",
            data_sources=[
                DataSourceConfig(name="source1", type="csv")
            ],
            algorithms=[
                AlgorithmConfig(name="alg1", weight=0.5)
            ],
            fusion=FusionConfig(strategy="weighted_average"),
            display=DisplayConfig(),
            training=TrainingConfig(enabled=False),
            labeling=LabelingConfig(enabled=False),
            metadata={"author": "test"}
        )
        assert config.version == "1.0"
        assert config.name == "Test Config"
        assert config.fusion is not None
        assert config.display is not None
        assert config.metadata["author"] == "test"
    
    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            FrameworkConfig(
                data_sources=[
                    DataSourceConfig(name="source1", type="csv")
                ],
                algorithms=[
                    AlgorithmConfig(name="alg1")
                ],
                invalid_field="should fail"
            )

