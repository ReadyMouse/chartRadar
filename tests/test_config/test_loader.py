"""Tests for configuration loader."""

import os
import tempfile
from pathlib import Path
import pytest
import yaml

from chartradar.core.exceptions import ConfigurationError
from chartradar.config.loader import (
    load_yaml_file,
    load_config,
    merge_configs,
    substitute_env_vars,
)
from chartradar.config.schema import FrameworkConfig


class TestLoadYamlFile:
    """Tests for load_yaml_file function."""
    
    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid YAML file."""
        yaml_content = """
        version: "1.0"
        name: "Test Config"
        data_sources:
          - name: "test"
            type: "csv"
        algorithms:
          - name: "test_alg"
        """
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        
        result = load_yaml_file(yaml_file)
        assert result["version"] == "1.0"
        assert result["name"] == "Test Config"
    
    def test_file_not_found(self):
        """Test that missing file raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_yaml_file("/nonexistent/file.yaml")
        assert "not found" in str(exc_info.value).lower()
    
    def test_invalid_yaml(self, tmp_path):
        """Test that invalid YAML raises error."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(ConfigurationError) as exc_info:
            load_yaml_file(yaml_file)
        assert "Failed to parse" in str(exc_info.value)


class TestSubstituteEnvVars:
    """Tests for environment variable substitution."""
    
    def test_simple_substitution(self):
        """Test simple environment variable substitution."""
        os.environ["TEST_VAR"] = "test_value"
        try:
            result = substitute_env_vars("${TEST_VAR}")
            assert result == "test_value"
        finally:
            del os.environ["TEST_VAR"]
    
    def test_substitution_with_default(self):
        """Test substitution with default value."""
        result = substitute_env_vars("${NONEXISTENT_VAR:-default_value}")
        assert result == "default_value"
    
    def test_missing_var_no_default(self):
        """Test that missing variable without default raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            substitute_env_vars("${NONEXISTENT_VAR}")
        assert "not set" in str(exc_info.value).lower()
    
    def test_nested_substitution(self):
        """Test substitution in nested structures."""
        os.environ["BASE_PATH"] = "/data"
        try:
            config = {
                "path": "${BASE_PATH}/subdir",
                "nested": {
                    "value": "${BASE_PATH}/nested"
                }
            }
            result = substitute_env_vars(config)
            assert result["path"] == "/data/subdir"
            assert result["nested"]["value"] == "/data/nested"
        finally:
            del os.environ["BASE_PATH"]
    
    def test_list_substitution(self):
        """Test substitution in lists."""
        os.environ["ITEM"] = "value"
        try:
            config = ["${ITEM}", "static"]
            result = substitute_env_vars(config)
            assert result == ["value", "static"]
        finally:
            del os.environ["ITEM"]


class TestMergeConfigs:
    """Tests for config merging."""
    
    def test_simple_merge(self):
        """Test merging two simple configs."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 3, "c": 4}
        result = merge_configs([config1, config2])
        assert result["a"] == 1
        assert result["b"] == 3  # Overridden
        assert result["c"] == 4
    
    def test_deep_merge(self):
        """Test deep merging of nested structures."""
        config1 = {"nested": {"a": 1, "b": 2}}
        config2 = {"nested": {"b": 3, "c": 4}}
        result = merge_configs([config1, config2])
        assert result["nested"]["a"] == 1
        assert result["nested"]["b"] == 3
        assert result["nested"]["c"] == 4
    
    def test_list_merge(self):
        """Test merging lists (appends by default)."""
        config1 = {"items": [1, 2]}
        config2 = {"items": [3, 4]}
        result = merge_configs([config1, config2])
        assert result["items"] == [1, 2, 3, 4]
    
    def test_list_replacement(self):
        """Test replacing lists with ! prefix."""
        config1 = {"items": [1, 2]}
        config2 = {"!items": [3, 4]}
        result = merge_configs([config1, config2])
        assert result["items"] == [3, 4]


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading a valid configuration."""
        yaml_content = """
        version: "1.0"
        data_sources:
          - name: "test_source"
            type: "csv"
            enabled: true
        algorithms:
          - name: "test_alg"
            enabled: true
        """
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        
        config = load_config(yaml_file)
        assert isinstance(config, FrameworkConfig)
        assert len(config.data_sources) == 1
        assert len(config.algorithms) == 1
    
    def test_load_with_env_substitution(self, tmp_path):
        """Test loading config with environment variable substitution."""
        os.environ["DATA_PATH"] = "/custom/path"
        try:
            yaml_content = """
            version: "1.0"
            data_sources:
              - name: "test"
                type: "csv"
                enabled: true
                parameters:
                  path: "${DATA_PATH}/data.csv"
            algorithms:
              - name: "test_alg"
                enabled: true
            """
            yaml_file = tmp_path / "config.yaml"
            yaml_file.write_text(yaml_content)
            
            config = load_config(yaml_file, substitute_env=True)
            assert config.data_sources[0].parameters["path"] == "/custom/path/data.csv"
        finally:
            del os.environ["DATA_PATH"]
    
    def test_load_multiple_files(self, tmp_path):
        """Test loading and merging multiple config files."""
        base_config = tmp_path / "base.yaml"
        base_config.write_text("""
        version: "1.0"
        name: "Base"
        data_sources:
          - name: "source1"
            type: "csv"
            enabled: true
        algorithms:
          - name: "alg1"
            enabled: true
        """)
        
        override_config = tmp_path / "override.yaml"
        override_config.write_text("""
        name: "Override"
        algorithms:
          - name: "alg2"
            enabled: true
        """)
        
        config = load_config([base_config, override_config])
        assert config.name == "Override"  # Overridden
        assert len(config.data_sources) == 1  # From base
        assert len(config.algorithms) == 2  # Both alg1 and alg2
    
    def test_invalid_config_validation(self, tmp_path):
        """Test that invalid config raises validation error."""
        yaml_content = """
        version: "1.0"
        data_sources: []
        algorithms: []
        """
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text(yaml_content)
        
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(yaml_file)
        assert "validation failed" in str(exc_info.value).lower()
    
    def test_load_without_validation(self, tmp_path):
        """Test loading config without validation."""
        yaml_content = """
        version: "1.0"
        data_sources: []
        algorithms: []
        """
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        
        # Should not raise error when validate=False
        config = load_config(yaml_file, validate=False)
        assert isinstance(config, FrameworkConfig)

