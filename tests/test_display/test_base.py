"""Tests for display base class."""

import pytest
import pandas as pd
from datetime import datetime

from chartradar.display.base import Display
from chartradar.core.exceptions import DisplayError


class ConcreteDisplay(Display):
    """Concrete display implementation for testing."""
    
    def visualize(self, results, data=None, **kwargs):
        """Test visualization."""
        return {"visualized": True}
    
    def export(self, results, format, output_path=None, **kwargs):
        """Test export."""
        if output_path:
            return output_path
        return "exported_data"


class TestDisplay:
    """Tests for Display base class."""
    
    def test_initialization(self):
        """Test display initialization."""
        display = ConcreteDisplay("test_display", param1="value1")
        assert display.name == "test_display"
        assert display.parameters["param1"] == "value1"
    
    def test_visualize(self):
        """Test visualization."""
        display = ConcreteDisplay("test")
        result = display.visualize({"test": "data"})
        assert result["visualized"] is True
    
    def test_export(self):
        """Test export."""
        display = ConcreteDisplay("test")
        result = display.export({"test": "data"}, "json")
        assert result == "exported_data"
    
    def test_export_with_path(self, tmp_path):
        """Test export with output path."""
        display = ConcreteDisplay("test")
        output_file = tmp_path / "output.json"
        result = display.export({"test": "data"}, "json", str(output_file))
        assert result == str(output_file)

