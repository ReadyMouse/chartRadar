"""Tests for exporter."""

import pytest
import json
from pathlib import Path
from datetime import datetime

from chartradar.display.exporter import Exporter
from chartradar.src.types import AlgorithmResult, PatternDetection
from chartradar.src.exceptions import DisplayError


class TestExporter:
    """Tests for Exporter class."""
    
    def test_export_json(self, tmp_path):
        """Test JSON export."""
        exporter = Exporter()
        data = {"test": "data", "value": 123}
        
        output_file = tmp_path / "output.json"
        result = exporter.export_json(data, str(output_file))
        
        assert result == str(output_file)
        assert output_file.exists()
        
        with open(output_file) as f:
            loaded = json.load(f)
        assert loaded == data
    
    def test_export_json_no_path(self):
        """Test JSON export without path (returns string)."""
        exporter = Exporter()
        data = {"test": "data"}
        
        result = exporter.export_json(data)
        assert isinstance(result, str)
        loaded = json.loads(result)
        assert loaded == data
    
    def test_export_csv(self, tmp_path):
        """Test CSV export."""
        exporter = Exporter()
        
        result = AlgorithmResult(
            algorithm_name="test_alg",
            success=True,
            results=[
                PatternDetection(
                    pattern_type="rising_wedge",
                    confidence=0.8,
                    start_index=0,
                    end_index=10,
                    start_timestamp=datetime.now(),
                    end_timestamp=datetime.now()
                )
            ],
            confidence_scores=[0.8],
            timestamp=datetime.now()
        )
        
        output_file = tmp_path / "output.csv"
        result_path = exporter.export_csv(result, str(output_file))
        
        assert result_path == str(output_file)
        assert output_file.exists()
    
    def test_export_unsupported_format(self):
        """Test export with unsupported format."""
        exporter = Exporter()
        
        with pytest.raises(DisplayError) as exc_info:
            exporter.export({"test": "data"}, "unsupported")
        assert "Unsupported export format" in str(exc_info.value)
    
    def test_export_image_requires_path(self):
        """Test that image export requires output path."""
        exporter = Exporter()
        
        # Mock figure object
        class MockFigure:
            def savefig(self, path, **kwargs):
                pass
        
        with pytest.raises(DisplayError) as exc_info:
            exporter.export_image(MockFigure(), "png")
        assert "output_path is required" in str(exc_info.value)

