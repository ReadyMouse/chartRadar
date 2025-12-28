"""Tests for label exporter."""

import pytest
import pandas as pd
from pathlib import Path

from chartradar.labeling.exporter import LabelExporter
from chartradar.src.exceptions import LabelingError


class TestLabelExporter:
    """Tests for LabelExporter class."""
    
    def test_export_json(self, tmp_path):
        """Test exporting to JSON."""
        exporter = LabelExporter()
        
        labels = [
            {
                "pattern_type": "rising_wedge",
                "start_index": 0,
                "end_index": 10
            }
        ]
        
        files = exporter.export(
            labels,
            format="json",
            output_path=str(tmp_path)
        )
        
        assert "labels" in files
        assert Path(files["labels"]).exists()
    
    def test_export_parquet(self, tmp_path):
        """Test exporting to Parquet."""
        exporter = LabelExporter()
        
        labels = [
            {
                "pattern_type": "rising_wedge",
                "start_index": 0,
                "end_index": 10
            }
        ]
        
        files = exporter.export(
            labels,
            format="parquet",
            output_path=str(tmp_path)
        )
        
        assert "labels" in files
        assert Path(files["labels"]).exists()
    
    def test_export_unsupported_format(self):
        """Test export with unsupported format."""
        exporter = LabelExporter()
        
        with pytest.raises(LabelingError) as exc_info:
            exporter.export([], format="unsupported")
        assert "Unsupported export format" in str(exc_info.value)
    
    def test_export_with_splits(self, tmp_path):
        """Test exporting with train/val/test splits."""
        exporter = LabelExporter()
        
        labels = [
            {"pattern_type": "test", "start_index": i, "end_index": i+1}
            for i in range(100)
        ]
        
        data = pd.DataFrame({'feature1': range(100)})
        
        files = exporter.export(
            labels,
            data=data,
            format="json",
            output_path=str(tmp_path),
            create_splits=True
        )
        
        # Should have files for each split
        assert len(files) > 0

