"""Tests for label storage."""

import pytest
import json
from pathlib import Path
from datetime import datetime

from chartradar.labeling.storage import LabelStorage
from chartradar.core.exceptions import LabelingError


class TestLabelStorage:
    """Tests for LabelStorage class."""
    
    def test_save_labels(self, tmp_path):
        """Test saving labels."""
        storage = LabelStorage(storage_path=str(tmp_path))
        
        labels = [
            {
                "pattern_type": "rising_wedge",
                "start_index": 0,
                "end_index": 10,
                "confidence": 0.8
            }
        ]
        
        label_file = storage.save_labels(labels, dataset_name="test_dataset")
        
        assert Path(label_file).exists()
        with open(label_file) as f:
            data = json.load(f)
            assert data["dataset_name"] == "test_dataset"
            assert len(data["labels"]) == 1
    
    def test_load_labels(self, tmp_path):
        """Test loading labels."""
        storage = LabelStorage(storage_path=str(tmp_path))
        
        labels = [{"pattern_type": "test", "start_index": 0, "end_index": 10}]
        storage.save_labels(labels, dataset_name="test")
        
        loaded = storage.load_labels("test")
        assert loaded["dataset_name"] == "test"
        assert len(loaded["labels"]) == 1
    
    def test_load_labels_not_found(self, tmp_path):
        """Test loading non-existent labels."""
        storage = LabelStorage(storage_path=str(tmp_path))
        
        with pytest.raises(LabelingError) as exc_info:
            storage.load_labels("nonexistent")
        # Check for either "not found" or "no labels found"
        error_msg = str(exc_info.value).lower()
        assert "not found" in error_msg or "no labels found" in error_msg
    
    def test_update_labels(self, tmp_path):
        """Test updating labels."""
        storage = LabelStorage(storage_path=str(tmp_path))
        
        labels = [{"pattern_type": "test", "start_index": 0, "end_index": 10}]
        storage.save_labels(labels, dataset_name="test")
        
        updates = [{"pattern_type": "test", "start_index": 0, "end_index": 10, "confidence": 0.9}]
        updated_file = storage.update_labels("test", updates)
        
        assert Path(updated_file).exists()
    
    def test_query_labels(self, tmp_path):
        """Test querying labels."""
        storage = LabelStorage(storage_path=str(tmp_path))
        
        labels = [
            {
                "pattern_type": "rising_wedge",
                "start_index": 0,
                "end_index": 10,
                "start_timestamp": "2024-01-01T00:00:00",
                "confidence": 0.8
            },
            {
                "pattern_type": "falling_wedge",
                "start_index": 20,
                "end_index": 30,
                "start_timestamp": "2024-01-02T00:00:00",
                "confidence": 0.9
            }
        ]
        
        storage.save_labels(labels, dataset_name="test")
        
        # Query by pattern type
        results = storage.query_labels("test", pattern_type="rising_wedge")
        assert len(results) == 1
        
        # Query by confidence
        results = storage.query_labels("test", min_confidence=0.85)
        assert len(results) == 1
    
    def test_list_datasets(self, tmp_path):
        """Test listing datasets."""
        storage = LabelStorage(storage_path=str(tmp_path))
        
        storage.save_labels([], dataset_name="dataset1")
        storage.save_labels([], dataset_name="dataset2")
        
        datasets = storage.list_datasets()
        assert "dataset1" in datasets
        assert "dataset2" in datasets

