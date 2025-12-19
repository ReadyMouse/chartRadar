"""Tests for labeling tool."""

import pytest
import pandas as pd

from chartradar.labeling.tool import LabelingTool
from chartradar.core.exceptions import LabelingError


class TestLabelingTool:
    """Tests for LabelingTool class."""
    
    def test_start_labeling_session(self, tmp_path):
        """Test starting a labeling session."""
        tool = LabelingTool(storage_path=str(tmp_path))
        
        session_id = tool.start_labeling_session("user1", "test_dataset")
        
        assert session_id is not None
        assert tool.current_session is not None
        assert tool.current_session["labeler"] == "user1"
    
    def test_add_label(self, tmp_path):
        """Test adding a label."""
        tool = LabelingTool(storage_path=str(tmp_path))
        tool.start_labeling_session("user1")
        
        label = tool.add_label(
            pattern_type="rising_wedge",
            start_index=0,
            end_index=10,
            confidence=0.8
        )
        
        assert label["pattern_type"] == "rising_wedge"
        assert len(tool.current_session["labels"]) == 1
    
    def test_add_label_no_session(self, tmp_path):
        """Test adding label without active session."""
        tool = LabelingTool(storage_path=str(tmp_path))
        
        with pytest.raises(LabelingError) as exc_info:
            tool.add_label("rising_wedge", 0, 10)
        assert "No active labeling session" in str(exc_info.value)
    
    def test_save_session(self, tmp_path):
        """Test saving labeling session."""
        tool = LabelingTool(storage_path=str(tmp_path))
        tool.start_labeling_session("user1", "test_dataset")
        
        tool.add_label("rising_wedge", 0, 10)
        
        label_file = tool.save_session()
        assert Path(label_file).exists()
    
    def test_get_session_summary(self, tmp_path):
        """Test getting session summary."""
        tool = LabelingTool(storage_path=str(tmp_path))
        tool.start_labeling_session("user1")
        
        tool.add_label("rising_wedge", 0, 10)
        tool.add_label("falling_wedge", 20, 30)
        
        summary = tool.get_session_summary()
        
        assert summary["total_labels"] == 2
        assert "pattern_type_distribution" in summary
    
    def test_suggest_labels(self, tmp_path):
        """Test generating label suggestions."""
        tool = LabelingTool(storage_path=str(tmp_path))
        
        data = pd.DataFrame({'feature1': range(100)})
        
        suggestions = tool.suggest_labels(data)
        
        # Should return a list (may be empty)
        assert isinstance(suggestions, list)

