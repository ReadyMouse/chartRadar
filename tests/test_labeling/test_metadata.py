"""Tests for label metadata."""

from chartradar.labeling.metadata import LabelMetadata


class TestLabelMetadata:
    """Tests for LabelMetadata class."""
    
    def test_create_label_metadata(self):
        """Test creating label metadata."""
        metadata = LabelMetadata()
        
        label_meta = metadata.create_label_metadata(
            labeler="user1",
            confidence=0.9,
            notes="Clear pattern"
        )
        
        assert label_meta["labeler"] == "user1"
        assert label_meta["labeling_confidence"] == 0.9
        assert "created_at" in label_meta
    
    def test_add_consensus(self):
        """Test adding consensus information."""
        metadata = LabelMetadata()
        
        label = {
            "pattern_type": "rising_wedge",
            "metadata": metadata.create_label_metadata("user1")
        }
        
        additional = [
            {
                "pattern_type": "rising_wedge",
                "metadata": metadata.create_label_metadata("user2")
            },
            {
                "pattern_type": "rising_wedge",
                "metadata": metadata.create_label_metadata("user3")
            }
        ]
        
        label_with_consensus = metadata.add_consensus(label, additional)
        
        assert "consensus" in label_with_consensus["metadata"]
        assert label_with_consensus["metadata"]["consensus"]["total_labelers"] == 3
    
    def test_get_labeler_statistics(self):
        """Test getting labeler statistics."""
        metadata = LabelMetadata()
        
        labels = [
            {
                "metadata": metadata.create_label_metadata("user1", confidence=0.8)
            },
            {
                "metadata": metadata.create_label_metadata("user1", confidence=0.9)
            },
            {
                "metadata": metadata.create_label_metadata("user2", confidence=0.7)
            }
        ]
        
        stats = metadata.get_labeler_statistics(labels)
        
        assert "user1" in stats
        assert stats["user1"]["total_labels"] == 2
        assert stats["user1"]["average_confidence"] == 0.85

