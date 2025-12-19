"""
Metadata tracking for labels in the ChartRadar framework.

This module provides functionality to track labeler identity, timestamps,
confidence, and support for multiple labelers and consensus tracking.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import Counter


class LabelMetadata:
    """
    Metadata tracker for labels.
    
    Tracks labeler identity, timestamps, confidence, and consensus.
    """
    
    def __init__(self, **kwargs: Any):
        """
        Initialize the metadata tracker.
        
        Args:
            **kwargs: Metadata-specific parameters
        """
        self.parameters = kwargs
    
    def create_label_metadata(
        self,
        labeler: str,
        confidence: Optional[float] = None,
        notes: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Create metadata for a label.
        
        Args:
            labeler: Identity of the labeler
            confidence: Labeling confidence (0.0-1.0)
            notes: Optional notes about the label
            **kwargs: Additional metadata fields
            
        Returns:
            Dictionary with label metadata
        """
        metadata = {
            "labeler": labeler,
            "created_at": datetime.now().isoformat(),
            **kwargs
        }
        
        if confidence is not None:
            metadata["labeling_confidence"] = confidence
        
        if notes:
            metadata["notes"] = notes
        
        return metadata
    
    def add_consensus(
        self,
        label: Dict[str, Any],
        additional_labelers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Add consensus information from multiple labelers.
        
        Args:
            label: Original label dictionary
            additional_labelers: List of labels from other labelers for the same pattern
            
        Returns:
            Label dictionary with consensus metadata
        """
        all_labelers = [label.get("metadata", {}).get("labeler", "unknown")]
        all_labelers.extend([
            lbl.get("metadata", {}).get("labeler", "unknown")
            for lbl in additional_labelers
        ])
        
        # Calculate consensus
        pattern_types = [label.get("pattern_type")]
        pattern_types.extend([lbl.get("pattern_type") for lbl in additional_labelers])
        
        consensus_type = Counter(pattern_types).most_common(1)[0][0]
        agreement_count = pattern_types.count(consensus_type)
        total_count = len(pattern_types)
        consensus_score = agreement_count / total_count if total_count > 0 else 0.0
        
        # Update label metadata
        if "metadata" not in label:
            label["metadata"] = {}
        
        label["metadata"]["consensus"] = {
            "labelers": all_labelers,
            "total_labelers": len(all_labelers),
            "agreement_count": agreement_count,
            "consensus_score": consensus_score,
            "consensus_pattern_type": consensus_type
        }
        
        return label
    
    def track_labeling_session(
        self,
        session_id: str,
        labeler: str,
        labels_created: int,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Track a labeling session.
        
        Args:
            session_id: Unique session identifier
            labeler: Labeler identity
            labels_created: Number of labels created in this session
            start_time: Session start time
            end_time: Session end time (current time if None)
            
        Returns:
            Dictionary with session metadata
        """
        if end_time is None:
            end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        return {
            "session_id": session_id,
            "labeler": labeler,
            "labels_created": labels_created,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "labels_per_minute": labels_created / (duration / 60) if duration > 0 else 0.0
        }
    
    def get_labeler_statistics(
        self,
        labels: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get statistics for labelers.
        
        Args:
            labels: List of labels
            
        Returns:
            Dictionary with labeler statistics
        """
        labeler_counts = Counter()
        labeler_confidences = {}
        
        for label in labels:
            metadata = label.get("metadata", {})
            labeler = metadata.get("labeler", "unknown")
            labeler_counts[labeler] += 1
            
            if labeler not in labeler_confidences:
                labeler_confidences[labeler] = []
            
            confidence = metadata.get("labeling_confidence")
            if confidence is not None:
                labeler_confidences[labeler].append(confidence)
        
        statistics = {}
        for labeler, count in labeler_counts.items():
            confidences = labeler_confidences.get(labeler, [])
            statistics[labeler] = {
                "total_labels": count,
                "average_confidence": (
                    sum(confidences) / len(confidences) if confidences else None
                ),
                "min_confidence": min(confidences) if confidences else None,
                "max_confidence": max(confidences) if confidences else None
            }
        
        return statistics

