"""
Labeling tool interface for the ChartRadar framework.

This module provides a CLI/API interface for interactive and programmatic labeling.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from datetime import datetime

from chartradar.labeling.storage import LabelStorage
from chartradar.labeling.validator import LabelValidator
from chartradar.labeling.metadata import LabelMetadata
from chartradar.src.exceptions import LabelingError


class LabelingTool:
    """
    Tool for interactive and programmatic labeling.
    
    Provides interface for loading data, annotating patterns, and saving labels.
    """
    
    def __init__(
        self,
        storage_path: str = "./labels",
        interface: str = "api"
    ):
        """
        Initialize the labeling tool.
        
        Args:
            storage_path: Path to label storage
            interface: Interface type ('api' or 'cli')
        """
        self.storage = LabelStorage(storage_path=storage_path)
        self.validator = LabelValidator()
        self.metadata = LabelMetadata()
        self.interface = interface
        self.current_session: Optional[Dict[str, Any]] = None
    
    def start_labeling_session(
        self,
        labeler: str,
        dataset_name: str = "default"
    ) -> str:
        """
        Start a new labeling session.
        
        Args:
            labeler: Labeler identity
            dataset_name: Name of dataset being labeled
            
        Returns:
            Session ID
        """
        session_id = f"{dataset_name}_{labeler}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = {
            "session_id": session_id,
            "labeler": labeler,
            "dataset_name": dataset_name,
            "start_time": datetime.now(),
            "labels": []
        }
        
        return session_id
    
    def add_label(
        self,
        pattern_type: str,
        start_index: int,
        end_index: int,
        confidence: float = 1.0,
        notes: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Add a label to the current session.
        
        Args:
            pattern_type: Type of pattern (e.g., 'rising_wedge')
            start_index: Start index of pattern
            end_index: End index of pattern
            confidence: Labeling confidence (0.0-1.0)
            notes: Optional notes
            **kwargs: Additional label fields
            
        Returns:
            Created label dictionary
        """
        if not self.current_session:
            raise LabelingError(
                "No active labeling session. Call start_labeling_session() first.",
                details={}
            )
        
        label = {
            "pattern_type": pattern_type,
            "start_index": start_index,
            "end_index": end_index,
            "confidence": confidence,
            "start_timestamp": datetime.now().isoformat(),
            "end_timestamp": datetime.now().isoformat(),
            "metadata": self.metadata.create_label_metadata(
                labeler=self.current_session["labeler"],
                confidence=confidence,
                notes=notes,
                **kwargs
            )
        }
        
        self.current_session["labels"].append(label)
        
        return label
    
    def suggest_labels(
        self,
        data: pd.DataFrame,
        algorithm_results: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate label suggestions using rule-based or algorithm-based methods.
        
        Args:
            data: OHLCV data
            algorithm_results: Optional algorithm detection results
            
        Returns:
            List of suggested labels
        """
        suggestions = []
        
        # Use algorithm results if provided
        if algorithm_results:
            for result in algorithm_results:
                if hasattr(result, 'results'):
                    for pattern in result.results:
                        if hasattr(pattern, 'pattern_type'):
                            suggestions.append({
                                "pattern_type": pattern.pattern_type,
                                "start_index": pattern.start_index,
                                "end_index": pattern.end_index,
                                "confidence": pattern.confidence,
                                "source": "algorithm",
                                "algorithm_name": getattr(result, 'algorithm_name', 'unknown')
                            })
        
        # Simple rule-based suggestions (placeholder)
        # In practice, this would implement basic pattern detection rules
        
        return suggestions
    
    def save_session(self, validate: bool = True) -> str:
        """
        Save the current labeling session.
        
        Args:
            validate: Whether to validate labels before saving
            
        Returns:
            Path to saved label file
        """
        if not self.current_session:
            raise LabelingError(
                "No active labeling session to save",
                details={}
            )
        
        labels = self.current_session["labels"]
        
        # Validate if requested (only if there are labels to validate)
        if validate:
            if not labels:
                # Empty labels are valid (user might save an empty session)
                pass
            else:
                validation_result = self.validator.validate(labels, raise_on_error=False)
                if not validation_result["valid"]:
                    raise LabelingError(
                        f"Label validation failed: {validation_result['errors']}",
                        details=validation_result
                    )
        
        # Save labels
        dataset_name = self.current_session["dataset_name"]
        label_file = self.storage.save_labels(
            labels=labels,
            dataset_name=dataset_name,
            metadata={
                "session_id": self.current_session["session_id"],
                "labeler": self.current_session["labeler"],
                "total_labels": len(labels)
            }
        )
        
        return label_file
    
    def load_data_for_labeling(
        self,
        data_source: Any,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data for labeling.
        
        Args:
            data_source: Data source (DataSource instance or path)
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame with OHLCV data
        """
        if hasattr(data_source, 'load_data'):
            # DataSource instance
            return data_source.load_data(start_date=start_date, end_date=end_date)
        elif isinstance(data_source, str):
            # File path - try to load as CSV
            return pd.read_csv(data_source, index_col=0, parse_dates=True)
        else:
            raise LabelingError(
                f"Unsupported data source type: {type(data_source)}",
                details={"data_source_type": type(data_source).__name__}
            )
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get summary of current labeling session.
        
        Returns:
            Dictionary with session summary
        """
        if not self.current_session:
            return {"message": "No active session"}
        
        labels = self.current_session["labels"]
        pattern_types = {}
        for label in labels:
            pattern_type = label.get("pattern_type", "unknown")
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + 1
        
        duration = (datetime.now() - self.current_session["start_time"]).total_seconds()
        
        return {
            "session_id": self.current_session["session_id"],
            "labeler": self.current_session["labeler"],
            "dataset_name": self.current_session["dataset_name"],
            "total_labels": len(labels),
            "pattern_type_distribution": pattern_types,
            "duration_seconds": duration,
            "labels_per_minute": len(labels) / (duration / 60) if duration > 0 else 0.0
        }

