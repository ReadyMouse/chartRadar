"""
Label storage for the ChartRadar framework.

This module provides functionality to save, load, update, and query labels
in a structured JSON-based format with versioning support.
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import hashlib

from chartradar.core.exceptions import LabelingError
from chartradar.core.types import PatternDetection


class LabelStorage:
    """
    Storage system for pattern labels.
    
    Uses JSON-based format with metadata and versioning support.
    """
    
    def __init__(
        self,
        storage_path: str = "./labels",
        storage_format: str = "json"
    ):
        """
        Initialize the label storage.
        
        Args:
            storage_path: Directory to store labels
            storage_format: Storage format ('json' for now)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.storage_format = storage_format
    
    def save_labels(
        self,
        labels: List[Dict[str, Any]],
        dataset_name: str = "default",
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save labels to storage.
        
        Args:
            labels: List of label dictionaries
            dataset_name: Name identifier for this dataset
            version: Optional version string (auto-generated if None)
            metadata: Optional metadata dictionary
            
        Returns:
            Path to saved label file
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create label structure
        label_data = {
            "dataset_name": dataset_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "total_labels": len(labels),
            "labels": labels,
            "metadata": metadata or {}
        }
        
        # Generate filename
        filename = f"{dataset_name}_v{version}.json"
        label_file = self.storage_path / filename
        
        # Save to file
        try:
            with open(label_file, 'w') as f:
                json.dump(label_data, f, indent=2, default=str)
        except Exception as e:
            raise LabelingError(
                f"Failed to save labels: {str(e)}",
                details={"label_file": str(label_file), "error": str(e)}
            ) from e
        
        return str(label_file)
    
    def load_labels(
        self,
        dataset_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load labels from storage.
        
        Args:
            dataset_name: Name of dataset to load
            version: Optional version (loads latest if None)
            
        Returns:
            Dictionary with label data
        """
        if version:
            filename = f"{dataset_name}_v{version}.json"
            label_file = self.storage_path / filename
        else:
            # Find latest version
            pattern = f"{dataset_name}_v*.json"
            label_files = list(self.storage_path.glob(pattern))
            if not label_files:
                raise LabelingError(
                    f"No labels found for dataset '{dataset_name}'",
                    details={"dataset_name": dataset_name}
                )
            label_file = max(label_files, key=lambda p: p.stat().st_mtime)
        
        if not label_file.exists():
            raise LabelingError(
                f"Label file not found: {label_file}",
                details={"label_file": str(label_file)}
            )
        
        try:
            with open(label_file, 'r') as f:
                label_data = json.load(f)
            return label_data
        except Exception as e:
            raise LabelingError(
                f"Failed to load labels: {str(e)}",
                details={"label_file": str(label_file), "error": str(e)}
            ) from e
    
    def update_labels(
        self,
        dataset_name: str,
        label_updates: List[Dict[str, Any]],
        create_new_version: bool = True
    ) -> str:
        """
        Update existing labels.
        
        Args:
            dataset_name: Name of dataset to update
            label_updates: List of label updates (must include 'id' or unique identifier)
            create_new_version: Whether to create a new version or update existing
            
        Returns:
            Path to updated label file
        """
        # Load existing labels
        existing_data = self.load_labels(dataset_name)
        existing_labels = existing_data["labels"]
        
        # Create label lookup
        label_dict = {self._get_label_id(label): label for label in existing_labels}
        
        # Apply updates
        for update in label_updates:
            label_id = self._get_label_id(update)
            if label_id in label_dict:
                # Update existing label
                label_dict[label_id].update(update)
                label_dict[label_id]["updated_at"] = datetime.now().isoformat()
            else:
                # Add new label
                update["created_at"] = datetime.now().isoformat()
                label_dict[label_id] = update
        
        # Save updated labels
        if create_new_version:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            version = existing_data["version"]
        
        updated_labels = list(label_dict.values())
        return self.save_labels(
            updated_labels,
            dataset_name=dataset_name,
            version=version,
            metadata=existing_data.get("metadata", {})
        )
    
    def _get_label_id(self, label: Dict[str, Any]) -> str:
        """Generate a unique ID for a label."""
        # Use pattern boundaries and type as ID
        key_parts = [
            str(label.get("pattern_type", "")),
            str(label.get("start_index", "")),
            str(label.get("end_index", ""))
        ]
        key_str = "_".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def query_labels(
        self,
        dataset_name: str,
        pattern_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        labeler: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Query labels by various criteria.
        
        Args:
            dataset_name: Name of dataset
            pattern_type: Filter by pattern type
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            labeler: Filter by labeler identity
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching labels
        """
        label_data = self.load_labels(dataset_name)
        labels = label_data["labels"]
        
        # Apply filters
        filtered = []
        for label in labels:
            # Pattern type filter
            if pattern_type and label.get("pattern_type") != pattern_type:
                continue
            
            # Date range filter
            if start_date or end_date:
                label_start = label.get("start_timestamp")
                if label_start:
                    label_start = datetime.fromisoformat(label_start) if isinstance(label_start, str) else label_start
                    if start_date:
                        start_dt = datetime.fromisoformat(start_date)
                        if label_start < start_dt:
                            continue
                    if end_date:
                        end_dt = datetime.fromisoformat(end_date)
                        if label_start > end_dt:
                            continue
            
            # Labeler filter
            if labeler:
                label_metadata = label.get("metadata", {})
                if label_metadata.get("labeler") != labeler:
                    continue
            
            # Confidence filter
            if min_confidence is not None:
                confidence = label.get("confidence", 0.0)
                if confidence < min_confidence:
                    continue
            
            filtered.append(label)
        
        return filtered
    
    def list_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
            List of dataset names
        """
        datasets = set()
        for label_file in self.storage_path.glob("*_v*.json"):
            # Extract dataset name (everything before _v)
            dataset_name = label_file.stem.rsplit("_v", 1)[0]
            datasets.add(dataset_name)
        
        return sorted(list(datasets))
    
    def list_versions(self, dataset_name: str) -> List[str]:
        """
        List all versions for a dataset.
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            List of version strings
        """
        pattern = f"{dataset_name}_v*.json"
        label_files = list(self.storage_path.glob(pattern))
        
        versions = []
        for label_file in label_files:
            # Extract version from filename
            version = label_file.stem.rsplit("_v", 1)[1]
            versions.append(version)
        
        return sorted(versions)

