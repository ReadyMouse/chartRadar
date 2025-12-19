"""
Training data export for the ChartRadar framework.

This module provides functionality to export labels in formats compatible
with ML frameworks, including JSON, HDF5, and Parquet.
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

from chartradar.core.exceptions import LabelingError
from chartradar.training.split import DataSplitter


class LabelExporter:
    """
    Exporter for training data.
    
    Exports labels in formats compatible with ML frameworks.
    """
    
    def __init__(self, **kwargs: Any):
        """
        Initialize the label exporter.
        
        Args:
            **kwargs: Exporter-specific parameters
        """
        self.parameters = kwargs
    
    def export(
        self,
        labels: List[Dict[str, Any]],
        data: Optional[pd.DataFrame] = None,
        format: str = "json",
        output_path: str = "./training_data",
        include_data: bool = True,
        create_splits: bool = False,
        split_ratios: Optional[Dict[str, float]] = None
    ) -> Dict[str, str]:
        """
        Export labels for training.
        
        Args:
            labels: List of label dictionaries
            data: Optional associated OHLCV data
            format: Export format ('json', 'hdf5', 'parquet')
            output_path: Base path for output files
            include_data: Whether to include data segments with labels
            create_splits: Whether to create train/val/test splits
            split_ratios: Ratios for splitting (if create_splits is True)
            
        Returns:
            Dictionary with paths to exported files
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if create_splits:
            return self._export_with_splits(
                labels, data, format, output_path,
                include_data, split_ratios
            )
        else:
            return self._export_single(
                labels, data, format, output_path, include_data
            )
    
    def _export_single(
        self,
        labels: List[Dict[str, Any]],
        data: Optional[pd.DataFrame],
        format: str,
        output_path: Path,
        include_data: bool
    ) -> Dict[str, str]:
        """Export labels as a single dataset."""
        if format == "json":
            return self._export_json(labels, data, output_path, include_data)
        elif format == "hdf5":
            return self._export_hdf5(labels, data, output_path, include_data)
        elif format == "parquet":
            return self._export_parquet(labels, data, output_path, include_data)
        else:
            raise LabelingError(
                f"Unsupported export format: {format}",
                details={"format": format, "supported": ["json", "hdf5", "parquet"]}
            )
    
    def _export_with_splits(
        self,
        labels: List[Dict[str, Any]],
        data: Optional[pd.DataFrame],
        format: str,
        output_path: Path,
        include_data: bool,
        split_ratios: Optional[Dict[str, float]]
    ) -> Dict[str, str]:
        """Export labels with train/val/test splits."""
        # Convert labels to DataFrame for splitting
        labels_df = pd.DataFrame(labels)
        
        if "start_index" in labels_df.columns and data is not None:
            # Create labels series based on data indices
            label_series = pd.Series(index=data.index, dtype=object)
            for label in labels:
                start_idx = label.get("start_index", 0)
                end_idx = label.get("end_index", 0)
                if start_idx < len(data) and end_idx < len(data):
                    label_series.iloc[start_idx:end_idx+1] = label.get("pattern_type")
        else:
            label_series = None
        
        # Split data
        splitter = DataSplitter()
        train_data, val_data, test_data, train_labels, val_labels, test_labels = \
            splitter.split_train_val_test(
                data if data is not None else labels_df,
                label_series,
                ratios=split_ratios,
                time_series=True
            )
        
        # Export each split
        exported_files = {}
        
        for split_name, split_labels in [
            ("train", train_labels),
            ("validation", val_labels),
            ("test", test_labels)
        ]:
            if split_labels is not None and len(split_labels) > 0:
                # Convert back to label format
                split_label_dicts = self._labels_series_to_dicts(split_labels, split_name)
                split_path = output_path / split_name
                split_path.mkdir(exist_ok=True)
                
                split_files = self._export_single(
                    split_label_dicts, None, format, split_path, include_data
                )
                exported_files.update({f"{split_name}_{k}": v for k, v in split_files.items()})
        
        return exported_files
    
    def _labels_series_to_dicts(
        self,
        labels_series: pd.Series,
        split_name: str
    ) -> List[Dict[str, Any]]:
        """Convert labels series to list of dictionaries."""
        # This is a simplified conversion
        # In practice, you'd want to preserve more label information
        labels = []
        for idx, pattern_type in labels_series.items():
            if pd.notna(pattern_type):
                labels.append({
                    "pattern_type": str(pattern_type),
                    "start_index": int(idx) if isinstance(idx, (int, float)) else 0,
                    "end_index": int(idx) if isinstance(idx, (int, float)) else 0,
                    "split": split_name
                })
        return labels
    
    def _export_json(
        self,
        labels: List[Dict[str, Any]],
        data: Optional[pd.DataFrame],
        output_path: Path,
        include_data: bool
    ) -> Dict[str, str]:
        """Export to JSON format."""
        export_data = {
            "labels": labels,
            "total_labels": len(labels)
        }
        
        if include_data and data is not None:
            export_data["data"] = data.to_dict(orient="records")
        
        output_file = output_path / "labels.json"
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return {"labels": str(output_file)}
    
    def _export_hdf5(
        self,
        labels: List[Dict[str, Any]],
        data: Optional[pd.DataFrame],
        output_path: Path,
        include_data: bool
    ) -> Dict[str, str]:
        """Export to HDF5 format."""
        if not HDF5_AVAILABLE:
            raise LabelingError(
                "h5py is not installed. Install it with: pip install h5py",
                details={"package": "h5py"}
            )
        
        output_file = output_path / "labels.h5"
        
        with h5py.File(output_file, 'w') as f:
            # Store labels as structured array
            labels_group = f.create_group("labels")
            for i, label in enumerate(labels):
                label_group = labels_group.create_group(f"label_{i}")
                for key, value in label.items():
                    if isinstance(value, (str, int, float)):
                        label_group.attrs[key] = value
            
            # Store data if provided
            if include_data and data is not None:
                data_group = f.create_group("data")
                for col in data.columns:
                    data_group.create_dataset(col, data=data[col].values)
        
        return {"labels": str(output_file)}
    
    def _export_parquet(
        self,
        labels: List[Dict[str, Any]],
        data: Optional[pd.DataFrame],
        output_path: Path,
        include_data: bool
    ) -> Dict[str, str]:
        """Export to Parquet format."""
        labels_df = pd.DataFrame(labels)
        
        labels_file = output_path / "labels.parquet"
        labels_df.to_parquet(labels_file)
        
        exported_files = {"labels": str(labels_file)}
        
        if include_data and data is not None:
            data_file = output_path / "data.parquet"
            data.to_parquet(data_file)
            exported_files["data"] = str(data_file)
        
        return exported_files

