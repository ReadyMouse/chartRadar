"""
Data splitting utilities for the ChartRadar framework.

This module provides functions for splitting data into train/validation/test sets
with support for time-series aware splitting, cross-validation, and stratification.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from chartradar.src.exceptions import TrainingError


class DataSplitter:
    """
    Utility class for splitting data into training, validation, and test sets.
    
    Supports time-series aware splitting, cross-validation, and stratification.
    """
    
    def __init__(self, **kwargs: Any):
        """
        Initialize the data splitter.
        
        Args:
            **kwargs: Splitter-specific parameters
        """
        self.parameters = kwargs
    
    def split_train_val_test(
        self,
        data: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        ratios: Optional[Dict[str, float]] = None,
        shuffle: bool = False,
        stratify: bool = False,
        time_series: bool = True,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: DataFrame to split
            labels: Optional labels for supervised learning
            ratios: Dictionary with 'train', 'validation', 'test' ratios (default: 0.7, 0.15, 0.15)
            shuffle: Whether to shuffle data (ignored for time-series)
            stratify: Whether to use stratified splitting (requires labels)
            time_series: Whether to use time-series aware splitting
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data, train_labels, val_labels, test_labels)
        """
        if ratios is None:
            ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}
        
        # Validate ratios
        total = sum(ratios.values())
        if abs(total - 1.0) > 0.01:
            raise TrainingError(
                f"Split ratios must sum to 1.0, got {total}",
                details={"ratios": ratios}
            )
        
        n_samples = len(data)
        
        if time_series:
            # Time-series aware splitting (no shuffling, sequential)
            train_end = int(n_samples * ratios["train"])
            val_end = train_end + int(n_samples * ratios["validation"])
            
            train_data = data.iloc[:train_end]
            val_data = data.iloc[train_end:val_end]
            test_data = data.iloc[val_end:]
            
            if labels is not None:
                train_labels = labels.iloc[:train_end]
                val_labels = labels.iloc[train_end:val_end]
                test_labels = labels.iloc[val_end:]
            else:
                train_labels = val_labels = test_labels = None
        else:
            # Standard splitting with optional shuffling
            if shuffle:
                indices = np.arange(n_samples)
                if random_state is not None:
                    np.random.seed(random_state)
                np.random.shuffle(indices)
                data = data.iloc[indices]
                if labels is not None:
                    labels = labels.iloc[indices]
            
            if stratify and labels is not None:
                # Use stratified splitting
                from sklearn.model_selection import train_test_split
                
                # First split: train vs (val + test)
                train_data, temp_data, train_labels, temp_labels = train_test_split(
                    data, labels,
                    test_size=(ratios["validation"] + ratios["test"]),
                    stratify=labels,
                    random_state=random_state
                )
                
                # Second split: val vs test
                val_ratio = ratios["validation"] / (ratios["validation"] + ratios["test"])
                val_data, test_data, val_labels, test_labels = train_test_split(
                    temp_data, temp_labels,
                    test_size=(1 - val_ratio),
                    stratify=temp_labels,
                    random_state=random_state
                )
            else:
                # Simple ratio-based splitting
                train_end = int(n_samples * ratios["train"])
                val_end = train_end + int(n_samples * ratios["validation"])
                
                train_data = data.iloc[:train_end]
                val_data = data.iloc[train_end:val_end]
                test_data = data.iloc[val_end:]
                
                if labels is not None:
                    train_labels = labels.iloc[:train_end]
                    val_labels = labels.iloc[train_end:val_end]
                    test_labels = labels.iloc[val_end:]
                else:
                    train_labels = val_labels = test_labels = None
        
        return train_data, val_data, test_data, train_labels, val_labels, test_labels
    
    def k_fold_cross_validation(
        self,
        data: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        n_splits: int = 5,
        shuffle: bool = False,
        stratify: bool = False,
        time_series: bool = False,
        random_state: Optional[int] = None
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]]:
        """
        Generate K-fold cross-validation splits.
        
        Args:
            data: DataFrame to split
            labels: Optional labels
            n_splits: Number of folds
            shuffle: Whether to shuffle (ignored for time-series)
            stratify: Whether to use stratified splitting
            time_series: Whether to use time-series splitting
            random_state: Random state for reproducibility
            
        Returns:
            List of (train_data, val_data, train_labels, val_labels) tuples
        """
        if time_series:
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            splits = []
            
            for train_idx, val_idx in tscv.split(data):
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]
                
                if labels is not None:
                    train_labels = labels.iloc[train_idx]
                    val_labels = labels.iloc[val_idx]
                else:
                    train_labels = val_labels = None
                
                splits.append((train_data, val_data, train_labels, val_labels))
            
            return splits
        else:
            # Standard or stratified K-fold
            if stratify and labels is not None:
                kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
                splits = []
                
                for train_idx, val_idx in kf.split(data, labels):
                    train_data = data.iloc[train_idx]
                    val_data = data.iloc[val_idx]
                    train_labels = labels.iloc[train_idx]
                    val_labels = labels.iloc[val_idx]
                    splits.append((train_data, val_data, train_labels, val_labels))
                
                return splits
            else:
                kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
                splits = []
                
                for train_idx, val_idx in kf.split(data):
                    train_data = data.iloc[train_idx]
                    val_data = data.iloc[val_idx]
                    
                    if labels is not None:
                        train_labels = labels.iloc[train_idx]
                        val_labels = labels.iloc[val_idx]
                    else:
                        train_labels = val_labels = None
                    
                    splits.append((train_data, val_data, train_labels, val_labels))
                
                return splits
    
    def split_by_date(
        self,
        data: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        train_end_date: Optional[str] = None,
        val_end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """
        Split data by date (for time-series data).
        
        Args:
            data: DataFrame with datetime index
            labels: Optional labels
            train_end_date: End date for training set (ISO format string)
            val_end_date: End date for validation set (ISO format string)
            
        Returns:
            Tuple of (train_data, val_data, test_data, train_labels, val_labels, test_labels)
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TrainingError(
                "Data index must be DatetimeIndex for date-based splitting",
                details={"index_type": type(data.index).__name__}
            )
        
        train_end = pd.to_datetime(train_end_date) if train_end_date else None
        val_end = pd.to_datetime(val_end_date) if val_end_date else None
        
        if train_end:
            train_data = data[data.index <= train_end]
            remaining_data = data[data.index > train_end]
        else:
            train_data = data
            remaining_data = pd.DataFrame()
        
        if val_end and len(remaining_data) > 0:
            val_data = remaining_data[remaining_data.index <= val_end]
            test_data = remaining_data[remaining_data.index > val_end]
        else:
            val_data = pd.DataFrame()
            test_data = remaining_data
        
        if labels is not None:
            if train_end:
                train_labels = labels[labels.index <= train_end]
                remaining_labels = labels[labels.index > train_end]
            else:
                train_labels = labels
                remaining_labels = pd.Series()
            
            if val_end and len(remaining_labels) > 0:
                val_labels = remaining_labels[remaining_labels.index <= val_end]
                test_labels = remaining_labels[remaining_labels.index > val_end]
            else:
                val_labels = pd.Series()
                test_labels = remaining_labels
        else:
            train_labels = val_labels = test_labels = None
        
        return train_data, val_data, test_data, train_labels, val_labels, test_labels

