"""Tests for data splitting."""

import pytest
import pandas as pd
import numpy as np

from chartradar.training.split import DataSplitter
from chartradar.src.exceptions import TrainingError


class TestDataSplitter:
    """Tests for DataSplitter class."""
    
    def test_split_train_val_test_basic(self):
        """Test basic train/val/test split."""
        splitter = DataSplitter()
        
        data = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200)})
        labels = pd.Series(range(100))
        
        train, val, test, train_labels, val_labels, test_labels = splitter.split_train_val_test(
            data, labels
        )
        
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert len(train_labels) == 70
    
    def test_split_custom_ratios(self):
        """Test split with custom ratios."""
        splitter = DataSplitter()
        
        data = pd.DataFrame({'feature1': range(100)})
        
        train, val, test, _, _, _ = splitter.split_train_val_test(
            data, ratios={"train": 0.8, "validation": 0.1, "test": 0.1}
        )
        
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10
    
    def test_split_invalid_ratios(self):
        """Test split with invalid ratios."""
        splitter = DataSplitter()
        
        data = pd.DataFrame({'feature1': range(100)})
        
        with pytest.raises(TrainingError) as exc_info:
            splitter.split_train_val_test(
                data, ratios={"train": 0.5, "validation": 0.3, "test": 0.3}
            )
        assert "must sum to 1.0" in str(exc_info.value)
    
    def test_split_time_series(self):
        """Test time-series aware splitting."""
        splitter = DataSplitter()
        
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({'feature1': range(100)}, index=dates)
        
        train, val, test, _, _, _ = splitter.split_train_val_test(
            data, time_series=True
        )
        
        # Time-series split should be sequential
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]
    
    def test_k_fold_cross_validation(self):
        """Test K-fold cross-validation."""
        splitter = DataSplitter()
        
        data = pd.DataFrame({'feature1': range(100)})
        labels = pd.Series(range(100))
        
        splits = splitter.k_fold_cross_validation(data, labels, n_splits=5)
        
        assert len(splits) == 5
        for train, val, train_labels, val_labels in splits:
            assert len(train) > 0
            assert len(val) > 0
            assert len(train) + len(val) == 100
    
    def test_split_by_date(self):
        """Test date-based splitting."""
        splitter = DataSplitter()
        
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({'feature1': range(100)}, index=dates)
        
        train, val, test, _, _, _ = splitter.split_by_date(
            data,
            train_end_date='2024-02-10',
            val_end_date='2024-02-20'
        )
        
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

