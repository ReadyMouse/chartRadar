"""Tests for metrics logging."""

import pytest
import json
from pathlib import Path

from chartradar.training.metrics import MetricsLogger


class TestMetricsLogger:
    """Tests for MetricsLogger class."""
    
    def test_log_metrics_csv(self, tmp_path):
        """Test logging metrics to CSV."""
        log_file = tmp_path / "metrics.csv"
        logger = MetricsLogger(log_file=str(log_file), log_format="csv")
        
        logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1)
        logger.close()
        
        assert log_file.exists()
        with open(log_file) as f:
            content = f.read()
            assert "loss" in content
            assert "accuracy" in content
    
    def test_log_metrics_json(self, tmp_path):
        """Test logging metrics to JSON."""
        log_file = tmp_path / "metrics.json"
        logger = MetricsLogger(log_file=str(log_file), log_format="json")
        
        logger.log_metrics({"loss": 0.5}, step=1)
        logger.close()
        
        assert log_file.exists()
        with open(log_file) as f:
            logs = json.load(f)
            assert len(logs) == 1
            assert logs[0]["loss"] == 0.5
    
    def test_log_params(self):
        """Test logging parameters."""
        logger = MetricsLogger(mlflow_tracking=False, wandb_tracking=False)
        
        # Should not raise error
        logger.log_params({"learning_rate": 0.001, "batch_size": 32})
        logger.close()

