"""Tests for experiment tracking."""

import pytest
from pathlib import Path

from chartradar.training.tracking import ExperimentTracker


class TestExperimentTracker:
    """Tests for ExperimentTracker class."""
    
    def test_log_parameters(self, tmp_path):
        """Test logging parameters."""
        tracker = ExperimentTracker(tracking_dir=str(tmp_path), experiment_name="test_exp")
        
        tracker.log_parameters({"learning_rate": 0.001, "epochs": 10})
        
        data = tracker.get_experiment_data()
        assert data["parameters"]["learning_rate"] == 0.001
        assert data["parameters"]["epochs"] == 10
    
    def test_log_results(self, tmp_path):
        """Test logging results."""
        tracker = ExperimentTracker(tracking_dir=str(tmp_path), experiment_name="test_exp")
        
        tracker.log_results({"accuracy": 0.9, "loss": 0.1})
        
        data = tracker.get_experiment_data()
        assert data["results"]["accuracy"] == 0.9
    
    def test_list_experiments(self, tmp_path):
        """Test listing experiments."""
        tracker1 = ExperimentTracker(tracking_dir=str(tmp_path), experiment_name="exp1")
        tracker1.log_parameters({"param1": "value1"})
        
        tracker2 = ExperimentTracker(tracking_dir=str(tmp_path), experiment_name="exp2")
        tracker2.log_parameters({"param2": "value2"})
        
        experiments = tracker1.list_experiments()
        assert len(experiments) >= 2
    
    def test_compare_experiments(self, tmp_path):
        """Test comparing experiments."""
        tracker1 = ExperimentTracker(tracking_dir=str(tmp_path), experiment_name="exp1")
        tracker1.log_results({"accuracy": 0.8})
        
        tracker2 = ExperimentTracker(tracking_dir=str(tmp_path), experiment_name="exp2")
        tracker2.log_results({"accuracy": 0.9})
        
        comparison = tracker1.compare_experiments(["exp1", "exp2"])
        
        assert "result_comparison" in comparison
        assert "best_experiment" in comparison

