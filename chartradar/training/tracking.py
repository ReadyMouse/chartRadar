"""
Experiment tracking for the ChartRadar framework.

This module provides functionality to track experiment parameters, results,
and metadata for ML training experiments.
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import hashlib

from chartradar.core.exceptions import TrainingError


class ExperimentTracker:
    """
    Tracker for ML training experiments.
    
    Provides functionality to track experiment parameters, results,
    and metadata with basic experiment comparison.
    """
    
    def __init__(
        self,
        tracking_dir: str = "./experiments",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the experiment tracker.
        
        Args:
            tracking_dir: Directory to store experiment data
            experiment_name: Name for this experiment (auto-generated if None)
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name:
            self.experiment_name = experiment_name
        else:
            # Generate experiment name from timestamp
            self.experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_dir = self.tracking_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_data = {
            "name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "parameters": {},
            "results": {},
            "metadata": {}
        }
    
    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Log experiment parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        self.experiment_data["parameters"].update(parameters)
        self._save_experiment_data()
    
    def log_results(self, results: Dict[str, Any]) -> None:
        """
        Log experiment results.
        
        Args:
            results: Dictionary of result names and values
        """
        self.experiment_data["results"].update(results)
        self.experiment_data["updated_at"] = datetime.now().isoformat()
        self._save_experiment_data()
    
    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Log experiment metadata.
        
        Args:
            metadata: Dictionary of metadata
        """
        self.experiment_data["metadata"].update(metadata)
        self._save_experiment_data()
    
    def _save_experiment_data(self) -> None:
        """Save experiment data to file."""
        experiment_file = self.experiment_dir / "experiment.json"
        with open(experiment_file, 'w') as f:
            json.dump(self.experiment_data, f, indent=2, default=str)
    
    def get_experiment_data(self) -> Dict[str, Any]:
        """
        Get current experiment data.
        
        Returns:
            Dictionary with experiment data
        """
        return self.experiment_data.copy()
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments in tracking directory.
        
        Returns:
            List of experiment metadata dictionaries
        """
        experiments = []
        
        for exp_dir in self.tracking_dir.iterdir():
            if exp_dir.is_dir():
                exp_file = exp_dir / "experiment.json"
                if exp_file.exists():
                    try:
                        with open(exp_file, 'r') as f:
                            exp_data = json.load(f)
                            experiments.append(exp_data)
                    except Exception:
                        continue
        
        # Sort by creation time
        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return experiments
    
    def compare_experiments(
        self,
        experiment_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_names: List of experiment names to compare (all if None)
            
        Returns:
            Dictionary with comparison results
        """
        if experiment_names is None:
            experiments = self.list_experiments()
        else:
            experiments = []
            for exp_name in experiment_names:
                exp_file = self.tracking_dir / exp_name / "experiment.json"
                if exp_file.exists():
                    with open(exp_file, 'r') as f:
                        experiments.append(json.load(f))
        
        if len(experiments) < 2:
            return {
                "message": "Need at least 2 experiments to compare",
                "experiments": len(experiments)
            }
        
        # Compare parameters
        param_comparison = self._compare_parameters(experiments)
        
        # Compare results
        result_comparison = self._compare_results(experiments)
        
        return {
            "experiments": [exp["name"] for exp in experiments],
            "parameter_comparison": param_comparison,
            "result_comparison": result_comparison,
            "best_experiment": self._find_best_experiment(experiments)
        }
    
    def _compare_parameters(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare parameters across experiments."""
        all_params = set()
        for exp in experiments:
            all_params.update(exp.get("parameters", {}).keys())
        
        comparison = {}
        for param in all_params:
            values = [exp.get("parameters", {}).get(param, "N/A") for exp in experiments]
            comparison[param] = {
                "values": values,
                "unique": len(set(str(v) for v in values)) == len(values)
            }
        
        return comparison
    
    def _compare_results(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results across experiments."""
        all_results = set()
        for exp in experiments:
            all_results.update(exp.get("results", {}).keys())
        
        comparison = {}
        for result in all_results:
            values = [exp.get("results", {}).get(result, None) for exp in experiments]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            
            comparison[result] = {
                "values": values,
                "min": min(numeric_values) if numeric_values else None,
                "max": max(numeric_values) if numeric_values else None,
                "mean": sum(numeric_values) / len(numeric_values) if numeric_values else None
            }
        
        return comparison
    
    def _find_best_experiment(self, experiments: List[Dict[str, Any]]) -> Optional[str]:
        """Find best experiment based on results."""
        # Simple heuristic: look for accuracy or f1_score
        best_exp = None
        best_score = -1.0
        
        for exp in experiments:
            results = exp.get("results", {})
            score = results.get("accuracy", results.get("f1_score", results.get("val_accuracy", -1.0)))
            if isinstance(score, (int, float)) and score > best_score:
                best_score = score
                best_exp = exp["name"]
        
        return best_exp

