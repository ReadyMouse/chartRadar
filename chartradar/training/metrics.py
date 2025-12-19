"""
Metrics logging for the ChartRadar framework.

This module provides functionality to log training and validation metrics
to files and optionally integrate with MLflow/Weights&Biases.
"""

import json
import csv
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
import logging

from chartradar.core.exceptions import TrainingError

logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Logger for training and validation metrics.
    
    Supports logging to CSV, JSON files, and optional integration with
    MLflow and Weights&Biases.
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_format: str = "csv",
        mlflow_tracking: bool = False,
        wandb_tracking: bool = False,
        **kwargs: Any
    ):
        """
        Initialize the metrics logger.
        
        Args:
            log_file: Path to log file
            log_format: Log format ('csv' or 'json')
            mlflow_tracking: Whether to use MLflow tracking
            wandb_tracking: Whether to use Weights&Biases tracking
            **kwargs: Additional parameters
        """
        self.log_file = Path(log_file) if log_file else None
        self.log_format = log_format
        self.mlflow_tracking = mlflow_tracking
        self.wandb_tracking = wandb_tracking
        self.parameters = kwargs
        
        # Initialize tracking libraries if requested
        self.mlflow_run = None
        if mlflow_tracking:
            try:
                import mlflow
                self.mlflow = mlflow
                self.mlflow_run = mlflow.start_run()
            except ImportError:
                logger.warning("MLflow not available, disabling MLflow tracking")
                self.mlflow_tracking = False
        
        self.wandb_run = None
        if wandb_tracking:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_run = wandb.init(**kwargs.get('wandb_config', {}))
            except ImportError:
                logger.warning("Weights&Biases not available, disabling W&B tracking")
                self.wandb_tracking = False
        
        # Initialize log file
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._initialize_log_file()
    
    def _initialize_log_file(self) -> None:
        """Initialize the log file with headers."""
        if self.log_format == "csv" and self.log_file:
            # CSV will be written row by row, headers in first write
            pass
        elif self.log_format == "json" and self.log_file:
            # JSON will be an array of log entries
            with open(self.log_file, 'w') as f:
                json.dump([], f)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step/epoch
            prefix: Prefix for metric names (e.g., 'train_', 'val_')
        """
        timestamp = datetime.now().isoformat()
        
        # Prepare log entry
        log_entry = {
            "timestamp": timestamp,
            "step": step,
            **{f"{prefix}{k}": v for k, v in metrics.items()}
        }
        
        # Log to file
        if self.log_file:
            if self.log_format == "csv":
                self._log_to_csv(log_entry)
            elif self.log_format == "json":
                self._log_to_json(log_entry)
        
        # Log to MLflow
        if self.mlflow_tracking and self.mlflow_run:
            try:
                prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
                self.mlflow.log_metrics(prefixed_metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {str(e)}")
        
        # Log to Weights&Biases
        if self.wandb_tracking and self.wandb_run:
            try:
                prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
                self.wandb.log(prefixed_metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to W&B: {str(e)}")
    
    def _log_to_csv(self, log_entry: Dict[str, Any]) -> None:
        """Log entry to CSV file."""
        if not self.log_file:
            return
        
        file_exists = self.log_file.exists()
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
    
    def _log_to_json(self, log_entry: Dict[str, Any]) -> None:
        """Log entry to JSON file."""
        if not self.log_file:
            return
        
        # Read existing logs
        logs = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
        
        # Append new log
        logs.append(log_entry)
        
        # Write back
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of parameter names and values
        """
        # Log to MLflow
        if self.mlflow_tracking and self.mlflow_run:
            try:
                self.mlflow.log_params(params)
            except Exception as e:
                logger.warning(f"Failed to log params to MLflow: {str(e)}")
        
        # Log to Weights&Biases
        if self.wandb_tracking and self.wandb_run:
            try:
                self.wandb.config.update(params)
            except Exception as e:
                logger.warning(f"Failed to log params to W&B: {str(e)}")
    
    def close(self) -> None:
        """Close logging connections."""
        if self.mlflow_tracking and self.mlflow_run:
            try:
                self.mlflow.end_run()
            except Exception:
                pass
        
        if self.wandb_tracking and self.wandb_run:
            try:
                self.wandb.finish()
            except Exception:
                pass

