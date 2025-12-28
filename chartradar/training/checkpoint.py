"""
Model checkpointing for the ChartRadar framework.

This module provides functionality to save and load model checkpoints
with versioning and best model selection.
"""

import pickle
import json
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime
import shutil

from chartradar.src.exceptions import TrainingError


class ModelCheckpointer:
    """
    Checkpointer for saving and loading model checkpoints.
    
    Supports model versioning, best model selection, and metadata storage.
    """
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_best: bool = True,
        monitor: str = "val_loss",
        mode: str = "min"
    ):
        """
        Initialize the checkpointer.
        
        Args:
            save_dir: Directory to save checkpoints
            save_best: Whether to save best model
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for monitor metric
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_model_path = None
    
    def save_checkpoint(
        self,
        model: Any,
        epoch: int,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            model: Model to save
            epoch: Training epoch
            metrics: Training metrics
            metadata: Optional metadata
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.save_dir / f"checkpoint_epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / "model.pkl"
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            raise TrainingError(
                f"Failed to save model: {str(e)}",
                details={"checkpoint_dir": str(checkpoint_dir), "error": str(e)}
            ) from e
        
        # Save metadata
        checkpoint_metadata = {
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "model_type": type(model).__name__,
            "is_best": is_best
        }
        
        if metadata:
            checkpoint_metadata.update(metadata)
        
        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
        
        # Update best model if needed
        if is_best or self._is_better(metrics.get(self.monitor, 0.0)):
            self.best_score = metrics.get(self.monitor, 0.0)
            self.best_model_path = str(model_path)
            
            if self.save_best:
                best_dir = self.save_dir / "best_model"
                best_dir.mkdir(exist_ok=True)
                
                # Copy model to best_model directory
                best_model_path = best_dir / "model.pkl"
                shutil.copy2(model_path, best_model_path)
                
                # Save best model metadata
                best_metadata_path = best_dir / "metadata.json"
                checkpoint_metadata["is_best"] = True
                with open(best_metadata_path, 'w') as f:
                    json.dump(checkpoint_metadata, f, indent=2)
        
        return str(model_path)
    
    def _is_better(self, score: float) -> bool:
        """Check if score is better than current best."""
        if self.mode == 'min':
            return score < self.best_score
        else:
            return score > self.best_score
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (uses best if None and load_best=True)
            load_best: Whether to load best model
            
        Returns:
            Dictionary with model and metadata
        """
        if load_best:
            checkpoint_path = str(self.save_dir / "best_model" / "model.pkl")
        elif checkpoint_path is None:
            raise TrainingError(
                "checkpoint_path must be provided if load_best is False",
                details={}
            )
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise TrainingError(
                f"Checkpoint not found: {checkpoint_path}",
                details={"checkpoint_path": str(checkpoint_path)}
            )
        
        # Load model
        try:
            with open(checkpoint_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            raise TrainingError(
                f"Failed to load model: {str(e)}",
                details={"checkpoint_path": str(checkpoint_path), "error": str(e)}
            ) from e
        
        # Load metadata
        metadata_path = checkpoint_path.parent / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return {
            "model": model,
            "metadata": metadata,
            "checkpoint_path": str(checkpoint_path)
        }
    
    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint metadata dictionaries
        """
        checkpoints = []
        
        for checkpoint_dir in self.save_dir.glob("checkpoint_epoch_*"):
            metadata_path = checkpoint_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    metadata["checkpoint_dir"] = str(checkpoint_dir)
                    checkpoints.append(metadata)
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x.get("epoch", 0))
        
        return checkpoints
    
    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the best model checkpoint.
        
        Returns:
            Dictionary with model and metadata, or None if no best model
        """
        best_dir = self.save_dir / "best_model"
        if not best_dir.exists():
            return None
        
        best_model_path = best_dir / "model.pkl"
        if not best_model_path.exists():
            return None
        
        return self.load_checkpoint(str(best_model_path))

