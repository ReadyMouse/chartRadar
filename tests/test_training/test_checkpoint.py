"""Tests for model checkpointing."""

import pytest
from pathlib import Path
from unittest.mock import Mock

from chartradar.training.checkpoint import ModelCheckpointer
from chartradar.core.exceptions import TrainingError


class TestModelCheckpointer:
    """Tests for ModelCheckpointer class."""
    
    def test_save_checkpoint(self, tmp_path):
        """Test saving a checkpoint."""
        checkpointer = ModelCheckpointer(save_dir=str(tmp_path))
        
        model = Mock()
        metrics = {"loss": 0.5, "accuracy": 0.9}
        
        checkpoint_path = checkpointer.save_checkpoint(model, epoch=1, metrics=metrics)
        
        assert Path(checkpoint_path).exists()
        assert (tmp_path / "checkpoint_epoch_1" / "model.pkl").exists()
        assert (tmp_path / "checkpoint_epoch_1" / "metadata.json").exists()
    
    def test_load_checkpoint(self, tmp_path):
        """Test loading a checkpoint."""
        checkpointer = ModelCheckpointer(save_dir=str(tmp_path))
        
        model = Mock()
        metrics = {"loss": 0.5}
        
        checkpoint_path = checkpointer.save_checkpoint(model, epoch=1, metrics=metrics)
        
        loaded = checkpointer.load_checkpoint(checkpoint_path)
        
        assert "model" in loaded
        assert "metadata" in loaded
    
    def test_load_checkpoint_not_found(self, tmp_path):
        """Test loading non-existent checkpoint."""
        checkpointer = ModelCheckpointer(save_dir=str(tmp_path))
        
        with pytest.raises(TrainingError) as exc_info:
            checkpointer.load_checkpoint("/nonexistent/path.pkl")
        assert "not found" in str(exc_info.value).lower()
    
    def test_save_best_model(self, tmp_path):
        """Test saving best model."""
        checkpointer = ModelCheckpointer(save_dir=str(tmp_path), save_best=True, monitor="val_loss")
        
        model1 = Mock()
        model2 = Mock()
        
        # Save first checkpoint
        checkpointer.save_checkpoint(model1, epoch=1, metrics={"val_loss": 0.5})
        
        # Save better checkpoint
        checkpointer.save_checkpoint(model2, epoch=2, metrics={"val_loss": 0.3}, is_best=True)
        
        assert (tmp_path / "best_model" / "model.pkl").exists()
    
    def test_list_checkpoints(self, tmp_path):
        """Test listing checkpoints."""
        checkpointer = ModelCheckpointer(save_dir=str(tmp_path))
        
        model = Mock()
        checkpointer.save_checkpoint(model, epoch=1, metrics={})
        checkpointer.save_checkpoint(model, epoch=2, metrics={})
        
        checkpoints = checkpointer.list_checkpoints()
        
        assert len(checkpoints) == 2
        assert checkpoints[0]["epoch"] == 1

