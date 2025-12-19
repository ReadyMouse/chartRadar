"""
Training module for the ChartRadar framework.

This module provides training and evaluation infrastructure for ML algorithms.
"""

from chartradar.training.base import TrainingLoop, EvaluationLoop
from chartradar.training.split import DataSplitter
from chartradar.training.loop import GenericTrainingLoop
from chartradar.training.evaluation import GenericEvaluationLoop
from chartradar.training.checkpoint import ModelCheckpointer
from chartradar.training.metrics import MetricsLogger
from chartradar.training.tracking import ExperimentTracker

__all__ = [
    # Base interfaces
    "TrainingLoop",
    "EvaluationLoop",
    # Implementations
    "DataSplitter",
    "GenericTrainingLoop",
    "GenericEvaluationLoop",
    "ModelCheckpointer",
    "MetricsLogger",
    "ExperimentTracker",
]

