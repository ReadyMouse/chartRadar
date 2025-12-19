"""
Data fusion module for the ChartRadar framework.

This module provides fusion strategies for combining results from multiple
algorithms into unified predictions.
"""

from chartradar.fusion.base import FusionStrategy
from chartradar.fusion.registry import (
    FusionStrategyRegistry,
    register_fusion_strategy,
    get_fusion_strategy,
    create_fusion_strategy,
)
from chartradar.fusion.executor import FusionExecutor

__all__ = [
    # Base classes
    "FusionStrategy",
    # Registry
    "FusionStrategyRegistry",
    "register_fusion_strategy",
    "get_fusion_strategy",
    "create_fusion_strategy",
    # Executor
    "FusionExecutor",
]

