"""
Metrics module for the ChartRadar framework.

This module provides pattern detection and analysis algorithms with
a plugin system for easy extension.
"""

from chartradar.metrics.base import Algorithm
from chartradar.metrics.registry import (
    AlgorithmRegistry,
    register_algorithm,
    get_algorithm,
    create_algorithm,
)
from chartradar.metrics.executor import AlgorithmExecutor

__all__ = [
    # Base classes
    "Algorithm",
    # Registry
    "AlgorithmRegistry",
    "register_algorithm",
    "get_algorithm",
    "create_algorithm",
    # Executor
    "AlgorithmExecutor",
]

