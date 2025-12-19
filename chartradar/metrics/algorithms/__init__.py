"""
Algorithm implementations for the ChartRadar framework.
"""

# Import rule-based algorithms to register them
from chartradar.metrics.algorithms.rule_based import wedge_detector, triangle_detector

__all__ = [
    "wedge_detector",
    "triangle_detector",
]

