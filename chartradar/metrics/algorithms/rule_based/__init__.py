"""
Rule-based algorithm implementations.
"""

# Import algorithms to trigger registration
from chartradar.metrics.algorithms.rule_based.wedge_detector import WedgeDetector
from chartradar.metrics.algorithms.rule_based.triangle_detector import TriangleDetector

__all__ = [
    "WedgeDetector",
    "TriangleDetector",
]

