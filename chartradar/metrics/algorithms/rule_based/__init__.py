"""
Rule-based algorithm implementations.
"""

# Import algorithms to trigger registration
from chartradar.metrics.algorithms.rule_based.wedge_detector import WedgeDetector
from chartradar.metrics.algorithms.rule_based.triangle_detector import TriangleDetector
from chartradar.metrics.algorithms.rule_based.ma_slope_detector import MovingAverageSlopeDetector

__all__ = [
    "WedgeDetector",
    "TriangleDetector",
    "MovingAverageSlopeDetector",
]

