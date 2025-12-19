"""
Fusion strategy implementations.
"""

# Import strategies to trigger registration
from chartradar.fusion.strategies.weighted_average import WeightedAverageFusion
from chartradar.fusion.strategies.voting import VotingFusion
from chartradar.fusion.strategies.stacking import StackingFusion

__all__ = [
    "WeightedAverageFusion",
    "VotingFusion",
    "StackingFusion",
]

