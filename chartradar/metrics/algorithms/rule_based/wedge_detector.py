"""
Wedge pattern detection algorithm.

This module implements detection algorithms for rising and falling wedge patterns
in price charts.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from chartradar.metrics.base import Algorithm
from chartradar.metrics.registry import register_algorithm
from chartradar.core.types import PatternDetection
from chartradar.core.exceptions import AlgorithmError


@register_algorithm(name="wedge_detector", version="1.0.0")
class WedgeDetector(Algorithm):
    """
    Algorithm for detecting rising and falling wedge patterns.
    
    A wedge pattern is characterized by converging trend lines with:
    - Rising wedge: Both trend lines slope upward, but the upper line
      has a steeper slope (bearish pattern)
    - Falling wedge: Both trend lines slope downward, but the lower line
      has a steeper slope (bullish pattern)
    """
    
    def __init__(
        self,
        name: str = "wedge_detector",
        version: str = "1.0.0",
        min_confidence: float = 0.6,
        lookback_period: int = 50,
        min_touches: int = 3,
        **kwargs: Any
    ):
        """
        Initialize the wedge detector.
        
        Args:
            name: Algorithm name
            version: Algorithm version
            min_confidence: Minimum confidence score for pattern detection
            lookback_period: Number of periods to look back for pattern detection
            min_touches: Minimum number of touches on each trend line
            **kwargs: Additional parameters
        """
        super().__init__(name, version, **kwargs)
        self.min_confidence = min_confidence
        self.lookback_period = lookback_period
        self.min_touches = min_touches
    
    def process(self, data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """
        Process data and detect wedge patterns.
        
        Args:
            data: DataFrame with OHLCV columns and datetime index
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with detection results
        """
        self.validate_data(data)
        
        # Use parameters from kwargs if provided
        min_confidence = kwargs.get('min_confidence', self.min_confidence)
        lookback_period = kwargs.get('lookback_period', self.lookback_period)
        min_touches = kwargs.get('min_touches', self.min_touches)
        
        if len(data) < lookback_period:
            return {
                "algorithm_name": self.name,
                "results": [],
                "confidence_scores": [],
                "metadata": {"message": "Insufficient data points"},
                "timestamp": datetime.now()
            }
        
        # Work with recent data
        recent_data = data.tail(lookback_period).copy()
        
        # Detect patterns
        patterns = []
        confidence_scores = []
        
        # Detect rising wedge
        rising_wedge = self._detect_rising_wedge(recent_data, min_touches)
        if rising_wedge:
            confidence = rising_wedge['confidence']
            if confidence >= min_confidence:
                pattern = PatternDetection(
                    pattern_type="rising_wedge",
                    confidence=confidence,
                    start_index=data.index.get_loc(rising_wedge['start_timestamp']),
                    end_index=data.index.get_loc(rising_wedge['end_timestamp']),
                    start_timestamp=rising_wedge['start_timestamp'],
                    end_timestamp=rising_wedge['end_timestamp'],
                    predicted_direction="bearish",
                    characteristics=rising_wedge.get('characteristics', {})
                )
                patterns.append(pattern)
                confidence_scores.append(confidence)
        
        # Detect falling wedge
        falling_wedge = self._detect_falling_wedge(recent_data, min_touches)
        if falling_wedge:
            confidence = falling_wedge['confidence']
            if confidence >= min_confidence:
                pattern = PatternDetection(
                    pattern_type="falling_wedge",
                    confidence=confidence,
                    start_index=data.index.get_loc(falling_wedge['start_timestamp']),
                    end_index=data.index.get_loc(falling_wedge['end_timestamp']),
                    start_timestamp=falling_wedge['start_timestamp'],
                    end_timestamp=falling_wedge['end_timestamp'],
                    predicted_direction="bullish",
                    characteristics=falling_wedge.get('characteristics', {})
                )
                patterns.append(pattern)
                confidence_scores.append(confidence)
        
        return {
            "algorithm_name": self.name,
            "results": patterns,
            "confidence_scores": confidence_scores,
            "metadata": {
                "lookback_period": lookback_period,
                "min_confidence": min_confidence,
                "patterns_detected": len(patterns)
            },
            "timestamp": datetime.now()
        }
    
    def _detect_rising_wedge(
        self,
        data: pd.DataFrame,
        min_touches: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect rising wedge pattern.
        
        Rising wedge: Both trend lines slope up, but upper line is steeper.
        This is typically a bearish pattern.
        """
        if len(data) < min_touches * 2:
            return None
        
        # Find local highs and lows
        highs = data['high'].values
        lows = data['low'].values
        
        # Simple approach: look for converging trend lines
        # Upper trend line: connecting higher highs
        # Lower trend line: connecting higher lows
        
        # Find peaks and troughs
        peaks = self._find_peaks(highs)
        troughs = self._find_troughs(lows)
        
        if len(peaks) < min_touches or len(troughs) < min_touches:
            return None
        
        # Fit trend lines
        peak_indices = peaks[-min_touches:]
        trough_indices = troughs[-min_touches:]
        
        # Calculate trend line slopes
        upper_slope = self._calculate_slope(peak_indices, highs[peak_indices])
        lower_slope = self._calculate_slope(trough_indices, lows[trough_indices])
        
        # Check if both slopes are positive (rising) and upper is steeper
        if upper_slope > 0 and lower_slope > 0 and upper_slope > lower_slope:
            # Calculate convergence
            convergence = abs(upper_slope - lower_slope) / max(abs(upper_slope), abs(lower_slope))
            
            # Calculate confidence based on convergence and number of touches
            confidence = min(0.95, 0.5 + convergence * 0.3 + (min_touches / 5) * 0.15)
            
            start_idx = min(min(peak_indices), min(trough_indices))
            end_idx = max(max(peak_indices), max(trough_indices))
            
            return {
                "confidence": confidence,
                "start_timestamp": data.index[start_idx],
                "end_timestamp": data.index[end_idx],
                "characteristics": {
                    "upper_slope": float(upper_slope),
                    "lower_slope": float(lower_slope),
                    "convergence": float(convergence),
                    "peak_touches": len(peak_indices),
                    "trough_touches": len(trough_indices)
                }
            }
        
        return None
    
    def _detect_falling_wedge(
        self,
        data: pd.DataFrame,
        min_touches: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect falling wedge pattern.
        
        Falling wedge: Both trend lines slope down, but lower line is steeper.
        This is typically a bullish pattern.
        """
        if len(data) < min_touches * 2:
            return None
        
        # Find local highs and lows
        highs = data['high'].values
        lows = data['low'].values
        
        # Find peaks and troughs
        peaks = self._find_peaks(highs)
        troughs = self._find_troughs(lows)
        
        if len(peaks) < min_touches or len(troughs) < min_touches:
            return None
        
        # Fit trend lines
        peak_indices = peaks[-min_touches:]
        trough_indices = troughs[-min_touches:]
        
        # Calculate trend line slopes
        upper_slope = self._calculate_slope(peak_indices, highs[peak_indices])
        lower_slope = self._calculate_slope(trough_indices, lows[trough_indices])
        
        # Check if both slopes are negative (falling) and lower is steeper (more negative)
        if upper_slope < 0 and lower_slope < 0 and abs(lower_slope) > abs(upper_slope):
            # Calculate convergence
            convergence = abs(abs(lower_slope) - abs(upper_slope)) / max(abs(upper_slope), abs(lower_slope))
            
            # Calculate confidence
            confidence = min(0.95, 0.5 + convergence * 0.3 + (min_touches / 5) * 0.15)
            
            start_idx = min(min(peak_indices), min(trough_indices))
            end_idx = max(max(peak_indices), max(trough_indices))
            
            return {
                "confidence": confidence,
                "start_timestamp": data.index[start_idx],
                "end_timestamp": data.index[end_idx],
                "characteristics": {
                    "upper_slope": float(upper_slope),
                    "lower_slope": float(lower_slope),
                    "convergence": float(convergence),
                    "peak_touches": len(peak_indices),
                    "trough_touches": len(trough_indices)
                }
            }
        
        return None
    
    def _find_peaks(self, values: np.ndarray, window: int = 3) -> List[int]:
        """Find local peaks in values."""
        peaks = []
        for i in range(window, len(values) - window):
            if all(values[i] >= values[i-j] for j in range(1, window+1)) and \
               all(values[i] >= values[i+j] for j in range(1, window+1)):
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, values: np.ndarray, window: int = 3) -> List[int]:
        """Find local troughs in values."""
        troughs = []
        for i in range(window, len(values) - window):
            if all(values[i] <= values[i-j] for j in range(1, window+1)) and \
               all(values[i] <= values[i+j] for j in range(1, window+1)):
                troughs.append(i)
        return troughs
    
    def _calculate_slope(self, indices: List[int], values: np.ndarray) -> float:
        """Calculate linear regression slope."""
        if len(indices) < 2:
            return 0.0
        
        x = np.array(indices)
        y = values[indices]
        
        # Simple linear regression
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return float(slope)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get algorithm metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Detects rising and falling wedge patterns in price charts",
            "author": "ChartRadar",
            "parameters": {
                "min_confidence": self.min_confidence,
                "lookback_period": self.lookback_period,
                "min_touches": self.min_touches
            },
            "requirements": self.get_requirements()
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get data requirements."""
        return {
            "min_data_points": self.lookback_period,
            "required_columns": ["open", "high", "low", "close", "volume"],
            "data_frequency": "any",
            "preprocessing": []
        }

