"""
Triangle pattern detection algorithm.

This module implements detection algorithms for rising, falling, and
symmetrical triangle patterns in price charts.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from chartradar.metrics.base import Algorithm
from chartradar.metrics.registry import register_algorithm
from chartradar.src.types import PatternDetection
from chartradar.src.exceptions import AlgorithmError


@register_algorithm(name="triangle_detector", version="1.0.0")
class TriangleDetector(Algorithm):
    """
    Algorithm for detecting triangle patterns.
    
    Triangle patterns are characterized by converging trend lines:
    - Rising triangle: Horizontal resistance line, ascending support line (bullish)
    - Falling triangle: Horizontal support line, descending resistance line (bearish)
    - Symmetrical triangle: Both lines converging at similar angles (neutral)
    """
    
    def __init__(
        self,
        name: str = "triangle_detector",
        version: str = "1.0.0",
        min_confidence: float = 0.6,
        lookback_period: int = 60,
        min_touches: int = 3,
        **kwargs: Any
    ):
        """
        Initialize the triangle detector.
        
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
        Process data and detect triangle patterns.
        
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
        
        # Detect rising triangle
        rising_triangle = self._detect_rising_triangle(recent_data, min_touches)
        if rising_triangle:
            confidence = rising_triangle['confidence']
            if confidence >= min_confidence:
                pattern = PatternDetection(
                    pattern_type="rising_triangle",
                    confidence=confidence,
                    start_index=data.index.get_loc(rising_triangle['start_timestamp']),
                    end_index=data.index.get_loc(rising_triangle['end_timestamp']),
                    start_timestamp=rising_triangle['start_timestamp'],
                    end_timestamp=rising_triangle['end_timestamp'],
                    predicted_direction="bullish",
                    characteristics=rising_triangle.get('characteristics', {})
                )
                patterns.append(pattern)
                confidence_scores.append(confidence)
        
        # Detect falling triangle
        falling_triangle = self._detect_falling_triangle(recent_data, min_touches)
        if falling_triangle:
            confidence = falling_triangle['confidence']
            if confidence >= min_confidence:
                pattern = PatternDetection(
                    pattern_type="falling_triangle",
                    confidence=confidence,
                    start_index=data.index.get_loc(falling_triangle['start_timestamp']),
                    end_index=data.index.get_loc(falling_triangle['end_timestamp']),
                    start_timestamp=falling_triangle['start_timestamp'],
                    end_timestamp=falling_triangle['end_timestamp'],
                    predicted_direction="bearish",
                    characteristics=falling_triangle.get('characteristics', {})
                )
                patterns.append(pattern)
                confidence_scores.append(confidence)
        
        # Detect symmetrical triangle
        symmetrical_triangle = self._detect_symmetrical_triangle(recent_data, min_touches)
        if symmetrical_triangle:
            confidence = symmetrical_triangle['confidence']
            if confidence >= min_confidence:
                pattern = PatternDetection(
                    pattern_type="symmetrical_triangle",
                    confidence=confidence,
                    start_index=data.index.get_loc(symmetrical_triangle['start_timestamp']),
                    end_index=data.index.get_loc(symmetrical_triangle['end_timestamp']),
                    start_timestamp=symmetrical_triangle['start_timestamp'],
                    end_timestamp=symmetrical_triangle['end_timestamp'],
                    predicted_direction=None,  # Neutral pattern
                    characteristics=symmetrical_triangle.get('characteristics', {})
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
    
    def _detect_rising_triangle(
        self,
        data: pd.DataFrame,
        min_touches: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect rising triangle pattern.
        
        Rising triangle: Horizontal resistance, ascending support (bullish).
        """
        if len(data) < min_touches * 2:
            return None
        
        highs = data['high'].values
        lows = data['low'].values
        
        # Find peaks and troughs
        peaks = self._find_peaks(highs)
        troughs = self._find_troughs(lows)
        
        if len(peaks) < min_touches or len(troughs) < min_touches:
            return None
        
        # Check for horizontal resistance (peaks at similar levels)
        peak_indices = peaks[-min_touches:]
        peak_values = highs[peak_indices]
        
        # Check if peaks are relatively flat (horizontal resistance)
        peak_std = np.std(peak_values)
        peak_mean = np.mean(peak_values)
        peak_cv = peak_std / peak_mean if peak_mean > 0 else float('inf')
        
        # Check for ascending support (troughs increasing)
        trough_indices = troughs[-min_touches:]
        trough_slope = self._calculate_slope(trough_indices, lows[trough_indices])
        
        # Rising triangle: flat top (low CV) and rising bottom (positive slope)
        if peak_cv < 0.02 and trough_slope > 0:
            convergence = abs(trough_slope) / (peak_mean * 0.01)  # Normalized
            confidence = min(0.95, 0.5 + (1 - peak_cv * 50) * 0.2 + (trough_slope / peak_mean) * 0.25)
            
            start_idx = min(min(peak_indices), min(trough_indices))
            end_idx = max(max(peak_indices), max(trough_indices))
            
            return {
                "confidence": confidence,
                "start_timestamp": data.index[start_idx],
                "end_timestamp": data.index[end_idx],
                "characteristics": {
                    "resistance_level": float(peak_mean),
                    "resistance_cv": float(peak_cv),
                    "support_slope": float(trough_slope),
                    "convergence": float(convergence),
                    "peak_touches": len(peak_indices),
                    "trough_touches": len(trough_indices)
                }
            }
        
        return None
    
    def _detect_falling_triangle(
        self,
        data: pd.DataFrame,
        min_touches: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect falling triangle pattern.
        
        Falling triangle: Horizontal support, descending resistance (bearish).
        """
        if len(data) < min_touches * 2:
            return None
        
        highs = data['high'].values
        lows = data['low'].values
        
        # Find peaks and troughs
        peaks = self._find_peaks(highs)
        troughs = self._find_troughs(lows)
        
        if len(peaks) < min_touches or len(troughs) < min_touches:
            return None
        
        # Check for horizontal support (troughs at similar levels)
        trough_indices = troughs[-min_touches:]
        trough_values = lows[trough_indices]
        
        trough_std = np.std(trough_values)
        trough_mean = np.mean(trough_values)
        trough_cv = trough_std / trough_mean if trough_mean > 0 else float('inf')
        
        # Check for descending resistance (peaks decreasing)
        peak_indices = peaks[-min_touches:]
        peak_slope = self._calculate_slope(peak_indices, highs[peak_indices])
        
        # Falling triangle: flat bottom (low CV) and falling top (negative slope)
        if trough_cv < 0.02 and peak_slope < 0:
            convergence = abs(peak_slope) / (trough_mean * 0.01)
            confidence = min(0.95, 0.5 + (1 - trough_cv * 50) * 0.2 + (abs(peak_slope) / trough_mean) * 0.25)
            
            start_idx = min(min(peak_indices), min(trough_indices))
            end_idx = max(max(peak_indices), max(trough_indices))
            
            return {
                "confidence": confidence,
                "start_timestamp": data.index[start_idx],
                "end_timestamp": data.index[end_idx],
                "characteristics": {
                    "support_level": float(trough_mean),
                    "support_cv": float(trough_cv),
                    "resistance_slope": float(peak_slope),
                    "convergence": float(convergence),
                    "peak_touches": len(peak_indices),
                    "trough_touches": len(trough_indices)
                }
            }
        
        return None
    
    def _detect_symmetrical_triangle(
        self,
        data: pd.DataFrame,
        min_touches: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect symmetrical triangle pattern.
        
        Symmetrical triangle: Both lines converging at similar angles (neutral).
        """
        if len(data) < min_touches * 2:
            return None
        
        highs = data['high'].values
        lows = data['low'].values
        
        # Find peaks and troughs
        peaks = self._find_peaks(highs)
        troughs = self._find_troughs(lows)
        
        if len(peaks) < min_touches or len(troughs) < min_touches:
            return None
        
        peak_indices = peaks[-min_touches:]
        trough_indices = troughs[-min_touches:]
        
        # Calculate slopes
        upper_slope = self._calculate_slope(peak_indices, highs[peak_indices])
        lower_slope = self._calculate_slope(trough_indices, lows[trough_indices])
        
        # Symmetrical: slopes should be opposite and similar magnitude
        if upper_slope < 0 and lower_slope > 0:
            slope_ratio = abs(upper_slope) / abs(lower_slope) if abs(lower_slope) > 0 else 0
            # Check if slopes are roughly symmetrical (ratio between 0.7 and 1.3)
            if 0.7 <= slope_ratio <= 1.3:
                convergence = (abs(upper_slope) + abs(lower_slope)) / 2
                confidence = min(0.95, 0.5 + (1 - abs(1 - slope_ratio)) * 0.3 + (convergence / np.mean(highs)) * 0.15)
                
                start_idx = min(min(peak_indices), min(trough_indices))
                end_idx = max(max(peak_indices), max(trough_indices))
                
                return {
                    "confidence": confidence,
                    "start_timestamp": data.index[start_idx],
                    "end_timestamp": data.index[end_idx],
                    "characteristics": {
                        "upper_slope": float(upper_slope),
                        "lower_slope": float(lower_slope),
                        "slope_ratio": float(slope_ratio),
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
        
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
        
        return float(slope)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get algorithm metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Detects rising, falling, and symmetrical triangle patterns in price charts",
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

