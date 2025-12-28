"""
Moving Average Slope Detector algorithm.

This module implements a trend detection algorithm based on calculating
moving averages and their slopes to identify uptrends, downtrends, and
sideways (ranging) markets.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from chartradar.metrics.base import Algorithm
from chartradar.metrics.registry import register_algorithm
from chartradar.src.types import PatternDetection
from chartradar.src.exceptions import AlgorithmError


@register_algorithm(name="ma_slope_detector", version="1.0.0")
class MovingAverageSlopeDetector(Algorithm):
    """
    Algorithm for detecting trends based on moving average slopes.
    
    This algorithm calculates multiple moving averages (e.g., 10, 20, 50, 200 periods)
    and computes their slopes to determine trend direction:
    - Uptrend: Positive slope above threshold
    - Downtrend: Negative slope below threshold
    - Sideways: Slope within neutral range (low volatility)
    
    The algorithm provides trend classification for each MA period and can
    detect alignment (all MAs trending in the same direction).
    """
    
    def __init__(
        self,
        name: str = "ma_slope_detector",
        version: str = "1.0.0",
        ma_periods: List[int] = None,
        slope_lookback: int = 5,
        slope_threshold: float = 0.001,
        min_confidence: float = 0.6,
        **kwargs: Any
    ):
        """
        Initialize the MA Slope Detector.
        
        Args:
            name: Algorithm name
            version: Algorithm version
            ma_periods: List of moving average periods to calculate (default: [10, 20, 50, 200])
            slope_lookback: Number of periods to use for slope calculation (default: 5)
            slope_threshold: Threshold for determining up/down vs sideways (default: 0.001)
            min_confidence: Minimum confidence score for trend detection
            **kwargs: Additional parameters
        """
        super().__init__(name, version, **kwargs)
        self.ma_periods = ma_periods if ma_periods is not None else [10, 20, 50, 200]
        self.slope_lookback = slope_lookback
        self.slope_threshold = slope_threshold
        self.min_confidence = min_confidence
    
    def process(self, data: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        """
        Process data and detect trends based on MA slopes.
        
        Args:
            data: DataFrame with OHLCV columns and datetime index
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with detection results
        """
        self.validate_data(data)
        
        # Use parameters from kwargs if provided
        ma_periods = kwargs.get('ma_periods', self.ma_periods)
        slope_lookback = kwargs.get('slope_lookback', self.slope_lookback)
        slope_threshold = kwargs.get('slope_threshold', self.slope_threshold)
        min_confidence = kwargs.get('min_confidence', self.min_confidence)
        
        # Check if we have enough data
        max_period = max(ma_periods)
        if len(data) < max_period + slope_lookback:
            return {
                "algorithm_name": self.name,
                "results": [],
                "confidence_scores": [],
                "metadata": {
                    "message": f"Insufficient data points (need at least {max_period + slope_lookback})"
                },
                "timestamp": datetime.now()
            }
        
        # Calculate all moving averages
        ma_data = self._calculate_moving_averages(data, ma_periods)
        
        # Calculate slopes for each MA
        ma_slopes = self._calculate_ma_slopes(ma_data, ma_periods, slope_lookback)
        
        # Classify trends for each MA
        trend_classifications = self._classify_trends(ma_slopes, slope_threshold)
        
        # Detect trend patterns and generate signals
        patterns = []
        confidence_scores = []
        
        # Current trend for each MA period
        current_idx = len(data) - 1
        current_timestamp = data.index[-1]
        
        for period in ma_periods:
            ma_col = f'ma_{period}'
            slope_col = f'slope_{period}'
            
            if ma_col not in ma_data.columns or slope_col not in ma_slopes.columns:
                continue
            
            trend = trend_classifications[period]
            slope_value = ma_slopes[slope_col].iloc[-1]
            ma_value = ma_data[ma_col].iloc[-1]
            
            # Calculate confidence based on slope magnitude and consistency
            confidence = self._calculate_confidence(
                ma_slopes[slope_col].tail(slope_lookback),
                slope_threshold
            )
            
            if confidence >= min_confidence:
                # Determine predicted direction
                predicted_direction = None
                if trend == "uptrend":
                    predicted_direction = "bullish"
                elif trend == "downtrend":
                    predicted_direction = "bearish"
                else:
                    predicted_direction = None  # Sideways/neutral
                
                pattern = PatternDetection(
                    pattern_type=f"ma_{period}_{trend}",
                    confidence=confidence,
                    start_index=max(0, current_idx - slope_lookback),
                    end_index=current_idx,
                    start_timestamp=data.index[max(0, current_idx - slope_lookback)],
                    end_timestamp=current_timestamp,
                    predicted_direction=predicted_direction,
                    characteristics={
                        "ma_period": period,
                        "ma_value": float(ma_value),
                        "slope": float(slope_value),
                        "trend": trend,
                        "slope_lookback": slope_lookback,
                        "slope_threshold": slope_threshold
                    }
                )
                patterns.append(pattern)
                confidence_scores.append(confidence)
        
        # Check for aligned trends (all MAs pointing same direction)
        aligned_pattern = self._detect_aligned_trends(
            trend_classifications,
            ma_slopes,
            slope_lookback,
            data,
            current_idx,
            current_timestamp,
            min_confidence
        )
        
        if aligned_pattern:
            patterns.append(aligned_pattern['pattern'])
            confidence_scores.append(aligned_pattern['confidence'])
        
        # Prepare metadata
        metadata = {
            "ma_periods": ma_periods,
            "slope_lookback": slope_lookback,
            "slope_threshold": slope_threshold,
            "min_confidence": min_confidence,
            "patterns_detected": len(patterns),
            "trend_summary": {
                str(period): trend_classifications[period]
                for period in ma_periods
            }
        }
        
        return {
            "algorithm_name": self.name,
            "results": patterns,
            "confidence_scores": confidence_scores,
            "metadata": metadata,
            "timestamp": datetime.now()
        }
    
    def _calculate_moving_averages(
        self,
        data: pd.DataFrame,
        ma_periods: List[int]
    ) -> pd.DataFrame:
        """Calculate simple moving averages for specified periods."""
        ma_data = data.copy()
        
        for period in ma_periods:
            ma_col = f'ma_{period}'
            ma_data[ma_col] = data['close'].rolling(window=period).mean()
        
        return ma_data
    
    def _calculate_ma_slopes(
        self,
        ma_data: pd.DataFrame,
        ma_periods: List[int],
        slope_lookback: int
    ) -> pd.DataFrame:
        """
        Calculate slopes for each moving average.
        
        Slope is calculated using linear regression over the lookback period.
        The slope represents the rate of change of the MA.
        """
        slopes_data = ma_data.copy()
        
        for period in ma_periods:
            ma_col = f'ma_{period}'
            slope_col = f'slope_{period}'
            
            if ma_col not in ma_data.columns:
                continue
            
            # Calculate rolling slope
            slopes = []
            ma_values = ma_data[ma_col].values
            
            for i in range(len(ma_values)):
                if i < slope_lookback - 1 or pd.isna(ma_values[i]):
                    slopes.append(np.nan)
                else:
                    # Get the last slope_lookback values
                    y = ma_values[i - slope_lookback + 1:i + 1]
                    x = np.arange(slope_lookback)
                    
                    # Check for NaN values
                    if np.any(np.isnan(y)):
                        slopes.append(np.nan)
                    else:
                        # Calculate linear regression slope
                        slope = self._calculate_slope(x, y)
                        # Normalize by the MA value to get percentage change
                        normalized_slope = slope / ma_values[i] if ma_values[i] != 0 else 0
                        slopes.append(normalized_slope)
            
            slopes_data[slope_col] = slopes
        
        return slopes_data
    
    def _calculate_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate linear regression slope using least squares."""
        if len(x) < 2:
            return 0.0
        
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
                (n * np.sum(x**2) - np.sum(x)**2)
        
        return float(slope)
    
    def _classify_trends(
        self,
        ma_slopes: pd.DataFrame,
        slope_threshold: float
    ) -> Dict[int, str]:
        """
        Classify trend for each MA based on its slope.
        
        Returns a dictionary mapping MA period to trend classification.
        """
        classifications = {}
        
        for col in ma_slopes.columns:
            if col.startswith('slope_'):
                period = int(col.split('_')[1])
                slope_value = ma_slopes[col].iloc[-1]
                
                if pd.isna(slope_value):
                    classifications[period] = "unknown"
                elif slope_value > slope_threshold:
                    classifications[period] = "uptrend"
                elif slope_value < -slope_threshold:
                    classifications[period] = "downtrend"
                else:
                    classifications[period] = "sideways"
        
        return classifications
    
    def _calculate_confidence(
        self,
        recent_slopes: pd.Series,
        slope_threshold: float
    ) -> float:
        """
        Calculate confidence score based on slope consistency.
        
        Higher confidence when:
        - Slope magnitude is larger
        - Slopes are consistent in direction
        - Less volatility in slope values
        """
        # Remove NaN values
        valid_slopes = recent_slopes.dropna()
        
        if len(valid_slopes) == 0:
            return 0.0
        
        current_slope = valid_slopes.iloc[-1]
        
        # Base confidence on slope magnitude
        magnitude_confidence = min(1.0, abs(current_slope) / (slope_threshold * 10))
        
        # Consistency: check if all slopes point in same direction
        if current_slope > slope_threshold:
            consistency = np.sum(valid_slopes > 0) / len(valid_slopes)
        elif current_slope < -slope_threshold:
            consistency = np.sum(valid_slopes < 0) / len(valid_slopes)
        else:
            # Sideways: check consistency of low slopes
            consistency = np.sum(np.abs(valid_slopes) < slope_threshold) / len(valid_slopes)
        
        # Stability: lower coefficient of variation = higher confidence
        slope_std = np.std(valid_slopes)
        slope_mean = np.mean(np.abs(valid_slopes))
        stability = 1.0 - min(1.0, slope_std / (slope_mean + 1e-6))
        
        # Combined confidence
        confidence = 0.4 * magnitude_confidence + 0.4 * consistency + 0.2 * stability
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _detect_aligned_trends(
        self,
        trend_classifications: Dict[int, str],
        ma_slopes: pd.DataFrame,
        slope_lookback: int,
        data: pd.DataFrame,
        current_idx: int,
        current_timestamp: Any,
        min_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """
        Detect when all moving averages are aligned in the same trend direction.
        
        This is a strong signal when short, medium, and long-term trends agree.
        """
        trends = list(trend_classifications.values())
        
        # Check if all MAs have the same trend (excluding "unknown")
        valid_trends = [t for t in trends if t != "unknown"]
        
        if len(valid_trends) < 2:
            return None
        
        # Count trend types
        uptrends = sum(1 for t in valid_trends if t == "uptrend")
        downtrends = sum(1 for t in valid_trends if t == "downtrend")
        sideways = sum(1 for t in valid_trends if t == "sideways")
        
        total = len(valid_trends)
        alignment_ratio = max(uptrends, downtrends, sideways) / total
        
        # Require at least 75% alignment
        if alignment_ratio < 0.75:
            return None
        
        # Determine the aligned trend
        if uptrends == max(uptrends, downtrends, sideways):
            aligned_trend = "uptrend"
            predicted_direction = "bullish"
        elif downtrends == max(uptrends, downtrends, sideways):
            aligned_trend = "downtrend"
            predicted_direction = "bearish"
        else:
            aligned_trend = "sideways"
            predicted_direction = None
        
        # Calculate average slope across all MAs
        slope_columns = [col for col in ma_slopes.columns if col.startswith('slope_')]
        avg_slope = ma_slopes[slope_columns].iloc[-1].mean()
        
        # Confidence based on alignment ratio and average slope magnitude
        confidence = alignment_ratio * 0.7 + min(0.3, abs(avg_slope) * 100)
        
        if confidence < min_confidence:
            return None
        
        pattern = PatternDetection(
            pattern_type=f"ma_aligned_{aligned_trend}",
            confidence=confidence,
            start_index=max(0, current_idx - slope_lookback),
            end_index=current_idx,
            start_timestamp=data.index[max(0, current_idx - slope_lookback)],
            end_timestamp=current_timestamp,
            predicted_direction=predicted_direction,
            characteristics={
                "aligned_trend": aligned_trend,
                "alignment_ratio": float(alignment_ratio),
                "avg_slope": float(avg_slope),
                "uptrends": uptrends,
                "downtrends": downtrends,
                "sideways": sideways,
                "total_mas": total
            }
        )
        
        return {
            "pattern": pattern,
            "confidence": confidence
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get algorithm metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Detects trends by calculating moving average slopes across multiple timeframes",
            "author": "ChartRadar",
            "parameters": {
                "ma_periods": self.ma_periods,
                "slope_lookback": self.slope_lookback,
                "slope_threshold": self.slope_threshold,
                "min_confidence": self.min_confidence
            },
            "requirements": self.get_requirements()
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get data requirements."""
        max_period = max(self.ma_periods) if self.ma_periods else 200
        min_data_points = max_period + self.slope_lookback
        
        return {
            "min_data_points": min_data_points,
            "required_columns": ["open", "high", "low", "close", "volume"],
            "data_frequency": "any",
            "preprocessing": []
        }

