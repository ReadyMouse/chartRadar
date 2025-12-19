"""
Type definitions and data models for the ChartRadar framework.

This module provides Pydantic models for standardized data formats
used throughout the framework.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import pandas as pd


class OHLCVData(BaseModel):
    """
    Model for normalized OHLCV (Open, High, Low, Close, Volume) price data.
    
    This model represents a single data point in a time series.
    """
    timestamp: datetime = Field(..., description="Timestamp for this data point")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Trading volume")
    
    @field_validator('high')
    @classmethod
    def validate_high(cls, v, info):
        """Validate that high >= low."""
        if hasattr(info, 'data') and 'low' in info.data:
            if v < info.data['low']:
                raise ValueError('high must be >= low')
        return v
    
    @field_validator('low')
    @classmethod
    def validate_low(cls, v, info):
        """Validate that low <= high."""
        if hasattr(info, 'data') and 'high' in info.data:
            if v > info.data['high']:
                raise ValueError('low must be <= high')
        return v
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List['OHLCVData']:
        """
        Create OHLCVData instances from a pandas DataFrame.
        
        Args:
            df: DataFrame with OHLCV columns and datetime index
            
        Returns:
            List of OHLCVData instances
        """
        records = []
        for idx, row in df.iterrows():
            timestamp = idx if isinstance(idx, datetime) else pd.to_datetime(idx)
            records.append(cls(
                timestamp=timestamp,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0))
            ))
        return records


class PatternDetection(BaseModel):
    """
    Model for pattern detection results.
    
    Represents a single detected pattern with its characteristics.
    """
    pattern_type: str = Field(..., description="Type of pattern (e.g., 'rising_wedge', 'falling_triangle')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    start_index: int = Field(..., ge=0, description="Start index of pattern in data")
    end_index: int = Field(..., ge=0, description="End index of pattern in data")
    start_timestamp: datetime = Field(..., description="Start timestamp of pattern")
    end_timestamp: datetime = Field(..., description="End timestamp of pattern")
    predicted_direction: Optional[str] = Field(None, description="Predicted price direction ('bullish', 'bearish', None)")
    price_target: Optional[float] = Field(None, description="Estimated price target")
    characteristics: Dict[str, Any] = Field(default_factory=dict, description="Pattern-specific characteristics")
    
    @field_validator('end_index')
    @classmethod
    def validate_indices(cls, v, info):
        """Validate that end_index >= start_index."""
        if hasattr(info, 'data') and 'start_index' in info.data:
            if v < info.data['start_index']:
                raise ValueError('end_index must be >= start_index')
        return v


class AlgorithmResult(BaseModel):
    """
    Model for algorithm output results.
    
    Standardized format for all algorithm outputs.
    """
    algorithm_name: str = Field(..., description="Name of the algorithm")
    algorithm_version: Optional[str] = Field(None, description="Algorithm version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
    results: List[PatternDetection] = Field(default_factory=list, description="List of detected patterns")
    confidence_scores: List[float] = Field(default_factory=list, description="Confidence scores for each result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional algorithm-specific metadata")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    success: bool = Field(True, description="Whether processing was successful")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            pd.Timestamp: lambda v: v.isoformat()
        }


class FusionResult(BaseModel):
    """
    Model for fused results from multiple algorithms.
    
    Represents the combined output from fusion strategies.
    """
    fusion_method: str = Field(..., description="Name of the fusion method used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Fusion timestamp")
    fused_result: Dict[str, Any] = Field(..., description="Combined/aggregated result")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Aggregated confidence score")
    contributing_algorithms: List[str] = Field(default_factory=list, description="List of algorithm names that contributed")
    individual_results: List[AlgorithmResult] = Field(default_factory=list, description="Individual algorithm results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional fusion-specific metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            pd.Timestamp: lambda v: v.isoformat()
        }

