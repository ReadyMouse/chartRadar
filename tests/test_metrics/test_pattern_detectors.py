"""Tests for pattern detection algorithms."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from chartradar.metrics.algorithms.rule_based.wedge_detector import WedgeDetector
from chartradar.metrics.algorithms.rule_based.triangle_detector import TriangleDetector


class TestWedgeDetector:
    """Tests for WedgeDetector algorithm."""
    
    def test_initialization(self):
        """Test wedge detector initialization."""
        detector = WedgeDetector(
            min_confidence=0.7,
            lookback_period=50,
            min_touches=3
        )
        assert detector.min_confidence == 0.7
        assert detector.lookback_period == 50
    
    def test_process_insufficient_data(self):
        """Test processing with insufficient data."""
        from chartradar.src.exceptions import AlgorithmError
        
        detector = WedgeDetector(lookback_period=50)
        data = pd.DataFrame({
            'open': range(10),
            'high': range(10, 20),
            'low': range(0, 10),
            'close': range(5, 15),
            'volume': range(100, 110)
        }, index=pd.date_range('2024-01-01', periods=10))
        
        # Should raise AlgorithmError for insufficient data
        with pytest.raises(AlgorithmError) as exc_info:
            detector.process(data)
        assert "requires at least" in str(exc_info.value).lower() or "data points" in str(exc_info.value).lower()
    
    def test_process_sufficient_data(self):
        """Test processing with sufficient data."""
        detector = WedgeDetector(lookback_period=20, min_confidence=0.3)
        
        # Create data with potential wedge pattern
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        # Simulate rising wedge: higher highs and higher lows, but converging
        highs = np.linspace(100, 120, 50) + np.random.randn(50) * 2
        lows = np.linspace(90, 110, 50) + np.random.randn(50) * 2
        
        data = pd.DataFrame({
            'open': (highs + lows) / 2,
            'high': highs,
            'low': lows,
            'close': (highs + lows) / 2 + np.random.randn(50),
            'volume': np.random.randint(1000, 2000, 50)
        }, index=dates)
        
        result = detector.process(data)
        assert "results" in result
        assert "confidence_scores" in result
    
    def test_get_metadata(self):
        """Test getting metadata."""
        detector = WedgeDetector()
        metadata = detector.get_metadata()
        
        assert metadata["name"] == "wedge_detector"
        assert "description" in metadata
        assert "parameters" in metadata
    
    def test_get_requirements(self):
        """Test getting requirements."""
        detector = WedgeDetector(lookback_period=50)
        requirements = detector.get_requirements()
        
        assert requirements["min_data_points"] == 50
        assert "open" in requirements["required_columns"]


class TestTriangleDetector:
    """Tests for TriangleDetector algorithm."""
    
    def test_initialization(self):
        """Test triangle detector initialization."""
        detector = TriangleDetector(
            min_confidence=0.7,
            lookback_period=60,
            min_touches=3
        )
        assert detector.min_confidence == 0.7
        assert detector.lookback_period == 60
    
    def test_process_insufficient_data(self):
        """Test processing with insufficient data."""
        from chartradar.src.exceptions import AlgorithmError
        
        detector = TriangleDetector(lookback_period=60)
        data = pd.DataFrame({
            'open': range(10),
            'high': range(10, 20),
            'low': range(0, 10),
            'close': range(5, 15),
            'volume': range(100, 110)
        }, index=pd.date_range('2024-01-01', periods=10))
        
        # Should raise AlgorithmError for insufficient data
        with pytest.raises(AlgorithmError) as exc_info:
            detector.process(data)
        assert "requires at least" in str(exc_info.value).lower() or "data points" in str(exc_info.value).lower()
    
    def test_process_sufficient_data(self):
        """Test processing with sufficient data."""
        detector = TriangleDetector(lookback_period=30, min_confidence=0.3)
        
        # Create data with potential triangle pattern
        dates = pd.date_range('2024-01-01', periods=70, freq='D')
        # Simulate rising triangle: flat highs, rising lows
        highs = np.full(70, 120) + np.random.randn(70) * 1
        lows = np.linspace(90, 110, 70) + np.random.randn(70) * 1
        
        data = pd.DataFrame({
            'open': (highs + lows) / 2,
            'high': highs,
            'low': lows,
            'close': (highs + lows) / 2 + np.random.randn(70),
            'volume': np.random.randint(1000, 2000, 70)
        }, index=dates)
        
        result = detector.process(data)
        assert "results" in result
        assert "confidence_scores" in result
    
    def test_get_metadata(self):
        """Test getting metadata."""
        detector = TriangleDetector()
        metadata = detector.get_metadata()
        
        assert metadata["name"] == "triangle_detector"
        assert "description" in metadata
    
    def test_get_requirements(self):
        """Test getting requirements."""
        detector = TriangleDetector(lookback_period=60)
        requirements = detector.get_requirements()
        
        assert requirements["min_data_points"] == 60

