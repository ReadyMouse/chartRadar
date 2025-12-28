"""
Tests for the Moving Average Slope Detector.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from chartradar.metrics.algorithms.rule_based.ma_slope_detector import MovingAverageSlopeDetector
from chartradar.src.exceptions import AlgorithmError


@pytest.fixture
def sample_uptrend_data():
    """Create sample data with an uptrend."""
    dates = pd.date_range(start='2024-01-01', periods=300, freq='h')
    
    # Create uptrending data
    base_price = 100
    trend = np.linspace(0, 50, 300)
    noise = np.random.normal(0, 2, 300)
    close_prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices - np.random.uniform(0, 1, 300),
        'high': close_prices + np.random.uniform(0, 2, 300),
        'low': close_prices - np.random.uniform(0, 2, 300),
        'close': close_prices,
        'volume': np.random.uniform(1000, 5000, 300)
    })
    data.set_index('timestamp', inplace=True)
    
    return data


@pytest.fixture
def sample_downtrend_data():
    """Create sample data with a downtrend."""
    dates = pd.date_range(start='2024-01-01', periods=300, freq='h')
    
    # Create downtrending data
    base_price = 150
    trend = np.linspace(0, -50, 300)
    noise = np.random.normal(0, 2, 300)
    close_prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.uniform(0, 1, 300),
        'high': close_prices + np.random.uniform(0, 2, 300),
        'low': close_prices - np.random.uniform(0, 2, 300),
        'close': close_prices,
        'volume': np.random.uniform(1000, 5000, 300)
    })
    data.set_index('timestamp', inplace=True)
    
    return data


@pytest.fixture
def sample_sideways_data():
    """Create sample data with sideways movement."""
    dates = pd.date_range(start='2024-01-01', periods=300, freq='h')
    
    # Create sideways data
    base_price = 100
    noise = np.random.normal(0, 2, 300)
    close_prices = base_price + noise
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.uniform(-0.5, 0.5, 300),
        'high': close_prices + np.random.uniform(0, 1, 300),
        'low': close_prices - np.random.uniform(0, 1, 300),
        'close': close_prices,
        'volume': np.random.uniform(1000, 5000, 300)
    })
    data.set_index('timestamp', inplace=True)
    
    return data


class TestMovingAverageSlopeDetector:
    """Tests for MovingAverageSlopeDetector class."""
    
    def test_initialization_default(self):
        """Test detector initialization with default parameters."""
        detector = MovingAverageSlopeDetector()
        
        assert detector.name == "ma_slope_detector"
        assert detector.version == "1.0.0"
        assert detector.ma_periods == [10, 20, 50, 200]
        assert detector.slope_lookback == 5
        assert detector.slope_threshold == 0.001
        assert detector.min_confidence == 0.6
    
    def test_initialization_custom(self):
        """Test detector initialization with custom parameters."""
        detector = MovingAverageSlopeDetector(
            ma_periods=[5, 10, 20],
            slope_lookback=3,
            slope_threshold=0.002,
            min_confidence=0.7
        )
        
        assert detector.ma_periods == [5, 10, 20]
        assert detector.slope_lookback == 3
        assert detector.slope_threshold == 0.002
        assert detector.min_confidence == 0.7
    
    def test_get_metadata(self):
        """Test get_metadata method."""
        detector = MovingAverageSlopeDetector()
        metadata = detector.get_metadata()
        
        assert metadata['name'] == "ma_slope_detector"
        assert metadata['version'] == "1.0.0"
        assert 'description' in metadata
        assert 'parameters' in metadata
        assert 'requirements' in metadata
    
    def test_get_requirements(self):
        """Test get_requirements method."""
        detector = MovingAverageSlopeDetector(
            ma_periods=[10, 20, 50],
            slope_lookback=5
        )
        requirements = detector.get_requirements()
        
        assert requirements['min_data_points'] == 55  # 50 + 5
        assert 'close' in requirements['required_columns']
        assert requirements['data_frequency'] == 'any'
    
    def test_process_uptrend_data(self, sample_uptrend_data):
        """Test processing uptrend data."""
        detector = MovingAverageSlopeDetector(
            ma_periods=[10, 20, 50],
            slope_lookback=5,
            slope_threshold=0.001,
            min_confidence=0.5
        )
        
        results = detector.process(sample_uptrend_data)
        
        assert results['algorithm_name'] == "ma_slope_detector"
        assert 'results' in results
        assert 'confidence_scores' in results
        assert 'metadata' in results
        assert 'timestamp' in results
        
        # Should detect uptrends
        trend_summary = results['metadata']['trend_summary']
        uptrends = sum(1 for trend in trend_summary.values() if trend == 'uptrend')
        assert uptrends > 0, "Should detect at least one uptrend"
    
    def test_process_downtrend_data(self, sample_downtrend_data):
        """Test processing downtrend data."""
        detector = MovingAverageSlopeDetector(
            ma_periods=[10, 20, 50],
            slope_lookback=5,
            slope_threshold=0.001,
            min_confidence=0.5
        )
        
        results = detector.process(sample_downtrend_data)
        
        # Should detect downtrends
        trend_summary = results['metadata']['trend_summary']
        downtrends = sum(1 for trend in trend_summary.values() if trend == 'downtrend')
        assert downtrends > 0, "Should detect at least one downtrend"
    
    def test_process_sideways_data(self, sample_sideways_data):
        """Test processing sideways data."""
        detector = MovingAverageSlopeDetector(
            ma_periods=[10, 20, 50],
            slope_lookback=5,
            slope_threshold=0.001,
            min_confidence=0.5
        )
        
        results = detector.process(sample_sideways_data)
        
        # Should detect sideways trends
        trend_summary = results['metadata']['trend_summary']
        sideways = sum(1 for trend in trend_summary.values() if trend == 'sideways')
        assert sideways > 0, "Should detect at least one sideways trend"
    
    def test_process_insufficient_data(self):
        """Test processing with insufficient data."""
        detector = MovingAverageSlopeDetector(
            ma_periods=[50, 100],
            slope_lookback=5
        )
        
        # Create small dataset
        dates = pd.date_range(start='2024-01-01', periods=50, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(99, 101, 50),
            'high': np.random.uniform(100, 102, 50),
            'low': np.random.uniform(98, 100, 50),
            'close': np.random.uniform(99, 101, 50),
            'volume': np.random.uniform(1000, 5000, 50)
        })
        data.set_index('timestamp', inplace=True)
        
        # Should raise AlgorithmError due to insufficient data
        with pytest.raises(AlgorithmError):
            detector.process(data)
    
    def test_process_with_kwargs(self, sample_uptrend_data):
        """Test processing with keyword arguments override."""
        detector = MovingAverageSlopeDetector(
            ma_periods=[10, 20],
            slope_threshold=0.001
        )
        
        # Override parameters via kwargs
        results = detector.process(
            sample_uptrend_data,
            ma_periods=[5, 10],
            slope_threshold=0.002,
            min_confidence=0.7
        )
        
        # Check that kwargs were used
        assert results['metadata']['ma_periods'] == [5, 10]
        assert results['metadata']['slope_threshold'] == 0.002
        assert results['metadata']['min_confidence'] == 0.7
    
    def test_confidence_calculation(self, sample_uptrend_data):
        """Test confidence score calculation."""
        detector = MovingAverageSlopeDetector(
            ma_periods=[20],
            slope_lookback=5,
            min_confidence=0.0  # Allow all detections
        )
        
        results = detector.process(sample_uptrend_data)
        
        # Check confidence scores
        for score in results['confidence_scores']:
            assert 0.0 <= score <= 1.0, "Confidence must be between 0 and 1"
    
    def test_pattern_characteristics(self, sample_uptrend_data):
        """Test that patterns have required characteristics."""
        detector = MovingAverageSlopeDetector(
            ma_periods=[20],
            min_confidence=0.5
        )
        
        results = detector.process(sample_uptrend_data)
        
        if results['results']:
            pattern = results['results'][0]
            chars = pattern.characteristics
            
            # Check required characteristics
            assert 'ma_period' in chars
            assert 'ma_value' in chars
            assert 'slope' in chars
            assert 'trend' in chars
            assert chars['trend'] in ['uptrend', 'downtrend', 'sideways']
    
    def test_aligned_trends_detection(self, sample_uptrend_data):
        """Test detection of aligned trends."""
        detector = MovingAverageSlopeDetector(
            ma_periods=[10, 20, 50],
            slope_lookback=5,
            slope_threshold=0.001,
            min_confidence=0.5
        )
        
        results = detector.process(sample_uptrend_data)
        
        # Check for aligned pattern
        aligned_patterns = [
            p for p in results['results']
            if 'aligned' in p.pattern_type
        ]
        
        # In strong uptrend, should detect alignment
        if aligned_patterns:
            aligned = aligned_patterns[0]
            assert 'alignment_ratio' in aligned.characteristics
            assert aligned.characteristics['alignment_ratio'] >= 0.75
    
    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        detector = MovingAverageSlopeDetector()
        
        # Create data with missing 'close' column
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(99, 101, 100),
            'high': np.random.uniform(100, 102, 100),
            'low': np.random.uniform(98, 100, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })
        data.set_index('timestamp', inplace=True)
        
        with pytest.raises(AlgorithmError):
            detector.validate_data(data)
    
    def test_predicted_direction(self, sample_uptrend_data, sample_downtrend_data):
        """Test that predicted direction matches trend."""
        detector = MovingAverageSlopeDetector(
            ma_periods=[20],
            min_confidence=0.5
        )
        
        # Test uptrend
        results_up = detector.process(sample_uptrend_data)
        if results_up['results']:
            for pattern in results_up['results']:
                if pattern.characteristics.get('trend') == 'uptrend':
                    assert pattern.predicted_direction == 'bullish'
        
        # Test downtrend
        results_down = detector.process(sample_downtrend_data)
        if results_down['results']:
            for pattern in results_down['results']:
                if pattern.characteristics.get('trend') == 'downtrend':
                    assert pattern.predicted_direction == 'bearish'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

