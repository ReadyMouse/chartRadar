"""Tests for statistics generator."""

import pytest
from datetime import datetime

from chartradar.display.statistics import StatisticsGenerator
from chartradar.core.types import AlgorithmResult, PatternDetection


class TestStatisticsGenerator:
    """Tests for StatisticsGenerator class."""
    
    def test_generate_algorithm_summary(self):
        """Test generating summary for single algorithm."""
        generator = StatisticsGenerator()
        
        result = AlgorithmResult(
            algorithm_name="test_alg",
            success=True,
            results=[
                PatternDetection(
                    pattern_type="rising_wedge",
                    confidence=0.8,
                    start_index=0,
                    end_index=10,
                    start_timestamp=datetime.now(),
                    end_timestamp=datetime.now()
                ),
                PatternDetection(
                    pattern_type="falling_wedge",
                    confidence=0.9,
                    start_index=20,
                    end_index=30,
                    start_timestamp=datetime.now(),
                    end_timestamp=datetime.now()
                )
            ],
            confidence_scores=[0.8, 0.9],
            processing_time_ms=100.0,
            timestamp=datetime.now()
        )
        
        summary = generator.generate_summary(result)
        
        assert summary["algorithm_name"] == "test_alg"
        assert summary["total_patterns"] == 2
        assert summary["average_confidence"] > 0
        assert "rising_wedge" in summary["pattern_type_distribution"]
    
    def test_generate_multi_algorithm_summary(self):
        """Test generating summary for multiple algorithms."""
        generator = StatisticsGenerator()
        
        results = [
            AlgorithmResult(
                algorithm_name="alg1",
                success=True,
                results=[
                    PatternDetection(
                        pattern_type="rising_wedge",
                        confidence=0.8,
                        start_index=0,
                        end_index=10,
                        start_timestamp=datetime.now(),
                        end_timestamp=datetime.now()
                    )
                ],
                confidence_scores=[0.8],
                timestamp=datetime.now()
            ),
            AlgorithmResult(
                algorithm_name="alg2",
                success=True,
                results=[],
                confidence_scores=[],
                timestamp=datetime.now()
            )
        ]
        
        summary = generator.generate_summary(results)
        
        assert summary["total_algorithms"] == 2
        assert summary["successful_algorithms"] == 2
        assert "algorithm_summaries" in summary
    
    def test_calculate_pattern_frequency(self):
        """Test calculating pattern frequency."""
        generator = StatisticsGenerator()
        
        result = AlgorithmResult(
            algorithm_name="test_alg",
            success=True,
            results=[
                PatternDetection(
                    pattern_type="rising_wedge",
                    confidence=0.8,
                    start_index=0,
                    end_index=10,
                    start_timestamp=datetime.now(),
                    end_timestamp=datetime.now()
                ),
                PatternDetection(
                    pattern_type="rising_wedge",
                    confidence=0.9,
                    start_index=20,
                    end_index=30,
                    start_timestamp=datetime.now(),
                    end_timestamp=datetime.now()
                ),
                PatternDetection(
                    pattern_type="falling_wedge",
                    confidence=0.7,
                    start_index=40,
                    end_index=50,
                    start_timestamp=datetime.now(),
                    end_timestamp=datetime.now()
                )
            ],
            confidence_scores=[0.8, 0.9, 0.7],
            timestamp=datetime.now()
        )
        
        frequency = generator.calculate_pattern_frequency(result)
        
        assert frequency["total_patterns"] == 3
        assert frequency["pattern_type_counts"]["rising_wedge"] == 2
        assert frequency["pattern_type_counts"]["falling_wedge"] == 1
    
    def test_generate_report_string(self):
        """Test generating report as string."""
        generator = StatisticsGenerator()
        
        result = AlgorithmResult(
            algorithm_name="test_alg",
            success=True,
            results=[],
            confidence_scores=[],
            timestamp=datetime.now()
        )
        
        report = generator.generate_report(result, format="string")
        assert isinstance(report, str)
        assert "ChartRadar Analysis Report" in report
    
    def test_generate_report_dict(self):
        """Test generating report as dict."""
        generator = StatisticsGenerator()
        
        result = AlgorithmResult(
            algorithm_name="test_alg",
            success=True,
            results=[],
            confidence_scores=[],
            timestamp=datetime.now()
        )
        
        report = generator.generate_report(result, format="dict")
        assert isinstance(report, dict)
        assert "summary" in report
        assert "frequency" in report

