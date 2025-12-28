"""Tests for algorithm comparator."""

import pytest
from datetime import datetime

from chartradar.display.comparator import AlgorithmComparator
from chartradar.src.types import AlgorithmResult, PatternDetection


class TestAlgorithmComparator:
    """Tests for AlgorithmComparator class."""
    
    def test_compare_empty_results(self):
        """Test comparison with empty results."""
        comparator = AlgorithmComparator()
        result = comparator.compare([])
        
        assert result["total_algorithms"] == 0
        assert len(result["agreements"]) == 0
    
    def test_compare_single_algorithm(self):
        """Test comparison with single algorithm."""
        comparator = AlgorithmComparator()
        
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
            )
        ]
        
        comparison = comparator.compare(results)
        assert comparison["total_algorithms"] == 1
        assert "Need at least 2" in comparison["statistics"]["message"]
    
    def test_compare_multiple_algorithms(self):
        """Test comparison with multiple algorithms."""
        comparator = AlgorithmComparator()
        
        base_time = datetime.now()
        
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
                        start_timestamp=base_time,
                        end_timestamp=base_time
                    )
                ],
                confidence_scores=[0.8],
                timestamp=base_time
            ),
            AlgorithmResult(
                algorithm_name="alg2",
                success=True,
                results=[
                    PatternDetection(
                        pattern_type="rising_wedge",
                        confidence=0.9,
                        start_index=2,
                        end_index=12,
                        start_timestamp=base_time,
                        end_timestamp=base_time
                    )
                ],
                confidence_scores=[0.9],
                timestamp=base_time
            )
        ]
        
        comparison = comparator.compare(results, overlap_threshold=0.3)
        assert comparison["total_algorithms"] == 2
        assert "statistics" in comparison
    
    def test_generate_comparison_report_dict(self):
        """Test generating comparison report as dict."""
        comparator = AlgorithmComparator()
        
        results = [
            AlgorithmResult(
                algorithm_name="alg1",
                success=True,
                results=[],
                confidence_scores=[],
                timestamp=datetime.now()
            )
        ]
        
        report = comparator.generate_comparison_report(results, output_format="dict")
        assert isinstance(report, dict)
        assert "total_algorithms" in report
    
    def test_generate_comparison_report_string(self):
        """Test generating comparison report as string."""
        comparator = AlgorithmComparator()
        
        results = [
            AlgorithmResult(
                algorithm_name="alg1",
                success=True,
                results=[],
                confidence_scores=[],
                timestamp=datetime.now()
            )
        ]
        
        report = comparator.generate_comparison_report(results, output_format="string")
        assert isinstance(report, str)
        assert "Algorithm Comparison Report" in report

