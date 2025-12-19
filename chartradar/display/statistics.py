"""
Summary statistics generation for the ChartRadar framework.

This module provides functions to calculate pattern detection frequency,
average confidence scores, performance metrics, and generate summary reports.
"""

from typing import Any, Dict, List, Optional
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd
import numpy as np

from chartradar.core.types import AlgorithmResult, FusionResult, PatternDetection


class StatisticsGenerator:
    """
    Generator for summary statistics and reports.
    
    Provides functionality to calculate various metrics and generate
    comprehensive summary reports.
    """
    
    def __init__(self, **kwargs: Any):
        """
        Initialize the statistics generator.
        
        Args:
            **kwargs: Generator-specific parameters
        """
        self.parameters = kwargs
    
    def generate_summary(
        self,
        results: Any,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate summary statistics from results.
        
        Args:
            results: AlgorithmResult, FusionResult, or list of results
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with summary statistics
        """
        if isinstance(results, AlgorithmResult):
            return self._generate_algorithm_summary(results)
        elif isinstance(results, FusionResult):
            return self._generate_fusion_summary(results)
        elif isinstance(results, list):
            return self._generate_multi_algorithm_summary(results)
        else:
            return {"error": "Unsupported results type"}
    
    def _generate_algorithm_summary(self, result: AlgorithmResult) -> Dict[str, Any]:
        """Generate summary for a single algorithm result."""
        pattern_types = Counter()
        confidences = []
        
        for pattern in result.results:
            if isinstance(pattern, PatternDetection):
                pattern_types[pattern.pattern_type] += 1
                confidences.append(pattern.confidence)
            else:
                pattern_types[pattern.get('pattern_type', 'unknown')] += 1
                confidences.append(pattern.get('confidence', 0.0))
        
        return {
            "algorithm_name": result.algorithm_name,
            "algorithm_version": result.algorithm_version,
            "success": result.success,
            "total_patterns": len(result.results),
            "pattern_type_distribution": dict(pattern_types),
            "average_confidence": np.mean(confidences) if confidences else 0.0,
            "min_confidence": np.min(confidences) if confidences else 0.0,
            "max_confidence": np.max(confidences) if confidences else 0.0,
            "processing_time_ms": result.processing_time_ms,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "error_message": result.error_message
        }
    
    def _generate_fusion_summary(self, result: FusionResult) -> Dict[str, Any]:
        """Generate summary for fusion result."""
        patterns = result.fused_result.get('patterns', [])
        pattern_types = Counter()
        confidences = []
        
        for pattern in patterns:
            if isinstance(pattern, PatternDetection):
                pattern_types[pattern.pattern_type] += 1
                confidences.append(pattern.confidence)
            else:
                pattern_types[pattern.get('pattern_type', 'unknown')] += 1
                confidences.append(pattern.get('confidence', 0.0))
        
        return {
            "fusion_method": result.fusion_method,
            "total_patterns": len(patterns),
            "pattern_type_distribution": dict(pattern_types),
            "fused_confidence": result.confidence_score,
            "contributing_algorithms": len(result.contributing_algorithms),
            "algorithm_names": result.contributing_algorithms,
            "average_confidence": np.mean(confidences) if confidences else 0.0,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None
        }
    
    def _generate_multi_algorithm_summary(self, results: List[AlgorithmResult]) -> Dict[str, Any]:
        """Generate summary for multiple algorithm results."""
        algorithm_summaries = []
        all_pattern_types = Counter()
        all_confidences = []
        
        for result in results:
            summary = self._generate_algorithm_summary(result)
            algorithm_summaries.append(summary)
            
            # Aggregate pattern types
            for pattern_type, count in summary["pattern_type_distribution"].items():
                all_pattern_types[pattern_type] += count
            
            # Aggregate confidences
            if summary["average_confidence"] > 0:
                all_confidences.append(summary["average_confidence"])
        
        return {
            "total_algorithms": len(results),
            "successful_algorithms": sum(1 for r in results if r.success),
            "failed_algorithms": sum(1 for r in results if not r.success),
            "total_patterns": sum(len(r.results) for r in results),
            "overall_pattern_type_distribution": dict(all_pattern_types),
            "overall_average_confidence": np.mean(all_confidences) if all_confidences else 0.0,
            "algorithm_summaries": algorithm_summaries,
            "performance_metrics": self._calculate_performance_metrics(results)
        }
    
    def _calculate_performance_metrics(self, results: List[AlgorithmResult]) -> Dict[str, Any]:
        """Calculate performance metrics per algorithm."""
        metrics = {}
        
        for result in results:
            if not result.success:
                continue
            
            processing_time = result.processing_time_ms or 0.0
            pattern_count = len(result.results)
            
            metrics[result.algorithm_name] = {
                "processing_time_ms": processing_time,
                "patterns_per_second": pattern_count / (processing_time / 1000) if processing_time > 0 else 0.0,
                "patterns_detected": pattern_count,
                "avg_confidence": (
                    np.mean(result.confidence_scores) if result.confidence_scores else 0.0
                ),
                "success_rate": 1.0 if result.success else 0.0
            }
        
        return metrics
    
    def calculate_pattern_frequency(
        self,
        results: Any,
        time_window: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate pattern detection frequency.
        
        Args:
            results: AlgorithmResult, FusionResult, or list of results
            time_window: Optional time window for frequency calculation
            
        Returns:
            Dictionary with frequency statistics
        """
        patterns = []
        
        if isinstance(results, AlgorithmResult):
            patterns = results.results
        elif isinstance(results, FusionResult):
            patterns = results.fused_result.get('patterns', [])
        elif isinstance(results, list):
            for result in results:
                if isinstance(result, AlgorithmResult):
                    patterns.extend(result.results)
                elif isinstance(result, FusionResult):
                    patterns.extend(result.fused_result.get('patterns', []))
        
        pattern_types = Counter()
        timestamps = []
        
        for pattern in patterns:
            if isinstance(pattern, PatternDetection):
                pattern_types[pattern.pattern_type] += 1
                timestamps.append(pattern.start_timestamp)
            else:
                pattern_types[pattern.get('pattern_type', 'unknown')] += 1
                if 'start_timestamp' in pattern:
                    timestamps.append(pattern['start_timestamp'])
        
        frequency_stats = {
            "total_patterns": len(patterns),
            "pattern_type_counts": dict(pattern_types),
            "pattern_type_percentages": {
                ptype: (count / len(patterns) * 100) if patterns else 0.0
                for ptype, count in pattern_types.items()
            }
        }
        
        if timestamps and time_window:
            # Calculate frequency over time window
            df = pd.DataFrame({"timestamp": timestamps})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            frequency_stats["temporal_distribution"] = self._calculate_temporal_frequency(
                df, time_window
            )
        
        return frequency_stats
    
    def _calculate_temporal_frequency(
        self,
        df: pd.DataFrame,
        time_window: str
    ) -> Dict[str, Any]:
        """Calculate frequency over time windows."""
        df = df.set_index('timestamp')
        resampled = df.resample(time_window).size()
        
        return {
            "window": time_window,
            "counts": resampled.to_dict(),
            "average_per_window": float(resampled.mean()) if len(resampled) > 0 else 0.0,
            "max_per_window": int(resampled.max()) if len(resampled) > 0 else 0,
            "min_per_window": int(resampled.min()) if len(resampled) > 0 else 0
        }
    
    def generate_report(
        self,
        results: Any,
        format: str = "dict"
    ) -> Any:
        """
        Generate a comprehensive report.
        
        Args:
            results: Results to generate report for
            format: Output format ('dict', 'dataframe', 'string')
            
        Returns:
            Report in requested format
        """
        summary = self.generate_summary(results)
        frequency = self.calculate_pattern_frequency(results)
        
        if format == "dataframe":
            # Convert to DataFrame
            rows = []
            if isinstance(summary, dict) and "algorithm_summaries" in summary:
                for alg_summary in summary["algorithm_summaries"]:
                    rows.append({
                        "algorithm": alg_summary["algorithm_name"],
                        "patterns": alg_summary["total_patterns"],
                        "avg_confidence": alg_summary["average_confidence"],
                        "processing_time_ms": alg_summary["processing_time_ms"]
                    })
            return pd.DataFrame(rows)
        elif format == "string":
            # Create human-readable string
            lines = [
                "ChartRadar Analysis Report",
                "=" * 50,
                f"Generated: {datetime.now().isoformat()}",
                "",
                "Summary:",
            ]
            
            if isinstance(summary, dict):
                if "total_algorithms" in summary:
                    lines.append(f"Total Algorithms: {summary['total_algorithms']}")
                    lines.append(f"Successful: {summary['successful_algorithms']}")
                    lines.append(f"Failed: {summary['failed_algorithms']}")
                    lines.append(f"Total Patterns: {summary['total_patterns']}")
                else:
                    lines.append(f"Algorithm: {summary.get('algorithm_name', 'Unknown')}")
                    lines.append(f"Patterns: {summary.get('total_patterns', 0)}")
                    lines.append(f"Avg Confidence: {summary.get('average_confidence', 0.0):.2f}")
            
            lines.append("")
            lines.append("Pattern Frequency:")
            for ptype, count in frequency.get("pattern_type_counts", {}).items():
                lines.append(f"  {ptype}: {count}")
            
            return "\n".join(lines)
        else:
            return {
                "summary": summary,
                "frequency": frequency
            }

