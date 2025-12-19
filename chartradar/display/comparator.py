"""
Algorithm comparison utilities for the ChartRadar framework.

This module provides functions to compare outputs from multiple algorithms,
generate comparison statistics, and identify agreement/disagreement.
"""

from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict, Counter
import pandas as pd

from chartradar.core.types import AlgorithmResult, PatternDetection


class AlgorithmComparator:
    """
    Comparator for analyzing and comparing algorithm outputs.
    
    Provides functionality to compare results from multiple algorithms,
    identify agreement/disagreement, and generate comparison statistics.
    """
    
    def __init__(self, **kwargs: Any):
        """
        Initialize the comparator.
        
        Args:
            **kwargs: Comparator-specific parameters
        """
        self.parameters = kwargs
    
    def compare(
        self,
        results: List[AlgorithmResult],
        overlap_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Compare outputs from multiple algorithms.
        
        Args:
            results: List of AlgorithmResult objects
            overlap_threshold: Minimum overlap ratio to consider patterns as matching
            
        Returns:
            Dictionary with comparison results
        """
        if not results:
            return {
                "total_algorithms": 0,
                "agreements": [],
                "disagreements": [],
                "statistics": {}
            }
        
        # Filter successful results
        successful_results = [r for r in results if r.success]
        
        if len(successful_results) < 2:
            return {
                "total_algorithms": len(successful_results),
                "agreements": [],
                "disagreements": [],
                "statistics": {
                    "message": "Need at least 2 successful algorithms for comparison"
                }
            }
        
        # Extract all patterns
        all_patterns = []
        for result in successful_results:
            for pattern in result.results:
                all_patterns.append({
                    "algorithm": result.algorithm_name,
                    "pattern": pattern
                })
        
        # Find agreements (patterns detected by multiple algorithms)
        agreements = self._find_agreements(all_patterns, overlap_threshold)
        
        # Find disagreements (patterns detected by only one algorithm)
        disagreements = self._find_disagreements(all_patterns, agreements)
        
        # Generate statistics
        statistics = self._generate_comparison_statistics(
            successful_results,
            agreements,
            disagreements
        )
        
        return {
            "total_algorithms": len(successful_results),
            "algorithm_names": [r.algorithm_name for r in successful_results],
            "agreements": agreements,
            "disagreements": disagreements,
            "statistics": statistics
        }
    
    def _find_agreements(
        self,
        all_patterns: List[Dict[str, Any]],
        overlap_threshold: float
    ) -> List[Dict[str, Any]]:
        """Find patterns that multiple algorithms agree on."""
        agreements = []
        processed = set()
        
        for i, pattern1 in enumerate(all_patterns):
            if i in processed:
                continue
            
            pattern_obj1 = pattern1["pattern"]
            if isinstance(pattern_obj1, PatternDetection):
                start1 = pattern_obj1.start_index
                end1 = pattern_obj1.end_index
                type1 = pattern_obj1.pattern_type
            else:
                start1 = pattern_obj1.get('start_index', 0)
                end1 = pattern_obj1.get('end_index', 0)
                type1 = pattern_obj1.get('pattern_type', 'unknown')
            
            # Find overlapping patterns from other algorithms
            matching_algorithms = [pattern1["algorithm"]]
            matching_patterns = [pattern_obj1]
            matching_indices = [i]
            
            for j, pattern2 in enumerate(all_patterns):
                if i == j or j in processed:
                    continue
                
                pattern_obj2 = pattern2["pattern"]
                if isinstance(pattern_obj2, PatternDetection):
                    start2 = pattern_obj2.start_index
                    end2 = pattern_obj2.end_index
                    type2 = pattern_obj2.pattern_type
                else:
                    start2 = pattern_obj2.get('start_index', 0)
                    end2 = pattern_obj2.get('end_index', 0)
                    type2 = pattern_obj2.get('pattern_type', 'unknown')
                
                # Check if patterns overlap and are same type
                if type1 == type2:
                    overlap = self._calculate_overlap(start1, end1, start2, end2)
                    if overlap >= overlap_threshold:
                        matching_algorithms.append(pattern2["algorithm"])
                        matching_patterns.append(pattern_obj2)
                        matching_indices.append(j)
            
            # If multiple algorithms agree, record as agreement
            if len(matching_algorithms) > 1:
                agreements.append({
                    "pattern_type": type1,
                    "algorithms": matching_algorithms,
                    "patterns": matching_patterns,
                    "overlap_ratio": overlap_threshold,
                    "agreement_count": len(matching_algorithms)
                })
                processed.update(matching_indices)
        
        return agreements
    
    def _find_disagreements(
        self,
        all_patterns: List[Dict[str, Any]],
        agreements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find patterns that only one algorithm detected."""
        # Get indices of patterns in agreements
        agreement_indices = set()
        for agreement in agreements:
            for pattern in agreement["patterns"]:
                # Find index in all_patterns
                for i, p in enumerate(all_patterns):
                    if p["pattern"] == pattern:
                        agreement_indices.add(i)
        
        # Patterns not in agreements are disagreements
        disagreements = []
        for i, pattern_info in enumerate(all_patterns):
            if i not in agreement_indices:
                disagreements.append({
                    "algorithm": pattern_info["algorithm"],
                    "pattern": pattern_info["pattern"]
                })
        
        return disagreements
    
    def _calculate_overlap(self, start1: int, end1: int, start2: int, end2: int) -> float:
        """Calculate overlap ratio between two ranges."""
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start > overlap_end:
            return 0.0
        
        overlap_length = overlap_end - overlap_start + 1
        union_length = max(end1, end2) - min(start1, start2) + 1
        
        return overlap_length / union_length if union_length > 0 else 0.0
    
    def _generate_comparison_statistics(
        self,
        results: List[AlgorithmResult],
        agreements: List[Dict[str, Any]],
        disagreements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comparison statistics."""
        total_patterns = sum(len(r.results) for r in results)
        total_agreements = len(agreements)
        total_disagreements = len(disagreements)
        
        # Pattern type distribution
        pattern_types = Counter()
        for result in results:
            for pattern in result.results:
                if isinstance(pattern, PatternDetection):
                    pattern_types[pattern.pattern_type] += 1
                else:
                    pattern_types[pattern.get('pattern_type', 'unknown')] += 1
        
        # Algorithm-specific statistics
        algorithm_stats = {}
        for result in results:
            algorithm_stats[result.algorithm_name] = {
                "total_patterns": len(result.results),
                "avg_confidence": (
                    sum(result.confidence_scores) / len(result.confidence_scores)
                    if result.confidence_scores else 0.0
                ),
                "success": result.success
            }
        
        return {
            "total_patterns": total_patterns,
            "total_agreements": total_agreements,
            "total_disagreements": total_disagreements,
            "agreement_rate": total_agreements / total_patterns if total_patterns > 0 else 0.0,
            "pattern_type_distribution": dict(pattern_types),
            "algorithm_statistics": algorithm_stats
        }
    
    def generate_comparison_report(
        self,
        results: List[AlgorithmResult],
        output_format: str = "dict"
    ) -> Any:
        """
        Generate a comprehensive comparison report.
        
        Args:
            results: List of AlgorithmResult objects
            output_format: Output format ('dict', 'dataframe', 'string')
            
        Returns:
            Comparison report in requested format
        """
        comparison = self.compare(results)
        
        if output_format == "dataframe":
            # Create DataFrame from comparison data
            rows = []
            for agreement in comparison["agreements"]:
                rows.append({
                    "type": "agreement",
                    "pattern_type": agreement["pattern_type"],
                    "algorithms": ", ".join(agreement["algorithms"]),
                    "count": agreement["agreement_count"]
                })
            for disagreement in comparison["disagreements"]:
                pattern = disagreement["pattern"]
                if isinstance(pattern, PatternDetection):
                    pattern_type = pattern.pattern_type
                else:
                    pattern_type = pattern.get('pattern_type', 'unknown')
                rows.append({
                    "type": "disagreement",
                    "pattern_type": pattern_type,
                    "algorithms": disagreement["algorithm"],
                    "count": 1
                })
            return pd.DataFrame(rows)
        elif output_format == "string":
            # Create human-readable string report
            lines = [
                "Algorithm Comparison Report",
                "=" * 50,
                f"Total Algorithms: {comparison['total_algorithms']}",
                f"Total Agreements: {comparison['statistics']['total_agreements']}",
                f"Total Disagreements: {comparison['statistics']['total_disagreements']}",
                f"Agreement Rate: {comparison['statistics']['agreement_rate']:.2%}",
                "",
                "Pattern Type Distribution:",
            ]
            for pattern_type, count in comparison['statistics']['pattern_type_distribution'].items():
                lines.append(f"  {pattern_type}: {count}")
            return "\n".join(lines)
        else:
            return comparison

