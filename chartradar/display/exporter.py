"""
Export functionality for the ChartRadar framework.

This module provides functions to export results to various formats
including JSON, CSV, and image files.
"""

import json
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from chartradar.core.exceptions import DisplayError
from chartradar.core.types import AlgorithmResult, FusionResult, PatternDetection


class Exporter:
    """
    Exporter for saving results to various file formats.
    
    Supports JSON, CSV, and image formats (PNG, SVG).
    """
    
    def __init__(self, **kwargs: Any):
        """
        Initialize the exporter.
        
        Args:
            **kwargs: Export-specific parameters
        """
        self.parameters = kwargs
    
    def export(
        self,
        results: Any,
        format: str,
        output_path: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Export results to a file.
        
        Args:
            results: Results to export (AlgorithmResult, FusionResult, dict, or list)
            format: Export format ('json', 'csv', 'png', 'svg', 'pdf')
            output_path: Path for output file (if None, returns data as string)
            **kwargs: Format-specific parameters
            
        Returns:
            Path to exported file, or exported data as string if output_path is None
        """
        format = format.lower()
        
        if format == 'json':
            return self.export_json(results, output_path, **kwargs)
        elif format == 'csv':
            return self.export_csv(results, output_path, **kwargs)
        elif format in ['png', 'svg', 'pdf']:
            return self.export_image(results, format, output_path, **kwargs)
        else:
            raise DisplayError(
                f"Unsupported export format: {format}",
                details={"format": format, "supported": ["json", "csv", "png", "svg", "pdf"]}
            )
    
    def export_json(
        self,
        results: Any,
        output_path: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Export results to JSON format.
        
        Args:
            results: Results to export
            output_path: Path for output file
            **kwargs: Additional parameters (indent, etc.)
            
        Returns:
            Path to file or JSON string
        """
        # Convert results to dictionary
        if isinstance(results, (AlgorithmResult, FusionResult)):
            data = results.model_dump()
        elif isinstance(results, list):
            data = [
                r.model_dump() if isinstance(r, (AlgorithmResult, FusionResult)) else r
                for r in results
            ]
        else:
            data = results
        
        # Serialize to JSON
        indent = kwargs.get('indent', 2)
        json_str = json.dumps(data, indent=indent, default=str)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(json_str)
            return str(output_path)
        else:
            return json_str
    
    def export_csv(
        self,
        results: Any,
        output_path: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Export results to CSV format.
        
        Args:
            results: Results to export
            output_path: Path for output file
            **kwargs: Additional parameters
            
        Returns:
            Path to file or CSV string
        """
        # Convert results to DataFrame
        rows = []
        
        if isinstance(results, AlgorithmResult):
            for pattern in results.results:
                if isinstance(pattern, PatternDetection):
                    rows.append({
                        'algorithm_name': results.algorithm_name,
                        'pattern_type': pattern.pattern_type,
                        'confidence': pattern.confidence,
                        'start_index': pattern.start_index,
                        'end_index': pattern.end_index,
                        'start_timestamp': pattern.start_timestamp,
                        'end_timestamp': pattern.end_timestamp,
                        'predicted_direction': pattern.predicted_direction,
                        'price_target': pattern.price_target
                    })
        elif isinstance(results, FusionResult):
            for pattern in results.fused_result.get('patterns', []):
                if isinstance(pattern, PatternDetection):
                    rows.append({
                        'fusion_method': results.fusion_method,
                        'pattern_type': pattern.pattern_type,
                        'confidence': pattern.confidence,
                        'start_index': pattern.start_index,
                        'end_index': pattern.end_index,
                        'start_timestamp': pattern.start_timestamp,
                        'end_timestamp': pattern.end_timestamp,
                        'predicted_direction': pattern.predicted_direction,
                        'price_target': pattern.price_target
                    })
        elif isinstance(results, list):
            for result in results:
                if isinstance(result, AlgorithmResult):
                    for pattern in result.results:
                        if isinstance(pattern, PatternDetection):
                            rows.append({
                                'algorithm_name': result.algorithm_name,
                                'pattern_type': pattern.pattern_type,
                                'confidence': pattern.confidence,
                                'start_index': pattern.start_index,
                                'end_index': pattern.end_index,
                                'start_timestamp': pattern.start_timestamp,
                                'end_timestamp': pattern.end_timestamp,
                                'predicted_direction': pattern.predicted_direction,
                                'price_target': pattern.price_target
                            })
        
        if not rows:
            # If no patterns, create a summary row
            if isinstance(results, AlgorithmResult):
                rows.append({
                    'algorithm_name': results.algorithm_name,
                    'success': results.success,
                    'total_patterns': len(results.results),
                    'avg_confidence': sum(results.confidence_scores) / len(results.confidence_scores) if results.confidence_scores else 0.0
                })
        
        df = pd.DataFrame(rows)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            return str(output_path)
        else:
            return df.to_csv(index=False)
    
    def export_image(
        self,
        figure: Any,
        format: str,
        output_path: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Export a figure to an image file.
        
        Args:
            figure: Matplotlib or Plotly figure object
            format: Image format ('png', 'svg', 'pdf')
            output_path: Path for output file
            **kwargs: Additional parameters (dpi, width, height, etc.)
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            raise DisplayError(
                "output_path is required for image export",
                details={"format": format}
            )
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if it's a matplotlib figure
        if hasattr(figure, 'savefig'):
            # Matplotlib figure
            dpi = kwargs.get('dpi', 300)
            bbox_inches = kwargs.get('bbox_inches', 'tight')
            figure.savefig(output_path, format=format, dpi=dpi, bbox_inches=bbox_inches)
        elif hasattr(figure, 'write_image'):
            # Plotly figure
            width = kwargs.get('width', 1200)
            height = kwargs.get('height', 800)
            figure.write_image(str(output_path), width=width, height=height)
        else:
            raise DisplayError(
                f"Unsupported figure type: {type(figure)}",
                details={"format": format, "figure_type": type(figure).__name__}
            )
        
        return str(output_path)

