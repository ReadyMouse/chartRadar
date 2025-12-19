"""
Base display interface for the ChartRadar framework.

This module provides the abstract base class for display and export functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

from chartradar.core.interfaces import DisplayBase
from chartradar.core.exceptions import DisplayError


class Display(DisplayBase):
    """
    Abstract base class for display and export functionality.
    
    All display implementations must inherit from this class and
    implement the required methods.
    """
    
    def __init__(self, name: str = "display", **kwargs: Any):
        """
        Initialize the display handler.
        
        Args:
            name: Display handler name
            **kwargs: Display-specific parameters
        """
        self.name = name
        self.parameters = kwargs
    
    @abstractmethod
    def visualize(
        self,
        results: Dict[str, Any],
        data: Optional[pd.DataFrame] = None,
        **kwargs: Any
    ) -> Any:
        """
        Visualize analysis results.
        
        Args:
            results: Results dictionary from algorithms or fusion
            data: Optional original OHLCV data for context
            **kwargs: Visualization-specific parameters
            
        Returns:
            Visualization object (matplotlib figure, plotly figure, etc.)
            
        Raises:
            DisplayError: If visualization fails
        """
        pass
    
    @abstractmethod
    def export(
        self,
        results: Dict[str, Any],
        format: str,
        output_path: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Export results to a file.
        
        Args:
            results: Results dictionary to export
            format: Export format ('json', 'csv', 'png', 'svg', etc.)
            output_path: Optional path for output file (if None, returns data as string)
            **kwargs: Export-specific parameters
            
        Returns:
            Path to exported file, or exported data as string if output_path is None
            
        Raises:
            DisplayError: If export fails
        """
        pass

