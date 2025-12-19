"""
Visualization functions for the ChartRadar framework.

This module provides functions to visualize price charts with detected patterns,
confidence scores, and algorithm comparisons.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from chartradar.core.exceptions import DisplayError
from chartradar.core.types import PatternDetection, AlgorithmResult, FusionResult


class Visualizer:
    """
    Visualizer for charting and displaying analysis results.
    
    Supports both matplotlib (static) and plotly (interactive) backends.
    """
    
    def __init__(
        self,
        backend: str = "matplotlib",
        style: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize the visualizer.
        
        Args:
            backend: Visualization backend ('matplotlib' or 'plotly')
            style: Style name for matplotlib
            **kwargs: Additional parameters
        """
        self.backend = backend
        self.style = style
        
        if backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise DisplayError(
                "matplotlib is not installed. Install it with: pip install matplotlib",
                details={"package": "matplotlib"}
            )
        elif backend == "plotly" and not PLOTLY_AVAILABLE:
            raise DisplayError(
                "plotly is not installed. Install it with: pip install plotly",
                details={"package": "plotly"}
            )
    
    def plot_price_chart(
        self,
        data: pd.DataFrame,
        patterns: Optional[List[PatternDetection]] = None,
        title: str = "Price Chart with Patterns",
        show_confidence: bool = True,
        **kwargs: Any
    ) -> Any:
        """
        Plot price chart with detected patterns overlaid.
        
        Args:
            data: OHLCV DataFrame
            patterns: List of PatternDetection objects
            title: Chart title
            show_confidence: Whether to show confidence scores
            **kwargs: Additional plotting parameters
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        if self.backend == "matplotlib":
            return self._plot_price_chart_matplotlib(data, patterns, title, show_confidence, **kwargs)
        elif self.backend == "plotly":
            return self._plot_price_chart_plotly(data, patterns, title, show_confidence, **kwargs)
        else:
            raise DisplayError(f"Unsupported backend: {self.backend}")
    
    def _plot_price_chart_matplotlib(
        self,
        data: pd.DataFrame,
        patterns: Optional[List[PatternDetection]],
        title: str,
        show_confidence: bool,
        **kwargs: Any
    ) -> Any:
        """Plot price chart using matplotlib."""
        if self.style:
            plt.style.use(self.style)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=kwargs.get('figsize', (12, 8)), sharex=True)
        
        # Plot price data
        ax1.plot(data.index, data['close'], label='Close', linewidth=1.5)
        ax1.fill_between(data.index, data['low'], data['high'], alpha=0.3, label='Range')
        ax1.set_ylabel('Price')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot volume
        ax2.bar(data.index, data['volume'], alpha=0.5, label='Volume')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Overlay patterns
        if patterns:
            colors = plt.cm.Set3(np.linspace(0, 1, len(patterns)))
            for i, pattern in enumerate(patterns):
                if isinstance(pattern, dict):
                    start_idx = pattern.get('start_index', 0)
                    end_idx = pattern.get('end_index', len(data) - 1)
                    pattern_type = pattern.get('pattern_type', 'unknown')
                    confidence = pattern.get('confidence', 0.0)
                else:
                    # PatternDetection object
                    start_idx = pattern.start_index
                    end_idx = pattern.end_index
                    pattern_type = pattern.pattern_type
                    confidence = pattern.confidence
                
                # Highlight pattern region
                if start_idx < len(data) and end_idx < len(data):
                    pattern_data = data.iloc[start_idx:end_idx+1]
                    ax1.fill_between(
                        pattern_data.index,
                        pattern_data['low'].min(),
                        pattern_data['high'].max(),
                        alpha=0.2,
                        color=colors[i],
                        label=f"{pattern_type} ({confidence:.2f})" if show_confidence else pattern_type
                    )
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def _plot_price_chart_plotly(
        self,
        data: pd.DataFrame,
        patterns: Optional[List[PatternDetection]],
        title: str,
        show_confidence: bool,
        **kwargs: Any
    ) -> Any:
        """Plot price chart using plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(title, 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Plot price
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Close',
                line=dict(width=2)
            ),
            row=1, col=1
        )
        
        # Add candlestick if available
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Plot volume
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # Overlay patterns
        if patterns:
            for i, pattern in enumerate(patterns):
                if isinstance(pattern, dict):
                    start_idx = pattern.get('start_index', 0)
                    end_idx = pattern.get('end_index', len(data) - 1)
                    pattern_type = pattern.get('pattern_type', 'unknown')
                    confidence = pattern.get('confidence', 0.0)
                else:
                    start_idx = pattern.start_index
                    end_idx = pattern.end_index
                    pattern_type = pattern.pattern_type
                    confidence = pattern.confidence
                
                if start_idx < len(data) and end_idx < len(data):
                    pattern_data = data.iloc[start_idx:end_idx+1]
                    fig.add_shape(
                        type="rect",
                        x0=pattern_data.index[0],
                        x1=pattern_data.index[-1],
                        y0=pattern_data['low'].min(),
                        y1=pattern_data['high'].max(),
                        fillcolor="rgba(255,0,0,0.2)",
                        line=dict(width=0),
                        row=1, col=1
                    )
                    fig.add_annotation(
                        x=pattern_data.index[len(pattern_data)//2],
                        y=pattern_data['high'].max(),
                        text=f"{pattern_type}<br>({confidence:.2f})" if show_confidence else pattern_type,
                        showarrow=True,
                        arrowhead=2,
                        row=1, col=1
                    )
        
        fig.update_layout(
            height=kwargs.get('height', 800),
            title_text=title,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def plot_confidence_scores(
        self,
        results: List[AlgorithmResult],
        title: str = "Confidence Scores by Algorithm",
        **kwargs: Any
    ) -> Any:
        """
        Visualize confidence scores from multiple algorithms.
        
        Args:
            results: List of AlgorithmResult objects
            title: Chart title
            **kwargs: Additional parameters
            
        Returns:
            Figure object
        """
        if self.backend == "matplotlib":
            return self._plot_confidence_matplotlib(results, title, **kwargs)
        elif self.backend == "plotly":
            return self._plot_confidence_plotly(results, title, **kwargs)
        else:
            raise DisplayError(f"Unsupported backend: {self.backend}")
    
    def _plot_confidence_matplotlib(
        self,
        results: List[AlgorithmResult],
        title: str,
        **kwargs: Any
    ) -> Any:
        """Plot confidence scores using matplotlib."""
        if self.style:
            plt.style.use(self.style)
        
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
        
        algorithm_names = []
        avg_confidences = []
        
        for result in results:
            if result.success and result.confidence_scores:
                algorithm_names.append(result.algorithm_name)
                avg_confidences.append(np.mean(result.confidence_scores))
        
        if algorithm_names:
            bars = ax.bar(algorithm_names, avg_confidences, alpha=0.7)
            ax.set_ylabel('Average Confidence Score')
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def _plot_confidence_plotly(
        self,
        results: List[AlgorithmResult],
        title: str,
        **kwargs: Any
    ) -> Any:
        """Plot confidence scores using plotly."""
        algorithm_names = []
        avg_confidences = []
        
        for result in results:
            if result.success and result.confidence_scores:
                algorithm_names.append(result.algorithm_name)
                avg_confidences.append(np.mean(result.confidence_scores))
        
        fig = go.Figure(data=[
            go.Bar(
                x=algorithm_names,
                y=avg_confidences,
                text=[f'{c:.2f}' for c in avg_confidences],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=title,
            yaxis_title='Average Confidence Score',
            yaxis=dict(range=[0, 1]),
            height=kwargs.get('height', 400)
        )
        
        return fig
    
    def plot_comparison(
        self,
        results: List[AlgorithmResult],
        data: Optional[pd.DataFrame] = None,
        title: str = "Algorithm Comparison",
        **kwargs: Any
    ) -> Any:
        """
        Create comparison charts for multiple algorithms.
        
        Args:
            results: List of AlgorithmResult objects
            data: Optional OHLCV data
            title: Chart title
            **kwargs: Additional parameters
            
        Returns:
            Figure object
        """
        if self.backend == "matplotlib":
            return self._plot_comparison_matplotlib(results, data, title, **kwargs)
        elif self.backend == "plotly":
            return self._plot_comparison_plotly(results, data, title, **kwargs)
        else:
            raise DisplayError(f"Unsupported backend: {self.backend}")
    
    def _plot_comparison_matplotlib(
        self,
        results: List[AlgorithmResult],
        data: Optional[pd.DataFrame],
        title: str,
        **kwargs: Any
    ) -> Any:
        """Plot algorithm comparison using matplotlib."""
        if self.style:
            plt.style.use(self.style)
        
        n_algorithms = len([r for r in results if r.success])
        fig, axes = plt.subplots(n_algorithms, 1, figsize=kwargs.get('figsize', (12, 4*n_algorithms)), sharex=True)
        
        if n_algorithms == 1:
            axes = [axes]
        
        for i, result in enumerate(results):
            if not result.success:
                continue
            
            ax = axes[i]
            
            # Plot price if data available
            if data is not None:
                ax.plot(data.index, data['close'], 'k-', alpha=0.3, linewidth=1)
            
            # Plot detected patterns
            for pattern in result.results:
                if isinstance(pattern, PatternDetection):
                    start_idx = pattern.start_index
                    end_idx = pattern.end_index
                    pattern_type = pattern.pattern_type
                    confidence = pattern.confidence
                else:
                    start_idx = pattern.get('start_index', 0)
                    end_idx = pattern.get('end_index', len(data) - 1) if data is not None else 0
                    pattern_type = pattern.get('pattern_type', 'unknown')
                    confidence = pattern.get('confidence', 0.0)
                
                if data is not None and start_idx < len(data) and end_idx < len(data):
                    pattern_data = data.iloc[start_idx:end_idx+1]
                    ax.fill_between(
                        pattern_data.index,
                        pattern_data['low'].min(),
                        pattern_data['high'].max(),
                        alpha=0.3,
                        label=f"{pattern_type} ({confidence:.2f})"
                    )
            
            ax.set_ylabel('Price')
            ax.set_title(f"{result.algorithm_name} - {len(result.results)} patterns")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        
        return fig
    
    def _plot_comparison_plotly(
        self,
        results: List[AlgorithmResult],
        data: Optional[pd.DataFrame],
        title: str,
        **kwargs: Any
    ) -> Any:
        """Plot algorithm comparison using plotly."""
        n_algorithms = len([r for r in results if r.success])
        
        fig = make_subplots(
            rows=n_algorithms, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[r.algorithm_name for r in results if r.success]
        )
        
        row = 1
        for result in results:
            if not result.success:
                continue
            
            # Plot price if data available
            if data is not None:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['close'], mode='lines', name='Price', line=dict(color='gray', width=1)),
                    row=row, col=1
                )
            
            # Plot patterns
            for pattern in result.results:
                if isinstance(pattern, PatternDetection):
                    start_idx = pattern.start_index
                    end_idx = pattern.end_index
                else:
                    start_idx = pattern.get('start_index', 0)
                    end_idx = pattern.get('end_index', len(data) - 1) if data is not None else 0
                
                if data is not None and start_idx < len(data) and end_idx < len(data):
                    pattern_data = data.iloc[start_idx:end_idx+1]
                    fig.add_shape(
                        type="rect",
                        x0=pattern_data.index[0],
                        x1=pattern_data.index[-1],
                        y0=pattern_data['low'].min(),
                        y1=pattern_data['high'].max(),
                        fillcolor="rgba(255,0,0,0.2)",
                        line=dict(width=0),
                        row=row, col=1
                    )
            
            row += 1
        
        fig.update_layout(
            title=title,
            height=kwargs.get('height', 300 * n_algorithms)
        )
        
        return fig

