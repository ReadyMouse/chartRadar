"""
Integrated pipeline that actually wires all components together.

This module provides a fully functional pipeline that integrates:
- Configuration loading
- Data ingestion from various sources
- Algorithm execution
- Data fusion
- Results display and export
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import algorithms to register them
import chartradar.metrics.algorithms.rule_based.wedge_detector
import chartradar.metrics.algorithms.rule_based.triangle_detector

# Import fusion strategies to register them
import chartradar.fusion.strategies.weighted_average
import chartradar.fusion.strategies.voting
import chartradar.fusion.strategies.stacking

from chartradar.src.logger import get_logger
from chartradar.src.exceptions import PipelineError, DataSourceError
from chartradar.src.types import AlgorithmResult
from chartradar.config.schema import FrameworkConfig
from chartradar.ingestion.sources.csv import CSVDataSource
from chartradar.ingestion.sources.freqtrade import FreqtradeDataSource
from chartradar.metrics.registry import create_algorithm
from chartradar.fusion.executor import FusionExecutor
from chartradar.display.exporter import Exporter
from chartradar.display.visualizer import Visualizer

logger = get_logger(__name__)


class IntegratedPipeline:
    """
    Fully integrated pipeline that connects all ChartRadar components.
    
    This pipeline actually implements the complete workflow:
    1. Load configuration
    2. Initialize and load data from configured sources
    3. Execute all enabled algorithms
    4. Apply fusion strategy
    5. Export and visualize results
    """
    
    def __init__(self, config: FrameworkConfig):
        """
        Initialize the integrated pipeline.
        
        Args:
            config: Validated FrameworkConfig object
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.data: Optional[pd.DataFrame] = None
        self.algorithm_results: List[AlgorithmResult] = []
        self.fusion_result: Optional[Any] = None
        
        self.logger.info(f"Integrated pipeline initialized: {config.name}")
    
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from configured source or provided path.
        
        Args:
            data_path: Optional path to override config data source
            
        Returns:
            DataFrame with OHLCV data
        """
        self.logger.info("Loading data...")
        
        if data_path:
            # Load from provided path (CSV)
            self.logger.info(f"Loading data from: {data_path}")
            source = CSVDataSource(
                name="provided_data",
                path=data_path,
                date_column="timestamp"
            )
            data = source.load_data()
        else:
            # Use configured data source
            if not self.config.data_sources:
                raise DataSourceError("No data source configured")
            
            # Get first enabled data source
            data_source_config = None
            for ds in self.config.data_sources:
                if ds.enabled:
                    data_source_config = ds
                    break
            
            if not data_source_config:
                raise DataSourceError("No enabled data source found in configuration")
            
            self.logger.info(f"Using configured data source: {data_source_config.name} ({data_source_config.type})")
            
            # Create appropriate data source
            if data_source_config.type == "csv":
                source = CSVDataSource(
                    name=data_source_config.name,
                    **data_source_config.parameters
                )
            elif data_source_config.type == "freqtrade":
                source = FreqtradeDataSource(
                    name=data_source_config.name,
                    **data_source_config.parameters
                )
            else:
                raise DataSourceError(f"Unsupported data source type: {data_source_config.type}")
            
            data = source.load_data()
        
        self.logger.info(f"✓ Loaded {len(data)} data points")
        self.logger.info(f"  Date range: {data.index[0]} to {data.index[-1]}")
        self.logger.info(f"  Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
        
        self.data = data
        return data
    
    def execute_algorithms(self) -> List[AlgorithmResult]:
        """
        Execute all enabled algorithms on the loaded data.
        
        Returns:
            List of AlgorithmResult objects
        """
        if self.data is None:
            raise PipelineError("No data loaded. Call load_data() first.")
        
        self.logger.info("Executing algorithms...")
        results = []
        
        for algo_config in self.config.algorithms:
            if not algo_config.enabled:
                self.logger.info(f"  Skipping {algo_config.name} (disabled)")
                continue
            
            self.logger.info(f"  Running: {algo_config.name}")
            
            try:
                # Create algorithm instance
                algorithm = create_algorithm(
                    name=algo_config.name,
                    version=algo_config.version,
                    **algo_config.parameters
                )
                
                # Process data
                result_dict = algorithm.process(self.data, **algo_config.parameters)
                
                # Convert to AlgorithmResult
                algorithm_result = AlgorithmResult(
                    algorithm_name=result_dict['algorithm_name'],
                    results=result_dict.get('results', []),
                    confidence_scores=result_dict.get('confidence_scores', []),
                    timestamp=result_dict.get('timestamp', datetime.now()),
                    success=True,
                    metadata=result_dict.get('metadata', {})
                )
                
                results.append(algorithm_result)
                
                self.logger.info(f"    ✓ Detected {len(algorithm_result.results)} patterns")
                
            except Exception as e:
                self.logger.error(f"    ✗ Error: {str(e)}")
                # Create failed result
                algorithm_result = AlgorithmResult(
                    algorithm_name=algo_config.name,
                    results=[],
                    confidence_scores=[],
                    timestamp=datetime.now(),
                    success=False,
                    error_message=str(e),
                    metadata={}
                )
                results.append(algorithm_result)
        
        self.algorithm_results = results
        self.logger.info(f"✓ Executed {len(results)} algorithms")
        
        return results
    
    def apply_fusion(self) -> Optional[Any]:
        """
        Apply configured fusion strategy to combine algorithm results.
        
        Returns:
            FusionResult object or None
        """
        if not self.config.fusion or not self.config.fusion.enabled:
            self.logger.info("Fusion disabled")
            return None
        
        if not self.algorithm_results:
            self.logger.warning("No algorithm results to fuse")
            return None
        
        self.logger.info(f"Applying fusion: {self.config.fusion.strategy}")
        
        try:
            executor = FusionExecutor()
            
            fusion_config = {
                'strategy': self.config.fusion.strategy,
                'parameters': dict(self.config.fusion.parameters),
                'enabled': self.config.fusion.enabled
            }
            
            fusion_result = executor.execute(
                algorithm_results=self.algorithm_results,
                fusion_config=fusion_config
            )
            
            self.fusion_result = fusion_result
            self.logger.info(f"✓ Fusion complete")
            
            return fusion_result
            
        except Exception as e:
            self.logger.error(f"Fusion failed: {str(e)}")
            return None
    
    def export_results(self, output_dir: str = "./output") -> None:
        """
        Export results to configured formats.
        
        Args:
            output_dir: Directory to save results
        """
        if not self.config.display or not self.config.display.enabled:
            self.logger.info("Display/export disabled")
            return
        
        self.logger.info(f"Exporting results to: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exporter = Exporter()
        export_formats = self.config.display.export.get('formats', ['json'])
        
        # Export input data
        if self.data is not None:
            data_file = output_path / "input_data.csv"
            self.data.to_csv(data_file)
            self.logger.info(f"  ✓ Saved input data: {data_file.name}")
        
        # Export algorithm results
        for algo_result in self.algorithm_results:
            if not algo_result.success:
                continue
            
            algo_name = algo_result.algorithm_name
            
            # Export to configured formats
            if 'json' in export_formats:
                json_file = output_path / f"{algo_name}_results.json"
                exporter.export_json(algo_result, str(json_file))
                self.logger.info(f"  ✓ Exported {algo_name}: {json_file.name}")
            
            if 'csv' in export_formats:
                csv_file = output_path / f"{algo_name}_patterns.csv"
                try:
                    exporter.export_csv(algo_result, str(csv_file))
                    self.logger.info(f"  ✓ Exported {algo_name}: {csv_file.name}")
                except Exception as e:
                    self.logger.warning(f"  CSV export failed for {algo_name}: {e}")
        
        # Create visualizations if enabled
        if self.config.display.visualization.get('backend') not in ['none', None]:
            self._create_visualizations(output_path)
    
    def _create_visualizations(self, output_path: Path) -> None:
        """Create visualizations of results."""
        try:
            backend = self.config.display.visualization.get('backend', 'matplotlib')
            self.logger.info(f"  Creating visualizations ({backend})...")
            
            visualizer = Visualizer(backend=backend)
            
            for algo_result in self.algorithm_results:
                if not algo_result.success or not algo_result.results:
                    continue
                
                algo_name = algo_result.algorithm_name
                
                try:
                    fig = visualizer.plot_price_chart(
                        data=self.data,
                        patterns=algo_result.results,
                        title=f"{algo_name} - Pattern Detection",
                        show_confidence=True
                    )
                    
                    # Save figure
                    if backend == 'matplotlib':
                        fig_file = output_path / f"{algo_name}_chart.png"
                        fig.savefig(fig_file, dpi=300, bbox_inches='tight')
                        self.logger.info(f"    ✓ Created visualization: {fig_file.name}")
                    elif backend == 'plotly':
                        fig_file = output_path / f"{algo_name}_chart.html"
                        fig.write_html(str(fig_file))
                        self.logger.info(f"    ✓ Created visualization: {fig_file.name}")
                    
                except Exception as e:
                    self.logger.warning(f"    Visualization failed for {algo_name}: {e}")
            
        except Exception as e:
            self.logger.warning(f"  Visualization creation failed: {e}")
    
    def run(
        self,
        data_path: Optional[str] = None,
        output_dir: str = "./output"
    ) -> Dict[str, Any]:
        """
        Run the complete integrated pipeline.
        
        Args:
            data_path: Optional path to data file (overrides config)
            output_dir: Directory for output files
            
        Returns:
            Dictionary with execution summary
        """
        try:
            # Step 1: Load data
            self.load_data(data_path)
            
            # Step 2: Execute algorithms
            self.execute_algorithms()
            
            # Step 3: Apply fusion
            self.apply_fusion()
            
            # Step 4: Export results
            self.export_results(output_dir)
            
            # Generate summary
            total_patterns = sum(
                len(r.results) for r in self.algorithm_results if r.success
            )
            
            successful_algos = sum(1 for r in self.algorithm_results if r.success)
            
            return {
                'success': True,
                'data_points': len(self.data) if self.data is not None else 0,
                'algorithms_executed': len(self.algorithm_results),
                'algorithms_successful': successful_algos,
                'total_patterns': total_patterns,
                'fusion_applied': self.fusion_result is not None,
                'output_dir': str(Path(output_dir).absolute()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise PipelineError(f"Pipeline execution failed: {str(e)}") from e

