"""
Main entry point for running ChartRadar from the command line.

Usage:
    python -m chartradar --config path/to/config.yaml --data path/to/data.csv
    python -m chartradar --config chartradar/config/examples/basic_config.yaml --data data.csv
"""

import argparse
import sys
from pathlib import Path

from chartradar.config.loader import load_config
from chartradar.src.integrated_pipeline import IntegratedPipeline
from chartradar.src.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for ChartRadar CLI."""
    parser = argparse.ArgumentParser(
        description="ChartRadar - Trading Pattern Recognition Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with basic config and CSV data
  python -m chartradar --config chartradar/config/examples/basic_config.yaml --data mydata.csv
  
  # Run with custom output directory
  python -m chartradar --config config.yaml --data data.csv --output ./results
  
  # Enable verbose logging
  python -m chartradar --config config.yaml --data data.csv --verbose
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        required=True,
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to CSV data file (overrides config file data source)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    try:
        # Configure logging
        if args.verbose:
            import logging
            logging.basicConfig(level=logging.DEBUG)
        
        logger.info("=" * 80)
        logger.info("ChartRadar - Trading Pattern Recognition")
        logger.info("=" * 80)
        
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config, validate=True)
        logger.info(f"✓ Configuration loaded: {config.name}")
        
        # Create and run pipeline
        pipeline = IntegratedPipeline(config)
        
        # Override data source if provided
        data_path = args.data if args.data else None
        
        # Run the pipeline
        logger.info("Starting pipeline execution...")
        results = pipeline.run(
            data_path=data_path,
            output_dir=args.output
        )
        
        logger.info("=" * 80)
        logger.info("Pipeline execution complete!")
        logger.info(f"✓ Processed {results.get('data_points', 0)} data points")
        logger.info(f"✓ Algorithms executed: {results.get('algorithms_executed', 0)}")
        logger.info(f"✓ Patterns detected: {results.get('total_patterns', 0)}")
        logger.info(f"✓ Results saved to: {Path(args.output).absolute()}")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"✗ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

