#!/usr/bin/env python3
"""
Run ChartRadar with basic_config.yaml on REAL trading data.

Usage:
    # Use your own CSV file:
    python run_basic_config.py --csv /path/to/your/data.csv
    
    # Or download sample real BTC data:
    python run_basic_config.py --download
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Register algorithms
import chartradar.metrics.algorithms.rule_based.wedge_detector
import chartradar.metrics.algorithms.rule_based.triangle_detector

from chartradar.config.loader import load_config
from chartradar.ingestion.sources.csv import CSVDataSource
from chartradar.metrics.registry import create_algorithm
from chartradar.display.exporter import Exporter


def download_sample_data():
    """Download real BTC/USDT data using yfinance."""
    print("="*80)
    print("Downloading Real BTC Data")
    print("="*80)
    
    try:
        import yfinance as yf
        print("\n✓ yfinance is installed")
    except ImportError:
        print("\n✗ yfinance not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
        import yfinance as yf
        print("✓ yfinance installed")
    
    print("\n▶ Downloading BTC-USD data (last 3 months, 1-hour intervals)...")
    ticker = yf.Ticker("BTC-USD")
    data = ticker.history(period="3mo", interval="1h")
    
    if data.empty:
        raise ValueError("Failed to download data from Yahoo Finance")
    
    # Rename columns to standard format
    data = data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    
    # Keep only OHLCV columns
    data = data[['open', 'high', 'low', 'close', 'volume']]
    
    # Save to file
    output_path = Path("data/btc_real_data.csv")
    output_path.parent.mkdir(exist_ok=True)
    data.to_csv(output_path)
    
    print(f"\n✓ Downloaded {len(data)} candles")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    print(f"  Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
    print(f"  Saved to: {output_path.absolute()}")
    
    return output_path


def load_real_data(csv_path):
    """Load real data from CSV file."""
    print("="*80)
    print("Loading Real Data")
    print("="*80)
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"\n✓ Loading data from: {csv_path.absolute()}")
    
    # Use the CSVDataSource to load data properly
    data_source = CSVDataSource(
        name="real_data",
        path=str(csv_path),
        date_column="timestamp"  # or "Date" or first column
    )
    
    # Load the data
    data = data_source.load_data()
    
    print(f"\n✓ Loaded {len(data)} candles")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    print(f"  Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
    print(f"\nFirst 5 rows:")
    print(data.head())
    print(f"\nData info:")
    print(data.info())
    
    return data


def run_algorithm_on_data(data, config):
    """Run algorithms from config on real data."""
    print("\n" + "="*80)
    print("Running Algorithms on Real Data")
    print("="*80)
    
    results = []
    
    for algo_config in config.algorithms:
        if not algo_config.enabled:
            continue
        
        print(f"\n{'─'*80}")
        print(f"Algorithm: {algo_config.name}")
        print(f"{'─'*80}")
        
        try:
            # Create algorithm
            algo = create_algorithm(
                name=algo_config.name,
                version=algo_config.version,
                **algo_config.parameters
            )
            
            print(f"\n✓ Created {algo.name} v{algo.version}")
            print(f"  Parameters: {algo_config.parameters}")
            
            # Process data
            print(f"\n▶ Analyzing {len(data)} candles...")
            result = algo.process(data, **algo_config.parameters)
            
            print(f"\n✓ Analysis complete!")
            print(f"  Patterns detected: {len(result['results'])}")
            
            if result['results']:
                print(f"\n  📊 DETECTED PATTERNS:")
                for i, pattern in enumerate(result['results'], 1):
                    print(f"\n  Pattern {i}:")
                    print(f"    Type: {pattern.pattern_type}")
                    print(f"    Confidence: {pattern.confidence:.4f} ({pattern.confidence*100:.1f}%)")
                    print(f"    Direction: {pattern.predicted_direction}")
                    print(f"    Start: {pattern.start_timestamp}")
                    print(f"    End: {pattern.end_timestamp}")
                    print(f"    Duration: {pattern.end_timestamp - pattern.start_timestamp}")
                    
                    # Show price action during pattern
                    pattern_data = data.iloc[pattern.start_index:pattern.end_index+1]
                    start_price = pattern_data['close'].iloc[0]
                    end_price = pattern_data['close'].iloc[-1]
                    price_change = ((end_price - start_price) / start_price) * 100
                    
                    print(f"    Price at start: ${start_price:,.2f}")
                    print(f"    Price at end: ${end_price:,.2f}")
                    print(f"    Price change: {price_change:+.2f}%")
                    print(f"    Characteristics:")
                    for key, val in pattern.characteristics.items():
                        if isinstance(val, float):
                            print(f"      {key}: {val:.4f}")
                        else:
                            print(f"      {key}: {val}")
            else:
                print(f"\n  ℹ No patterns detected with current parameters")
                print(f"  💡 Try adjusting:")
                print(f"     - min_confidence: {algo_config.parameters.get('min_confidence', 'N/A')}")
                print(f"     - lookback_period: {algo_config.parameters.get('lookback_period', 'N/A')}")
            
            results.append({
                'algorithm': algo.name,
                'result': result
            })
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    return results


def save_results(data, results, output_dir="output"):
    """Save analysis results."""
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save input data
    data_file = output_path / "input_data.csv"
    data.to_csv(data_file)
    print(f"\n✓ Saved input data: {data_file}")
    
    # Save results
    exporter = Exporter()
    
    for item in results:
        algo_name = item['algorithm']
        result = item['result']
        
        # Export to JSON
        json_file = output_path / f"{algo_name}_results.json"
        
        # Prepare data for export
        import json
        export_data = {
            'algorithm': algo_name,
            'timestamp': str(result['timestamp']),
            'metadata': result['metadata'],
            'patterns': []
        }
        
        for pattern in result['results']:
            pattern_dict = {
                'pattern_type': pattern.pattern_type,
                'confidence': float(pattern.confidence),
                'start_timestamp': str(pattern.start_timestamp),
                'end_timestamp': str(pattern.end_timestamp),
                'start_index': int(pattern.start_index),
                'end_index': int(pattern.end_index),
                'predicted_direction': pattern.predicted_direction,
                'characteristics': pattern.characteristics
            }
            export_data['patterns'].append(pattern_dict)
        
        with open(json_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Saved {algo_name} results: {json_file}")
    
    print(f"\n✓ All results saved to: {output_path.absolute()}/")


def main():
    parser = argparse.ArgumentParser(
        description="Run ChartRadar basic_config.yaml on real trading data"
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to your CSV file with OHLCV data'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download real BTC data from Yahoo Finance'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='chartradar/config/examples/basic_config.yaml',
        help='Path to config file (default: basic_config.yaml)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ChartRadar: Running on REAL Trading Data")
    print("="*80)
    
    try:
        # Step 1: Get data
        if args.download:
            csv_path = download_sample_data()
        elif args.csv:
            csv_path = args.csv
        else:
            print("\n✗ Error: You must specify either --csv or --download")
            print("\nUsage:")
            print("  python run_basic_config.py --csv /path/to/data.csv")
            print("  python run_basic_config.py --download")
            return 1
        
        # Step 2: Load data
        data = load_real_data(csv_path)
        
        # Step 3: Load config
        print("\n" + "="*80)
        print("Loading Configuration")
        print("="*80)
        print(f"\n✓ Loading: {args.config}")
        config = load_config(args.config)
        print(f"✓ Configuration loaded")
        print(f"  Algorithms: {[a.name for a in config.algorithms if a.enabled]}")
        
        # Step 4: Run algorithms
        results = run_algorithm_on_data(data, config)
        
        # Step 5: Save results
        save_results(data, results)
        
        # Summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\n✓ Analyzed {len(data)} real candles")
        total_patterns = sum(len(r['result']['results']) for r in results)
        print(f"✓ Total patterns detected: {total_patterns}")
        print(f"✓ Results saved to: ./output/")
        
        if total_patterns > 0:
            print(f"\n📊 Summary of patterns:")
            for item in results:
                patterns = item['result']['results']
                if patterns:
                    print(f"  {item['algorithm']}: {len(patterns)} patterns")
                    for p in patterns:
                        print(f"    - {p.pattern_type} (confidence: {p.confidence:.2f})")
        
        print("\n" + "="*80)
        return 0
        
    except Exception as e:
        print(f"\n✗ Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

