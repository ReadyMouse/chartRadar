# ChartRadar

**Extracting the signal from the noise, in crypto trading.**

ChartRadar is a modular, extensible Python framework for trading pattern recognition and analysis. Detect wedges, triangles, and other chart patterns in cryptocurrency price data using rule-based and ML algorithms. Configure everything via YAML, no code changes needed.

## Features

- **Modular Architecture**: Pluggable components for data ingestion, algorithms, fusion, and display
- **YAML Configuration**: Complete system configuration via YAML files without code changes
- **Algorithm Bank**: Library of pattern detection and analysis algorithms that can be mixed and matched
- **Data Fusion**: Mechanisms to combine outputs from multiple algorithms
- **Dual Data Modes**: Support for both batch (chunked) and streaming data processing
- **ML Infrastructure**: Training and testing loops for ML-based algorithms

## Available Algorithms

### Rule-Based Detectors

- [UNTESTED]**Wedge Detector**: Rising and falling wedge pattern detection
- [UNTESTED]**Triangle Detector**: Rising, falling, and symmetrical triangle patterns
- [UNTESTED]**MA Slope Detector**: Multi-timeframe moving average trend analysis

Run any detector using the framework:

```bash
# Download real market data first
python download_kraken_data.py

# Wedge pattern detection
python -m chartradar --config chartradar/config/examples/basic_config.yaml --data data/kraken_BTC_USD_1h.csv

# Moving Average Slope analysis
python -m chartradar --config chartradar/config/examples/ma_slope_config.yaml --data data/kraken_BTC_USD_1h.csv

# Multiple detectors (wedge, triangle, and MA slope)
python -m chartradar --config chartradar/config/examples/full_config.yaml --data data/kraken_BTC_USD_1h.csv
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/chartradar.git
cd chartradar

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"

# Or install with all optional dependencies
pip install -e ".[dev,ml,data-sources,visualization]"
```

### Install Dependencies Only

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

## Getting Real Data

Download live market data from Kraken exchange:

```bash
# Download BTC/USD data (default: 1000 candles, 1h timeframe)
python download_kraken_data.py

# Download different pairs or timeframes
python download_kraken_data.py --symbol ETH/USD --timeframe 4h --limit 2000
python download_kraken_data.py --symbol ZEC/USD --timeframe 1d
```

Data is saved to `data/kraken_*.csv` and ready to use with the pipeline.

## Quick Start

```bash
# Download real market data
python download_kraken_data.py

# Run pattern detection with basic config
python -m chartradar --config chartradar/config/examples/basic_config.yaml --data data/kraken_BTC_USD_1h.csv

# Results saved to ./output/ directory
```

Or use the Python API:

```python
from chartradar.config.loader import load_config
from chartradar.src.integrated_pipeline import IntegratedPipeline

# Load configuration
config = load_config("chartradar/config/examples/basic_config.yaml")

# Run pipeline
pipeline = IntegratedPipeline(config)
pipeline.run(data_path="data/kraken_BTC_USD_1h.csv")
```

## Project Structure

```
chartradar/
├── src/            # Core framework components (interfaces, types, pipeline)
├── config/         # Configuration system (YAML loading, validation)
├── ingestion/      # Data ingestion module (batch and streaming)
├── metrics/        # Algorithm bank and execution engine
├── fusion/         # Data fusion strategies
├── display/        # Visualization and export
└── training/       # ML training and testing infrastructure
```

## Configuration

ChartRadar uses YAML files for configuration. See `chartradar/config/examples/` for example configurations.

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black chartradar/
ruff check chartradar/
```

### Type Checking

```bash
mypy chartradar/
```

## License

Apache License 2.0 - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Status

This project is in early development (Alpha). APIs may change without notice.
