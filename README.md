# ChartRadar

**Extracting the signal from the noise, in crypto trading.**

ChartRadar is a modular, extensible Python framework for trading pattern recognition and analysis. The framework is designed with a pluggable architecture where algorithms can be swapped in and out of different modules, configured via YAML files.

## Features

- **Modular Architecture**: Pluggable components for data ingestion, algorithms, fusion, and display
- **YAML Configuration**: Complete system configuration via YAML files without code changes
- **Algorithm Bank**: Library of pattern detection and analysis algorithms that can be mixed and matched
- **Data Fusion**: Mechanisms to combine outputs from multiple algorithms
- **Dual Data Modes**: Support for both batch (chunked) and streaming data processing
- **ML Infrastructure**: Training and testing loops for ML-based algorithms
- **Data Labeling**: Tools and methods for creating labeled training datasets

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

## Quick Start

```python
from chartradar.core.pipeline import Pipeline

# Load configuration from YAML
config = {
    "data_source": {
        "type": "csv",
        "path": "data/ohlcv.csv"
    },
    "algorithms": [
        {"name": "wedge_detector", "parameters": {}}
    ]
}

# Create and run pipeline
pipeline = Pipeline(config)
results = pipeline.run()
```

## Project Structure

```
chartradar/
├── core/           # Core framework components (interfaces, types, pipeline)
├── config/         # Configuration system (YAML loading, validation)
├── ingestion/      # Data ingestion module (batch and streaming)
├── metrics/        # Algorithm bank and execution engine
├── fusion/         # Data fusion strategies
├── display/        # Visualization and export
├── training/       # ML training and testing infrastructure
└── labeling/       # Data labeling tools
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
