# Running ChartRadar with basic_config.yaml

## ✅ IT'S SET UP AND WORKING!

The framework is **fully wired up** and ready to run with the `basic_config.yaml` configuration.

## Quick Start

```bash
# Run ChartRadar with the basic config
python -m chartradar --config chartradar/config/examples/basic_config.yaml
```

This will:
1. ✅ Load configuration from `basic_config.yaml`
2. ✅ Load data from the configured CSV file
3. ✅ Run the `wedge_detector` algorithm
4. ✅ Export results to `./output/`

## What Just Happened?

When you run the command above, you'll see:

```
================================================================================
ChartRadar - Trading Pattern Recognition
================================================================================
Loading configuration from: chartradar/config/examples/basic_config.yaml
✓ Configuration loaded: Basic Trading Pattern Recognition
Integrated pipeline initialized: Basic Trading Pattern Recognition
Starting pipeline execution...
Loading data...
Using configured data source: csv_data (csv)
✓ Loaded 200 data points
  Date range: 2024-01-01 00:00:00 to 2024-01-09 07:00:00
  Price range: $40051.02 to $42063.43
Executing algorithms...
  Running: wedge_detector
    ✓ Detected X patterns
✓ Executed 1 algorithms
Exporting results to: ./output
  ✓ Saved input data: input_data.csv
  ✓ Exported wedge_detector: wedge_detector_results.json
  ✓ Exported wedge_detector: wedge_detector_patterns.csv
================================================================================
Pipeline execution complete!
✓ Processed 200 data points
✓ Algorithms executed: 1
✓ Patterns detected: X
✓ Results saved to: /path/to/output
================================================================================
```

## Output Files

After running, check the `./output/` directory:

```
output/
├── input_data.csv                    # The OHLCV data that was analyzed
├── wedge_detector_results.json       # Full results with pattern details
└── wedge_detector_patterns.csv       # Patterns in CSV format
```

### wedge_detector_results.json
```json
{
  "algorithm_name": "wedge_detector",
  "timestamp": "2025-12-27 17:53:45",
  "results": [...],  // Array of detected patterns
  "confidence_scores": [...],
  "metadata": {
    "lookback_period": 50,
    "min_confidence": 0.6,
    "patterns_detected": 0
  },
  "success": true
}
```

## Using Your Own Data

### Option 1: Update the Config File
Edit `chartradar/config/examples/basic_config.yaml`:

```yaml
data_sources:
  - name: "csv_data"
    type: "csv"
    enabled: true
    parameters:
      path: "/path/to/your/data.csv"  # Change this
      date_column: null  # or "timestamp" if your CSV has it as a column
```

### Option 2: Pass Data on Command Line
```bash
python -m chartradar --config chartradar/config/examples/basic_config.yaml --data /path/to/your/data.csv
```

## Your CSV Data Format

Your CSV should have these columns:
- `open` - Opening price
- `high` - High price  
- `low` - Low price
- `close` - Closing price
- `volume` - Trading volume
- Datetime index or a date column

Example:
```csv
,open,high,low,close,volume
2024-01-01 00:00:00,40090.74,40117.76,40009.47,40081.93,796.08
2024-01-01 01:00:00,40137.16,40200.56,40028.31,40090.26,357.16
...
```

## Configuration Options

The `basic_config.yaml` controls everything:

### Algorithms
```yaml
algorithms:
  - name: "wedge_detector"
    enabled: true
    parameters:
      min_confidence: 0.6      # Lower to detect more patterns (0.4-0.8)
      lookback_period: 50      # How many candles to analyze
```

### Display & Export
```yaml
display:
  enabled: true
  visualization:
    backend: "matplotlib"     # or "plotly"
  export:
    formats: ["json", "csv"]  # Output formats
```

### Fusion (Combine Multiple Algorithms)
```yaml
fusion:
  strategy: "weighted_average"
  enabled: true
  parameters:
    normalize_weights: true
```

## Command Line Options

```bash
python -m chartradar --help

Options:
  --config, -c   Path to YAML configuration file (required)
  --data, -d     Path to CSV data file (overrides config)
  --output, -o   Output directory (default: ./output)
  --verbose, -v  Enable verbose logging
```

## Examples

### Basic usage
```bash
python -m chartradar --config chartradar/config/examples/basic_config.yaml
```

### With your own data
```bash
python -m chartradar --config chartradar/config/examples/basic_config.yaml --data ~/trading/btc_data.csv
```

### Custom output directory
```bash
python -m chartradar --config chartradar/config/examples/basic_config.yaml --output ./my_results
```

### Verbose mode (see all details)
```bash
python -m chartradar --config chartradar/config/examples/basic_config.yaml --verbose
```

## Try the Full Config

For more algorithms and options:
```bash
python -m chartradar --config chartradar/config/examples/full_config.yaml --data your_data.csv
```

This runs both `wedge_detector` AND `triangle_detector` algorithms.

## Understanding Results

### Inputs (Shown During Execution)
- Number of data points loaded
- Date range of data
- Price range
- Configuration settings

### Intermediate Results (Logged)
- Each algorithm's execution status
- Number of patterns detected per algorithm
- Confidence scores

### Outputs (Saved to Files)
- `input_data.csv` - The data that was analyzed
- `{algorithm}_results.json` - Complete results with:
  - Pattern types
  - Confidence scores (0.0 to 1.0)
  - Start/end timestamps
  - Pattern characteristics (slopes, convergence, etc.)
- `{algorithm}_patterns.csv` - Patterns in tabular format

## Pattern Information

When patterns are detected, you'll see:
- **Pattern Type**: e.g., "rising_wedge", "falling_wedge"
- **Confidence**: How certain the algorithm is (0.0 to 1.0)
- **Direction**: "bullish" or "bearish"
- **Time Period**: Start and end timestamps
- **Characteristics**: Technical details (slopes, convergence, touches, etc.)

## Troubleshooting

### No patterns detected?
Try adjusting parameters in the config:
```yaml
parameters:
  min_confidence: 0.4  # Lower threshold (was 0.6)
  lookback_period: 80  # Look at more data (was 50)
```

### Wrong date format?
If your CSV has the date as a column (not index):
```yaml
parameters:
  date_column: "timestamp"  # or "Date", "datetime", etc.
```

### Need more data?
The algorithms work best with:
- At least 100 data points
- Clear trending patterns
- Hourly or daily timeframes

## What's Configured in basic_config.yaml?

1. **Data Source**: CSV file reader
2. **Algorithm**: Wedge pattern detector
3. **Fusion**: Weighted average (for when you add more algorithms)
4. **Display**: JSON and CSV export enabled

## Next Steps

1. **Add more algorithms**: Edit the config to enable `triangle_detector`
2. **Adjust parameters**: Lower `min_confidence` to find more patterns
3. **Use real data**: Point to your actual trading data CSV
4. **Try full_config.yaml**: See all available options

## Summary

✅ **The framework is fully functional**
✅ **Config file controls everything**
✅ **Command line tool is ready**: `python -m chartradar`
✅ **All inputs/outputs are shown and saved**
✅ **Works with your own CSV data**

Just run:
```bash
python -m chartradar --config chartradar/config/examples/basic_config.yaml
```

And check `./output/` for results!

