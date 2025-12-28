# Moving Average Slope Detector - Quick Reference

## One-Line Summary
Analyzes moving average slopes across multiple timeframes to classify trends as uptrend, downtrend, or sideways.

## Basic Usage

```python
from chartradar.metrics.algorithms.rule_based.ma_slope_detector import MovingAverageSlopeDetector

detector = MovingAverageSlopeDetector()
results = detector.process(data)
```

## Quick Configuration Presets

### Scalping (Fast Response)
```python
detector = MovingAverageSlopeDetector(
    ma_periods=[5, 10, 20],
    slope_lookback=3,
    slope_threshold=0.002,
    min_confidence=0.7
)
```

### Day Trading (Balanced)
```python
detector = MovingAverageSlopeDetector(
    ma_periods=[10, 20, 50],
    slope_lookback=5,
    slope_threshold=0.001,
    min_confidence=0.6
)
```

### Swing Trading (Longer-term)
```python
detector = MovingAverageSlopeDetector(
    ma_periods=[20, 50, 100, 200],
    slope_lookback=7,
    slope_threshold=0.0008,
    min_confidence=0.6
)
```

## Parameters Cheat Sheet

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `ma_periods` | List[int] | [10,20,50,200] | MA periods to calculate |
| `slope_lookback` | int | 5 | Periods for slope calculation |
| `slope_threshold` | float | 0.001 | Up/down vs sideways threshold |
| `min_confidence` | float | 0.6 | Minimum confidence to report |

## Reading Results

### Trend Summary
```python
trend_summary = results['metadata']['trend_summary']
# {'10': 'uptrend', '20': 'uptrend', '50': 'sideways', '200': 'downtrend'}
```

### Individual Patterns
```python
for pattern in results['results']:
    print(f"{pattern.pattern_type}: {pattern.confidence:.2%}")
    print(f"Direction: {pattern.predicted_direction}")
    print(f"Trend: {pattern.characteristics['trend']}")
    print(f"Slope: {pattern.characteristics['slope']:.6f}")
```

### Aligned Trends
```python
aligned = [p for p in results['results'] if 'aligned' in p.pattern_type]
if aligned:
    trend = aligned[0].characteristics['aligned_trend']
    confidence = aligned[0].confidence
    ratio = aligned[0].characteristics['alignment_ratio']
    print(f"Strong {trend} (confidence: {confidence:.0%}, alignment: {ratio:.0%})")
```

## Signal Interpretation

### 🚀 Strong Bullish
```
MA-10 : UPTREND
MA-20 : UPTREND
MA-50 : UPTREND
MA-200: UPTREND
Pattern: ma_aligned_uptrend (>80% confidence)
```
**Action**: Strong buy signal, trend following

### 📉 Strong Bearish
```
MA-10 : DOWNTREND
MA-20 : DOWNTREND
MA-50 : DOWNTREND
MA-200: DOWNTREND
Pattern: ma_aligned_downtrend (>80% confidence)
```
**Action**: Strong sell signal, exit longs

### ↔️ Consolidation
```
MA-10 : SIDEWAYS
MA-20 : SIDEWAYS
MA-50 : SIDEWAYS
MA-200: SIDEWAYS
Pattern: ma_aligned_sideways (>70% confidence)
```
**Action**: Range trading, wait for breakout

### ⚠️ Mixed Signals
```
MA-10 : UPTREND
MA-20 : UPTREND
MA-50 : SIDEWAYS
MA-200: DOWNTREND
No aligned pattern
```
**Action**: Caution, trend transition possible

## Common Use Cases

### 1. Trend Confirmation
```python
results = detector.process(data)
trend_10 = results['metadata']['trend_summary']['10']
trend_50 = results['metadata']['trend_summary']['50']

if trend_10 == 'uptrend' and trend_50 == 'uptrend':
    print("Confirmed uptrend across timeframes")
```

### 2. Entry/Exit Signals
```python
# Enter long when short-term turns up while long-term is up
if trend_10 == 'uptrend' and trend_200 == 'uptrend':
    print("Buy signal")
    
# Exit when short-term turns down
if trend_10 == 'downtrend':
    print("Sell signal")
```

### 3. Filter False Signals
```python
# Only trade when aligned
aligned = [p for p in results['results'] if 'aligned' in p.pattern_type]
if aligned and aligned[0].confidence > 0.8:
    trend = aligned[0].characteristics['aligned_trend']
    if trend != 'sideways':
        print(f"High-confidence {trend} trade setup")
```

## Troubleshooting

### No Patterns Detected
- **Check data length**: Need at least `max(ma_periods) + slope_lookback` rows
- **Lower min_confidence**: Try 0.5 or lower
- **Adjust slope_threshold**: Try 0.0005 for more sensitive detection

### Too Many Patterns
- **Raise min_confidence**: Try 0.7 or higher
- **Increase slope_threshold**: Try 0.002 for more selective
- **Use longer slope_lookback**: Try 7 or 10 for more stable signals

### Whipsaws in Choppy Markets
- **Increase slope_threshold**: Requires stronger trends
- **Use longer MA periods**: [20, 50, 100, 200] for less noise
- **Require alignment**: Only trade on aligned patterns

## Performance Tips

- **Vectorized operations**: Process entire DataFrame at once (not row-by-row)
- **Reuse detector**: Create once, call `process()` multiple times
- **Limit MA periods**: More periods = more computation time
- **Appropriate data**: Hourly/daily data works best, tick data may be noisy

## Command Line Usage

```bash
# Download data first
python download_kraken_data.py

# Run through framework
python -m chartradar --config chartradar/config/examples/ma_slope_config.yaml --data data/kraken_BTC_USD_1h.csv

# Check outputs
cat output/ma_slope_detector_results.json
cat output/ma_slope_detector_results.csv
```

## Configuration File

Edit `chartradar/config/examples/ma_slope_config.yaml` to customize:

```yaml
algorithms:
  - name: "ma_slope_detector"
    enabled: true
    parameters:
      ma_periods: [10, 20, 50, 200]  # Adjust periods
      slope_lookback: 5               # Response speed
      slope_threshold: 0.001          # Sensitivity
      min_confidence: 0.6             # Quality filter
```

## Integration Example

```python
# Use with other detectors
from chartradar.metrics.algorithms.rule_based import (
    MovingAverageSlopeDetector,
    WedgeDetector,
    TriangleDetector
)

ma_detector = MovingAverageSlopeDetector()
wedge_detector = WedgeDetector()

ma_results = ma_detector.process(data)
wedge_results = wedge_detector.process(data)

# Combine signals
ma_trend = ma_results['metadata']['trend_summary']['50']
has_wedge = len(wedge_results['results']) > 0

if ma_trend == 'uptrend' and has_wedge:
    print("Pattern confirmation: uptrend + wedge pattern")
```

## Export Results

```python
import pandas as pd
import json

# To DataFrame
patterns_data = []
for pattern in results['results']:
    row = {
        'pattern_type': pattern.pattern_type,
        'confidence': pattern.confidence,
        'trend': pattern.characteristics.get('trend'),
        'slope': pattern.characteristics.get('slope'),
    }
    patterns_data.append(row)

df = pd.DataFrame(patterns_data)
df.to_csv('ma_slopes.csv', index=False)

# To JSON
with open('ma_slopes.json', 'w') as f:
    json.dump({
        'timestamp': str(results['timestamp']),
        'trends': results['metadata']['trend_summary'],
        'patterns': [p.__dict__ for p in results['results']]
    }, f, indent=2)
```

## Testing

```bash
# Run tests
pytest tests/test_metrics/test_ma_slope_detector.py -v

# Run with coverage
pytest tests/test_metrics/test_ma_slope_detector.py --cov --cov-report=html
```

## Further Reading

- Full Documentation: `docs/MA_SLOPE_DETECTOR.md`
- Integration Guide: `INTEGRATION_SUMMARY.md`
- Config Examples: `chartradar/config/examples/ma_slope_config.yaml`
- Test Suite: `tests/test_metrics/test_ma_slope_detector.py`

## Support

For issues or questions:
1. Check the full documentation in `docs/MA_SLOPE_DETECTOR.md`
2. Review integration guide: `INTEGRATION_SUMMARY.md`
3. Review test examples in `tests/test_metrics/test_ma_slope_detector.py`
4. Check config examples in `chartradar/config/examples/`
5. Examine sample output in `output/ma_slope_detector_results.json`

