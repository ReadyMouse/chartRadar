# Moving Average Slope Detector

## Overview

The Moving Average Slope Detector is a rule-based trading pattern recognition algorithm that analyzes market trends by calculating multiple moving averages and their slopes. It classifies trends as **uptrend**, **downtrend**, or **sideways** for each time period and can detect when all moving averages are aligned in the same direction.

## How It Works

### 1. Moving Average Calculation

The algorithm calculates Simple Moving Averages (SMA) for multiple periods. Default periods are:
- **MA-10**: Short-term (10 periods)
- **MA-20**: Short-medium term (20 periods)
- **MA-50**: Medium-term (50 periods)
- **MA-200**: Long-term (200 periods)

### 2. Slope Calculation

For each moving average, the algorithm:
1. Takes the last `N` values (default: 5 periods)
2. Performs linear regression to calculate the slope
3. Normalizes the slope by dividing by the current MA value to get a percentage change rate

The slope represents the rate of change of the moving average:
- **Positive slope**: Price is trending upward
- **Negative slope**: Price is trending downward
- **Near-zero slope**: Price is moving sideways

### 3. Trend Classification

Based on the calculated slope and a configurable threshold:

```
if slope > threshold:
    trend = "uptrend"
elif slope < -threshold:
    trend = "downtrend"
else:
    trend = "sideways"
```

Default threshold is `0.001` (0.1% change per period).

### 4. Confidence Calculation

Confidence scores are calculated based on three factors:

1. **Magnitude** (40%): How strong is the slope?
2. **Consistency** (40%): Are recent slopes pointing in the same direction?
3. **Stability** (20%): Is the slope stable (low volatility)?

Confidence ranges from 0.0 to 1.0.

### 5. Aligned Trends Detection

The algorithm also detects when multiple moving averages align in the same direction:
- At least 75% of MAs must show the same trend
- Generates a special "aligned" pattern with higher confidence
- Strong signal for trend following strategies

## Usage

### Basic Example

```python
from chartradar.metrics.algorithms.rule_based.ma_slope_detector import MovingAverageSlopeDetector
import pandas as pd

# Load your OHLCV data
data = pd.read_csv('your_data.csv')
data.set_index('timestamp', inplace=True)

# Initialize detector with default settings
detector = MovingAverageSlopeDetector()

# Process the data
results = detector.process(data)

# Access results
for pattern in results['results']:
    print(f"Pattern: {pattern.pattern_type}")
    print(f"Confidence: {pattern.confidence:.2%}")
    print(f"Direction: {pattern.predicted_direction}")
    print(f"Trend: {pattern.characteristics['trend']}")
```

### Custom Configuration

```python
# Initialize with custom parameters
detector = MovingAverageSlopeDetector(
    ma_periods=[5, 10, 20, 50],     # Shorter periods for day trading
    slope_lookback=3,                # Faster response to changes
    slope_threshold=0.002,           # Higher threshold (more selective)
    min_confidence=0.7               # Require higher confidence
)

results = detector.process(data)
```

### Running Through the Framework

Use the ChartRadar framework with the provided configuration:

```bash
# Download real market data
python download_kraken_data.py

# Run MA slope detector
python -m chartradar --config chartradar/config/examples/ma_slope_config.yaml --data data/kraken_BTC_USD_1h.csv

# Results saved to ./output/ directory
```

This will:
1. Load the specified OHLCV data
2. Run the MA slope detector with configured parameters
3. Export results to CSV and JSON
4. Generate visualizations (if enabled)

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ma_periods` | List[int] | [10, 20, 50, 200] | Moving average periods to calculate |
| `slope_lookback` | int | 5 | Number of periods for slope calculation |
| `slope_threshold` | float | 0.001 | Threshold for up/down vs sideways (0.1%) |
| `min_confidence` | float | 0.6 | Minimum confidence for pattern detection |

## Output Format

### Results Dictionary

```python
{
    'algorithm_name': 'ma_slope_detector',
    'timestamp': datetime,
    'results': [PatternDetection, ...],
    'confidence_scores': [float, ...],
    'metadata': {
        'ma_periods': [10, 20, 50, 200],
        'slope_lookback': 5,
        'slope_threshold': 0.001,
        'patterns_detected': int,
        'trend_summary': {
            '10': 'uptrend',
            '20': 'uptrend',
            '50': 'sideways',
            '200': 'downtrend'
        }
    }
}
```

### Pattern Detection Object

Each detected pattern includes:

```python
PatternDetection(
    pattern_type='ma_20_uptrend',           # or 'ma_aligned_uptrend'
    confidence=0.85,
    start_index=295,
    end_index=300,
    start_timestamp=datetime,
    end_timestamp=datetime,
    predicted_direction='bullish',          # 'bullish', 'bearish', or None
    characteristics={
        'ma_period': 20,
        'ma_value': 45123.45,
        'slope': 0.00234,                   # Normalized slope
        'trend': 'uptrend',
        'slope_lookback': 5,
        'slope_threshold': 0.001
    }
)
```

## Trading Signals

### Bullish Signals

1. **Strong Uptrend**: All MAs showing uptrend with aligned pattern
   - High confidence (>0.8)
   - All slopes positive
   - Good for trend following

2. **Short-term Reversal**: Short-term MAs (10, 20) turn upward while long-term flat
   - Medium confidence (0.6-0.8)
   - Potential early entry

### Bearish Signals

1. **Strong Downtrend**: All MAs showing downtrend with aligned pattern
   - High confidence (>0.8)
   - All slopes negative
   - Exit long positions or short entry

2. **Short-term Reversal**: Short-term MAs (10, 20) turn downward
   - Medium confidence (0.6-0.8)
   - Warning signal

### Neutral/Sideways

1. **Ranging Market**: Most MAs showing sideways trend
   - Price consolidating
   - Wait for breakout
   - Consider range trading strategies

## Recommended Timeframes

| Trading Style | MA Periods | Slope Lookback | Data Frequency |
|---------------|------------|----------------|----------------|
| Scalping | [5, 10, 20] | 3 | 1-5 minutes |
| Day Trading | [10, 20, 50] | 5 | 15-60 minutes |
| Swing Trading | [20, 50, 100, 200] | 5 | 1-4 hours |
| Position Trading | [50, 100, 200] | 10 | Daily |

## Interpreting Results

### Example 1: Strong Bullish Trend

```
Trend Summary:
  MA-10 : UPTREND
  MA-20 : UPTREND
  MA-50 : UPTREND
  MA-200: UPTREND

Pattern: ma_aligned_uptrend
Confidence: 0.92
Alignment Ratio: 100%
```

**Interpretation**: Very strong uptrend. All timeframes aligned. High probability of continuation. Good for trend following strategies.

### Example 2: Trend Transition

```
Trend Summary:
  MA-10 : UPTREND
  MA-20 : UPTREND
  MA-50 : SIDEWAYS
  MA-200: DOWNTREND

No aligned pattern detected
```

**Interpretation**: Mixed signals. Short-term showing strength but long-term still bearish. Could be:
- Early reversal forming
- Bear market rally
- Requires more confirmation

### Example 3: Ranging Market

```
Trend Summary:
  MA-10 : SIDEWAYS
  MA-20 : SIDEWAYS
  MA-50 : SIDEWAYS
  MA-200: SIDEWAYS

Pattern: ma_aligned_sideways
Confidence: 0.78
```

**Interpretation**: Consolidation phase. Price moving sideways. Wait for breakout or use range trading strategies.

## Integration with Other Indicators

The MA Slope Detector works well in combination with:

1. **Volume Analysis**: Confirm trends with volume
2. **Support/Resistance**: Use MA values as dynamic support/resistance
3. **Momentum Indicators**: RSI, MACD for confirmation
4. **Volatility Indicators**: ATR, Bollinger Bands

## Advantages

✅ **Multi-timeframe Analysis**: Analyzes trends across multiple timeframes simultaneously
✅ **Quantitative**: Objective slope-based classification (no subjective interpretation)
✅ **Confidence Scoring**: Provides confidence levels for risk management
✅ **Trend Alignment**: Detects when all timeframes agree (strong signal)
✅ **Configurable**: Easily adjust parameters for different trading styles
✅ **No Lookahead Bias**: Only uses historical data (safe for backtesting)

## Limitations

⚠️ **Lagging Indicator**: Moving averages inherently lag price
⚠️ **Whipsaws**: Can generate false signals in choppy markets
⚠️ **Parameter Sensitivity**: Results depend on chosen periods and thresholds
⚠️ **No Price Targets**: Indicates direction but not magnitude of move
⚠️ **Requires Tuning**: Optimal parameters vary by market and timeframe

## Best Practices

1. **Backtest First**: Test parameters on historical data before live trading
2. **Use Multiple Timeframes**: Don't rely on single MA period
3. **Combine with Risk Management**: Use stop losses regardless of confidence
4. **Adjust for Market Conditions**: Volatile markets may need higher thresholds
5. **Monitor Confidence**: Higher confidence = stronger signal
6. **Watch for Alignment**: Aligned patterns are more reliable
7. **Consider Context**: Economic events can override technical signals

## Testing

### Unit Tests

Run the test suite:

```bash
pytest tests/test_metrics/test_ma_slope_detector.py -v
```

Tests cover:
- Uptrend detection
- Downtrend detection
- Sideways detection
- Aligned trends
- Confidence calculation
- Parameter validation
- Edge cases

### Integration Test

Test through the framework:

```bash
# Download data
python download_kraken_data.py

# Run detector
python -m chartradar --config chartradar/config/examples/ma_slope_config.yaml --data data/kraken_BTC_USD_1h.csv

# Verify output
ls -l output/ma_slope_detector_*
```

## References

- **Moving Averages**: Classic trend-following indicators
- **Slope Analysis**: Quantitative measure of trend strength
- **Multiple Timeframe Analysis**: Confirms trends across different periods
- **Trend Following**: Core concept in technical analysis

## Future Enhancements

Potential improvements:
- [ ] Exponential Moving Average (EMA) support
- [ ] Adaptive threshold based on volatility (ATR)
- [ ] MA crossover detection
- [ ] Price distance from MA analysis
- [ ] Support/resistance level identification
- [ ] Divergence detection between price and MA slope
- [ ] Integration with volume-weighted averages

## Support

For issues, questions, or contributions, please refer to the main ChartRadar documentation.

