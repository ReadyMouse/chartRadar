# Getting Real Kraken Data for ZEC or BTC

You have **3 options** to get real Kraken data:

## Option 1: Download Manually (Easiest)

### Using Kraken's Website
1. Go to https://www.cryptodatadownload.com/data/kraken/
2. Download CSV files for BTC/USD or ZEC/USD
3. Save to `data/` folder
4. Run ChartRadar:
   ```bash
   python -m chartradar --config chartradar/config/examples/basic_config.yaml --data data/your_file.csv
   ```

## Option 2: Use CCXT Library (Programmatic)

Run the download script **outside the sandbox**:

```bash
# Download BTC/USD from Kraken (last 500 hours)
python download_kraken_data.py --symbol BTC/USD --timeframe 1h --limit 500

# Download ZEC/USD from Kraken
python download_kraken_data.py --symbol ZEC/USD --timeframe 1h --limit 500

# Then run ChartRadar
python -m chartradar --config chartradar/config/examples/basic_config.yaml --data data/kraken_BTC_USD_1h.csv
```

## Option 3: Use Freqtrade Data (If You Have It)

If you have Freqtrade installed and have downloaded data:

1. Update `basic_config.yaml`:
```yaml
data_sources:
  - name: "freqtrade_data"
    type: "freqtrade"
    enabled: true
    parameters:
      data_dir: "~/freqtrade/user_data/data"  # Your freqtrade data directory
      exchange: "kraken"
      pair: "BTC/USDT"  # or "ZEC/USDT"
      timeframe: "1h"
```

2. Run:
```bash
python -m chartradar --config chartradar/config/examples/basic_config.yaml
```

## Quick Test with Real Data

Here's a quick way to test with real data:

### Step 1: Create a simple data fetcher that works

```python
# Simple method using pandas_datareader or yfinance
pip install yfinance

python3 << 'EOF'
import yfinance as yf
import pandas as pd

# Download BTC-USD (Yahoo Finance has real crypto data)
ticker = yf.Ticker("BTC-USD")
data = ticker.history(period="1mo", interval="1h")

# Format for ChartRadar
data = data.rename(columns=str.lower)
data = data[['open', 'high', 'low', 'close', 'volume']]

# Save
data.to_csv('data/btc_real.csv')
print(f"✓ Downloaded {len(data)} candles")
print(f"  Saved to: data/btc_real.csv")
EOF
```

### Step 2: Run ChartRadar

```bash
python -m chartradar --config chartradar/config/examples/basic_config.yaml --data data/btc_real.csv
```

## CSV Format Required

Your CSV must have these columns:
```csv
,open,high,low,close,volume
2024-01-01 00:00:00,40000.50,40500.00,39800.00,40200.00,1234.56
2024-01-01 01:00:00,40200.00,40600.00,40000.00,40400.00,2345.67
...
```

## For ZEC Specifically

ZEC is not available on Yahoo Finance, so you'll need:

1. **CryptoDataDownload.com** - Free historical data
2. **CCXT** - `python download_kraken_data.py --symbol ZEC/USD`
3. **Kraken API directly** - Download from their REST API
4. **Freqtrade** - If you use Freqtrade bot

## Example: Testing Right Now

Create a test file with real BTC prices (manually):

```bash
cat > data/test_real_btc.csv << 'EOF'
,open,high,low,close,volume
2024-12-01 00:00:00,95234.50,96000.00,94800.00,95800.00,1234.56
2024-12-01 01:00:00,95800.00,96500.00,95200.00,96200.00,2345.67
2024-12-01 02:00:00,96200.00,97000.00,95800.00,96800.00,1456.78
2024-12-01 03:00:00,96800.00,97500.00,96200.00,97200.00,1567.89
EOF

python -m chartradar --config chartradar/config/examples/basic_config.yaml --data data/test_real_btc.csv
```

## Recommended Approach

**For immediate testing:**
1. Download from CryptoDataDownload.com (no coding needed)
2. Save CSV to `data/` folder
3. Run: `python -m chartradar --config basic_config.yaml --data data/your_file.csv`

**For automated pipeline:**
1. Use the `download_kraken_data.py` script outside the sandbox
2. Or integrate with your existing data pipeline
3. Point ChartRadar to the CSV file

The framework accepts any CSV with OHLCV columns and a datetime index!

