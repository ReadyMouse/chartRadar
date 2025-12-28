#!/usr/bin/env python3
"""
Download real Kraken data using ccxt library.

This will download actual historical OHLCV data from Kraken exchange.
"""

import sys

def download_kraken_data(symbol="BTC/USD", timeframe="1h", limit=1000):
    """Download real data from Kraken."""
    
    print("="*80)
    print(f"Downloading Real Kraken Data: {symbol}")
    print("="*80)
    
    # Check if ccxt is installed
    try:
        import ccxt
        print("\n✓ ccxt library is installed")
    except ImportError:
        print("\n✗ ccxt not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt"])
        import ccxt
        print("✓ ccxt installed")
    
    import pandas as pd
    from datetime import datetime
    
    # Initialize Kraken exchange
    print(f"\n▶ Connecting to Kraken exchange...")
    exchange = ccxt.kraken()
    
    # Download data
    print(f"▶ Downloading {limit} candles of {symbol} ({timeframe})...")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Save to CSV
    filename = f"data/kraken_{symbol.replace('/', '_')}_{timeframe}.csv"
    df.to_csv(filename)
    
    print(f"\n✓ Downloaded {len(df)} candles")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    print(f"  Saved to: {filename}")
    
    return filename


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download real Kraken data")
    parser.add_argument('--symbol', default='BTC/USD', help='Trading pair (e.g., BTC/USD, ZEC/USD)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (e.g., 1m, 5m, 1h, 1d)')
    parser.add_argument('--limit', type=int, default=1000, help='Number of candles to download')
    
    args = parser.parse_args()
    
    try:
        filename = download_kraken_data(args.symbol, args.timeframe, args.limit)
        print(f"\n✓ Success! Now run:")
        print(f"  python -m chartradar --config chartradar/config/examples/basic_config.yaml --data {filename}")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

