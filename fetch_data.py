import sys
import os

# Add parent directory to Python path to import config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame


def fetch_cmg_data(start_date="2003-01-27", end_date="2025-04-06"):
    """Fetches CMG daily stock data from Alpaca API and saves as CSV."""

    print(" Connecting to Alpaca API...")
    api = tradeapi.REST(key_id=config.api_key, secret_key=config.secret_key)

    print(f" Fetching CMG data from {start_date} to {end_date}...")
    bars = api.get_bars(
        symbol='CMG',
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    ).df

    bars.index = pd.to_datetime(bars.index)

    print(" Fetching SPY data...")
    spy_bars = api.get_bars(
        symbol='SPY',
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    ).df
    spy_bars.index = pd.to_datetime(spy_bars.index)
    spy_bars = spy_bars[['close']].rename(columns={'close': 'spy_close'})

    print(" Fetching VIX data...")
    vix_bars = api.get_bars(
        symbol='VIXY',  # Alpaca does not support ^VIX directly, so VIXY is used
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    ).df
    vix_bars.index = pd.to_datetime(vix_bars.index)
    vix_bars = vix_bars[['close']].rename(columns={'close': 'vix_close'})

    # Hardcoded path to save CSV
    save_path = "/Users/gianlucapannozzo/Desktop/university/VS Code/dltr-volatility-arb/data/cmg_raw.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(" Merging datasets...")
    combined = bars[['close']].rename(columns={'close': 'cmg_close'}).join(
        spy_bars, how='inner').join(
        vix_bars, how='inner'
    )

    combined.to_csv(save_path)

    print(f" CMG data saved to: {save_path}")

if __name__ == "__main__":
    fetch_cmg_data()
