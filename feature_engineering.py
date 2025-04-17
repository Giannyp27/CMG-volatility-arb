import os
import sys
import pandas as pd
import numpy as np

# Add root path so this script can be run independently
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def build_features(input_path, output_path):
    """Loads raw price data, builds volatility features, and saves to CSV."""

    print(" Loading raw CMG data...")
    df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    # Calculate log returns
    df['log_return'] = np.log(df['cmg_close'] / df['cmg_close'].shift(1))

    # Realized volatility (annualized)
    df['rv_5d'] = df['log_return'].rolling(window=5).std() * np.sqrt(252)
    df['rv_10d'] = df['log_return'].rolling(window=10).std() * np.sqrt(252)
    df['rv_21d'] = df['log_return'].rolling(window=21).std() * np.sqrt(252)

    # Rolling stats and other features
    df['rolling_mean_5'] = df['cmg_close'].rolling(5).mean()
    df['rolling_std_5'] = df['cmg_close'].rolling(5).std()
    df['rolling_mean_10'] = df['cmg_close'].rolling(10).mean()
    df['rolling_std_10'] = df['cmg_close'].rolling(10).std()
    df['return_squared'] = df['log_return'] ** 2
    df['abs_return'] = df['log_return'].abs()

    # Lagged features and interactions
    df['log_return_lag1'] = df['log_return'].shift(1)
    df['rv_5d_lag1'] = df['rv_5d'].shift(1)
    df['vol_of_vol'] = df['rv_5d'].rolling(5).std()
    df['abs_return_x_vol'] = df['abs_return'] * df['rv_5d']
    
    # New lagged features
    df['abs_return_lag1'] = df['abs_return'].shift(1)
    df['abs_return_lag2'] = df['abs_return'].shift(2)
    df['rv_10d_lag1'] = df['rv_10d'].shift(1)
    df['rv_21d_lag1'] = df['rv_21d'].shift(1)

    # New features
    df['rolling_std_change'] = df['rolling_std_5'] - df['rolling_std_5'].shift(1)
    df['vol_spike'] = (df['rv_5d'] / df['rv_5d_lag1']) > 1.5
    df['vol_spike'] = df['vol_spike'].astype(int)

    # === SPY and VIX Volatility Features ===
    df['spy_return'] = np.log(df['spy_close'] / df['spy_close'].shift(1))
    df['spy_vol'] = df['spy_return'].rolling(5).std() * np.sqrt(252)

    df['vix_return'] = df['vix_close'].pct_change()
    df['vix_vol'] = df['vix_return'].rolling(5).std() * np.sqrt(252)

    # New target features
    df['target_rv_5d'] = df['rv_5d'].shift(-5)
    df['target_rv_5d_log'] = np.log1p(df['target_rv_5d'])

    # Also create a log-transformed version of the target
    df.dropna(inplace=True)

    # === Temporal Context Features ===

    # Ratio of short vs long volatility
    df['vol_ratio_5_21'] = df['rv_5d'] / (df['rv_21d'] + 1e-6)
    df['vol_ratio_10_21'] = df['rv_10d'] / (df['rv_21d'] + 1e-6)

    # Save processed features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f" Feature-engineered data saved to: {output_path}")


if __name__ == "__main__":
    INPUT_FILE = "/Users/gianlucapannozzo/Desktop/university/VS Code/dltr-volatility-arb/data/cmg_raw.csv"
    OUTPUT_FILE = "/Users/gianlucapannozzo/Desktop/university/VS Code/dltr-volatility-arb/data/cmg_features.csv"

    build_features(INPUT_FILE, OUTPUT_FILE)