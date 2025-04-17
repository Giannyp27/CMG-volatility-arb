import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import timedelta
import matplotlib.pyplot as plt

def black_scholes_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put, d1

def black_scholes_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def run_strategy():
    print("📥 Loading price and volatility data...")
    price_df = pd.read_csv("../data/cmg_raw.csv", index_col=0, parse_dates=True)
    vol_df = pd.read_csv("../data/vol_predictions.csv", index_col=0, parse_dates=True)
    
    df = price_df.join(vol_df, how="inner")
    df = df.dropna()

    initial_capital = 100000
    capital = initial_capital
    trade_log = []

    r = 0.05  # risk-free rate
    T = 5/252  # 5-day holding period

    for i in range(len(df) - 5):
        today = df.index[i]
        if today.weekday() != 0:  # Only trade on Mondays
            continue

        S0 = df.iloc[i]["cmg_close"]
        K = round(S0)
        sigma = df.iloc[i]["predicted_vol"]
        
        vix_scaled = df.iloc[i]["vix_close"] / 100
        implied_vol = min(max(vix_scaled, 0.15), 0.50)
        
        vol_edge = implied_vol - sigma
        if vol_edge <= 0.05:
            continue  # skip low edge trades

        if i < 6:  # need at least 5 days of history to compute lagged RV
            continue
        window_returns = np.log(df["cmg_close"].iloc[i-5:i] / df["cmg_close"].iloc[i-5:i].shift(1)).dropna()
        rv_lagged = np.std(window_returns) * np.sqrt(252)  # annualized realized vol from past 5 days
        if sigma < rv_lagged * 1.1:
            continue  # skip if edge is weak

        if "spike_prob" in df.columns and df.iloc[i]["spike_prob"] > 0.7:
            continue  # skip if spike risk is high

        call_price, put_price, _ = black_scholes_price(S0, K, T, r, implied_vol)

        # Monte Carlo simulation
        N = 1000
        dt = 1 / 252
        S_paths = np.zeros((N, 6))
        S_paths[:, 0] = S0
        for t in range(1, 6):
            rand = np.random.normal(0, 1, N)
            S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
        final_prices = S_paths[:, -1]
        simulated_pnls = (call_price + put_price) - np.abs(final_prices - K)
        expected_pnl = simulated_pnls.mean()
        worst_pnl = simulated_pnls.min()
        prob_of_loss = (simulated_pnls < 0).mean()

        if expected_pnl < 0 or prob_of_loss > 0.5:
            continue  # skip high-risk, low-reward trades

        # Dynamic capital allocation: risk 10% of capital scaled by vol edge
        edge = sigma - rv_lagged
        edge_score = min(max(edge / rv_lagged, 0), 1)  # clamp between 0 and 1
        capital_allocated = capital * 0.1 * edge_score
        position_size = capital_allocated / (call_price + put_price)

        S5 = df.iloc[i + 5]["cmg_close"]
        pnl_straddle = (call_price + put_price) - abs(S5 - K)
        total_pnl = pnl_straddle * position_size
        capital += total_pnl

        trade_log.append({
            "date": today,
            "stock_price": S0,
            "strike": K,
            "pred_vol": sigma,
            "call+put_premium": call_price + put_price,
            "final_stock_price": S5,
            "pnl_straddle": pnl_straddle,
            "total_pnl": total_pnl,
            "capital": capital,
            "expected_pnl_mc": expected_pnl,
            "worst_pnl_mc": worst_pnl,
            "prob_of_loss_mc": prob_of_loss,
        })

    results = pd.DataFrame(trade_log)
    results.set_index("date", inplace=True)
    results.to_csv("../data/backtest_results.csv")
    print(" Backtest complete. Results saved to: ../data/backtest_results.csv")
    print(f"\n Final Capital: ${capital:.2f} | Total Return: {((capital - initial_capital) / initial_capital) * 100:.2f}%")

    returns = results["total_pnl"]
    daily_returns = returns / initial_capital

    # Sharpe ratio (weekly frequency, 52 weeks/year)
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(52)

    # Max drawdown
    cumulative_cap = results["capital"].cummax()
    drawdown = results["capital"] / cumulative_cap - 1
    max_drawdown = drawdown.min()

    # Win/Loss ratio
    wins = (returns > 0).sum()
    losses = (returns <= 0).sum()
    win_loss_ratio = wins / losses if losses > 0 else np.inf

    print(f" Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f" Max Drawdown: {max_drawdown:.2%}")
    print(f" Win/Loss Ratio: {win_loss_ratio:.2f}")

    # Monte Carlo performance visualization
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    results["expected_pnl_mc"].plot(ax=axs[0], marker='o', title="Expected PnL per Trade (Monte Carlo)")
    axs[0].axhline(0, color='red', linestyle='--')
    axs[0].set_ylabel("Expected PnL")

    results["prob_of_loss_mc"].plot(ax=axs[1], marker='o', color='orange', title="Probability of Loss per Trade (Monte Carlo)")
    axs[1].axhline(0.5, color='red', linestyle='--')
    axs[1].set_ylabel("Prob of Loss")
    axs[1].set_xlabel("Trade Index")

    plt.tight_layout()


if __name__ == "__main__":
    run_strategy()
