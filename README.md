#  DLTR Volatility Arbitrage Strategy

This repo contains a quantitative options trading strategy that targets **short volatility** opportunities using CMG (Chipotle) options. The model combines **machine learning**, **volatility filters**, and **Monte Carlo simulation** to identify trades with high risk-adjusted return.

---

##  Strategy Summary

- Sells **weekly short straddles** on Mondays, closes 5 days later
- Uses ML-predicted **5-day realized volatility**
- Trades **only when**:
  - Implied volatility > Predicted RV + 5%
  - Predicted RV > Past RV × 1.1
  - Spike probability < 0.7 (from trained model)
  - Monte Carlo shows:
    - Positive expected PnL
    - < 50% probability of loss
- **Position size** dynamically adjusts based on volatility edge

---

##  Backtest Results 

- **Final Capital**: `$123,958.25`
- **Total Return**: `+23.96%`
- **Sharpe Ratio**: `3.47`
- **Max Drawdown**: `-2.75%`
- **Win/Loss Ratio**: `1.00`
  ![image](https://github.com/user-attachments/assets/e99dd1e3-0ea4-4ab6-adcb-a715bff4789e)



These results reflect a strategy that:
- Avoids risky trades using a volatility spike classifier
- Uses Monte Carlo simulation to filter low-expected-value setups
- Maintains tight drawdown control with high risk-adjusted return

---

##  Key Files

| File | Description |
|------|-------------|
| `main.py` | Executes the strategy backtest and applies trade logic |
| `pred_vol.py` | Trains and applies volatility prediction models |
| `feature_engineering.py` | Generates lagged returns, rolling vols, volatility ratios |
| `vol_spike_pred.py` | Trains spike detection classifier and outputs `spike_prob` |
| `fetch_data.py` | Downloads CMG, SPY, and VIXY price data |
| `data/` | Contains raw and processed `.csv` data files (excluded from Git) |

---

##  How to Run

```bash
pip install -r requirements.txt
python fetch_data.py
python feature_engineering.py
python pred_vol.py
python vol_spike_pred.py
python main.py
