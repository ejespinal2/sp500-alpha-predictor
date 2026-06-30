"""
SP_Return_Data.py
--------------------
Downloads S&P 500 index("^GSPC") historical data and computes:
  - Proper compound monthly/yearly returns
  - Annualised yearly volatility
  - Sharpe ratio (using 3-month T-bill as risk-free rate)
  - Max drawdown per year
"""

import yfinance as yf
import pandas as pd
import numpy as np

START = "2013-01-01"
END   = "2025-07-01"
RISK_FREE_ANNUAL = 0.04          # approximate 10-yr average 3-mo T-bill

def compound_return(series: pd.Series) -> float:
    """True compound return from a daily price series."""
    return (series.iloc[-1] / series.iloc[0]) - 1

def annualised_volatility(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(252)

def max_drawdown(prices: pd.Series) -> float:
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return drawdown.min()


# ── Download ──────────────────────────────────────────────────────────────────
print("Downloading S&P 500 index data …")
raw = yf.download("^GSPC", start=START, end=END, auto_adjust=True)
prices = raw["Close"].squeeze()
daily_ret = prices.pct_change().dropna()

# ── Monthly returns ───────────────────────────────────────────────────────────
monthly_returns = (
    prices
    .resample("ME")
    .apply(compound_return)
    .rename("Monthly Return")
)

# ── Yearly aggregates ─────────────────────────────────────────────────────────
yearly_returns = (
    prices
    .resample("YE")
    .apply(compound_return)
    .rename("Yearly Return")
)

yearly_vol = (
    daily_ret
    .resample("YE")
    .apply(annualised_volatility)
    .rename("Yearly Volatility")
)

yearly_sharpe = (
    ((yearly_returns - RISK_FREE_ANNUAL) / yearly_vol)
    .rename("Yearly Sharpe")
)

yearly_drawdown = (
    prices
    .resample("YE")
    .apply(max_drawdown)
    .rename("Max Drawdown")
)

# ── Combine & save ────────────────────────────────────────────────────────────
index_df = pd.concat(
    [monthly_returns, yearly_returns, yearly_vol, yearly_sharpe, yearly_drawdown],
    axis=1
)
index_df.index.name = "Date"

output = "sp500_returns.xlsx"
index_df.to_excel(output)
print(f"Saved → {output}")
print(index_df[["Yearly Return", "Yearly Volatility", "Yearly Sharpe"]].dropna().to_string())
