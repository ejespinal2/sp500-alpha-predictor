"""
SP500_Stock_Return_Data.py
------------------------------
Downloads every S&P 500 constituent and computes per-stock:
  - Compound monthly returns
  - Compound yearly returns
  - Annualised yearly volatility
  - Yearly Sharpe ratio
  - Max drawdown per year
  - GICS Sector (from Wikipedia table)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests

START = "2013-01-01"
END   = "2025-07-01"
RISK_FREE_ANNUAL = 0.04


def compound_return(series: pd.Series) -> float:
    if len(series) < 2 or series.iloc[0] == 0:
        return np.nan
    return (series.iloc[-1] / series.iloc[0]) - 1

def annualised_vol(daily_rets: pd.Series) -> float:
    return daily_rets.std() * np.sqrt(252)

def max_drawdown(prices: pd.Series) -> float:
    peak = prices.cummax()
    return ((prices - peak) / peak).min()


# ── Load ticker / sector table from Wikipedia ─────────────────────────────────
print("Loading S&P 500 constituent list …")
sp500_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
data_table = pd.read_html(sp500_URL, storage_options={"User-Agent": "Mozilla/5.0"})[0]
tickers = data_table['Symbol'].tolist()
sectors = data_table['GICS Sector'].tolist()

# ── Download & compute ────────────────────────────────────────────────────────
monthly_rows, yearly_rows, errors = [], [], []

for i, (ticker, sector) in enumerate(zip(tickers, sectors), 1):
    print(f"  [{i:3d}/{len(tickers)}] {ticker}", end="\r")
    try:
        raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False, multi_level_index=False)
        if raw.empty or len(raw) < 60:
            raise ValueError("Insufficient data")
        prices = raw["Close"]
        daily_ret = prices.pct_change().dropna()

        # Monthly returns
        for period_end, grp in prices.resample("ME"):
            if len(grp) < 5:
                continue
            monthly_rows.append({
                "Stock":          ticker,
                "Sector":         sector,
                "Date":           period_end,
                "Monthly Return": compound_return(grp),
            })

        # Yearly aggregates
        for period_end, price_grp in prices.resample("YE"):
            price_grp = price_grp.squeeze()
            ret_grp = daily_ret.reindex(price_grp.index).dropna()
            if len(price_grp) < 50:
                continue
            yr  = compound_return(price_grp)
            vol = annualised_vol(ret_grp)
            yearly_rows.append({
                "Stock":              ticker,
                "Sector":             sector,
                "Year":               period_end.year,
                "Yearly Return":      yr,
                "Yearly Volatility":  vol,
                "Yearly Sharpe":      (yr - RISK_FREE_ANNUAL) / vol if vol > 0 else np.nan,
                "Max Drawdown":       max_drawdown(price_grp),
            })

    except Exception as exc:
        errors.append({"Stock": ticker, "Error": str(exc)})

print("\nDone downloading.")

# ── Save outputs ──────────────────────────────────────────────────────────────
monthly_df = pd.DataFrame(monthly_rows)
yearly_df  = pd.DataFrame(yearly_rows)

monthly_df.to_excel("sp500_stock_returns.xlsx", index=False, sheet_name="Monthly")
yearly_df.to_excel("sp500_yearly_performance.xlsx", index=False, sheet_name="Yearly")

if errors:
    pd.DataFrame(errors).to_csv("download_errors.csv", index=False)
    print(f"  {len(errors)} tickers failed — see download_errors.csv")

print(f"Saved {len(monthly_df):,} monthly rows → sp500_stock_returns.xlsx")
print(f"Saved {len(yearly_df):,} yearly rows  → sp500_yearly_performance.xlsx")
