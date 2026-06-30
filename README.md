# S&P 500 Predictive Analysis & Market Direction Model

Was the 2019 S&P 500 rally predictable? This project investigates that question using two complementary machine learning models: one that predicts the **market direction** (Bull / Neutral / Bear) and one that **ranks individual S&P 500 stocks** by expected alpha. Together they form a two-stage framework for annual portfolio allocation decisions.

**[View the Tableau Dashboard →](https://public.tableau.com/views/SP5002019-WastheBigJumpPredictable/Wasthe2019SP500ChangesPredictable?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)**

---

## What This Project Is For

This is not a day-trading or market-timing tool. It is designed for **annual portfolio construction** — helping an investor decide:

1. **How much** to allocate to equities next year (market direction model)
2. **Which stocks** to favor within that allocation (stock ranking model)

The practical use case: each December, feed in current macro data and prior-year stock metrics, run both models, and use the output to tilt your portfolio heading into the new year.

---

## How the Models Work

### Stage 1 — Market Direction Model (`SP500_Market_Direction_Model.py`)

Predicts whether the S&P 500 will be in a **Bull (≥8%), Bear (≤-5%), or Neutral** regime the following year using five macro signals observable at year-end:

| Signal | Source |
|---|---|
| CAPE (Shiller P/E) | multpl.com / Shiller data |
| Yield curve spread (10yr − 2yr) | FRED |
| Credit spread (BAA − 10yr) | FRED |
| VIX (year-end) | CBOE |
| Fed funds rate change | Federal Reserve |

Uses **Logistic Regression (C=0.5)** for direction classification and **Ridge Regression (alpha=2.0)** for return magnitude, both inside sklearn Pipelines with median imputation and standard scaling.

### Stage 2 — Stock Ranking Model (`SP500_2019_Predictive_Analysis.py`)

Ranks all ~500 S&P 500 constituents by predicted **alpha** (excess return over the cross-sectional market mean) using prior-year stock features:

- 1-year and 3-year momentum
- Sharpe proxy, prior volatility, volatility change
- Return consistency (trailing 3-year standard deviation)
- Sector-relative momentum (stock vs sector average)
- Cross-sectional rank features (regime-stable percentile ranks within each year)
- Macro context features (CAPE, yield spread, credit spread, VIX)

Uses **Ridge Regression (alpha=5.0)** as the primary model with a **Gradient Boosting** model for comparison, both inside Pipelines with median imputation, feature winsorization (1st/99th percentile, no look-ahead bias), and standard scaling.

---

## Key Statistics

### Stock Ranking Model (2019 out-of-sample)

| Metric | Value |
|---|---|
| Information Coefficient (IC) | 0.170 |
| Top-quintile hit rate | 33.7% |
| Outlier-excluded R² | reported in fig1 |

The top-quintile hit rate of 33.7% means roughly 1 in 3 of the model's top-ranked stocks actually landed in the top 20% of performers — approximately **1.7× better than random selection**.

### Walk-Forward Validation (2017–2024, 8 years)

| Metric | Value |
|---|---|
| Mean IC across years | 0.082 |
| Mean top-quintile hit rate | 30.1% |
| Best year IC | ~0.20+ |
| Worst year IC | near 0 |

Walk-forward validation uses an expanding training window with no look-ahead bias — each year's model is trained only on data available before that year.

### Market Direction Model (2000–2024, 25 test years)

| Metric | Value |
|---|---|
| Overall directional accuracy | 68% |
| Bull year accuracy | 89.5% (19 bull years) |
| Bear year accuracy | 0% (6 bear years) |
| Random baseline | 50% |
| Always-up baseline | 76% |

The model is strong at confirming bull markets and weak at predicting bear markets — a known limitation of macro signals for timing sharp drawdowns.

---

## How to Run

Run the scripts in this order:

```bash
# Step 1 — Download S&P 500 index data (~1 min)
python SP_Return_Data.py

# Step 2 — Download all ~500 individual stock histories (~15–30 min)
python SP500_Stock_Return_Data.py

# Step 3 — Train stock ranking model, generate predictions and figures fig1–fig6
python SP500_2019_Predictive_Analysis.py

# Step 4 — Run market direction model, generate macro dashboard and figures fig7–fig9
# (can be run independently at any time — does not depend on steps 1–3)
python SP500_Market_Direction_Model.py
```

Steps 1 and 2 only need to be re-run when you want fresh market data. Steps 3 and 4 can be re-run at any time against the existing Excel files.

---

## Project Structure

```
├── SP_Return_Data.py                    # Step 1 — S&P 500 index data download
├── SP500_Stock_Return_Data.py           # Step 2 — Individual constituent download
├── SP500_2019_Predictive_Analysis.py    # Step 3 — Stock ranking model
├── SP500_Market_Direction_Model.py      # Step 4 — Market direction model
│
├── sp500_returns.xlsx                   # Output of Step 1
├── sp500_stock_returns.xlsx             # Output of Step 2 (monthly)
├── sp500_yearly_performance.xlsx        # Output of Step 2 (yearly)
├── sp500_2019_predictions.xlsx          # Output of Step 3
├── sp500_market_direction.xlsx          # Output of Step 4
│
└── figures/
    ├── fig1_predicted_vs_actual.png     # Predicted vs actual alpha (Ridge), R² and IC
    ├── fig2_feature_importances.png     # GBM feature importances by signal group
    ├── fig3_yearly_context.png          # Annual S&P 500 return & breadth 2015–2024
    ├── fig4_mean_reversion_deciles.png  # 2018 losers vs 2019 winners by sector
    ├── fig5_walkforward_r2.png          # Walk-forward R² by year (2017–2024)
    ├── fig6_walkforward_ic_hitrate.png  # Walk-forward IC and top-quintile hit rate
    ├── fig7_market_direction.png        # Market direction predictions 2000–2024
    ├── fig8_macro_signals.png           # 6-panel macro signal dashboard
    └── fig9_combined_strategy.png       # Two-stage framework combined view
```

---

## Real-World Usage

At each year-end:

1. Observe CAPE, yield spread, credit spread, VIX, and Fed rate change
2. Run `SP500_Market_Direction_Model.py` → get **P(Bull)** and predicted regime
3. Set allocation size based on regime:
   - P(Bull) > 65% → Full equity allocation
   - P(Bull) 40–65% → Reduced / neutral allocation
   - P(Bull) < 40% → Defensive positioning
4. Run `SP500_2019_Predictive_Analysis.py` → get ranked stock list
5. Build a 20–30 stock portfolio from the top quintile, diversified across sectors

---

## Limitations

- **Annual cadence only** — not suitable for intraday or weekly trading
- **Bear market timing** — the direction model cannot reliably predict sharp drawdowns
- **S&P 500 constituents only** — not applicable to small/mid cap stocks
- **Survivorship bias** — constituent list sourced from current Wikipedia snapshot

---

## Requirements

```
yfinance
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
openpyxl
requests
```

Install with:

```bash
pip install yfinance pandas numpy scikit-learn scipy matplotlib seaborn openpyxl requests
```
