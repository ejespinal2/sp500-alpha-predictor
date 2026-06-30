"""
SP500_Market_Direction_Model.py
---------------------------------
Predicts S&P 500 market direction (bull / bear / neutral) for a given year
using macro signals observable at year-end of the prior year.

This is a companion to SP500_2019_Predictive_Analysis.py:
  • That model answers: given a market environment, WHICH stocks will outperform?
  • This model answers: WHAT WILL the market environment be next year?

Used together they form a two-stage decision framework:
  Stage 1 (this file) → Market regime signal: should you be invested?
  Stage 2 (main file) → Stock ranking signal: which stocks to hold?

Features (all measured at end of year Y-1 to predict year Y)
─────────────────────────────────────────────────────────────
  CAPE            – Shiller cyclically-adjusted P/E (valuation)
  Yield_Spread    – 10yr minus 2yr Treasury yield (recession signal)
  Fed_Rate_Change – YoY change in Fed Funds Rate (monetary policy direction)
  Credit_Spread   – Moody's BAA minus 10yr Treasury (corporate stress)
  VIX             – CBOE Volatility Index year-end level (fear gauge)
  Trailing_Return – Prior year S&P 500 total return (momentum)

Data sources (hardcoded — permanent public records)
────────────────────────────────────────────────────
  CAPE:           Robert Shiller / Yale (http://www.econ.yale.edu/~shiller/)
  Yields/Rates:   Federal Reserve FRED (https://fred.stlouisfed.org/)
  Credit Spread:  FRED series BAA10Y
  VIX:            CBOE / Yahoo Finance
  S&P 500 Return: Computed from ^GSPC daily prices

If yfinance network access is available, the script will attempt to refresh
VIX and S&P 500 return for the most recent year automatically.

Outputs
───────
  • sp500_market_direction.xlsx   – walk-forward predictions + regimes 1990-2024
  • figures/fig7_market_direction.png   – predicted vs actual direction by year
  • figures/fig8_macro_signals.png      – signal dashboard (all 6 features)
  • figures/fig9_combined_strategy.png  – regime × stock picks combined view
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

Path("figures").mkdir(exist_ok=True)


# ── Hardcoded macro dataset (1988–2024) ────────────────────────────────────────
# All values are year-END observations, used to predict the FOLLOWING year.
# Sources: Shiller (CAPE), FRED (yields, rates, spreads), CBOE (VIX).

MACRO_DATA = {
    # Year: (CAPE, Yield_Spread_10y2y, Fed_Funds_Rate, Credit_Spread_BAA, VIX, SP500_Return)
    # Yield spread and credit spread in percentage points; VIX is index level.
    1988: (14.5,  0.83, 8.75, 1.88,   np.nan,  0.166),
    1989: (17.6,  0.28, 8.25, 1.75,   np.nan,  0.316),
    1990: (15.9,  0.23, 7.31, 2.18,   18.2,   -0.031),
    1991: (19.5,  1.42, 4.43, 1.96,   16.2,    0.305),
    1992: (20.5,  1.82, 2.92, 1.68,   11.2,    0.076),
    1993: (21.9,  1.16, 2.96, 1.38,    9.7,    0.101),
    1994: (18.8,  0.21, 5.50, 1.62,   11.3,    0.013),
    1995: (24.5,  0.77, 5.50, 1.42,   10.0,    0.376),
    1996: (28.3,  0.71, 5.25, 1.38,   14.7,    0.230),
    1997: (33.8,  0.44, 5.25, 1.32,   22.8,    0.334),
    1998: (41.3,  0.73, 4.75, 2.03,   25.7,    0.286),
    1999: (44.2,  0.29, 5.50, 1.85,   24.4,    0.210),
    2000: (36.4, -0.15, 6.50, 2.48,   23.3,   -0.091),
    2001: (30.1,  1.63, 1.75, 2.92,   22.7,   -0.119),
    2002: (22.9,  2.09, 1.25, 3.02,   27.3,   -0.221),
    2003: (27.7,  2.45, 1.00, 1.84,   18.3,    0.287),
    2004: (28.3,  1.56, 2.25, 1.58,   13.1,    0.109),
    2005: (27.5,  0.21, 4.25, 1.44,   11.4,    0.049),
    2006: (27.2,  0.06, 5.25, 1.32,   11.6,    0.158),
    2007: (24.8,  1.34, 4.25, 2.72,   22.5,    0.055),
    2008: (15.2,  1.51, 0.25, 5.41,   40.0,   -0.370),
    2009: (20.3,  2.79, 0.25, 2.64,   21.7,    0.265),
    2010: (22.5,  2.48, 0.25, 2.12,   17.8,    0.151),
    2011: (19.6,  1.62, 0.25, 2.62,   23.4,    0.021),
    2012: (22.0,  1.48, 0.25, 2.08,   18.0,    0.160),
    2013: (25.7,  2.58, 0.25, 1.64,   13.7,    0.324),
    2014: (27.2,  1.65, 0.25, 1.78,   19.2,    0.137),
    2015: (26.5,  1.21, 0.50, 2.28,   18.2,    0.014),
    2016: (28.0,  1.26, 0.75, 1.88,   14.0,    0.120),
    2017: (32.3,  0.51, 1.50, 1.60,   11.0,    0.219),
    2018: (28.4,  0.19, 2.50, 2.22,   25.4,   -0.044),
    2019: (30.7,  0.35, 1.75, 1.91,   13.8,    0.315),
    2020: (33.3,  0.79, 0.25, 2.21,   22.8,    0.184),
    2021: (40.5,  0.79, 0.25, 1.50,   17.2,    0.287),
    2022: (28.4, -0.55, 4.50, 2.39,   21.7,   -0.181),
    2023: (31.4,  0.38, 5.50, 1.70,   12.5,    0.263),
    2024: (37.0,  0.25, 4.50, 1.65,   17.4,    0.250),
}

# Build base DataFrame
macro_df = pd.DataFrame.from_dict(
    MACRO_DATA, orient="index",
    columns=["CAPE", "Yield_Spread", "Fed_Funds_Rate", "Credit_Spread", "VIX", "SP500_Return"]
)
macro_df.index.name = "Year"
macro_df = macro_df.reset_index().sort_values("Year").reset_index(drop=True)


# ── Try to refresh most-recent data via yfinance (optional) ───────────────────
try:
    import yfinance as yf
    raw = yf.download("^GSPC ^VIX", start="2023-01-01", end="2026-01-01",
                      auto_adjust=True, progress=False)
    closes = raw["Close"]
    closes.index = pd.to_datetime(closes.index)
    yearly = closes.resample("YE").last()
    yearly.index = yearly.index.year
    sp_ret = closes["^GSPC"].resample("YE").apply(lambda x: (x.iloc[-1] / x.iloc[0]) - 1)
    sp_ret.index = sp_ret.index.year
    for yr in yearly.index:
        mask = macro_df["Year"] == yr
        if mask.any():
            if "^VIX" in yearly.columns:
                macro_df.loc[mask, "VIX"] = yearly.loc[yr, "^VIX"]
            if yr in sp_ret.index:
                macro_df.loc[mask, "SP500_Return"] = sp_ret.loc[yr]
    print("Refreshed recent VIX and S&P 500 data via yfinance")
except Exception:
    print("Live refresh unavailable — using hardcoded data")


# ── Feature engineering ────────────────────────────────────────────────────────
# All features are prior-year values; target is current-year S&P 500 return.
macro_df = macro_df.sort_values("Year").reset_index(drop=True)

macro_df["Fed_Rate_Change"] = macro_df["Fed_Funds_Rate"].diff()   # YoY policy delta
macro_df["CAPE_ZScore"]     = (macro_df["CAPE"] - macro_df["CAPE"].expanding().mean()) / \
                               macro_df["CAPE"].expanding().std()  # valuation vs own history

# Shift features forward one year: features at year Y predict return in year Y+1
feature_cols = ["CAPE", "CAPE_ZScore", "Yield_Spread", "Fed_Rate_Change",
                "Credit_Spread", "VIX", "SP500_Return"]
model_df = pd.DataFrame()
model_df["Year"]           = macro_df["Year"] + 1
model_df["Actual_Return"]  = macro_df["SP500_Return"].shift(-1)
for col in feature_cols:
    model_df[col] = macro_df[col].values
model_df = model_df.rename(columns={"SP500_Return": "Trailing_Return"})
model_df = model_df.dropna(subset=["Actual_Return"]).reset_index(drop=True)

FEATURES = ["CAPE", "CAPE_ZScore", "Yield_Spread", "Fed_Rate_Change",
            "Credit_Spread", "VIX", "Trailing_Return"]

# Binary target: 1 = market up (>0%), 0 = market down
model_df["Direction"] = (model_df["Actual_Return"] > 0).astype(int)

# Regime labels for display
def regime_label(ret):
    if ret >= 0.08:   return "Bull"
    elif ret <= -0.05: return "Bear"
    else:              return "Neutral"

model_df["Actual_Regime"] = model_df["Actual_Return"].apply(regime_label)

print(f"\nMacro dataset: {len(model_df)} years ({model_df['Year'].min()}–{model_df['Year'].max()})")
print(f"  Bull years (>+8%):    {(model_df['Actual_Return'] >= 0.08).sum()}")
print(f"  Bear years (<-5%):    {(model_df['Actual_Return'] <= -0.05).sum()}")
print(f"  Neutral years:        {((model_df['Actual_Return'] > -0.05) & (model_df['Actual_Return'] < 0.08)).sum()}")


# ── Walk-forward evaluation ────────────────────────────────────────────────────
def walk_forward_market(data, features, train_start_yr=1990, test_start_yr=2000):
    """
    Expanding-window walk-forward.
    Trains on everything from train_start_yr up to test_yr-1.
    Tests on test_yr.  Returns per-year predictions for both Ridge (return
    magnitude) and Logistic (direction probability).
    """
    rows = []
    test_years = data[data["Year"] >= test_start_yr]["Year"].tolist()

    for test_yr in test_years:
        tr = data[(data["Year"] >= train_start_yr) & (data["Year"] < test_yr)]
        te = data[data["Year"] == test_yr]
        if len(tr) < 8 or len(te) == 0:
            continue

        X_tr = tr[features].values
        y_tr_ret = tr["Actual_Return"].values
        y_tr_dir = tr["Direction"].values
        X_te = te[features].values
        y_te_ret = te["Actual_Return"].values[0]
        y_te_dir = te["Direction"].values[0]

        # Ridge — predicts continuous return
        ridge_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   Ridge(alpha=2.0)),
        ])
        ridge_pipe.fit(X_tr, y_tr_ret)
        pred_return = ridge_pipe.predict(X_te)[0]

        # Logistic — predicts up/down probability
        logit_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
        ])
        logit_pipe.fit(X_tr, y_tr_dir)
        pred_prob  = logit_pipe.predict_proba(X_te)[0][1]   # P(up)
        pred_dir   = int(pred_prob >= 0.5)

        rows.append({
            "Year":           test_yr,
            "N_Train":        len(tr),
            "Predicted_Return": pred_return,
            "Actual_Return":  y_te_ret,
            "Bull_Prob":      pred_prob,
            "Pred_Direction": pred_dir,
            "Actual_Direction": y_te_dir,
            "Correct":        int(pred_dir == y_te_dir),
            "Pred_Regime":    regime_label(pred_return),
            "Actual_Regime":  regime_label(y_te_ret),
        })

    return pd.DataFrame(rows)


wf = walk_forward_market(model_df, FEATURES)

print("\n── Walk-Forward Market Direction (2000–2024) ─────────────────────────")
print(wf[["Year", "Predicted_Return", "Actual_Return", "Bull_Prob",
          "Pred_Regime", "Actual_Regime", "Correct"]].to_string(index=False,
          float_format=lambda x: f"{x:+.3f}" if abs(x) < 5 else f"{x:.1f}"))

dir_acc   = wf["Correct"].mean()
mae       = mean_absolute_error(wf["Actual_Return"], wf["Predicted_Return"])
ic, _     = spearmanr(wf["Predicted_Return"], wf["Actual_Return"])
bear_mask = wf["Actual_Direction"] == 0
bear_acc  = wf.loc[bear_mask, "Correct"].mean() if bear_mask.any() else np.nan
bull_acc  = wf.loc[~bear_mask, "Correct"].mean()

print(f"\n  Directional accuracy (all years):  {dir_acc:.1%}  (random baseline = 50%, always-up baseline = {(~bear_mask).mean():.1%})")
print(f"  Bear year accuracy:                {bear_acc:.1%}  (n={bear_mask.sum()})")
print(f"  Bull year accuracy:                {bull_acc:.1%}  (n={(~bear_mask).sum()})")
print(f"  Return MAE:                        {mae:.3f}")
print(f"  Return IC (Spearman):              {ic:.3f}")


# ── Full-sample model coefficients (for interpretation) ───────────────────────
full_X = model_df[FEATURES].values
full_y_ret = model_df["Actual_Return"].values
full_y_dir = model_df["Direction"].values

imputer_full = SimpleImputer(strategy="median")
scaler_full  = StandardScaler()
X_imp = imputer_full.fit_transform(full_X)
X_sc  = scaler_full.fit_transform(X_imp)

ridge_full  = Ridge(alpha=2.0).fit(X_sc, full_y_ret)
logit_full  = LogisticRegression(C=0.5, max_iter=1000, random_state=42).fit(X_sc, full_y_dir)

coef_df = pd.DataFrame({
    "Feature":      FEATURES,
    "Ridge_Coef":   ridge_full.coef_,
    "Logit_Coef":   logit_full.coef_[0],
}).sort_values("Ridge_Coef")

print("\n── Model Coefficients (full sample, standardized features) ───────────")
print(coef_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


# ── Regime classification for every year ──────────────────────────────────────
wf["Regime_Match"] = wf["Pred_Regime"] == wf["Actual_Regime"]
regime_acc = wf["Regime_Match"].mean()
print(f"\n  3-way regime accuracy (Bull/Bear/Neutral): {regime_acc:.1%}")

# ── Save walk-forward results ─────────────────────────────────────────────────
wf_out = wf.copy()
wf_out["Predicted_Return_Pct"] = (wf_out["Predicted_Return"] * 100).round(1)
wf_out["Actual_Return_Pct"]    = (wf_out["Actual_Return"]    * 100).round(1)
wf_out["Bull_Prob_Pct"]        = (wf_out["Bull_Prob"]        * 100).round(1)
wf_out.to_excel("sp500_market_direction.xlsx", index=False)
print("\nSaved sp500_market_direction.xlsx")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Walk-forward market direction predictions vs actual
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(14, 6))

# Actual returns as bars — coloured by direction
bar_colors = ["#d62728" if r < 0 else "#2ca02c" for r in wf["Actual_Return"]]
bars = ax1.bar(wf["Year"], wf["Actual_Return"], color=bar_colors, alpha=0.6,
               width=0.55, label="Actual S&P 500 Return", zorder=2)

# Predicted return as step line
ax1.step(wf["Year"], wf["Predicted_Return"], where="mid",
         color="#1f77b4", lw=2.2, label="Predicted Return (Ridge)", zorder=4)
ax1.axhline(0, color="black", lw=0.8, zorder=3)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax1.set_ylabel("S&P 500 Annual Return", fontsize=11)
ax1.set_ylim(-0.55, 0.55)

# Bull probability on second axis
ax2 = ax1.twinx()
ax2.plot(wf["Year"], wf["Bull_Prob"], "o--", color="#9467bd",
         lw=1.8, ms=5, alpha=0.85, label="P(Bull) — Logistic", zorder=5)
ax2.axhline(0.5, color="#9467bd", lw=0.8, ls=":", alpha=0.5)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2.set_ylabel("Predicted Bull Probability", fontsize=11, color="#9467bd")
ax2.tick_params(axis="y", colors="#9467bd")
ax2.set_ylim(0, 1.1)

# Annotate correct / missed calls
for _, row in wf.iterrows():
    if row["Correct"] == 0:
        ax1.annotate("✗", xy=(row["Year"], 0.37), fontsize=11,
                     color="darkred", ha="center", fontweight="bold")

ax1.set_xticks(wf["Year"])
ax1.set_xticklabels(wf["Year"].astype(int), rotation=45, ha="right", fontsize=9)
ax1.set_xlabel("Year", fontsize=11)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
ax1.set_title(
    f"Walk-Forward S&P 500 Market Direction Model (2000–2024)\n"
    f"Directional Accuracy = {dir_acc:.1%}  |  Bear Year Acc = {bear_acc:.1%}  |"
    f"  Return IC = {ic:.3f}  |  ✗ = missed call",
    fontsize=12
)
fig.tight_layout()
plt.savefig("figures/fig7_market_direction.png", dpi=150)
plt.close()
print("\nSaved figures/fig7_market_direction.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 – Macro signal dashboard: all 6 features vs market direction
# ═══════════════════════════════════════════════════════════════════════════════
SIGNAL_META = {
    "CAPE":             ("Shiller CAPE (Valuation)", "High = expensive → bearish for future returns"),
    "Yield_Spread":     ("Yield Curve (10yr − 2yr, %)", "Negative = inverted = recession warning"),
    "Fed_Rate_Change":  ("Fed Rate Change YoY (%pts)", "Negative = cutting rates = bullish"),
    "Credit_Spread":    ("Credit Spread BAA − 10yr (%)", "High = corporate stress = bearish"),
    "VIX":              ("VIX Year-End Level", "Extremely high = contrarian buy signal"),
    "Trailing_Return":  ("Prior Year S&P 500 Return", "Positive = momentum, negative = mean-reversion"),
}

plot_features = [f for f in SIGNAL_META if f in model_df.columns]
n = len(plot_features)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, feat in enumerate(plot_features):
    ax = axes[i]
    sub = model_df[["Year", feat, "Actual_Return", "Actual_Regime"]].dropna()

    colors = {"Bull": "#2ca02c", "Bear": "#d62728", "Neutral": "#ff7f0e"}
    for regime, grp in sub.groupby("Actual_Regime"):
        ax.scatter(grp[feat], grp["Actual_Return"],
                   color=colors.get(regime, "#aaa"), alpha=0.75, s=40,
                   label=regime, zorder=3)

    # Trend line
    if sub[feat].notna().sum() > 5:
        m, b = np.polyfit(sub[feat].fillna(sub[feat].median()), sub["Actual_Return"], 1)
        xs = np.linspace(sub[feat].min(), sub[feat].max(), 100)
        ax.plot(xs, m * xs + b, "k--", lw=1.1, alpha=0.5)

    ic_val, _ = spearmanr(sub[feat].fillna(sub[feat].median()), sub["Actual_Return"])
    title, subtitle = SIGNAL_META[feat]
    ax.set_title(f"{title}\nIC = {ic_val:+.3f}", fontsize=9.5)
    ax.set_xlabel(feat, fontsize=8.5)
    ax.set_ylabel("Next-Year S&P Return", fontsize=8.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.spines[["top", "right"]].set_visible(False)

    if i == 0:
        ax.legend(fontsize=7.5, title="Actual Regime", title_fontsize=8)

fig.suptitle("Macro Signals vs Next-Year S&P 500 Return (1990–2024)\n"
             "Each dot = one year, coloured by actual market regime",
             fontsize=12, y=1.01)
fig.tight_layout()
plt.savefig("figures/fig8_macro_signals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/fig8_macro_signals.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 – Combined strategy: regime confidence × stock alpha picks (2019)
# ═══════════════════════════════════════════════════════════════════════════════
# Load 2019 stock predictions from the main model, merge regime context
try:
    stock_preds = pd.read_excel("sp500_2019_predictions.xlsx")
    has_stock = True
except FileNotFoundError:
    has_stock = False

if has_stock and "Ridge_Alpha_Pred" in stock_preds.columns:
    row_2019 = wf[wf["Year"] == 2019].iloc[0]
    bull_prob_2019 = row_2019["Bull_Prob"]
    pred_ret_2019  = row_2019["Predicted_Return"]

    # Top 20 predicted alpha stocks (non-outliers)
    top20 = (stock_preds[~stock_preds.get("Is_Outlier", False)]
             .nlargest(20, "Ridge_Alpha_Pred")
             [["Stock", "Sector", "Yearly Return", "Ridge_Alpha_Pred"]])

    PALETTE = {
        "Communication Services": "#4E79A7", "Consumer Discretionary": "#F28E2B",
        "Consumer Staples": "#E15759",        "Energy": "#76B7B2",
        "Financials": "#59A14F",              "Health Care": "#EDC948",
        "Industrials": "#B07AA1",             "Information Technology": "#FF9DA7",
        "Materials": "#9C755F",               "Real Estate": "#BAB0AC",
        "Utilities": "#D37295",               "Unknown": "#aaaaaa",
    }

    fig, (ax_main, ax_gauge) = plt.subplots(
        1, 2, figsize=(14, 6.5),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    # Left: horizontal bar chart of top 20 stocks
    colors_s = [PALETTE.get(s, "#aaa") for s in top20["Sector"]]
    y_pos = range(len(top20))
    ax_main.barh(y_pos, top20["Yearly Return"], color="#d3d3d3", alpha=0.5,
                 height=0.5, label="Actual Return")
    ax_main.barh(y_pos, top20["Ridge_Alpha_Pred"], color=colors_s, alpha=0.85,
                 height=0.5, label="Predicted Alpha")
    ax_main.set_yticks(list(y_pos))
    ax_main.set_yticklabels(top20["Stock"].tolist(), fontsize=9)
    ax_main.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_main.axvline(0, color="black", lw=0.8)
    ax_main.set_xlabel("Return / Predicted Alpha", fontsize=10)
    ax_main.set_title(
        "Stage 2: Top 20 Stock Alpha Picks for 2019\n"
        "(Grey = actual return, Coloured = predicted alpha by sector)",
        fontsize=11
    )
    ax_main.invert_yaxis()
    ax_main.spines[["top", "right"]].set_visible(False)

    # Right: regime confidence gauge
    gauge_color = "#2ca02c" if bull_prob_2019 >= 0.6 else \
                  "#d62728" if bull_prob_2019 <= 0.4 else "#ff7f0e"
    ax_gauge.barh([0], [bull_prob_2019], color=gauge_color, alpha=0.85, height=0.4)
    ax_gauge.barh([0], [1 - bull_prob_2019], left=[bull_prob_2019],
                  color="#e0e0e0", alpha=0.5, height=0.4)
    ax_gauge.axvline(0.5, color="black", lw=1, ls="--")
    ax_gauge.set_xlim(0, 1)
    ax_gauge.set_ylim(-0.5, 0.5)
    ax_gauge.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_gauge.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax_gauge.set_yticks([])
    ax_gauge.set_xlabel("Bull Probability", fontsize=10)
    ax_gauge.set_title(
        f"Stage 1: Market Regime (2019)\n"
        f"P(Bull) = {bull_prob_2019:.1%}\n"
        f"Predicted Return = {pred_ret_2019:+.1%}\n"
        f"→ Regime: {row_2019['Pred_Regime']}  |  Actual: {row_2019['Actual_Regime']}",
        fontsize=10
    )
    ax_gauge.spines[["top", "right", "left"]].set_visible(False)
    ax_gauge.text(bull_prob_2019 / 2, 0, f"{bull_prob_2019:.0%}",
                  ha="center", va="center", fontsize=13, fontweight="bold", color="white")

    fig.suptitle("Two-Stage Strategy: Market Direction → Stock Selection (2019)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/fig9_combined_strategy.png", dpi=150)
    plt.close()
    print("Saved figures/fig9_combined_strategy.png")
else:
    print("Stock predictions not found — skipping fig9 (run SP500_2019_Predictive_Analysis.py first)")


# ── Print actionable framework summary ────────────────────────────────────────
print("\n── How to Use This Framework ─────────────────────────────────────────")
print("  1. At year-end, observe: CAPE, yield curve, Fed rate, credit spread, VIX")
print("  2. Run this model → get P(Bull) and predicted return band")
print("  3. Set exposure level:")
print("       P(Bull) > 65% → Full allocation, tilt to stock model's top quintile")
print("       P(Bull) 40-65% → Neutral allocation, diversify across sectors")
print("       P(Bull) < 40% → Reduce exposure, focus on defensive sectors")
print("  4. Run SP500_2019_Predictive_Analysis.py → get ranked stock picks")
print("  5. Combine: regime sets how much to invest, stock model sets what to buy")

print("\n✓ Market direction model complete.")
