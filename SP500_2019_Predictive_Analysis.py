"""
SP500_2019_Predictive_Analysis.py
-----------------------------------
Uses 2015-2018 S&P 500 stock data to build a predictive model for 2019
and evaluates how "foreseeable" the 2019 rally actually was.

Prediction target: Alpha (stock return minus the cross-sectional mean for
that year).  This removes market-level noise — the +34% average 2019 return
is driven by macro forces the model can't see.  What IS predictable is
*which stocks beat the market*, and that's what we measure.

Narrative hypothesis
────────────────────
2019 was the S&P 500's best year since 2013.  The thesis here is that
while individual stock magnitudes are hard to forecast, *which stocks and
sectors would outperform* in 2019 was discernible from three structural
signals visible at the end of 2018:

  1. Mean-reversion pressure – stocks that fell hardest in 2018's Q4
     sell-off had the most room to recover once the Fed pivoted.

  2. Volatility regime – high 2018 volatility accompanied compressed
     valuations; stocks with elevated σ but solid multi-year track records
     historically rebound in risk-on environments.

  3. Multi-year momentum vs. reversion – 3-year momentum (2016-2018) was
     a stronger predictor than single-year momentum, capturing the
     Technology sector's sustained outperformance.

  4. Macro environment – the Fed's January 2019 pivot to rate cuts was
     foreshadowed by rising volatility and yield compression in 2018.

Evaluation metrics
──────────────────
  • R² and MAE on alpha (excess return over market)
  • Information Coefficient (Spearman rank correlation, predicted vs actual)
  • Top-quintile hit rate (what fraction of top-20% predictions actually land
    in the top 20% of actual returns)
  • Walk-forward validation across 2017–2024 (8 market environments)

Outputs
───────
  • sp500_2019_predictions.xlsx  – predictions, alpha, IC, outlier flag
  • figures/                     – 6 publication-quality charts for Tableau
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


class FeatureWinsorizer(BaseEstimator, TransformerMixin):
    """
    Clips each feature to [low_pct, high_pct] percentiles computed on
    the training data only — no look-ahead into the test set.
    Targets are never touched so the model learns from real extreme outcomes.
    """
    def __init__(self, low=0.01, high=0.99):
        self.low  = low
        self.high = high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.low  * 100, axis=0)
        self.hi_ = np.nanpercentile(X, self.high * 100, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def information_coefficient(y_true, y_pred):
    """Spearman rank correlation between predictions and actuals."""
    ic, _ = spearmanr(y_pred, y_true)
    return ic


def top_quintile_hit_rate(y_true, y_pred):
    """
    Fraction of stocks predicted to be in the top 20% that actually
    land in the top 20% of actual returns.
    """
    n = len(y_true)
    cutoff = int(n * 0.80)
    pred_top = set(np.argsort(y_pred)[-cutoff:]) if cutoff < n else set(range(n))
    # top 20% by prediction
    n_top = max(1, int(n * 0.20))
    pred_top = set(np.argsort(y_pred)[-n_top:])
    actual_top = set(np.argsort(np.asarray(y_true))[-n_top:])
    return len(pred_top & actual_top) / n_top


# ── Config ─────────────────────────────────────────────────────────────────────
YEARLY_FILE   = "sp500_yearly_performance.xlsx"
FALLBACK_FILE = "Yearly_Performance.xlsx"

PALETTE = {
    "Communication Services": "#4E79A7",
    "Consumer Discretionary":  "#F28E2B",
    "Consumer Staples":        "#E15759",
    "Energy":                  "#76B7B2",
    "Financials":              "#59A14F",
    "Health Care":             "#EDC948",
    "Industrials":             "#B07AA1",
    "Information Technology":  "#FF9DA7",
    "Materials":               "#9C755F",
    "Real Estate":             "#BAB0AC",
    "Utilities":               "#D37295",
    "Unknown":                 "#aaaaaa",
}

Path("figures").mkdir(exist_ok=True)


# ── Load stock data ────────────────────────────────────────────────────────────
try:
    df = pd.read_excel(YEARLY_FILE)
    print(f"Loaded {YEARLY_FILE}")
except FileNotFoundError:
    df = pd.read_excel(FALLBACK_FILE)
    df["Sector"] = "Unknown"
    print(f"Loaded fallback {FALLBACK_FILE} (no Sector column)")

df = df.sort_values(["Stock", "Year"]).reset_index(drop=True)

stock_counts = df.groupby("Stock")["Year"].count()
valid_stocks = stock_counts[stock_counts >= 4].index
df = df[df["Stock"].isin(valid_stocks)].copy()
print(f"Stocks with ≥4 years of data: {len(valid_stocks)}")


# ── Macro features (VIX, 10-yr yield, Fed rate change) ────────────────────────
def fetch_macro_features(years):
    """
    Download year-end macro indicators via yfinance.
    Returns DataFrame with columns: VIX_Level, Treasury_10yr, Fed_Rate_Change.
    Gracefully skipped if network is unavailable.
    """
    macro_rows = {yr: {} for yr in years}
    tickers = {"^VIX": "VIX_Level", "^TNX": "Treasury_10yr", "^IRX": "Fed_Proxy"}
    try:
        raw = yf.download(
            list(tickers.keys()),
            start="2012-01-01", end="2026-01-01",
            auto_adjust=True, progress=False,
        )
        closes = raw["Close"] if "Close" in raw else raw
        closes.index = pd.to_datetime(closes.index)
        yearly_end = closes.resample("YE").last()
        yearly_end.index = yearly_end.index.year

        for ticker, col in tickers.items():
            for yr in years:
                if yr in yearly_end.index and ticker in yearly_end.columns:
                    macro_rows[yr][col] = yearly_end.loc[yr, ticker]

        for yr in years:
            cur  = macro_rows[yr].get("Fed_Proxy", np.nan)
            prev = macro_rows.get(yr - 1, {}).get("Fed_Proxy", np.nan)
            macro_rows[yr]["Fed_Rate_Change"] = (
                cur - prev if not (np.isnan(cur) or np.isnan(prev)) else np.nan
            )
        for yr in years:
            macro_rows[yr].pop("Fed_Proxy", None)

    except Exception as exc:
        print(f"  Macro download warning: {exc} — macro features will be skipped")

    macro_df = pd.DataFrame.from_dict(macro_rows, orient="index")
    macro_df.index.name = "Year"
    return macro_df.reset_index()


all_years = sorted(df["Year"].unique())
macro_df  = fetch_macro_features(all_years)
has_macro = macro_df.drop(columns="Year").notna().any(axis=1).any()

if has_macro:
    df = df.merge(macro_df, on="Year", how="left")
    MACRO_FEATURES = [c for c in ["VIX_Level", "Treasury_10yr", "Fed_Rate_Change"]
                      if c in df.columns and df[c].notna().any()]
    print(f"Macro features loaded: {MACRO_FEATURES}")
else:
    MACRO_FEATURES = []
    print("Macro features unavailable — continuing without them")


# ── Feature engineering ────────────────────────────────────────────────────────
g = df.groupby("Stock")

df["Prev_Return"]  = g["Yearly Return"].shift(1)
df["Prev2_Return"] = g["Yearly Return"].shift(2)
df["Prev3_Return"] = g["Yearly Return"].shift(3)
df["Prev_Vol"]     = g["Yearly Volatility"].shift(1)
df["Prev2_Vol"]    = g["Yearly Volatility"].shift(2)

# Raw signals
df["Momentum_1yr"]      = df["Prev_Return"]
df["Momentum_3yr"]      = df[["Prev_Return", "Prev2_Return", "Prev3_Return"]].mean(axis=1)
df["Sharpe_Proxy"]      = df["Prev_Return"] / (df["Prev_Vol"] + 1e-6)
df["Vol_Change"]        = (df["Prev_Vol"] - df["Prev2_Vol"]).fillna(0)
df["Prev_Vol"]          = df["Prev_Vol"]
df["Return_Consistency"] = df[["Prev_Return", "Prev2_Return", "Prev3_Return"]].std(axis=1)

# Sector-relative momentum: how did this stock perform vs its own sector?
df["Sector_Avg_Prev_Return"] = df.groupby(["Year", "Sector"])["Prev_Return"].transform("mean")
df["Sector_Rel_Momentum"]    = df["Prev_Return"] - df["Sector_Avg_Prev_Return"]

# Cross-sectional rank features (within each year) — regime-stable signals
for raw_feat in ["Momentum_1yr", "Momentum_3yr", "Sharpe_Proxy", "Vol_Change",
                 "Prev_Vol", "Sector_Rel_Momentum"]:
    df[f"{raw_feat}_Rank"] = df.groupby("Year")[raw_feat].rank(pct=True)

# Alpha target: excess return over the cross-sectional mean for that year
df["Market_Avg_Return"] = df.groupby("Year")["Yearly Return"].transform("mean")
df["Alpha"]             = df["Yearly Return"] - df["Market_Avg_Return"]

BASE_FEATURES = [
    "Momentum_1yr",
    "Momentum_3yr",
    "Sharpe_Proxy",
    "Vol_Change",
    "Prev_Vol",
    "Return_Consistency",
    "Sector_Rel_Momentum",
    # Cross-sectional rank versions (regime-stable)
    "Momentum_1yr_Rank",
    "Momentum_3yr_Rank",
    "Sharpe_Proxy_Rank",
    "Vol_Change_Rank",
    "Prev_Vol_Rank",
    "Sector_Rel_Momentum_Rank",
] + MACRO_FEATURES


# ── Prepare model dataset ──────────────────────────────────────────────────────
model_data = df.dropna(subset=["Momentum_1yr", "Prev_Vol", "Alpha"]).copy()

sector_dummies = pd.get_dummies(model_data["Sector"], prefix="Sect", drop_first=True)
model_data = pd.concat([model_data, sector_dummies], axis=1)
SECTOR_COLS = sector_dummies.columns.tolist()
FEATURES    = BASE_FEATURES + SECTOR_COLS


# ── Walk-forward cross-validation (2017–2024) ──────────────────────────────────
def walk_forward_eval(data, features, train_start=2016, test_end=2024):
    """
    Expanding-window walk-forward evaluation on Alpha target.
    Trains on all years up to test_year-1, tests on test_year.
    Reports R², MAE, IC, and top-quintile hit rate per fold.
    """
    rows = []
    for test_yr in range(train_start + 1, test_end + 1):
        train_yrs = list(range(train_start, test_yr))
        tr = data[data["Year"].isin(train_yrs)]
        te = data[data["Year"] == test_yr]
        if len(tr) < 30 or len(te) < 10:
            continue

        X_tr, y_tr = tr[features], tr["Alpha"]
        X_te, y_te = te[features], te["Alpha"]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor",  FeatureWinsorizer(low=0.01, high=0.99)),
            ("scaler",  StandardScaler()),
            ("ridge",   Ridge(alpha=5.0)),
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)

        rows.append({
            "Test_Year":      test_yr,
            "Train_Years":    f"{train_start}–{test_yr - 1}",
            "N_Train":        len(tr),
            "N_Test":         len(te),
            "R2":             r2_score(y_te, preds),
            "MAE":            mean_absolute_error(y_te, preds),
            "IC":             information_coefficient(y_te.values, preds),
            "TopQ_HitRate":   top_quintile_hit_rate(y_te.values, preds),
        })
    return pd.DataFrame(rows)


wf_results = walk_forward_eval(model_data, FEATURES)
print("\n── Walk-Forward Evaluation (Ridge, alpha target, 2017–2024) ──────────")
print(wf_results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"  Mean R²:              {wf_results['R2'].mean():.3f}")
print(f"  Mean MAE:             {wf_results['MAE'].mean():.3f}")
print(f"  Mean IC:              {wf_results['IC'].mean():.3f}  (>0.05 = useful, >0.10 = strong)")
print(f"  Mean Top-Q Hit Rate:  {wf_results['TopQ_HitRate'].mean():.1%}  (random baseline = 20%)")


# ── Final 2019 model (trained on 2016–2018, predicts Alpha) ───────────────────
train = model_data[model_data["Year"].isin([2016, 2017, 2018])]
test  = model_data[model_data["Year"] == 2019]

X_train, y_train = train[FEATURES], train["Alpha"]
X_test,  y_test  = test[FEATURES],  test["Alpha"]
y_test_raw       = test["Yearly Return"]

imputer = SimpleImputer(strategy="median")
Xtr_imp = imputer.fit_transform(X_train)
Xte_imp = imputer.transform(X_test)

winsor = FeatureWinsorizer(low=0.01, high=0.99)
Xtr_w  = winsor.fit_transform(Xtr_imp)
Xte_w  = winsor.transform(Xte_imp)

scaler = StandardScaler()
Xtr_s  = scaler.fit_transform(Xtr_w)
Xte_s  = scaler.transform(Xte_w)

ridge = Ridge(alpha=5.0)
ridge.fit(Xtr_s, y_train)
y_pred_ridge = ridge.predict(Xte_s)

gbm = GradientBoostingRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, random_state=42
)
gbm.fit(Xtr_s, y_train)
y_pred_gbm = gbm.predict(Xte_s)

cv_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("winsor",  FeatureWinsorizer(low=0.01, high=0.99)),
    ("scaler",  StandardScaler()),
    ("ridge",   Ridge(alpha=5.0)),
])
cv_r2 = cross_val_score(cv_pipe, X_train, y_train, cv=5, scoring="r2").mean()

# Outlier mask: bottom/top 2% of actual 2019 raw returns (reporting filter only)
lo_out    = y_test_raw.quantile(0.02)
hi_out    = y_test_raw.quantile(0.98)
excl_mask = (y_test_raw >= lo_out) & (y_test_raw <= hi_out)
n_excluded = (~excl_mask).sum()

y_test_excl       = y_test[excl_mask]
y_pred_ridge_excl = pd.Series(y_pred_ridge, index=y_test.index)[excl_mask]
y_pred_gbm_excl   = pd.Series(y_pred_gbm,   index=y_test.index)[excl_mask]

print("\n── 2019 Out-of-Sample Performance (Alpha target) ─────────────────────")
print(f"  Ridge  R²: {r2_score(y_test, y_pred_ridge):.3f}  "
      f"MAE: {mean_absolute_error(y_test, y_pred_ridge):.3f}  "
      f"IC: {information_coefficient(y_test.values, y_pred_ridge):.3f}  "
      f"TopQ: {top_quintile_hit_rate(y_test.values, y_pred_ridge):.1%}  "
      f"(all {len(y_test)} stocks)")
print(f"  GBM    R²: {r2_score(y_test, y_pred_gbm):.3f}  "
      f"MAE: {mean_absolute_error(y_test, y_pred_gbm):.3f}  "
      f"IC: {information_coefficient(y_test.values, y_pred_gbm):.3f}  "
      f"TopQ: {top_quintile_hit_rate(y_test.values, y_pred_gbm):.1%}  "
      f"(all {len(y_test)} stocks)")
print(f"  Ridge 5-fold CV R² (alpha, train): {cv_r2:.3f}")
print(f"  Baseline (predict mean alpha=0):  R²=0.000  IC=0.000  TopQ=20.0%")
print(f"\n── Excluding {n_excluded} outlier stocks (bottom/top 2% of raw 2019 returns) ──")
print(f"  Ridge  R²: {r2_score(y_test_excl, y_pred_ridge_excl):.3f}  "
      f"MAE: {mean_absolute_error(y_test_excl, y_pred_ridge_excl):.3f}  "
      f"IC: {information_coefficient(y_test_excl.values, y_pred_ridge_excl.values):.3f}  "
      f"({len(y_test_excl)} stocks)")
print(f"  GBM    R²: {r2_score(y_test_excl, y_pred_gbm_excl):.3f}  "
      f"MAE: {mean_absolute_error(y_test_excl, y_pred_gbm_excl):.3f}  "
      f"IC: {information_coefficient(y_test_excl.values, y_pred_gbm_excl.values):.3f}  "
      f"({len(y_test_excl)} stocks)")

# Feature importances — aggregate sector and macro into single bars
feat_imp_all = pd.Series(gbm.feature_importances_, index=FEATURES)
sector_imp   = feat_imp_all[SECTOR_COLS].sum()
base_only    = [f for f in BASE_FEATURES if f not in MACRO_FEATURES]
feat_imp     = feat_imp_all[base_only].copy()
if MACRO_FEATURES:
    feat_imp["Macro (VIX/Rates)"] = feat_imp_all[MACRO_FEATURES].sum()
feat_imp["Sector"] = sector_imp
feat_imp = feat_imp.sort_values(ascending=False)
print("\n── Feature Importances (GBM) ──────────────────────────")
print(feat_imp.to_string())

# Build results table
results = test[["Stock", "Sector", "Yearly Return", "Alpha"]].copy()
results["Ridge_Alpha_Pred"] = y_pred_ridge
results["GBM_Alpha_Pred"]   = y_pred_gbm
results["Ridge_Error"]      = (results["Alpha"] - results["Ridge_Alpha_Pred"]).abs()
results["GBM_Error"]        = (results["Alpha"] - results["GBM_Alpha_Pred"]).abs()
results["Is_Outlier"]       = ~excl_mask.values

# Sector-level summary
sector_2019 = (
    results.groupby("Sector")
    .agg(
        Avg_Actual_Alpha=("Alpha", "mean"),
        Avg_Predicted_Alpha=("Ridge_Alpha_Pred", "mean"),
        Avg_Raw_Return=("Yearly Return", "mean"),
        Stock_Count=("Stock", "count"),
    )
    .round(3)
    .sort_values("Avg_Raw_Return", ascending=False)
)
print("\n── 2019 Sector Performance vs Predicted Alpha ───────────")
print(sector_2019.to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Predicted Alpha vs Actual Alpha (scatter, coloured by sector)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))
for sector, grp in results.groupby("Sector"):
    color = PALETTE.get(sector, "#888888")
    ax.scatter(grp["Ridge_Alpha_Pred"], grp["Alpha"],
               color=color, alpha=0.65, s=28, label=sector, linewidths=0)

lims = [
    min(results["Ridge_Alpha_Pred"].min(), results["Alpha"].min()) - 0.05,
    max(results["Ridge_Alpha_Pred"].max(), results["Alpha"].max()) + 0.05,
]
ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.axvline(0, color="grey", lw=0.6, ls=":")
ax.set_xlabel("Predicted Alpha (Excess over Market, Ridge)", fontsize=12)
ax.set_ylabel("Actual Alpha (Excess over Market)", fontsize=12)
ax.set_title("2019 Predicted vs Actual Alpha (Ridge Regression)\nTrained on 2016–2018  |  Feature-winsorized inputs", fontsize=13)
r2_all  = r2_score(y_test, y_pred_ridge)
r2_excl = r2_score(y_test_excl, y_pred_ridge_excl)
ic_all  = information_coefficient(y_test.values, y_pred_ridge)
ax.text(0.04, 0.93,
        f"R² = {r2_all:.3f}  (all {len(y_test)})\n"
        f"R² = {r2_excl:.3f}  (excl. {n_excluded} outliers)\n"
        f"IC = {ic_all:.3f}",
        transform=ax.transAxes, fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), fontsize=7.5, loc="lower right",
          title="Sector", title_fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig1_predicted_vs_actual_2019.png", dpi=150)
plt.close()
print("\nSaved figures/fig1_predicted_vs_actual_2019.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Feature importances
# ═══════════════════════════════════════════════════════════════════════════════
FEAT_LABELS = {
    "Momentum_1yr":              "1-Yr Momentum (raw)",
    "Momentum_3yr":              "3-Yr Momentum (raw)",
    "Sharpe_Proxy":              "Risk-Adj Return (raw)",
    "Vol_Change":                "Volatility Change (raw)",
    "Prev_Vol":                  "Prior Volatility (raw)",
    "Return_Consistency":        "Return Consistency\n(trailing std dev)",
    "Sector_Rel_Momentum":       "Sector-Relative\nMomentum (raw)",
    "Momentum_1yr_Rank":         "1-Yr Momentum (rank)",
    "Momentum_3yr_Rank":         "3-Yr Momentum (rank)",
    "Sharpe_Proxy_Rank":         "Risk-Adj Return (rank)",
    "Vol_Change_Rank":           "Volatility Change (rank)",
    "Prev_Vol_Rank":             "Prior Volatility (rank)",
    "Sector_Rel_Momentum_Rank":  "Sector-Relative\nMomentum (rank)",
    "Macro (VIX/Rates)":         "Macro Environment\n(VIX & Rates)",
    "Sector":                    "GICS Sector\n(combined)",
}
fig, ax = plt.subplots(figsize=(9, 6.5))
colors = ["#2166AC" if x > 0.08 else "#92C5DE" for x in feat_imp.values]
bars = ax.barh(
    [FEAT_LABELS.get(f, f) for f in feat_imp.index],
    feat_imp.values,
    color=colors, edgecolor="white", height=0.6
)
ax.set_xlabel("Feature Importance (Gradient Boosting)", fontsize=11)
ax.set_title("What Signals Predicted 2019 Alpha?\n(Trained on 2016–2018, alpha target, feature-winsorized)", fontsize=12)
for bar, val in zip(bars, feat_imp.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=8.5)
ax.set_xlim(0, feat_imp.max() * 1.2)
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/fig2_feature_importances.png", dpi=150)
plt.close()
print("Saved figures/fig2_feature_importances.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Year-over-year market context: avg return + % positive by year
# ═══════════════════════════════════════════════════════════════════════════════
yr_summary = (
    df[df["Year"].between(2015, 2024)]
    .groupby("Year")
    .agg(Avg_Return=("Yearly Return", "mean"),
         Pct_Positive=("Yearly Return", lambda x: (x > 0).mean()))
    .reset_index()
)

fig, ax1 = plt.subplots(figsize=(10, 5))
bar_colors = ["#d62728" if r < 0 else "#2ca02c" for r in yr_summary["Avg_Return"]]
ax1.bar(yr_summary["Year"], yr_summary["Avg_Return"],
        color=bar_colors, alpha=0.75, width=0.5, label="Avg Annual Return")
ax1.axhline(0, color="black", lw=0.8)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax1.set_ylabel("Average Annual Return", fontsize=11)
ax1.set_xlabel("")
ax1.set_xticks(yr_summary["Year"])

ax2 = ax1.twinx()
ax2.plot(yr_summary["Year"], yr_summary["Pct_Positive"],
         "o--", color="#1f77b4", lw=2, ms=6, label="% Stocks Positive")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2.set_ylabel("% Stocks with Positive Return", fontsize=11, color="#1f77b4")
ax2.tick_params(axis="y", colors="#1f77b4")
ax2.set_ylim(0, 1.1)

yr19 = yr_summary[yr_summary["Year"] == 2019].iloc[0]
ax1.annotate("2019 Peak\n(best since 2013)",
             xy=(2019, yr19["Avg_Return"]),
             xytext=(2020.5, yr19["Avg_Return"] + 0.04),
             fontsize=9, color="darkgreen",
             arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
ax1.set_title("S&P 500 Constituents – Annual Performance (2015–2024)", fontsize=13)
fig.tight_layout()
plt.savefig("figures/fig3_yearly_context.png", dpi=150)
plt.close()
print("Saved figures/fig3_yearly_context.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Sector-level: 2018 signal vs 2019 outcome (mean-reversion story)
# ═══════════════════════════════════════════════════════════════════════════════
data_2018 = model_data[model_data["Year"] == 2018][
    ["Stock", "Sector", "Yearly Return", "Yearly Volatility"]
].rename(columns={"Yearly Return": "Return_2018", "Yearly Volatility": "Vol_2018"})
data_2019 = model_data[model_data["Year"] == 2019][["Stock", "Yearly Return"]].rename(
    columns={"Yearly Return": "Return_2019"}
)
mv = data_2018.merge(data_2019, on="Stock")
sector_mv = mv.groupby("Sector")[["Return_2018", "Return_2019"]].mean().reset_index()

fig, ax = plt.subplots(figsize=(9, 6))
for _, row in sector_mv.iterrows():
    color = PALETTE.get(row["Sector"], "#888888")
    ax.scatter(row["Return_2018"], row["Return_2019"],
               color=color, s=180, zorder=3, edgecolors="white", linewidths=0.8)
    ax.annotate(row["Sector"].replace(" ", "\n"),
                xy=(row["Return_2018"], row["Return_2019"]),
                fontsize=7.5, ha="center", va="bottom",
                xytext=(0, 7), textcoords="offset points")

m, b = np.polyfit(sector_mv["Return_2018"], sector_mv["Return_2019"], 1)
xs = np.linspace(sector_mv["Return_2018"].min() - 0.01,
                 sector_mv["Return_2018"].max() + 0.01, 100)
ax.plot(xs, m * xs + b, "k--", lw=1.2, alpha=0.6, label=f"Trend (slope={m:.2f})")
ax.axvline(0, color="grey", lw=0.8, ls=":")
ax.axhline(0, color="grey", lw=0.8, ls=":")

r_val = np.corrcoef(sector_mv["Return_2018"], sector_mv["Return_2019"])[0, 1]
ax.text(0.04, 0.93, f"Sector r = {r_val:.2f}", transform=ax.transAxes,
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_xlabel("Average 2018 Return (by Sector)", fontsize=11)
ax.set_ylabel("Average 2019 Return (by Sector)", fontsize=11)
ax.set_title("Sector Mean-Reversion: 2018 Loss → 2019 Recovery\n"
             "(Each dot = one GICS sector, averaged across constituents)", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
plt.savefig("figures/fig4_mean_reversion_sectors.png", dpi=150)
plt.close()
print("Saved figures/fig4_mean_reversion_sectors.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Walk-forward R² by year (2017–2024)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 4.5))
bar_colors = ["#2ca02c" if r >= 0 else "#d62728" for r in wf_results["R2"]]
bars = ax.bar(wf_results["Test_Year"], wf_results["R2"],
              color=bar_colors, alpha=0.8, width=0.55, zorder=3)
ax.axhline(0, color="black", lw=1, zorder=2)
ax.axhline(wf_results["R2"].mean(), color="#1f77b4", lw=1.5, ls="--",
           label=f"Mean R² = {wf_results['R2'].mean():.3f}", zorder=4)

for bar, row in zip(bars, wf_results.itertuples()):
    ypos = bar.get_height() + 0.005 if row.R2 >= 0 else bar.get_height() - 0.025
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{row.R2:.3f}\n(n={row.N_Test})", ha="center", fontsize=8)

ax.set_xlabel("Test Year", fontsize=11)
ax.set_ylabel("Out-of-Sample R² (Alpha)", fontsize=11)
ax.set_title("Walk-Forward Alpha Prediction — 8-Year Validation (2017–2024)\n"
             "(Ridge Regression, expanding window, feature-winsorized)", fontsize=12)
ax.set_xticks(wf_results["Test_Year"])
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/fig5_walk_forward_r2.png", dpi=150)
plt.close()
print("Saved figures/fig5_walk_forward_r2.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Walk-forward IC and Top-Quintile Hit Rate by year
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(11, 4.5))

ax1.bar(wf_results["Test_Year"], wf_results["IC"],
        color="#4E79A7", alpha=0.75, width=0.4, label="IC (Spearman rank corr.)", zorder=3)
ax1.axhline(0,    color="black", lw=0.8, zorder=2)
ax1.axhline(0.05, color="#4E79A7", lw=1.2, ls=":", alpha=0.6, label="IC = 0.05 (useful threshold)")
ax1.axhline(0.10, color="#2166AC", lw=1.2, ls=":", alpha=0.6, label="IC = 0.10 (strong threshold)")
ax1.set_ylabel("Information Coefficient (IC)", fontsize=11, color="#4E79A7")
ax1.tick_params(axis="y", colors="#4E79A7")
ax1.set_ylim(-0.25, 0.35)

ax2 = ax1.twinx()
ax2.plot(wf_results["Test_Year"], wf_results["TopQ_HitRate"],
         "o--", color="#E15759", lw=2, ms=7, label="Top-Quintile Hit Rate", zorder=4)
ax2.axhline(0.20, color="#E15759", lw=1.2, ls=":", alpha=0.6, label="Random baseline = 20%")
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax2.set_ylabel("Top-Quintile Hit Rate", fontsize=11, color="#E15759")
ax2.tick_params(axis="y", colors="#E15759")
ax2.set_ylim(0, 0.55)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
ax1.set_xlabel("Test Year", fontsize=11)
ax1.set_xticks(wf_results["Test_Year"])
ax1.set_title("Walk-Forward Ranking Skill — IC & Top-Quintile Hit Rate (2017–2024)\n"
              "(IC > 0.05 = useful; Hit Rate > 20% = beats random selection)", fontsize=12)
ax1.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.savefig("figures/fig6_ic_and_hitrate.png", dpi=150)
plt.close()
print("Saved figures/fig6_ic_and_hitrate.png")


# ── Save predictions spreadsheet ──────────────────────────────────────────────
results = results.sort_values("Ridge_Alpha_Pred", ascending=False)
results.to_excel("sp500_2019_predictions.xlsx", index=False)
print("\nSaved sp500_2019_predictions.xlsx")
print("\n✓ All outputs complete.")
