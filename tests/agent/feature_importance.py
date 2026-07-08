#!/usr/bin/env python3
"""Feature importance analysis: which of the 40 features actually predict next-day returns?"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import json

# Load dataset
dataset_path = Path("data/processed/ml_dataset.parquet")
df = pd.read_parquet(dataset_path)
df = df.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

# Compute target (tomorrow's return)
df["ret"] = df.groupby("ticker")["adj_close"].pct_change().apply(np.log1p)
df["ret_tomorrow"] = df.groupby("ticker")["ret"].shift(-1)

# Feature columns (exclude OHLCV, dates, target, identifiers)
skip = {
    "ticker", "trade_date", "ret", "ret_tomorrow", "close", "open", "high", "low",
    "adj_open", "adj_high", "adj_low", "adj_close", "volume", "volume_adjusted",
    "traded_amount", "num_trades", "reference_date"
}
X_cols = [c for c in df.columns if c not in skip and df[c].dtype in (np.float64, np.int64)]

# Prepare data (drop NaNs and inf values)
data = df[X_cols + ["ret_tomorrow"]].dropna()
# Replace inf with NaN, then drop
data = data.replace([np.inf, -np.inf], np.nan).dropna()
X = data[X_cols].values
y = data["ret_tomorrow"].values

# Final sanity check: remove any remaining non-finite values
mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
X = X[mask]
y = y[mask]

print(f"Dataset: {len(data):,} rows, {len(X_cols)} features")
print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
print()

# ============================================================================
# Method 1: Random Forest Feature Importance
# ============================================================================
print("=" * 80)
print("METHOD 1: Random Forest Feature Importance")
print("=" * 80)
rf = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
rf.fit(X, y)
rf_importance = pd.DataFrame({
    "feature": X_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print(f"R² on train set: {rf.score(X, y):.6f}")
print()
print("Top 15 features by Random Forest importance:")
for i, row in rf_importance.head(15).iterrows():
    print(f"  {row['feature']:35s}: {row['importance']:8.6f}")

# ============================================================================
# Method 2: Linear Regression Coefficients
# ============================================================================
print()
print("=" * 80)
print("METHOD 2: Linear Regression Coefficients (absolute magnitude)")
print("=" * 80)
lr = LinearRegression()
lr.fit(X, y)
lr_importance = pd.DataFrame({
    "feature": X_cols,
    "coefficient": lr.coef_,
    "abs_coefficient": np.abs(lr.coef_)
}).sort_values("abs_coefficient", ascending=False)

print(f"R² on train set: {lr.score(X, y):.6f}")
print(f"Intercept: {lr.intercept_:.8f}")
print()
print("Top 15 features by absolute linear coefficient:")
for i, row in lr_importance.head(15).iterrows():
    print(f"  {row['feature']:35s}: {row['coefficient']:+.8f}")

# ============================================================================
# Method 3: Correlation
# ============================================================================
print()
print("=" * 80)
print("METHOD 3: Correlation with target")
print("=" * 80)
corr_importance = pd.DataFrame({
    "feature": X_cols,
    "correlation": [data[[feat, "ret_tomorrow"]].corr().iloc[0, 1] for feat in X_cols],
}).copy()
corr_importance["abs_correlation"] = corr_importance["correlation"].abs()
corr_importance = corr_importance.sort_values("abs_correlation", ascending=False)

print()
print("Top 15 features by absolute correlation:")
for i, row in corr_importance.head(15).iterrows():
    print(f"  {row['feature']:35s}: {row['correlation']:+.6f}")

# ============================================================================
# Summary: Consensus ranking (average rank across all 3 methods)
# ============================================================================
print()
print("=" * 80)
print("CONSENSUS RANKING (average rank across all 3 methods)")
print("=" * 80)

# Assign ranks
rf_ranks = {feat: rank for rank, feat in enumerate(rf_importance["feature"], 1)}
lr_ranks = {feat: rank for rank, feat in enumerate(lr_importance["feature"], 1)}
corr_ranks = {feat: rank for rank, feat in enumerate(corr_importance["feature"], 1)}

consensus = pd.DataFrame({
    "feature": X_cols,
    "rf_rank": [rf_ranks[f] for f in X_cols],
    "lr_rank": [lr_ranks[f] for f in X_cols],
    "corr_rank": [corr_ranks[f] for f in X_cols],
}).copy()
consensus["avg_rank"] = consensus[["rf_rank", "lr_rank", "corr_rank"]].mean(axis=1)
consensus = consensus.sort_values("avg_rank")

print()
print("Top 20 features by consensus ranking:")
for i, row in consensus.head(20).iterrows():
    print(f"  {row['feature']:35s}: avg_rank={row['avg_rank']:6.2f} "
          f"(RF:{row['rf_rank']:3.0f}, LR:{row['lr_rank']:3.0f}, Corr:{row['corr_rank']:3.0f})")

# ============================================================================
# Save results
# ============================================================================
output_path = Path("data/models/feature_importance_analysis.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

results = {
    "metadata": {
        "n_samples": len(data),
        "n_features": len(X_cols),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
    },
    "random_forest": rf_importance.to_dict(orient="records"),
    "linear_regression": lr_importance.to_dict(orient="records"),
    "correlation": corr_importance.to_dict(orient="records"),
    "consensus": consensus.to_dict(orient="records"),
    "top_20_consensus": consensus.head(20)[["feature", "avg_rank"]].to_dict(orient="records"),
}

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print()
print(f"✓ Results saved → {output_path}")
print()
print("Next step: Review top 15–20 features and consider:")
print("  - Drop features with low consensus ranking (noise)")
print("  - Keep features with high ranking across methods (signal)")
print("  - Retrain agent with pruned feature set + 50-ticker universe")
