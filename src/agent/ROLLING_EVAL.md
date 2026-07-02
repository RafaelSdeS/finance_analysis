# Anchored Rolling Window Evaluation

Robustness testing via continuous retraining simulation.

## Motivation

The fixed train/val/test split (2000–2015 | 2015–2021 | 2021–2026) tests whether a single policy trained in the past can forecast the future. But real investors retrain continuously as new market data arrives.

**Rolling windows simulate this:**
- Each window trains on progressively larger historical data (always anchored at 2000)
- Tests on a non-overlapping future period
- No lookahead bias (train/test never overlap)
- Produces 8+ independent backtest evaluations across different market regimes

## Architecture

```
Window 0: train [2000–2010] → test [2010–2012]
Window 1: train [2000–2012] → test [2012–2014]
Window 2: train [2000–2014] → test [2014–2016]
...
Window 7: train [2000–2024] → test [2024–2026]
```

**Key properties:**
- ✅ **No lookahead**: test data for window N is not in train for window N+1
- ✅ **Expanding history**: each model sees more years than the last (realistic for learning)
- ✅ **Multiple regimes**: windows span 2008 crash, COVID, rate hikes, etc.
- ✅ **Independent backtest**: each window is a separate evaluation

## Usage

### Quick test (2 windows, 50K timesteps each):
```bash
python -m src.agent.rolling_eval 2>&1 | tee data/logs/rolling_eval.log
```
(Takes ~10–20 min on GPU)

### Full evaluation (8 windows, 1M timesteps per window):
Edit `rolling_eval.py`, change `timesteps_per_window=100_000` to `1_000_000` and run.
(Takes ~2–4 hours on GPU; can parallelize across multiple runs)

### Eval-only mode (load pre-trained, skip training):
```bash
python -c "
from src.agent.rolling_eval import run_rolling_eval
from src.agent.config import DEFAULT_CONFIG
results = run_rolling_eval(DEFAULT_CONFIG, skip_training=True)
"
```

## Output

Produces `data/models/rolling_eval_results.json` with:
- **summary**: aggregated stats (mean/std/min/max) for each metric and strategy across all windows
- **windows**: per-window metrics (detailed breakdown)

Example summary structure:
```json
{
  "summary": {
    "agent": {
      "sharpe": {"mean": 0.72, "std": 0.15, "min": 0.48, "max": 0.91},
      "max_drawdown": {"mean": 0.25, "std": 0.08, "min": 0.15, "max": 0.35}
    },
    "equal_weight": { ... }
  },
  "windows": [ ... ]
}
```

## Interpretation

**What matters:**
- **Agent vs equal_weight Sharpe**: if mean is +0.2+ and consistent (low std), agent has learned something robust.
- **Drawdown std**: if high, performance is regime-dependent (not robust).
- **Min/max spread**: wide spread = high variance (lucky vs unlucky windows).

**Example reads:**
- ✅ Agent mean Sharpe 0.75±0.08, equal-weight 0.70±0.10 → agent outperforms reliably
- ⚠️ Agent mean Sharpe 0.80±0.25 → volatile (good in some regimes, bad in others)
- ❌ Agent mean Sharpe 0.65±0.15, equal-weight 0.70±0.10 → agent underperforms

## Comparison to Fixed Split

| Aspect | Fixed (2021–2026) | Rolling (8 windows) |
|---|---|---|
| # backtests | 1 | 8 |
| Regimes covered | 1 (post-COVID) | 8 (2008–2026) |
| Statistical power | Low (n=1) | High (n=8) |
| Training time | ~1 hour | ~2–4 hours |

## Implementation Details

See `rolling_eval.py`:
- `generate_windows()`: create train/test windows
- `train_window()`: train PPO for one window
- `eval_window()`: backtest all strategies (agent, equal-weight, market-cap, 1/vol)
- `summarize_rolling_results()`: aggregate across windows

## Future Enhancements

- **Parallel training**: spawn N windows in parallel (each on separate GPU if available)
- **Hyperparameter tuning**: per-window or global meta-validation
- **Walk-forward proper**: non-anchored windows if you want to test "never retrain" vs "always retrain" tradeoffs
