#!/bin/bash
# Quick verification script for M1.1 + M2.1 fixes

echo "=========================================="
echo "Verifying M1.1 + M2.1 Fixes"
echo "=========================================="

# Check 1: Excess metrics in backtest
echo ""
echo "1. Checking excess metrics in latest backtest..."
python3 << 'EOF'
import json
from pathlib import Path

metrics_path = Path("artifacts/backtest/metrics.json")
if metrics_path.exists():
    with open(metrics_path) as f:
        m = json.load(f)

    agent = m.get("agent", {})
    selic = m.get("selic", {})

    print(f"   Agent raw Sharpe: {agent.get('sharpe', 'N/A'):>8.4f}")
    print(f"   Agent excess Sharpe: {agent.get('excess_sharpe', 'N/A'):>8.4f} ← honest signal")
    print(f"   SELIC excess Sharpe: {selic.get('excess_sharpe', 'N/A'):>8.4f} (should be ≈0)")

    if agent.get('excess_sharpe', 0) < 0:
        print("   ✓ Negative excess Sharpe confirmed (agent destroys alpha)")
    if abs(selic.get('excess_sharpe', 1)) < 0.01:
        print("   ✓ SELIC excess Sharpe ≈ 0 (correct)")
else:
    print("   ✗ metrics.json not found - run evaluate first")
EOF

# Check 2: Test cap weights fix
echo ""
echo "2. Testing cap weights redistribution..."
python3 << 'EOF'
import numpy as np
from src.agent.env import PortfolioEnv
from src.agent.config import DEFAULT_CONFIG

try:
    env = PortfolioEnv(DEFAULT_CONFIG, date_range="test")
    env.reset(seed=42)

    # Concentrated action
    action = np.zeros(len(env.tickers), dtype=np.float32)
    action[0] = 10.0

    obs, _, _, _, info = env.step(action)
    weights = info["weights"]

    stock_mask = ~env._is_cash_mask
    stock_weights = weights[stock_mask]
    num_stocks = (stock_weights > 1e-6).sum()
    cash = weights[-1]

    print(f"   Active stocks: {num_stocks:>2} (should be > 10, not 1)")
    print(f"   Max stock weight: {stock_weights.max():.4f} (cap={DEFAULT_CONFIG.max_position_weight})")
    print(f"   CASH weight: {cash:.4f} (should NOT be ~0.9)")

    if num_stocks > 10 and cash < 0.5:
        print("   ✓ Cap redistribution working (overflow to stocks, not CASH)")
    else:
        print("   ✗ Cap bug may not be fixed")
except Exception as e:
    print(f"   ✗ Error: {e}")
EOF

# Check 3: Model file exists
echo ""
echo "3. Checking model files..."
if [ -f "artifacts/models/agent_best.zip" ]; then
    echo "   ✓ agent_best.zip exists"
    size=$(ls -lh artifacts/models/agent_best.zip | awk '{print $5}')
    echo "     Size: $size"
else
    echo "   ✗ agent_best.zip not found"
fi

# Check 4: Rolling eval results with excess metrics
echo ""
echo "4. Checking rolling_eval_results.json..."
python3 << 'EOF'
import json
from pathlib import Path

path = Path("artifacts/models/rolling_eval_results.json")
if path.exists():
    with open(path) as f:
        r = json.load(f)

    if r.get("windows"):
        w0 = r["windows"][0]
        agent_excess = w0["metrics"]["agent"].get("excess_sharpe", "N/A")
        print(f"   Window 0 agent excess_sharpe: {agent_excess}")
        if isinstance(agent_excess, (int, float)):
            print("   ✓ Excess metrics present in rolling windows")
        else:
            print("   ⚠ Excess metrics not yet in rolling_eval (old results)")
    else:
        print("   ✗ No windows in rolling_eval_results.json")
else:
    print("   ⚠ rolling_eval_results.json not found (run full trainer)")
EOF

echo ""
echo "=========================================="
echo "Verification complete!"
echo "=========================================="
