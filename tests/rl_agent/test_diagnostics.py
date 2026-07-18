"""
Test: diagnostics.py's 8-D2 (cross-seed consistency) and 8-D3 (ranking
quality) computations (EIIE_DIAGNOSIS_PLAN.md Phase 8). Synthetic data only
-- `_load_run_weights`/`main` need real on-disk experiment artifacts
(config.json, weights.npz or model.pt, the real dataset) and are exercised
manually against actual run directories, not by this fast suite.

Run from project root:
    python tests/rl_agent/test_diagnostics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.rl_agent import diagnostics as diag  # noqa: E402
from src.rl_agent.data import GlobalAssetIndex, PricePanel  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402

N_ASSETS = 15


def _ranked_panel(T=10):
    """15 tickers, each growing at a distinct fixed daily rate (ticker i's
    rate = i * 0.001, i=1..15) -- so the forward-return ranking is identical
    and known on every day: ticker 15 always best, ticker 1 always worst."""
    tickers = tuple(f"T{i:02d}" for i in range(1, N_ASSETS + 1))
    asset_index = GlobalAssetIndex(tickers=tickers, ticker_to_gidx={t: i + 1 for i, t in enumerate(tickers)})
    rates = np.arange(1, N_ASSETS + 1) * 0.001
    t_arr = np.arange(T)[:, None]
    close = np.column_stack([np.ones(T), 100.0 * (1.0 + rates[None, :]) ** t_arr])
    dates = pd.bdate_range("2021-01-01", periods=T)
    n_slots = N_ASSETS
    return PricePanel(
        asset_index=asset_index, dates=dates, close=close, high=close.copy(), low=close.copy(),
        cdi_factor=np.full(T, 1.0001),
        slot_gidx=np.tile(np.arange(1, N_ASSETS + 1), (T, 1)),
        valid=np.ones((T, n_slots), dtype=bool),
        window=2, start_idx=0, end_idx=T - 1,
    )


def test_forward_return(passed, failed):
    panel = _ranked_panel(T=10)
    fwd = diag.forward_return(panel, t=0, k=3)
    expected_last = (100.0 * 1.015 ** 3) / 100.0 - 1.0  # ticker 15, rate 0.015
    ok = np.isclose(fwd[-1], expected_last)
    print_check("forward_return: last (fastest) ticker matches hand-computed 3-day growth", ok,
                f"got {fwd[-1]:.6f}, expected {expected_last:.6f}")
    passed, failed = passed + ok, failed + (not ok)

    expected_cash = 1.0001 ** 3 - 1.0
    ok = np.isclose(fwd[0], expected_cash)
    print_check("forward_return: cash slot matches compounded CDI factor", ok)
    passed, failed = passed + ok, failed + (not ok)

    fwd_end = diag.forward_return(panel, t=8, k=3)
    ok = np.all(np.isnan(fwd_end))
    print_check("forward_return: NaN when t+k runs past the calendar", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_spearman(passed, failed):
    x = np.arange(10.0)
    ok = np.isclose(diag._spearman(x, x), 1.0)
    print_check("_spearman: identical monotonic arrays give rho=1.0", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = np.isclose(diag._spearman(x, x[::-1]), -1.0)
    print_check("_spearman: reversed array gives rho=-1.0", ok)
    passed, failed = passed + ok, failed + (not ok)

    ok = np.isnan(diag._spearman(np.zeros(5), x[:5]))
    print_check("_spearman: constant input returns NaN, not a crash", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_ranking_quality_perfect(passed, failed):
    panel = _ranked_panel(T=6)
    dates = panel.dates.values.astype("datetime64[D]")
    tickers = np.array(["cash"] + list(panel.asset_index.tickers))

    # weight ranking mirrors the true growth-rate ranking exactly -> perfect Spearman.
    w_non_cash = np.arange(1, N_ASSETS + 1, dtype=float)
    w_non_cash /= w_non_cash.sum()
    row = np.concatenate([[0.0], w_non_cash])
    weights = np.tile(row, (6, 1))

    out = diag.ranking_quality(weights, dates, tickers, panel, k_list=(1,))
    ok = np.isclose(out[1]["mean_spearman"], 1.0)
    print_check("ranking_quality: weight ranking == truth ranking gives mean_spearman=1.0", ok,
                f"got {out[1]['mean_spearman']}")
    passed, failed = passed + ok, failed + (not ok)

    ok = np.isclose(out[1]["mean_top10_hit_rate"], 1.0)
    print_check("ranking_quality: perfect ranking puts every top-decile winner in the top-10", ok,
                f"got {out[1]['mean_top10_hit_rate']}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_ranking_quality_inverted(passed, failed):
    panel = _ranked_panel(T=6)
    dates = panel.dates.values.astype("datetime64[D]")
    tickers = np.array(["cash"] + list(panel.asset_index.tickers))

    # weight ranking is the EXACT REVERSE of the true growth-rate ranking.
    w_non_cash = np.arange(N_ASSETS, 0, -1, dtype=float)
    w_non_cash /= w_non_cash.sum()
    row = np.concatenate([[0.0], w_non_cash])
    weights = np.tile(row, (6, 1))

    out = diag.ranking_quality(weights, dates, tickers, panel, k_list=(1,))
    ok = np.isclose(out[1]["mean_spearman"], -1.0)
    print_check("ranking_quality: inverted weight ranking gives mean_spearman=-1.0", ok,
                f"got {out[1]['mean_spearman']}")
    passed, failed = passed + ok, failed + (not ok)

    ok = np.isclose(out[1]["mean_top10_hit_rate"], 0.0)
    print_check("ranking_quality: inverted ranking excludes every top-decile winner from the top-10", ok,
                f"got {out[1]['mean_top10_hit_rate']}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_ranking_quality_ticker_mismatch(passed, failed):
    panel = _ranked_panel(T=6)
    dates = panel.dates.values.astype("datetime64[D]")
    bad_tickers = np.array(["cash"] + [f"WRONG{i}" for i in range(N_ASSETS)])
    weights = np.tile(np.full(N_ASSETS + 1, 1.0 / (N_ASSETS + 1)), (6, 1))

    try:
        diag.ranking_quality(weights, dates, bad_tickers, panel, k_list=(1,))
        ok = False
    except AssertionError:
        ok = True
    print_check("ranking_quality: mismatched tickers vs panel asset index raises AssertionError", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_cross_seed_consistency(passed, failed):
    dates = pd.bdate_range("2021-01-01", periods=8).values.astype("datetime64[D]")
    rng = np.random.default_rng(0)
    base = rng.dirichlet(np.ones(6), size=8)

    identical_runs = [{"weights": base.copy(), "dates": dates.copy()},
                       {"weights": base.copy(), "dates": dates.copy()}]
    out = diag.cross_seed_consistency(identical_runs)
    ok = (np.isclose(out["mean_pairwise_cosine"], 1.0) and np.isclose(out["mean_pairwise_top10_jaccard"], 1.0)
          and np.isclose(out["mean_pairwise_meanweight_corr"], 1.0))
    print_check("cross_seed_consistency: identical runs give cosine=jaccard=corr=1.0", ok, str(out))
    passed, failed = passed + ok, failed + (not ok)

    # A run whose weights are a fixed permutation of another's non-cash columns is neither
    # a plain rescale nor identical -- cosine/corr should measurably drop below 1.
    perm = np.roll(np.arange(1, 6), 1)
    shuffled = base.copy()
    shuffled[:, 1:] = base[:, perm]
    different_runs = [{"weights": base.copy(), "dates": dates.copy()},
                       {"weights": shuffled, "dates": dates.copy()}]
    out2 = diag.cross_seed_consistency(different_runs)
    ok = out2["mean_pairwise_cosine"] < 0.999
    print_check("cross_seed_consistency: a column-permuted run scores measurably below 1.0", ok, str(out2))
    passed, failed = passed + ok, failed + (not ok)

    ok = out2["n_common_days"] == 8
    print_check("cross_seed_consistency: full date overlap detected", ok)
    passed, failed = passed + ok, failed + (not ok)

    try:
        diag.cross_seed_consistency([identical_runs[0]])
        ok = False
    except ValueError:
        ok = True
    print_check("cross_seed_consistency: fewer than 2 runs raises ValueError", ok)
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def main():
    print_header("test_diagnostics")
    passed = failed = 0

    passed, failed = test_forward_return(passed, failed)
    passed, failed = test_spearman(passed, failed)
    passed, failed = test_ranking_quality_perfect(passed, failed)
    passed, failed = test_ranking_quality_inverted(passed, failed)
    passed, failed = test_ranking_quality_ticker_mismatch(passed, failed)
    passed, failed = test_cross_seed_consistency(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
