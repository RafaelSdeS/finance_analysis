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


N_PHANTOM = 5


def _ranked_panel_with_phantoms(T=10):
    """Same 15 holdable tickers as _ranked_panel (rate = i*0.001, i=1..15,
    all always active/valid), PLUS 5 "phantom" global tickers (T16-T20)
    that are never in any day's slot_gidx (never holdable -- like a
    delisted/pre-IPO/out-of-membership name) and always carry weight 0.
    The phantoms grow far faster than any holdable ticker (rate 0.5-0.9)
    specifically so a Spearman/top-decile computation that (bug) includes
    the full global space would have its ranking dominated by names the
    agent could never have picked -- the exact dilution in Finding 1.1."""
    n_holdable = N_ASSETS
    n_total = N_ASSETS + N_PHANTOM
    tickers = tuple(f"T{i:02d}" for i in range(1, n_total + 1))
    asset_index = GlobalAssetIndex(tickers=tickers, ticker_to_gidx={t: i + 1 for i, t in enumerate(tickers)})
    holdable_rates = np.arange(1, n_holdable + 1) * 0.001
    phantom_rates = 0.5 + np.arange(N_PHANTOM) * 0.1  # 0.5, 0.6, ..., 0.9 -- far above any holdable rate
    rates = np.concatenate([holdable_rates, phantom_rates])
    t_arr = np.arange(T)[:, None]
    close = np.column_stack([np.ones(T), 100.0 * (1.0 + rates[None, :]) ** t_arr])
    dates = pd.bdate_range("2021-01-01", periods=T)
    return PricePanel(
        asset_index=asset_index, dates=dates, close=close, high=close.copy(), low=close.copy(),
        cdi_factor=np.full(T, 1.0001),
        slot_gidx=np.tile(np.arange(1, n_holdable + 1), (T, 1)),  # phantoms never appear here
        valid=np.ones((T, n_holdable), dtype=bool),
        window=2, start_idx=0, end_idx=T - 1,
    )


def _noisy_ranked_panel(T=250, seed=0, noise_sigma=0.01):
    """Like _ranked_panel, but each ticker's daily log-return is its fixed
    skill rate (i*0.001) plus i.i.d. daily noise. _ranked_panel's realized
    ranking is bit-identical every single day (fixed rates -> the same
    relative order forever), which makes spearman_permutation_null's
    day-shuffle null degenerate (every permutation reproduces the same
    observed value). Real forward returns vary day to day; this fixture
    gives the null something to actually shuffle."""
    rng = np.random.default_rng(seed)
    tickers = tuple(f"T{i:02d}" for i in range(1, N_ASSETS + 1))
    asset_index = GlobalAssetIndex(tickers=tickers, ticker_to_gidx={t: i + 1 for i, t in enumerate(tickers)})
    mu = np.arange(1, N_ASSETS + 1) * 0.001
    log_rets = mu[None, :] + rng.normal(0, noise_sigma, size=(T, N_ASSETS))
    close_assets = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    close = np.column_stack([np.ones(T), close_assets])
    dates = pd.bdate_range("2021-01-01", periods=T)
    return PricePanel(
        asset_index=asset_index, dates=dates, close=close, high=close.copy(), low=close.copy(),
        cdi_factor=np.full(T, 1.0001),
        slot_gidx=np.tile(np.arange(1, N_ASSETS + 1), (T, 1)),
        valid=np.ones((T, N_ASSETS), dtype=bool),
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


def test_rankdata_tie_averaging(passed, failed):
    # a = [5, 1, 1, 1, 9]: sorted is [1,1,1,5,9] -> the three tied 1's occupy
    # 0-indexed rank positions 0,1,2, so tie-averaged rank = (0+1+2)/3 = 1.0
    # each; 5 -> rank 3; 9 -> rank 4. Plain argsort would instead hand out
    # three DISTINCT ranks {0,1,2} to the tied block (arbitrary, order-
    # dependent) -- this is the exact bug Finding 1.2 flagged.
    a = np.array([5.0, 1.0, 1.0, 1.0, 9.0])
    expected = np.array([3.0, 1.0, 1.0, 1.0, 4.0])
    got = diag._rankdata(a)
    ok = np.allclose(got, expected)
    print_check("_rankdata: tied values get the AVERAGE rank of their block, not distinct ranks", ok,
                f"got {got}, expected {expected}")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_ranking_quality_active_only_scope(passed, failed):
    # Same perfect-ranking-among-holdables setup as test_ranking_quality_perfect,
    # but the panel also carries 5 phantom (never-holdable) tickers that grow
    # far faster than any holdable name. A metric that (bug) scores over the
    # full global space would see its top-decile cut and Spearman dominated by
    # the unreachable phantoms; the fixed, active-only metric must be blind to
    # them and reproduce the exact perfect-ranking result.
    panel = _ranked_panel_with_phantoms(T=6)
    dates = panel.dates.values.astype("datetime64[D]")
    tickers = np.array(["cash"] + list(panel.asset_index.tickers))

    w_holdable = np.arange(1, N_ASSETS + 1, dtype=float)
    w_holdable /= w_holdable.sum()
    row = np.concatenate([[0.0], w_holdable, np.zeros(N_PHANTOM)])  # phantoms always weight 0
    weights = np.tile(row, (6, 1))

    out = diag.ranking_quality(weights, dates, tickers, panel, k_list=(1,))
    ok = np.isclose(out[1]["mean_spearman"], 1.0)
    print_check("ranking_quality: active-only scope is immune to faster-growing phantom tickers (Spearman)", ok,
                f"got {out[1]['mean_spearman']}")
    passed, failed = passed + ok, failed + (not ok)

    ok = np.isclose(out[1]["mean_top10_hit_rate"], 1.0)
    print_check("ranking_quality: active-only scope is immune to phantom tickers (top-10 hit rate)", ok,
                f"got {out[1]['mean_top10_hit_rate']} (would be diluted toward 0 under the old, undiluted bug)")
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_selection_alpha(passed, failed):
    panel = _ranked_panel(T=20)
    dates = panel.dates.values.astype("datetime64[D]")
    tickers = np.array(["cash"] + list(panel.asset_index.tickers))

    # Skilled: always holds the single fastest-growing ticker (T15).
    skilled_row = np.zeros(N_ASSETS + 1)
    skilled_row[N_ASSETS] = 1.0  # last non-cash column = T15, the fastest grower
    skilled_weights = np.tile(skilled_row, (20, 1))
    out_skilled = diag.selection_alpha(skilled_weights, dates, tickers, panel, k_list=(1,), top_k_list=(1,), n_perm=500, seed=0)
    stats = out_skilled[1]["top1"]
    ok = stats["mean_selection_alpha"] > 0 and stats["p_value"] < 0.05
    print_check("selection_alpha: always holding the true best asset gives positive alpha, low p-value", ok, str(stats))
    passed, failed = passed + ok, failed + (not ok)

    # Unskilled: always holds the single WORST-growing ticker (T01).
    unskilled_row = np.zeros(N_ASSETS + 1)
    unskilled_row[1] = 1.0  # first non-cash column = T01, the slowest grower
    unskilled_weights = np.tile(unskilled_row, (20, 1))
    out_unskilled = diag.selection_alpha(unskilled_weights, dates, tickers, panel, k_list=(1,), top_k_list=(1,), n_perm=500, seed=0)
    stats_u = out_unskilled[1]["top1"]
    ok = stats_u["mean_selection_alpha"] < 0 and stats_u["p_value"] > 0.5
    print_check("selection_alpha: always holding the true worst asset gives negative alpha, high p-value", ok, str(stats_u))
    passed, failed = passed + ok, failed + (not ok)
    return passed, failed


def test_spearman_permutation_null(passed, failed):
    # Noisy panel: day-to-day realized ranking varies (unlike _ranked_panel's
    # bit-identical-every-day ranking, which degenerates the day-shuffle null --
    # every permutation would reproduce the same observed value).
    #
    # Weights must ALSO vary day to day here, for the same reason: a fixed
    # (day-invariant) weight vector makes every row of the day-by-day
    # correlation matrix identical, so ANY permutation of day-pairing sums the
    # exact same multiset of terms -- the "null" degenerates to the observed
    # value plus floating-point noise, which is meaningless. Real trained
    # policies emit a different weight vector every day; these fixtures must
    # too for the day-shuffle null to test anything real.
    T = 250
    panel = _noisy_ranked_panel(T=T, seed=1, noise_sigma=0.01)
    dates = panel.dates.values.astype("datetime64[D]")
    tickers = np.array(["cash"] + list(panel.asset_index.tickers))
    rng = np.random.default_rng(7)

    # Skilled: weight on day t tracks THAT DAY's actual forward return ranking
    # (a noisy oracle) -- genuine day-specific predictive skill, the thing a
    # real trained policy would need to show. A merely STATIC factor tilt
    # (same base ranking reused every day, only lightly jittered) is a
    # degenerate case for this null: since the tilt alone already correlates
    # with essentially every day's average return, shuffling which day it's
    # paired with barely changes anything -- the permutation null specifically
    # tests for day-SPECIFIC information, and a real adaptive policy's weight
    # vector genuinely differs day to day for exactly this reason.
    skilled_weights = np.zeros((T, N_ASSETS + 1))
    for t in range(T - 1):
        fwd = diag.forward_return(panel, t, 1)[1:]  # non-cash forward returns, day t
        score = fwd + rng.normal(0, 0.02, size=N_ASSETS)  # noisy oracle on day t's own outcome
        rank_proxy = np.argsort(np.argsort(score)) + 1.0
        skilled_weights[t, 1:] = rank_proxy / rank_proxy.sum()
    skilled_weights[T - 1, 1:] = 1.0 / N_ASSETS  # last day has no forward return; arbitrary
    out_skilled = diag.spearman_permutation_null(skilled_weights, dates, tickers, panel, k_list=(1,), n_perm=500, seed=0)
    ok = (out_skilled[1]["observed_mean_spearman"] > out_skilled[1]["null_975pct"]
          and out_skilled[1]["p_value"] < 0.05)
    print_check("spearman_permutation_null: weights matching the true (noisy) skill ranking clear the null", ok,
                str(out_skilled[1]))
    passed, failed = passed + ok, failed + (not ok)

    # Unskilled: a fresh random weight vector every day, uncorrelated with mu.
    w_random_daily = rng.dirichlet(np.ones(N_ASSETS), size=T)
    random_weights = np.column_stack([np.zeros(T), w_random_daily])
    out_random = diag.spearman_permutation_null(random_weights, dates, tickers, panel, k_list=(1,), n_perm=500, seed=0)
    ok = out_random[1]["p_value"] > 0.1
    print_check("spearman_permutation_null: a skill-uncorrelated weighting shows no evidence of positive skill", ok,
                str(out_random[1]))
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
    passed, failed = test_rankdata_tie_averaging(passed, failed)
    passed, failed = test_ranking_quality_active_only_scope(passed, failed)
    passed, failed = test_selection_alpha(passed, failed)
    passed, failed = test_spearman_permutation_null(passed, failed)
    passed, failed = test_cross_seed_consistency(passed, failed)

    print_section_end(passed, failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
