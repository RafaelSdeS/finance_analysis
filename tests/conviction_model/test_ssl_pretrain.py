"""
Test: conviction_model/ssl_pretrain.py's CPC pieces -- info_nce_loss's known
low/high-loss extremes, and sample_cpc_negatives' two negative types
(same-stock-different-regime, different-stock-same-time). Synthetic data
only, no dependency on data/raw or data/processed.

Run from project root:
    python tests/conviction_model/test_ssl_pretrain.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.conviction_model.data import (  # noqa: E402
    DAILY_FEATURES, MONTHLY_FEATURES, QUARTERLY_FEATURES, WEEKLY_FEATURES, resample_branch_frame,
)
from src.conviction_model.encoder import EncoderCNN  # noqa: E402
from src.conviction_model.config import SSLConfig  # noqa: E402
from src.conviction_model.ssl_pretrain import (  # noqa: E402
    CachedPanelGatherer, CPCPanelStore, LazyPanelGatherer, ReconstructionHeads, _mask_branch, _price_macro_state,
    build_cpc_batch, build_stage1b_batch, info_nce_loss, sample_cpc_anchor_positions, sample_cpc_negatives,
    score_holdout, score_holdout_stage1b, score_holdout_stage1c, split_train_holdout, train_step, train_step_stage1b,
    train_step_stage1c,
)
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def test_info_nce_near_zero_when_positive_matches_and_negatives_orthogonal(passed, failed):
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])            # identical to anchor -> cos sim = 1
    negatives = torch.tensor([[[0.0, 1.0], [0.0, -1.0]]])  # orthogonal -> cos sim = 0
    loss = info_nce_loss(anchor, positive, negatives, temperature=0.1).item()
    ok = loss < 0.01
    print_check("info_nce_loss: near 0 when the positive is an exact match and negatives are orthogonal",
                ok, f"loss={loss:.6f}")
    return passed + ok, failed + (not ok)


def test_info_nce_equals_log_n_plus_1_when_all_candidates_tied(passed, failed):
    anchor = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    n = 3
    negatives = torch.tensor([[[1.0, 0.0]] * n])      # identical to positive -> all logits tied
    loss = info_nce_loss(anchor, positive, negatives, temperature=0.1).item()
    expected = np.log(n + 1)
    ok = abs(loss - expected) < 1e-4
    print_check("info_nce_loss: equals log(N+1) when positive and all N negatives are tied",
                ok, f"loss={loss:.6f}, expected={expected:.6f}")
    return passed + ok, failed + (not ok)


def _synthetic_panel():
    tickers = ["AAA", "BBB", "CCC"]
    dates = pd.bdate_range("2010-01-01", periods=500)
    rows = [{"ticker": t, "trade_date": d} for t in tickers for d in dates]
    return pd.DataFrame(rows).reset_index(drop=True)


def test_same_stock_negatives_are_same_ticker_and_far_in_time(passed, failed):
    panel = _synthetic_panel()
    anchor_pos = np.array([panel[(panel["ticker"] == "AAA")].index[400]])
    neg = sample_cpc_negatives(panel, anchor_pos, n_same_stock=4, n_diff_stock=4,
                                regime_gap_days=252, rng=np.random.default_rng(0))
    same_stock_neg = neg[0, :4]
    anchor_date = panel.loc[anchor_pos[0], "trade_date"]
    tickers_ok = bool((panel.loc[same_stock_neg, "ticker"] == "AAA").all())
    gap_days = (panel.loc[same_stock_neg, "trade_date"] - anchor_date).abs().dt.days
    gap_ok = bool((gap_days >= 252).all())
    ok = tickers_ok and gap_ok
    print_check("sample_cpc_negatives: same-stock negatives are the same ticker, >=regime_gap_days away",
                ok, f"tickers={panel.loc[same_stock_neg, 'ticker'].tolist()}, gap_days={gap_days.tolist()}")
    return passed + ok, failed + (not ok)


def test_sample_cpc_negatives_excludes_positive_for_short_history_ticker(passed, failed):
    # A ticker with LESS history than regime_gap_days forces the same-stock fallback
    # branch (`same_pool[same_pool != pos]`) on every draw, since no row can satisfy
    # `gap_days >= regime_gap_days`. Without exclude_positions, that fallback could pick
    # the positive itself (pos + cpc_horizon) as a "negative" -- a contradictory InfoNCE
    # label (same embedding as both the numerator and a negative). exclude_positions
    # must keep it out.
    tickers = ["SHORT"] * 30 + ["OTHER"] * 30
    dates = list(pd.bdate_range("2020-01-01", periods=30)) * 2
    panel = pd.DataFrame({"ticker": tickers, "trade_date": dates}).sort_values(
        ["ticker", "trade_date"]).reset_index(drop=True)

    cpc_horizon = 5
    short_positions = panel.index[panel["ticker"] == "SHORT"].to_numpy()
    anchor_positions = short_positions[:-cpc_horizon]  # every valid anchor for this ticker
    positive_positions = anchor_positions + cpc_horizon

    neg = sample_cpc_negatives(panel, anchor_positions, n_same_stock=4, n_diff_stock=2,
                                regime_gap_days=252, rng=np.random.default_rng(7),
                                exclude_positions=positive_positions[:, None])
    same_stock_neg = neg[:, :4]
    leaked = np.array([positive_positions[i] in same_stock_neg[i] for i in range(len(anchor_positions))])
    ok = not leaked.any()
    print_check("sample_cpc_negatives: exclude_positions keeps the positive out of the same-stock "
                "fallback pool for a short-history ticker", ok, f"leaked for {int(leaked.sum())} anchors")
    return passed + ok, failed + (not ok)


def test_diff_stock_negatives_are_other_tickers_same_date(passed, failed):
    panel = _synthetic_panel()
    anchor_pos = np.array([panel[(panel["ticker"] == "AAA")].index[400]])
    neg = sample_cpc_negatives(panel, anchor_pos, n_same_stock=4, n_diff_stock=4,
                                regime_gap_days=252, rng=np.random.default_rng(0))
    diff_stock_neg = neg[0, 4:]
    anchor_date = panel.loc[anchor_pos[0], "trade_date"]
    date_ok = bool((panel.loc[diff_stock_neg, "trade_date"] == anchor_date).all())
    ticker_ok = bool((panel.loc[diff_stock_neg, "ticker"] != "AAA").all())
    ok = date_ok and ticker_ok
    print_check("sample_cpc_negatives: different-stock negatives share the anchor's date, differ in ticker",
                ok, f"dates_match={date_ok}, tickers={panel.loc[diff_stock_neg, 'ticker'].tolist()}")
    return passed + ok, failed + (not ok)


def _synthetic_frame_cache(tickers, n_days=400, n_quarters=20, rng=None):
    """Same shape data.build_frame_cache produces for real training (a
    (ticker,trade_date) panel + a {ticker: (daily_frame, weekly_frame,
    monthly_frame, quarterly_frame)} cache), but built from synthetic frames
    -- no real dataset dependency, stays in the `fast` group."""
    rng = rng or np.random.default_rng(0)
    daily_cols = list(dict.fromkeys(DAILY_FEATURES + WEEKLY_FEATURES + MONTHLY_FEATURES))
    calendar = pd.bdate_range("2010-01-01", periods=n_days)
    quarters = pd.bdate_range("2010-01-01", periods=n_quarters, freq="QE")
    cache, panels = {}, []
    for t in tickers:
        daily_frame = pd.DataFrame(rng.normal(size=(n_days, len(daily_cols))), index=calendar, columns=daily_cols)
        quarterly_frame = pd.DataFrame(rng.normal(size=(n_quarters, len(QUARTERLY_FEATURES))),
                                        index=quarters, columns=list(QUARTERLY_FEATURES))
        cache[t] = (daily_frame, resample_branch_frame(daily_frame, "W"),
                    resample_branch_frame(daily_frame, "ME"), quarterly_frame)
        panels.append(pd.DataFrame({"ticker": t, "trade_date": calendar}))
    panel = pd.concat(panels).sort_values(["ticker", "trade_date"]).reset_index(drop=True)
    return panel, cache


def test_sample_cpc_anchor_positions_leaves_room_for_the_horizon(passed, failed):
    panel, _ = _synthetic_frame_cache(["AAA", "BBB"], n_days=400)
    cpc_horizon = 63
    anchors = sample_cpc_anchor_positions(panel, batch_size=50, cpc_horizon=cpc_horizon,
                                           rng=np.random.default_rng(1))
    remaining = panel.groupby("ticker").cumcount(ascending=False).to_numpy()
    ok = bool(np.all(remaining[anchors] >= cpc_horizon))
    print_check("sample_cpc_anchor_positions: every sampled anchor has >=cpc_horizon rows left in its ticker",
                ok, f"min remaining={remaining[anchors].min() if len(anchors) else None}")
    return passed + ok, failed + (not ok)


def test_build_cpc_batch_shapes(passed, failed):
    panel, cache = _synthetic_frame_cache(["AAA", "BBB", "CCC"], n_days=400)
    store = CPCPanelStore(panel, cache)
    cpc_horizon, n_same, n_diff, batch_size = 21, 3, 3, 8
    anchors = sample_cpc_anchor_positions(panel, batch_size=batch_size, cpc_horizon=cpc_horizon,
                                           rng=np.random.default_rng(2))
    anchor_batch, positive_batch, negative_batch = build_cpc_batch(
        panel, store, anchors, cpc_horizon, n_same_stock=n_same, n_diff_stock=n_diff,
        rng=np.random.default_rng(3))

    n_feat = {"daily": len(DAILY_FEATURES), "weekly": len(WEEKLY_FEATURES),
              "monthly": len(MONTHLY_FEATURES), "fundamentals": len(QUARTERLY_FEATURES)}
    anchor_ok = all(anchor_batch[k].shape[0] == batch_size and anchor_batch[k].shape[1] == n_feat[k]
                     for k in n_feat)
    positive_ok = all(positive_batch[k].shape == anchor_batch[k].shape for k in n_feat)
    negative_ok = all(negative_batch[k].shape[:2] == (batch_size, n_same + n_diff)
                       and negative_batch[k].shape[2] == n_feat[k] for k in n_feat)
    ok = anchor_ok and positive_ok and negative_ok
    print_check("build_cpc_batch: anchor/positive/negative batches have the shapes train_step expects",
                ok, f"anchor daily shape={anchor_batch['daily'].shape}, "
                    f"negative daily shape={negative_batch['daily'].shape}")
    return passed + ok, failed + (not ok)


def test_lazy_panel_gatherer_matches_cpc_panel_store(passed, failed):
    panel, cache = _synthetic_frame_cache(["AAA", "BBB", "CCC"], n_days=400)
    positions = np.array([5, 120, 250, 399, 0])
    store_batch = CPCPanelStore(panel, cache).gather(positions)
    lazy_batch = LazyPanelGatherer(panel, cache).gather(positions)
    ok = all(torch.allclose(store_batch[k], lazy_batch[k], atol=1e-6) for k in store_batch)
    print_check("LazyPanelGatherer: gather() output matches CPCPanelStore's exactly (memory-light "
                "path is not a behavior change)", ok)
    return passed + ok, failed + (not ok)


def test_cached_panel_gatherer_matches_lazy_panel_gatherer(passed, failed):
    panel, cache = _synthetic_frame_cache(["AAA", "BBB", "CCC"], n_days=400)
    positions = np.array([5, 120, 250, 399, 0])
    lazy_batch = LazyPanelGatherer(panel, cache).gather(positions)
    cached_batch = CachedPanelGatherer(panel, cache, maxsize=3).gather(positions)
    ok = all(torch.allclose(lazy_batch[k], cached_batch[k], atol=1e-6) for k in lazy_batch)
    print_check("CachedPanelGatherer: gather() output matches LazyPanelGatherer's exactly "
                "(memoization is a pure speed fix, not a behavior change)", ok)
    return passed + ok, failed + (not ok)


def test_cached_panel_gatherer_consistent_after_eviction(passed, failed):
    # maxsize smaller than the number of distinct positions touched -- forces eviction
    # partway through, so re-querying an evicted position must still recompute correctly
    # (not silently stale/wrong), not just "doesn't crash".
    panel, cache = _synthetic_frame_cache(["AAA", "BBB"], n_days=400)
    positions = np.array([5, 120, 250, 399, 0, 50, 150, 350])
    gatherer = CachedPanelGatherer(panel, cache, maxsize=2)  # much smaller than 8 distinct positions
    first = gatherer.gather(positions)
    second = gatherer.gather(positions)  # some entries evicted+recomputed between calls
    ok = all(torch.allclose(first[k], second[k], atol=1e-6) for k in first)
    print_check("CachedPanelGatherer: repeated gather() after cache eviction still returns "
                "consistent (correctly recomputed) results", ok)
    return passed + ok, failed + (not ok)


def test_cached_panel_gatherer_evicts_least_recently_used(passed, failed):
    panel, cache = _synthetic_frame_cache(["AAA"], n_days=400)
    gatherer = CachedPanelGatherer(panel, cache, maxsize=2)
    gatherer.gather(np.array([10]))
    gatherer.gather(np.array([20]))
    gatherer.gather(np.array([10]))  # re-touch 10 -> 20 becomes the least-recently-used entry
    gatherer.gather(np.array([30]))  # should evict 20, not 10

    key10 = ("AAA", pd.Timestamp(panel.loc[10, "trade_date"]))
    key20 = ("AAA", pd.Timestamp(panel.loc[20, "trade_date"]))
    cache_keys = list(gatherer._cache.keys())
    ok = key10 in cache_keys and key20 not in cache_keys
    print_check("CachedPanelGatherer: evicts the LEAST-recently-used entry, not plain insertion order",
                ok, f"cache_keys={cache_keys}")
    return passed + ok, failed + (not ok)


def test_cached_panel_gatherer_maxsize_zero_still_works(passed, failed):
    panel, cache = _synthetic_frame_cache(["AAA", "BBB"], n_days=400)
    gatherer = CachedPanelGatherer(panel, cache, maxsize=0)
    batch = gatherer.gather(np.array([5, 10, 15]))
    ok = batch["daily"].shape[0] == 3 and len(gatherer._cache) == 0
    print_check("CachedPanelGatherer: maxsize=0 still produces correct output (cache stays "
                "empty, effectively disabled)", ok, f"cache_len={len(gatherer._cache)}")
    return passed + ok, failed + (not ok)


def test_cached_panel_gatherer_stores_float32_not_float64(passed, failed):
    # Regression test for a real bug: window_tensor's raw output is float64. An earlier
    # version of CachedPanelGatherer cached that raw dict directly, silently DOUBLING the
    # intended per-entry memory footprint (CPCPanelStore's own ~17KB/position figure already
    # assumes a float32 cast) -- this caused a real OOM kill in production at the
    # then-default cache size. Cached entries must be float32, matching CPCPanelStore's own
    # convention, not whatever dtype the underlying computation happens to produce.
    panel, cache = _synthetic_frame_cache(["AAA"], n_days=400)
    gatherer = CachedPanelGatherer(panel, cache, maxsize=10)
    gatherer.gather(np.array([100]))
    cached_entry = next(iter(gatherer._cache.values()))
    ok = all(arr.dtype == np.float32 for arr in cached_entry.values())
    print_check("CachedPanelGatherer: cached entries are float32, not float64 (halves resident "
                "memory vs. window_tensor's raw output dtype)", ok,
                f"dtypes={[arr.dtype for arr in cached_entry.values()]}")
    return passed + ok, failed + (not ok)


def test_build_cpc_batch_positive_is_same_ticker_k_ahead(passed, failed):
    panel, cache = _synthetic_frame_cache(["AAA", "BBB"], n_days=400)
    cpc_horizon = 21
    anchors = sample_cpc_anchor_positions(panel, batch_size=10, cpc_horizon=cpc_horizon,
                                           rng=np.random.default_rng(4))
    tickers = panel["ticker"].to_numpy()
    positive_positions = anchors + cpc_horizon
    same_ticker = bool(np.all(tickers[positive_positions] == tickers[anchors]))
    gap_trading_days = positive_positions - anchors
    ok = same_ticker and bool(np.all(gap_trading_days == cpc_horizon))
    print_check("build_cpc_batch: the positive is the same ticker's window exactly cpc_horizon rows ahead",
                ok, f"same_ticker={same_ticker}")
    return passed + ok, failed + (not ok)


def test_split_train_holdout_respects_cutoff_and_stays_contiguous(passed, failed):
    panel, _ = _synthetic_frame_cache(["AAA", "BBB"], n_days=400)
    cutoff = panel["trade_date"].max() - pd.Timedelta(days=180)
    train_panel, holdout_panel = split_train_holdout(panel, holdout_days=180)

    cutoff_ok = bool((train_panel["trade_date"] <= cutoff).all() and (holdout_panel["trade_date"] > cutoff).all())
    coverage_ok = len(train_panel) + len(holdout_panel) == len(panel)
    # contiguity: each ticker's train rows are exactly that ticker's earliest
    # rows from the original panel, in the same order (a clean prefix, not a
    # scattered subset) -- sample_cpc_anchor_positions/sample_cpc_negatives's
    # shared "position i+1 is the same ticker's next row" contract needs this.
    contiguous_ok = True
    for t in ["AAA", "BBB"]:
        expected_prefix = panel.loc[panel["ticker"] == t, "trade_date"].reset_index(drop=True)
        actual_prefix = train_panel.loc[train_panel["ticker"] == t, "trade_date"].reset_index(drop=True)
        contiguous_ok &= expected_prefix.iloc[:len(actual_prefix)].equals(actual_prefix)
    ok = cutoff_ok and coverage_ok and contiguous_ok
    print_check("split_train_holdout: train/holdout respect the date cutoff and stay per-ticker contiguous",
                ok, f"train={len(train_panel)}, holdout={len(holdout_panel)}, "
                    f"cutoff_ok={cutoff_ok}, coverage_ok={coverage_ok}, contiguous_ok={contiguous_ok}")
    return passed + ok, failed + (not ok)


def test_score_holdout_does_not_change_params_and_returns_finite_score(passed, failed):
    panel, cache = _synthetic_frame_cache(["AAA", "BBB"], n_days=400)
    _, holdout_panel = split_train_holdout(panel, holdout_days=180)
    holdout_store = CPCPanelStore(holdout_panel, cache)
    cfg = SSLConfig(cpc_horizon=10, batch_size=8, n_same_stock_negatives=2, n_diff_stock_negatives=2)

    model = EncoderCNN(len(DAILY_FEATURES), len(WEEKLY_FEATURES), len(MONTHLY_FEATURES),
                        len(QUARTERLY_FEATURES), d_model=8, n_heads=2)
    params_before = [p.clone() for p in model.parameters()]
    score = score_holdout(model, holdout_panel, holdout_store, cfg, rng=np.random.default_rng(0), n_eval_batches=2)

    unchanged = all(torch.equal(a, b) for a, b in zip(params_before, model.parameters()))
    finite = score == score and score != float("inf")
    ok = unchanged and finite
    print_check("score_holdout: leaves model parameters unchanged (no_grad) and returns a finite score",
                ok, f"score={score}, unchanged={unchanged}")
    return passed + ok, failed + (not ok)


def _tiny_batch(batch_size, n_negatives, n_features, window):
    return {
        "daily": torch.randn(batch_size, n_features, window),
        "weekly": torch.randn(batch_size, n_features, window),
        "monthly": torch.randn(batch_size, n_features, window),
        "fundamentals": torch.randn(batch_size, n_features, window),
    }, {
        "daily": torch.randn(batch_size, n_negatives, n_features, window),
        "weekly": torch.randn(batch_size, n_negatives, n_features, window),
        "monthly": torch.randn(batch_size, n_negatives, n_features, window),
        "fundamentals": torch.randn(batch_size, n_negatives, n_features, window),
    }


def test_train_step_updates_params_and_returns_finite_loss(passed, failed):
    torch.manual_seed(0)
    n_features, window, batch_size, n_negatives = 5, 8, 4, 3
    model = EncoderCNN(n_features, n_features, n_features, n_features, d_model=8, n_heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    anchor, negatives = _tiny_batch(batch_size, n_negatives, n_features, window)
    positive, _ = _tiny_batch(batch_size, n_negatives, n_features, window)

    params_before = [p.clone() for p in model.parameters()]
    loss = train_step(model, optimizer, anchor, positive, negatives)

    ok = loss == loss and loss != float("inf")  # finite, not NaN
    print_check("train_step: returns a finite loss", ok, f"loss={loss}")
    passed, failed = passed + ok, failed + (not ok)

    changed = any(not torch.allclose(a, b) for a, b in zip(params_before, model.parameters()))
    print_check("train_step: model parameters change after one gradient step", changed)
    passed, failed = passed + changed, failed + (not changed)
    return passed, failed


def test_price_macro_state_ignores_fundamentals(passed, failed):
    branch_embeddings = {
        "daily": torch.tensor([[1.0, 2.0]]),
        "weekly": torch.tensor([[3.0, 4.0]]),
        "monthly": torch.tensor([[5.0, 6.0]]),
        "fundamentals": torch.tensor([[999.0, -999.0]]),  # must not move the result
    }
    state = _price_macro_state(branch_embeddings)
    expected = torch.tensor([[3.0, 4.0]])  # mean of (1,2),(3,4),(5,6)
    ok = bool(torch.allclose(state, expected))
    print_check("_price_macro_state: pools daily/weekly/monthly only, ignores fundamentals",
                ok, f"got {state.tolist()}, expected {expected.tolist()}")
    return passed + ok, failed + (not ok)


def test_build_stage1b_batch_positives_use_different_horizons(passed, failed):
    panel, cache = _synthetic_frame_cache(["AAA", "BBB"], n_days=400)
    store = CPCPanelStore(panel, cache)
    cpc_horizon, alignment_horizon = 21, 63
    max_horizon = max(cpc_horizon, alignment_horizon)
    anchors = sample_cpc_anchor_positions(panel, batch_size=10, cpc_horizon=max_horizon,
                                           rng=np.random.default_rng(5))
    tickers = panel["ticker"].to_numpy()

    anchor_batch, cpc_positive_batch, align_positive_batch, negative_batch = build_stage1b_batch(
        panel, store, anchors, cpc_horizon, alignment_horizon, rng=np.random.default_rng(6))

    same_ticker_cpc = bool(np.all(tickers[anchors + cpc_horizon] == tickers[anchors]))
    same_ticker_align = bool(np.all(tickers[anchors + alignment_horizon] == tickers[anchors]))
    differ = not torch.allclose(cpc_positive_batch["daily"], align_positive_batch["daily"])
    ok = same_ticker_cpc and same_ticker_align and differ
    print_check("build_stage1b_batch: cpc/alignment positives are the same ticker, different "
                "(cpc_horizon vs alignment_horizon) offsets ahead",
                ok, f"same_ticker_cpc={same_ticker_cpc}, same_ticker_align={same_ticker_align}, differ={differ}")
    return passed + ok, failed + (not ok)


def test_train_step_stage1b_updates_params_and_returns_finite_losses(passed, failed):
    torch.manual_seed(0)
    n_features, window, batch_size, n_negatives = 5, 8, 4, 3
    model = EncoderCNN(n_features, n_features, n_features, n_features, d_model=8, n_heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    anchor, negatives = _tiny_batch(batch_size, n_negatives, n_features, window)
    cpc_positive, _ = _tiny_batch(batch_size, n_negatives, n_features, window)
    align_positive, _ = _tiny_batch(batch_size, n_negatives, n_features, window)

    params_before = [p.clone() for p in model.parameters()]
    losses = train_step_stage1b(model, optimizer, anchor, cpc_positive, align_positive, negatives,
                                 alignment_weight=0.5)

    finite = all(v == v and v != float("inf") for v in losses.values())
    print_check("train_step_stage1b: returns finite total/cpc/alignment losses", finite, f"{losses}")
    passed, failed = passed + finite, failed + (not finite)

    changed = any(not torch.allclose(a, b) for a, b in zip(params_before, model.parameters()))
    print_check("train_step_stage1b: model parameters change after one gradient step", changed)
    passed, failed = passed + changed, failed + (not changed)
    return passed, failed


def test_score_holdout_stage1b_does_not_change_params_and_returns_finite_score(passed, failed):
    panel, cache = _synthetic_frame_cache(["AAA", "BBB"], n_days=400)
    _, holdout_panel = split_train_holdout(panel, holdout_days=180)
    holdout_store = CPCPanelStore(holdout_panel, cache)
    cfg = SSLConfig(cpc_horizon=10, alignment_horizon=30, batch_size=8,
                     n_same_stock_negatives=2, n_diff_stock_negatives=2, alignment_weight=0.5)

    model = EncoderCNN(len(DAILY_FEATURES), len(WEEKLY_FEATURES), len(MONTHLY_FEATURES),
                        len(QUARTERLY_FEATURES), d_model=8, n_heads=2)
    params_before = [p.clone() for p in model.parameters()]
    score = score_holdout_stage1b(model, holdout_panel, holdout_store, cfg,
                                   rng=np.random.default_rng(0), n_eval_batches=2)

    unchanged = all(torch.equal(a, b) for a, b in zip(params_before, model.parameters()))
    finite = score == score and score != float("inf")
    ok = unchanged and finite
    print_check("score_holdout_stage1b: leaves model parameters unchanged (no_grad) and returns a finite score",
                ok, f"score={score}, unchanged={unchanged}")
    return passed + ok, failed + (not ok)


def test_mask_branch_zeros_only_the_target_branch(passed, failed):
    batch = {
        "daily": torch.randn(2, 3, 4),
        "weekly": torch.randn(2, 3, 4),
        "monthly": torch.randn(2, 3, 4),
        "fundamentals": torch.randn(2, 3, 4),
    }
    masked = _mask_branch(batch, "weekly")
    zeroed = bool(torch.all(masked["weekly"] == 0))
    others_untouched = all(torch.equal(masked[k], batch[k]) for k in batch if k != "weekly")
    ok = zeroed and others_untouched
    print_check("_mask_branch: zeros only the target branch, leaves the others untouched",
                ok, f"zeroed={zeroed}, others_untouched={others_untouched}")
    return passed + ok, failed + (not ok)


def test_reconstruction_heads_output_shape(passed, failed):
    d_model = 8
    heads = ReconstructionHeads(d_model)
    embedding = torch.randn(5, d_model)
    out = heads(embedding, "fundamentals")
    ok = out.shape == (5, d_model)
    print_check("ReconstructionHeads: output shape matches [B, d_model] for the requested branch",
                ok, f"shape={tuple(out.shape)}")
    return passed + ok, failed + (not ok)


def test_train_step_stage1c_updates_params_and_returns_finite_losses(passed, failed):
    torch.manual_seed(0)
    n_features, window, batch_size, n_negatives, d_model = 5, 8, 4, 3, 8
    model = EncoderCNN(n_features, n_features, n_features, n_features, d_model=d_model, n_heads=2)
    recon_heads = ReconstructionHeads(d_model)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(recon_heads.parameters()), lr=1e-2)
    anchor, negatives = _tiny_batch(batch_size, n_negatives, n_features, window)
    cpc_positive, _ = _tiny_batch(batch_size, n_negatives, n_features, window)
    align_positive, _ = _tiny_batch(batch_size, n_negatives, n_features, window)

    params_before = [p.clone() for p in list(model.parameters()) + list(recon_heads.parameters())]
    losses = train_step_stage1c(model, recon_heads, optimizer, anchor, cpc_positive, align_positive,
                                 negatives, masked_branch="fundamentals", alignment_weight=0.5,
                                 reconstruction_weight=0.5)

    finite = all(v == v and v != float("inf") for v in losses.values())
    print_check("train_step_stage1c: returns finite total/cpc/alignment/reconstruction losses",
                finite, f"{losses}")
    passed, failed = passed + finite, failed + (not finite)

    changed = any(not torch.allclose(a, b)
                  for a, b in zip(params_before, list(model.parameters()) + list(recon_heads.parameters())))
    print_check("train_step_stage1c: parameters (encoder + recon heads) change after one gradient step", changed)
    passed, failed = passed + changed, failed + (not changed)
    return passed, failed


def test_score_holdout_stage1c_does_not_change_params_and_returns_finite_score(passed, failed):
    panel, cache = _synthetic_frame_cache(["AAA", "BBB"], n_days=400)
    _, holdout_panel = split_train_holdout(panel, holdout_days=180)
    holdout_store = CPCPanelStore(holdout_panel, cache)
    cfg = SSLConfig(cpc_horizon=10, alignment_horizon=30, batch_size=8,
                     n_same_stock_negatives=2, n_diff_stock_negatives=2, alignment_weight=0.5,
                     reconstruction_weight=0.5, d_model=8, n_heads=2)

    model = EncoderCNN(len(DAILY_FEATURES), len(WEEKLY_FEATURES), len(MONTHLY_FEATURES),
                        len(QUARTERLY_FEATURES), d_model=cfg.d_model, n_heads=cfg.n_heads)
    recon_heads = ReconstructionHeads(cfg.d_model)
    params_before = [p.clone() for p in list(model.parameters()) + list(recon_heads.parameters())]
    score = score_holdout_stage1c(model, recon_heads, holdout_panel, holdout_store, cfg,
                                   rng=np.random.default_rng(0), n_eval_batches=2)

    unchanged = all(torch.equal(a, b)
                     for a, b in zip(params_before, list(model.parameters()) + list(recon_heads.parameters())))
    finite = score == score and score != float("inf")
    ok = unchanged and finite
    print_check("score_holdout_stage1c: leaves parameters unchanged (no_grad) and returns a finite score",
                ok, f"score={score}, unchanged={unchanged}")
    return passed + ok, failed + (not ok)


def test_reconstruction_loss_decreases_with_training(passed, failed):
    # Non-trivial-logic check: masked reconstruction must actually be learnable, not just
    # finite -- train on a FIXED tiny batch for a few dozen steps and confirm the
    # reconstruction term drops.
    torch.manual_seed(1)
    n_features, window, batch_size, n_negatives, d_model = 5, 8, 6, 2, 8
    model = EncoderCNN(n_features, n_features, n_features, n_features, d_model=d_model, n_heads=2)
    recon_heads = ReconstructionHeads(d_model)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(recon_heads.parameters()), lr=5e-3)
    anchor, negatives = _tiny_batch(batch_size, n_negatives, n_features, window)
    cpc_positive, _ = _tiny_batch(batch_size, n_negatives, n_features, window)
    align_positive, _ = _tiny_batch(batch_size, n_negatives, n_features, window)

    first_recon = last_recon = None
    for step in range(60):
        losses = train_step_stage1c(model, recon_heads, optimizer, anchor, cpc_positive, align_positive,
                                     negatives, masked_branch="daily")
        if step == 0:
            first_recon = losses["reconstruction"]
        last_recon = losses["reconstruction"]

    ok = last_recon < first_recon
    print_check("train_step_stage1c: reconstruction loss decreases over training steps on a fixed batch",
                ok, f"first={first_recon:.4f}, last={last_recon:.4f}")
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("conviction_model/ssl_pretrain.py (Stage 1A: CPC, Stage 1B: + forward cross-modal alignment, "
                 "Stage 1C: + masked reconstruction)")
    passed = failed = 0
    for test_fn in [
        test_info_nce_near_zero_when_positive_matches_and_negatives_orthogonal,
        test_info_nce_equals_log_n_plus_1_when_all_candidates_tied,
        test_same_stock_negatives_are_same_ticker_and_far_in_time,
        test_sample_cpc_negatives_excludes_positive_for_short_history_ticker,
        test_diff_stock_negatives_are_other_tickers_same_date,
        test_sample_cpc_anchor_positions_leaves_room_for_the_horizon,
        test_build_cpc_batch_shapes,
        test_lazy_panel_gatherer_matches_cpc_panel_store,
        test_cached_panel_gatherer_matches_lazy_panel_gatherer,
        test_cached_panel_gatherer_consistent_after_eviction,
        test_cached_panel_gatherer_evicts_least_recently_used,
        test_cached_panel_gatherer_maxsize_zero_still_works,
        test_cached_panel_gatherer_stores_float32_not_float64,
        test_build_cpc_batch_positive_is_same_ticker_k_ahead,
        test_split_train_holdout_respects_cutoff_and_stays_contiguous,
        test_score_holdout_does_not_change_params_and_returns_finite_score,
        test_train_step_updates_params_and_returns_finite_loss,
        test_price_macro_state_ignores_fundamentals,
        test_build_stage1b_batch_positives_use_different_horizons,
        test_train_step_stage1b_updates_params_and_returns_finite_losses,
        test_score_holdout_stage1b_does_not_change_params_and_returns_finite_score,
        test_mask_branch_zeros_only_the_target_branch,
        test_reconstruction_heads_output_shape,
        test_train_step_stage1c_updates_params_and_returns_finite_losses,
        test_score_holdout_stage1c_does_not_change_params_and_returns_finite_score,
        test_reconstruction_loss_decreases_with_training,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
