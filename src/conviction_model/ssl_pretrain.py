"""
ssl_pretrain.py -- Phase 1, Stage 1A (docs/conviction_model/CONVICTION_MODEL_PLAN.md):
the CPC (Contrastive Predictive Coding) loss, the first of the 4 SSL losses,
introduced alone per the plan's staged 1A->1D design (isolate one loss at a
time so any diagnostic change is attributable to the loss just added).
Stages 1B-1D (forward cross-modal alignment, masked reconstruction, the
auxiliary valuation-probe nudge) are added incrementally in later work, not
this file yet.

CPC predicts FUTURE LATENT STATE (an embedding k steps ahead), not future
price -- pushes the encoder toward temporal consistency of "market state"
rather than pure price autocorrelation ("What the latent representation is
for", CONVICTION_MODEL_PLAN.md). InfoNCE with two negative types, specified
in the plan rather than left generic:
  - same-stock-different-regime: same ticker, a distant time window
  - different-stock-same-time: a different ticker at the same date, so
    market-wide co-movement alone can't trivially satisfy the objective
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import (
    DAILY_FEATURES, DAILY_WINDOW, MONTHLY_FEATURES, MONTHLY_WINDOW, QUARTERLY_FEATURES,
    QUARTERLY_WINDOW, WEEKLY_FEATURES, WEEKLY_WINDOW, branch_windows_from_precomputed,
)


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor,
                   negatives: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """anchor [B, d], positive [B, d] (anchor's true future embedding),
    negatives [B, N, d]. Cross-entropy treating the positive as the correct
    class among {positive} U negatives, similarity = cosine / temperature."""
    anchor_n = F.normalize(anchor, dim=-1)
    pos_n = F.normalize(positive, dim=-1)
    neg_n = F.normalize(negatives, dim=-1)

    pos_sim = (anchor_n * pos_n).sum(dim=-1, keepdim=True) / temperature      # [B, 1]
    neg_sim = torch.einsum("bd,bnd->bn", anchor_n, neg_n) / temperature       # [B, N]
    logits = torch.cat([pos_sim, neg_sim], dim=1)                             # [B, 1+N]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def _group_positions(keys: np.ndarray) -> dict:
    """{key: array of row positions with that key}, via a single O(n log n)
    argsort instead of one O(n) boolean scan PER unique key -- the naive
    dict comprehension (`{k: np.flatnonzero(keys == k) for k in
    np.unique(keys)}`) is O(n_unique * n), which measured as the actual
    bottleneck in the Stage 1A pilot (~155ms/step, ~5.4k unique trade_dates
    x 27k rows =~ 147M comparisons, rebuilt from scratch every step)."""
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    groups = np.split(order, boundaries)
    unique_keys = sorted_keys[np.r_[0, boundaries]] if len(sorted_keys) else sorted_keys
    return dict(zip(unique_keys, groups))


def sample_cpc_negatives(panel: pd.DataFrame, anchor_positions: np.ndarray,
                          n_same_stock: int = 4, n_diff_stock: int = 4,
                          regime_gap_days: int = 252,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """`panel`: DataFrame with columns ['ticker', 'trade_date'], one row per
    (ticker, date) embedding position -- row position (0..len-1) IS the
    embedding batch index elsewhere in the training loop. `anchor_positions`:
    [B] int array of row positions to sample negatives for. Returns
    [B, n_same_stock + n_diff_stock] int array of negative row positions.

    ponytail: "different regime" is approximated by a plain time-gap
    threshold (>= regime_gap_days), not an actual valuation/volatility-bucket
    match -- the plan's own wording ("a distant time window in a different
    valuation/volatility state") wants the latter; upgrade to a real regime
    filter if Stage 1A's diagnostics show these negatives aren't hard enough
    to be informative.
    """
    rng = rng or np.random.default_rng()
    tickers = panel["ticker"].to_numpy()
    dates = panel["trade_date"].to_numpy()
    positions_by_ticker = _group_positions(tickers)
    positions_by_date = _group_positions(dates)

    out = np.empty((len(anchor_positions), n_same_stock + n_diff_stock), dtype=np.int64)
    for row, pos in enumerate(anchor_positions):
        ticker, date = tickers[pos], dates[pos]

        same_pool = positions_by_ticker[ticker]
        gap_days = np.abs((dates[same_pool] - date) / np.timedelta64(1, "D"))
        far = same_pool[gap_days >= regime_gap_days]
        if len(far) == 0:
            far = same_pool[same_pool != pos]
        out[row, :n_same_stock] = rng.choice(far, size=n_same_stock, replace=len(far) < n_same_stock)

        same_time_pool = positions_by_date[date]
        diff = same_time_pool[tickers[same_time_pool] != ticker]
        if len(diff) == 0:
            diff = np.delete(np.arange(len(panel)), pos)
        out[row, n_same_stock:] = rng.choice(diff, size=n_diff_stock, replace=len(diff) < n_diff_stock)
    return out


def sample_cpc_anchor_positions(panel: pd.DataFrame, batch_size: int, cpc_horizon: int,
                                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Random row positions from `panel` (must be sorted by
    ['ticker','trade_date'] and RangeIndex'd -- same contract as
    sample_cpc_negatives, since each ticker's rows need to be contiguous)
    that have a valid same-ticker positive at position + cpc_horizon, i.e.
    excludes each ticker's trailing `cpc_horizon` rows."""
    rng = rng or np.random.default_rng()
    remaining = panel.groupby("ticker").cumcount(ascending=False).to_numpy()
    eligible = np.flatnonzero(remaining >= cpc_horizon)
    if len(eligible) < batch_size:
        raise ValueError(f"only {len(eligible)} eligible anchors for cpc_horizon={cpc_horizon}, "
                          f"need batch_size={batch_size}")
    return rng.choice(eligible, size=batch_size, replace=False)


class CPCPanelStore:
    """Precomputed window tensors for EVERY position in `panel`, one tensor
    per branch, kept resident -- batch assembly becomes pure index_select,
    no per-position pandas calls. Mirrors rl_agent/train.py's _PanelStore
    (CLAUDE.md, TRAINING_SPEEDUP_PLAN): this encoder's per-step compute is
    tiny, so repeated small pandas lookups dominated wall-clock (measured in
    the Stage 1A pilot: ~950ms/step batch assembly vs ~43ms/step gradient
    step before this) -- pay the lookup cost once for the whole panel, not
    once per step forever. ponytail: ~n_positions * (11+11+14+19) floats *
    window-length -- fine for a 5-ticker pilot (~32k positions); revisit
    with chunked construction (like _PanelStore's `chunk` arg) if scaling to
    the full ~515-ticker universe makes the intermediate memory a problem."""

    _SHAPES = {"daily": (len(DAILY_FEATURES), DAILY_WINDOW),
               "weekly": (len(WEEKLY_FEATURES), WEEKLY_WINDOW),
               "monthly": (len(MONTHLY_FEATURES), MONTHLY_WINDOW),
               "fundamentals": (len(QUARTERLY_FEATURES), QUARTERLY_WINDOW)}

    def __init__(self, panel: pd.DataFrame, frame_cache: dict):
        n = len(panel)
        tickers = panel["ticker"].to_numpy()
        dates = panel["trade_date"].to_numpy()
        arrays = {name: np.empty((n, *shape), dtype=np.float32) for name, shape in self._SHAPES.items()}
        for pos in range(n):
            ticker, as_of = tickers[pos], pd.Timestamp(dates[pos])
            daily_frame, weekly_frame, monthly_frame, quarterly_frame = frame_cache[ticker]
            windows = branch_windows_from_precomputed(daily_frame, weekly_frame, monthly_frame,
                                                        quarterly_frame, as_of)
            for name, arr in windows.items():
                arrays[name][pos] = arr
        self.tensors = {name: torch.tensor(arr, dtype=torch.float32) for name, arr in arrays.items()}

    def gather(self, positions: np.ndarray) -> dict:
        idx = torch.as_tensor(positions, dtype=torch.long)
        return {name: t.index_select(0, idx) for name, t in self.tensors.items()}


class LazyPanelGatherer:
    """Same .gather(positions) interface as CPCPanelStore (duck-typed --
    build_cpc_batch doesn't care which one it gets), but computes each
    position's windows on demand from a per-ticker frame_cache instead of
    precomputing every position upfront. Trades speed for memory: adjacent
    positions for the same ticker overlap almost their entire window (a
    daily window shares 59/60 rows with its neighbor), so CPCPanelStore's
    per-position storage is fundamentally redundant -- ~17KB/position adds
    up fast (measured: ~9GB for a 150-ticker/540k-position universe). This
    class holds only the per-ticker frames (tens of MB for 150 tickers),
    recomputing a window every time it's touched. Use when the universe is
    too large for CPCPanelStore's full precompute to fit in memory."""

    def __init__(self, panel: pd.DataFrame, frame_cache: dict):
        self.tickers = panel["ticker"].to_numpy()
        self.dates = panel["trade_date"].to_numpy()
        self.frame_cache = frame_cache

    def gather(self, positions: np.ndarray) -> dict:
        per_branch = {"daily": [], "weekly": [], "monthly": [], "fundamentals": []}
        for pos in positions:
            ticker, as_of = self.tickers[pos], pd.Timestamp(self.dates[pos])
            daily_frame, weekly_frame, monthly_frame, quarterly_frame = self.frame_cache[ticker]
            windows = branch_windows_from_precomputed(daily_frame, weekly_frame, monthly_frame,
                                                        quarterly_frame, as_of)
            for name, arr in windows.items():
                per_branch[name].append(arr)
        return {name: torch.tensor(np.stack(arrs), dtype=torch.float32) for name, arrs in per_branch.items()}


def build_cpc_batch(panel: pd.DataFrame, store, anchor_positions: np.ndarray,
                     cpc_horizon: int, n_same_stock: int = 4, n_diff_stock: int = 4,
                     regime_gap_days: int = 252,
                     rng: np.random.Generator | None = None) -> tuple[dict, dict, dict]:
    """Assemble one real CPC train_step's (anchor_batch, positive_batch,
    negative_batch) from a real (ticker, trade_date) panel + a `store`
    exposing .gather(positions) -> dict (CPCPanelStore, fast/memory-heavy, or
    LazyPanelGatherer, slower/memory-light -- pick per universe size). `panel`
    must be sorted by ['ticker','trade_date'] and RangeIndex'd
    (sample_cpc_negatives' contract). Positive = anchor_position
    + cpc_horizon: valid only because each ticker's rows are contiguous in
    `panel` and sample_cpc_anchor_positions already filtered for enough
    trailing rows -- this function trusts that filtering, it doesn't redo it.
    """
    rng = rng or np.random.default_rng()
    positive_positions = anchor_positions + cpc_horizon
    negative_positions = sample_cpc_negatives(panel, anchor_positions, n_same_stock,
                                               n_diff_stock, regime_gap_days, rng)

    anchor_batch = store.gather(anchor_positions)
    positive_batch = store.gather(positive_positions)
    flat_negatives = store.gather(negative_positions.reshape(-1))
    n_negatives = n_same_stock + n_diff_stock
    negative_batch = {name: t.reshape(len(anchor_positions), n_negatives, *t.shape[1:])
                       for name, t in flat_negatives.items()}
    return anchor_batch, positive_batch, negative_batch


def _pool_embedding(branch_embeddings: dict) -> torch.Tensor:
    """CPC predicts one fused "market state" vector, not 4 separate branch
    embeddings (Architecture: CPC's InfoNCE operates on future LATENT STATE,
    singular) -- mean-pool the 4 branch tokens for this loss only. The 4
    separate sub-embeddings themselves (encoder.py's actual output) stay
    unpooled for the downstream regressor; this pooling is local to CPC."""
    return torch.stack(list(branch_embeddings.values()), dim=1).mean(dim=1)


def train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                anchor_batch: dict, positive_batch: dict, negative_batch: dict,
                temperature: float = 0.1, grad_clip_norm: float | None = None) -> float:
    """One CPC gradient step. `anchor_batch`/`positive_batch`: each a dict
    {'daily': [B,f,w], 'weekly': [B,f,w], 'monthly': [B,f,w],
    'fundamentals': [B,f,w]} (data.py's window_tensor, batched) for the
    anchor row and its t+k positive. `negative_batch`: same keys, each
    [B,N,f,w] (N negatives per anchor, from sample_cpc_negatives) --
    flattened to [B*N,f,w] to run through the encoder once, reshaped back to
    [B,N,d] to match info_nce_loss's contract."""
    optimizer.zero_grad()
    anchor_emb = _pool_embedding(model(**anchor_batch))
    positive_emb = _pool_embedding(model(**positive_batch))

    batch_size = anchor_batch["daily"].shape[0]
    n_negatives = negative_batch["daily"].shape[1]
    flat_negatives = {name: t.reshape(batch_size * n_negatives, *t.shape[2:])
                       for name, t in negative_batch.items()}
    negative_emb = _pool_embedding(model(**flat_negatives)).reshape(batch_size, n_negatives, -1)

    loss = info_nce_loss(anchor_emb, positive_emb, negative_emb, temperature)
    loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return float(loss.item())
