"""
ssl_pretrain.py -- Phase 1, Stages 1A-1C (docs/conviction_model/CONVICTION_MODEL_PLAN.md):
CPC (Contrastive Predictive Coding), forward cross-modal alignment, and masked
reconstruction across branches -- the first 3 of the 4 SSL losses, introduced one at a
time per the plan's staged 1A->1D design (isolate one loss at a time so any diagnostic
change is attributable to the loss just added). Stage 1D (the auxiliary valuation-probe
nudge) is added incrementally in later work, not this file yet.

CPC predicts FUTURE LATENT STATE (an embedding k steps ahead), not future
price -- pushes the encoder toward temporal consistency of "market state"
rather than pure price autocorrelation ("What the latent representation is
for", CONVICTION_MODEL_PLAN.md). InfoNCE with two negative types, specified
in the plan rather than left generic:
  - same-stock-different-regime: same ticker, a distant time window
  - different-stock-same-time: a different ticker at the same date, so
    market-wide co-movement alone can't trivially satisfy the objective

Forward cross-modal alignment (Stage 1B) reuses CPC's own (anchor, positive, negative)
batches -- same underlying (t, t+cpc_horizon) pairs, just a different branch selection:
price/macro state (daily+weekly+monthly, pooled) at t must predict the FUNDAMENTALS
branch's embedding at t+k, not merely agree with it at t (Architecture / "What the latent
representation is for") -- biases the representation toward what's fundamentally
predictive, not just price-autocorrelation-predictive. Same InfoNCE machinery, same
negative-sampling scheme, just scored against a different pair of embeddings.

Masked reconstruction (Stage 1C) reuses Stage 1B's own anchor batch too -- no new batch
assembly, no new negative sampling (it isn't a contrastive loss). One branch of the
ANCHOR is zeroed out; a small per-branch linear head must recover that branch's TRUE
pre-attention token from the post-attention embedding the encoder produces at the masked
slot -- which, with its own input zeroed, can only be built from what cross-attention
pulled in from the OTHER 3 branches. That's "reconstruction across branches": can the
encoder infer a missing branch's content from the rest of the state. Target is
`.detach()`-ed (stop-gradient) so the loss can't cheat by moving the target to match the
prediction instead of the other way around.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .data import (
    DAILY_FEATURES, DAILY_WINDOW, MONTHLY_FEATURES, MONTHLY_WINDOW, QUARTERLY_FEATURES,
    QUARTERLY_WINDOW, WEEKLY_FEATURES, WEEKLY_WINDOW, branch_windows_from_precomputed,
)
from .encoder import BRANCHES


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
                          rng: np.random.Generator | None = None,
                          exclude_positions: np.ndarray | None = None) -> np.ndarray:
    """`panel`: DataFrame with columns ['ticker', 'trade_date'], one row per
    (ticker, date) embedding position -- row position (0..len-1) IS the
    embedding batch index elsewhere in the training loop. `anchor_positions`:
    [B] int array of row positions to sample negatives for. Returns
    [B, n_same_stock + n_diff_stock] int array of negative row positions.

    `exclude_positions`: optional [B, K] int array of extra row positions
    (the CPC/alignment positive(s) for each anchor) to keep out of the
    same-stock NEGATIVE pool. Without this, a ticker with less than
    regime_gap_days of history falls back to `same_pool[same_pool != pos]`,
    which can select the positive itself (pos + horizon) as a "negative" --
    a contradictory InfoNCE label (the positive embedding would appear as
    both the numerator and a member of the negative set). Only the fallback
    branch needs this: regime_gap_days (252) is always larger than every
    horizon in play (cpc_horizon=21, alignment_horizon=63), so the positive
    can never satisfy `gap_days >= regime_gap_days` and land in `far`.

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
            if exclude_positions is not None:
                # Last-resort: if excluding the positive(s) would empty the pool
                # entirely (a ticker with almost no history beyond the anchor
                # itself), keep the pre-exclusion pool rather than crashing --
                # a rare degenerate case, not the common path this guards.
                filtered = far[~np.isin(far, exclude_positions[row])]
                far = filtered if len(filtered) else far
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


class CachedPanelGatherer:
    """Same .gather(positions) interface as CPCPanelStore/LazyPanelGatherer (duck-typed),
    but a middle ground between them: memoizes each (ticker, as_of) position's computed
    window tensors in a bounded LRU cache instead of either recomputing every time
    (LazyPanelGatherer, safe memory but pays the ~.loc-lookup-plus-normalize cost on every
    single touch) or precomputing the ENTIRE panel upfront (CPCPanelStore, ~17KB/position --
    ~9GB at the old 150-ticker snapshot universe, more at the current ~360-ticker
    point-in-time union -- too large for most machines to hold resident).

    CPC/alignment sample positions WITH REPLACEMENT from the same panel every step, so by
    later steps in a several-thousand-step run many positions have already been computed
    at least once -- caching those results trades a BOUNDED amount of memory for skipping
    the repeat computation on a cache hit, without CPCPanelStore's unbounded-with-universe-
    size memory risk. A pure memoization of a deterministic function (the underlying frame
    data never changes mid-run) -- results are bit-identical to LazyPanelGatherer's, this
    only changes wall-clock, never behavior (tested: test_ssl_pretrain.py).

    Cached entries are cast to float32 before storing (~17.65KB/entry, empirically measured
    via tracemalloc) -- window_tensor's raw output is float64, and an earlier version of
    this class cached that raw float64 dict directly, silently DOUBLING the intended
    footprint and causing a real OOM kill in production at the (then-)default 200_000-entry
    size across two independent stores (train + holdout each cache separately -- ~13.6GB
    combined pre-fix, not the ~3.4GB a per-store-only estimate implied). Casting per-window
    here vs. after gather()'s final stack+torch.tensor(..., dtype=torch.float32) produces
    bit-identical values either way (elementwise rounding doesn't depend on stacking order).

    ponytail: a plain OrderedDict LRU keyed by (ticker, as_of), evicting the least-recently-
    used entry once maxsize is reached -- not thread-safe (fine, this training loop is
    single-threaded); revisit with a proper cache library only if this bookkeeping itself
    becomes a measurable cost, which is unlikely at these sizes."""

    def __init__(self, panel: pd.DataFrame, frame_cache: dict, maxsize: int = 50_000):
        self.tickers = panel["ticker"].to_numpy()
        self.dates = panel["trade_date"].to_numpy()
        self.frame_cache = frame_cache
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()

    def _windows_for(self, pos: int) -> dict:
        ticker, as_of = self.tickers[pos], pd.Timestamp(self.dates[pos])
        key = (ticker, as_of)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        daily_frame, weekly_frame, monthly_frame, quarterly_frame = self.frame_cache[ticker]
        windows = branch_windows_from_precomputed(daily_frame, weekly_frame, monthly_frame,
                                                    quarterly_frame, as_of)
        # window_tensor returns float64 -- cast to float32 BEFORE caching (halves resident
        # memory, matches CPCPanelStore's own float32 convention for its resident tensors).
        # Casting per-window here vs. after gather()'s final stack+torch.tensor(...,
        # dtype=torch.float32) produces bit-identical results either way (elementwise
        # rounding doesn't depend on stacking order) -- verified in test_ssl_pretrain.py.
        windows = {name: arr.astype(np.float32) for name, arr in windows.items()}
        self._cache[key] = windows
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)  # evict least-recently-used
        return windows

    def gather(self, positions: np.ndarray) -> dict:
        per_branch = {"daily": [], "weekly": [], "monthly": [], "fundamentals": []}
        for pos in positions:
            windows = self._windows_for(pos)
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
    negative_positions = sample_cpc_negatives(panel, anchor_positions, n_same_stock, n_diff_stock,
                                               regime_gap_days, rng,
                                               exclude_positions=positive_positions[:, None])

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


def _cpc_loss(model: torch.nn.Module, anchor_batch: dict, positive_batch: dict,
              negative_batch: dict, temperature: float) -> torch.Tensor:
    """Shared by train_step (grad) and score_holdout (no_grad): embed
    anchor/positive/negative batches and compute the InfoNCE loss.
    `anchor_batch`/`positive_batch`: each a dict {'daily': [B,f,w], 'weekly':
    [B,f,w], 'monthly': [B,f,w], 'fundamentals': [B,f,w]} (data.py's
    window_tensor, batched) for the anchor row and its t+k positive.
    `negative_batch`: same keys, each [B,N,f,w] (N negatives per anchor, from
    sample_cpc_negatives) -- flattened to [B*N,f,w] to run through the
    encoder once, reshaped back to [B,N,d] to match info_nce_loss's
    contract."""
    anchor_emb = _pool_embedding(model(**anchor_batch))
    positive_emb = _pool_embedding(model(**positive_batch))

    batch_size = anchor_batch["daily"].shape[0]
    n_negatives = negative_batch["daily"].shape[1]
    flat_negatives = {name: t.reshape(batch_size * n_negatives, *t.shape[2:])
                       for name, t in negative_batch.items()}
    negative_emb = _pool_embedding(model(**flat_negatives)).reshape(batch_size, n_negatives, -1)

    return info_nce_loss(anchor_emb, positive_emb, negative_emb, temperature)


def train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                anchor_batch: dict, positive_batch: dict, negative_batch: dict,
                temperature: float = 0.1, grad_clip_norm: float | None = None) -> float:
    """One CPC gradient step -- see _cpc_loss for the batch dict shapes."""
    optimizer.zero_grad()
    loss = _cpc_loss(model, anchor_batch, positive_batch, negative_batch, temperature)
    loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return float(loss.item())


def split_train_holdout(panel: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-ticker date-based split: the trailing `holdout_days` calendar days
    of `panel`'s history become the holdout tail, carved out of CPC training
    entirely (never an anchor, positive, or negative in a train_step call)
    and used only to score checkpoints -- same purpose as
    rl_agent/train.py's checkpoint_holdout_days (CLAUDE.md: caught a real
    case where the same seed went from +47% return at 100k steps to -71% at
    2M, pure overfitting past a peak that a fixed step count alone can't
    detect). Each half is a per-ticker prefix/suffix, freshly RangeIndex'd,
    so sample_cpc_anchor_positions/sample_cpc_negatives's contiguous-
    per-ticker contract (position i+1 is the same ticker's next row) still
    holds within each half."""
    cutoff = panel["trade_date"].max() - pd.Timedelta(days=holdout_days)
    train_panel = panel.loc[panel["trade_date"] <= cutoff].reset_index(drop=True)
    holdout_panel = panel.loc[panel["trade_date"] > cutoff].reset_index(drop=True)
    return train_panel, holdout_panel


def score_holdout(model: torch.nn.Module, holdout_panel: pd.DataFrame, holdout_store,
                   cfg, rng: np.random.Generator, n_eval_batches: int = 4) -> float:
    """Mean CPC loss over `n_eval_batches` batches drawn from the holdout
    panel, frozen weights (no gradient, model.eval()) -- lower is better
    (it's a loss, unlike rl_agent's total_return score). Averaging a few
    batches instead of one reduces sampling noise in the score used to pick
    the "best" checkpoint. `cfg`: SSLConfig (batch_size, cpc_horizon,
    negative-sampling params, temperature)."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_eval_batches):
            anchors = sample_cpc_anchor_positions(holdout_panel, cfg.batch_size, cfg.cpc_horizon, rng=rng)
            anchor_batch, positive_batch, negative_batch = build_cpc_batch(
                holdout_panel, holdout_store, anchors, cfg.cpc_horizon,
                n_same_stock=cfg.n_same_stock_negatives, n_diff_stock=cfg.n_diff_stock_negatives,
                regime_gap_days=cfg.regime_gap_days, rng=rng)
            loss = _cpc_loss(model, anchor_batch, positive_batch, negative_batch, cfg.temperature)
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


# --- Stage 1B: + forward cross-modal alignment -------------------------------

def _price_macro_state(branch_embeddings: dict) -> torch.Tensor:
    """Price/macro state = daily+weekly+monthly branches pooled, deliberately
    EXCLUDING fundamentals -- Stage 1B's alignment anchor ("price/macro state
    at t must predict the fundamentals branch's future embedding"). Distinct
    from _pool_embedding (CPC's "market state"), which pools all 4 branches
    including fundamentals -- that inclusion would let the anchor trivially
    see a piece of the very thing it's supposed to predict."""
    keys = ("daily", "weekly", "monthly")
    return torch.stack([branch_embeddings[k] for k in keys], dim=1).mean(dim=1)


def build_stage1b_batch(panel: pd.DataFrame, store, anchor_positions: np.ndarray,
                         cpc_horizon: int, alignment_horizon: int,
                         n_same_stock: int = 4, n_diff_stock: int = 4,
                         regime_gap_days: int = 252,
                         rng: np.random.Generator | None = None) -> tuple[dict, dict, dict, dict]:
    """Stage 1B's batch assembly: CPC and forward cross-modal alignment read
    DIFFERENT forward horizons off the SAME anchor -- alignment_horizon
    (default ~63 trading days, one fiscal quarter) deliberately differs from
    cpc_horizon (21 trading days), matching the fundamentals branch's actual
    update cadence instead of a horizon picked for CPC's own reasons (see
    SSLConfig.alignment_horizon for why sharing cpc_horizon was wrong).
    Negatives ARE shared -- sample_cpc_negatives doesn't depend on horizon at
    all, so the same same-stock-different-regime/different-stock-same-time
    negatives serve both losses; only the positive differs. Returns
    (anchor_batch, cpc_positive_batch, align_positive_batch, negative_batch)."""
    rng = rng or np.random.default_rng()
    cpc_positive_positions = anchor_positions + cpc_horizon
    align_positive_positions = anchor_positions + alignment_horizon
    exclude_positions = np.stack([cpc_positive_positions, align_positive_positions], axis=1)
    negative_positions = sample_cpc_negatives(panel, anchor_positions, n_same_stock, n_diff_stock,
                                               regime_gap_days, rng, exclude_positions=exclude_positions)

    anchor_batch = store.gather(anchor_positions)
    cpc_positive_batch = store.gather(cpc_positive_positions)
    align_positive_batch = store.gather(align_positive_positions)
    flat_negatives = store.gather(negative_positions.reshape(-1))
    n_negatives = n_same_stock + n_diff_stock
    negative_batch = {name: t.reshape(len(anchor_positions), n_negatives, *t.shape[1:])
                       for name, t in flat_negatives.items()}
    return anchor_batch, cpc_positive_batch, align_positive_batch, negative_batch


def _alignment_loss(model: torch.nn.Module, anchor_batch: dict, align_positive_batch: dict,
                     negative_batch: dict, temperature: float) -> torch.Tensor:
    """Stage 1B's forward cross-modal alignment loss. anchor pools price/macro
    branches only (_price_macro_state) at t; align_positive_batch/negatives
    read the FUNDAMENTALS branch alone (not pooled) at t+alignment_horizon
    (build_stage1b_batch). Same InfoNCE contrastive form as CPC
    (info_nce_loss), scored against a different pair of embeddings and a
    different horizon, not new machinery."""
    anchor_emb = _price_macro_state(model(**anchor_batch))
    positive_emb = model(**align_positive_batch)["fundamentals"]

    batch_size = anchor_batch["daily"].shape[0]
    n_negatives = negative_batch["daily"].shape[1]
    flat_negatives = {name: t.reshape(batch_size * n_negatives, *t.shape[2:])
                       for name, t in negative_batch.items()}
    negative_emb = model(**flat_negatives)["fundamentals"].reshape(batch_size, n_negatives, -1)

    return info_nce_loss(anchor_emb, positive_emb, negative_emb, temperature)


def _stage1b_loss(model: torch.nn.Module, anchor_batch: dict, cpc_positive_batch: dict,
                   align_positive_batch: dict, negative_batch: dict, temperature: float,
                   alignment_weight: float) -> dict:
    """CPC + forward cross-modal alignment, combined as a weighted sum (Module
    layout: ssl_pretrain.py's losses combine as a weighted sum). Shared by
    train_step_stage1b (grad) and score_holdout_stage1b (no_grad) -- same
    split Stage 1A's _cpc_loss already uses. Returns all three terms (not
    just the total) so the training loop can log whether alignment is
    actually learning something, not just riding CPC's gradient."""
    cpc = _cpc_loss(model, anchor_batch, cpc_positive_batch, negative_batch, temperature)
    alignment = _alignment_loss(model, anchor_batch, align_positive_batch, negative_batch, temperature)
    total = cpc + alignment_weight * alignment
    return {"total": total, "cpc": cpc, "alignment": alignment}


def train_step_stage1b(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        anchor_batch: dict, cpc_positive_batch: dict, align_positive_batch: dict,
                        negative_batch: dict, temperature: float = 0.1, alignment_weight: float = 1.0,
                        grad_clip_norm: float | None = None) -> dict:
    """One Stage 1B gradient step. Returns {'total','cpc','alignment'} floats
    -- unlike Stage 1A's train_step (a single float), since seeing each term
    separately is the whole point of logging a combined-loss run."""
    optimizer.zero_grad()
    losses = _stage1b_loss(model, anchor_batch, cpc_positive_batch, align_positive_batch,
                            negative_batch, temperature, alignment_weight)
    losses["total"].backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return {k: float(v.item()) for k, v in losses.items()}


def score_holdout_stage1b(model: torch.nn.Module, holdout_panel: pd.DataFrame, holdout_store,
                           cfg, rng: np.random.Generator, n_eval_batches: int = 4) -> float:
    """Same purpose/shape as Stage 1A's score_holdout, scoring the combined
    Stage 1B loss -- checkpoint-at-peak needs one scalar to compare
    checkpoints by, so this returns the combined total (cpc + weighted
    alignment), not the two terms separately."""
    model.eval()
    totals = []
    max_horizon = max(cfg.cpc_horizon, cfg.alignment_horizon)
    with torch.no_grad():
        for _ in range(n_eval_batches):
            anchors = sample_cpc_anchor_positions(holdout_panel, cfg.batch_size, max_horizon, rng=rng)
            anchor_batch, cpc_positive_batch, align_positive_batch, negative_batch = build_stage1b_batch(
                holdout_panel, holdout_store, anchors, cfg.cpc_horizon, cfg.alignment_horizon,
                n_same_stock=cfg.n_same_stock_negatives, n_diff_stock=cfg.n_diff_stock_negatives,
                regime_gap_days=cfg.regime_gap_days, rng=rng)
            losses = _stage1b_loss(model, anchor_batch, cpc_positive_batch, align_positive_batch,
                                    negative_batch, cfg.temperature, cfg.alignment_weight)
            totals.append(float(losses["total"].item()))
    model.train()
    return sum(totals) / len(totals)


# --- Stage 1C: + masked reconstruction across branches -----------------------

class ReconstructionHeads(nn.Module):
    """One small Linear(d_model, d_model) per branch (BRANCHES). CPC/alignment compare
    embeddings that already live in the same space (post-attention vs. post-attention, or
    a pooled vector vs. the raw fundamentals token) -- no head needed there. Here the two
    sides genuinely differ: the prediction is a POST-attention embedding built from the
    other 3 branches, the target is that branch's own PRE-attention token
    (encoder.py::EncoderCNN.branch_tokens) -- a small linear bridge between the two spaces,
    not a full decoder back to the raw [n_features, window] input (reconstructing a
    d_model-wide token is the design's actual goal; the plan never asks for pixel/tick-level
    input recovery)."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Linear(d_model, d_model) for name in BRANCHES})

    def forward(self, embedding: torch.Tensor, branch_name: str) -> torch.Tensor:
        return self.heads[branch_name](embedding)


def _mask_branch(batch: dict, branch_name: str) -> dict:
    """Zero out one branch's window tensor -- matches this pipeline's existing NaN->0 fill
    convention (CLAUDE.md's NaN policy) rather than a learned mask-token parameter; no new
    encoder.py machinery needed for this."""
    return {name: (torch.zeros_like(t) if name == branch_name else t) for name, t in batch.items()}


def _branch_token(model: torch.nn.Module, batch: dict, branch_name: str) -> torch.Tensor:
    """The TRUE pre-attention token for one branch of `batch` -- the reconstruction
    target. Runs all 4 branch CNNs (encoder.py::branch_tokens doesn't expose a
    single-branch shortcut); tiny CNNs, not worth a special-cased path."""
    tokens = model.branch_tokens(batch["daily"], batch["weekly"], batch["monthly"], batch["fundamentals"])
    return tokens[:, BRANCHES.index(branch_name), :]


def _reconstruction_loss(model: torch.nn.Module, recon_heads: ReconstructionHeads,
                          anchor_batch: dict, masked_branch: str) -> torch.Tensor:
    """Mask `masked_branch` out of the anchor, run the encoder, and try to recover that
    branch's true (unmasked) pre-attention token from the masked post-attention embedding
    -- see module docstring. `.detach()` on the target: a plain stop-gradient so the
    encoder can't lower this loss by dragging the target token toward whatever the
    prediction already says, only by making the OTHER branches' information (visible via
    cross-attention) actually predictive of the masked one."""
    masked_batch = _mask_branch(anchor_batch, masked_branch)
    masked_embedding = model(**masked_batch)[masked_branch]
    predicted = recon_heads(masked_embedding, masked_branch)
    target = _branch_token(model, anchor_batch, masked_branch).detach()
    return F.mse_loss(predicted, target)


def _stage1c_loss(model: torch.nn.Module, recon_heads: ReconstructionHeads, anchor_batch: dict,
                   cpc_positive_batch: dict, align_positive_batch: dict, negative_batch: dict,
                   masked_branch: str, temperature: float, alignment_weight: float,
                   reconstruction_weight: float) -> dict:
    """CPC + alignment + masked reconstruction, combined as a weighted sum (same
    "weighted sum" convention as _stage1b_loss). Shared by train_step_stage1c (grad) and
    score_holdout_stage1c (no_grad). Returns all four terms, not just the total, so a
    training loop can log whether reconstruction is actually learning anything."""
    cpc = _cpc_loss(model, anchor_batch, cpc_positive_batch, negative_batch, temperature)
    alignment = _alignment_loss(model, anchor_batch, align_positive_batch, negative_batch, temperature)
    reconstruction = _reconstruction_loss(model, recon_heads, anchor_batch, masked_branch)
    total = cpc + alignment_weight * alignment + reconstruction_weight * reconstruction
    return {"total": total, "cpc": cpc, "alignment": alignment, "reconstruction": reconstruction}


def train_step_stage1c(model: torch.nn.Module, recon_heads: ReconstructionHeads,
                        optimizer: torch.optim.Optimizer, anchor_batch: dict, cpc_positive_batch: dict,
                        align_positive_batch: dict, negative_batch: dict, masked_branch: str,
                        temperature: float = 0.1, alignment_weight: float = 1.0,
                        reconstruction_weight: float = 1.0, grad_clip_norm: float | None = None) -> dict:
    """One Stage 1C gradient step. `optimizer` must cover BOTH model.parameters() and
    recon_heads.parameters() -- the caller builds one optimizer over both (run_stage1c.py),
    same as any other multi-module training loop. Returns
    {'total','cpc','alignment','reconstruction'} floats."""
    optimizer.zero_grad()
    losses = _stage1c_loss(model, recon_heads, anchor_batch, cpc_positive_batch, align_positive_batch,
                            negative_batch, masked_branch, temperature, alignment_weight, reconstruction_weight)
    losses["total"].backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(recon_heads.parameters()), grad_clip_norm)
    optimizer.step()
    return {k: float(v.item()) for k, v in losses.items()}


def score_holdout_stage1c(model: torch.nn.Module, recon_heads: ReconstructionHeads,
                           holdout_panel: pd.DataFrame, holdout_store, cfg,
                           rng: np.random.Generator, n_eval_batches: int = 4) -> float:
    """Same purpose/shape as score_holdout_stage1b, scoring the combined Stage 1C loss
    (cpc + weighted alignment + weighted reconstruction). `masked_branch` is re-sampled
    per eval batch (same rng, same "average a few batches to cut sampling noise"
    reasoning n_eval_batches already uses elsewhere in this file) rather than fixed to one
    branch, so the score isn't skewed by whichever single branch happened to be picked."""
    model.eval()
    recon_heads.eval()
    totals = []
    max_horizon = max(cfg.cpc_horizon, cfg.alignment_horizon)
    with torch.no_grad():
        for _ in range(n_eval_batches):
            anchors = sample_cpc_anchor_positions(holdout_panel, cfg.batch_size, max_horizon, rng=rng)
            anchor_batch, cpc_positive_batch, align_positive_batch, negative_batch = build_stage1b_batch(
                holdout_panel, holdout_store, anchors, cfg.cpc_horizon, cfg.alignment_horizon,
                n_same_stock=cfg.n_same_stock_negatives, n_diff_stock=cfg.n_diff_stock_negatives,
                regime_gap_days=cfg.regime_gap_days, rng=rng)
            masked_branch = BRANCHES[int(rng.integers(len(BRANCHES)))]
            losses = _stage1c_loss(model, recon_heads, anchor_batch, cpc_positive_batch, align_positive_batch,
                                    negative_batch, masked_branch, cfg.temperature, cfg.alignment_weight,
                                    cfg.reconstruction_weight)
            totals.append(float(losses["total"].item()))
    model.train()
    recon_heads.train()
    return sum(totals) / len(totals)
