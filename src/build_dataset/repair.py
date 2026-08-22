"""repair.py — rescale adj_* price history where a split/inplit was left unadjusted."""

import json
import numpy as np
import pandas as pd

from .paths import CORPORATE_EVENTS_PATH, CONTINUITY_PATH

ADJ_PRICE_COLS = ["adj_open", "adj_high", "adj_low", "adj_close"]
RAW_PRICE_COLS = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume", "volume_adjusted"]

# Tickers where the vendor's plain (non-adj_*) OHLC is ALSO unadjusted for a
# real split -- not just adj_close, which is what this repair normally
# assumes is the only column needing it (raw close/open/high/low are meant to
# show real historical nominal jumps at genuine corporate events, so they're
# left alone for every other ticker). Confirmed for TIMS3 by directly
# cross-checking against TIMP3 (the real pre-2020-08-31 entity, spliced in via
# ticker_continuity.json): TIMP3's OHLC is exactly 100.00x TIMS3's own native
# OHLC on all 2,273 overlapping trading dates 2011-08-03..2020-10-09 (std
# <0.4%, not market noise), and TIMP3's basis is the one consistent with CVM's
# real fundamentals (shares_outstanding x close_price implies a plausible
# ~R$28B TIM market cap in 2019 off TIMP3's close; TIMS3's own native close
# implies ~R$280M, implausible for TIM). This -- not a missed corporate-share-
# count update -- is what caused test_unit_scale_invariants.py's
# market_cap/shares==close check to flag TIMS3 worst (ratio ~99x): CVM
# fundamentals were correctly spliced from TIMP3, but merge.py's close-price-
# at-filing lookup and this repair both compare against raw `close`, which
# for TIMS3 alone carries the same defect adj_close already gets fixed for
# ("TIMS3's /10000 arrives as two /100 jumps", see below) -- just never
# applied to the non-adj_* columns (confirmed 2026-08-22).
RAW_OHLC_ALSO_UNADJUSTED = {"TIMS3"}

# An event is only detectable when its raw jump ln(1/factor) stands out from
# normal market moves (0.3 ≈ ±35%); the observed return must match it within
# JUMP_MATCH_TOL. The window is wide because corporate_events dates are
# month-granular (most are recorded as the 1st of the month).
MIN_DETECTABLE_JUMP = 0.3
JUMP_MATCH_TOL = 0.15
EVENT_WINDOW_DAYS = (-10, 35)

# ponytail: a persistence guard (reject a matched jump unless the new price
# level actually holds for ~a month afterward, not just the triggering day)
# was investigated 2026-07-24 as defense-in-depth against a coincidental
# market move being mistaken for a split. Two independent designs were tried
# and both produced false rejections against the REAL 67-event dataset: every
# one of BGIP4/CASH3/LUXM4/PATI4/RANI3/SBSP3's matches traces to a genuinely
# recorded corporate_events.parquet entry (confirmed by inspection, incl.
# PATI4's ~annual small bonus-share splits and SBSP3's clustered restructuring
# sequence), not a coincidental move -- ordinary volatility on illiquid/
# small-ratio tickers swamps any window/tolerance loose enough to admit them,
# so no threshold both keeps these and would reject a hypothetical misfire.
# Zero actual misfires have been found in the current dataset (see the audit
# for the full persistence check performed by hand). Not implemented -- would
# add real complexity to already-delicate matching logic for a risk that
# remains theoretical. Revisit only if a future ticker's repair is found to
# have actually misfired.


def repair_unadjusted_splits(prices):
    """Rescale adj_* history where the source left a split/inplit unadjusted.

    corporate_events.parquet is the audit log of all splits. Most are already
    baked into adj_close upstream, but ~45 events are not: the raw jump
    ln(1/factor) shows up verbatim in the daily return (a fake ±90-99.99%
    move that poisons returns, volatility, drawdown and any reward built on
    them). Detect that jump near each recorded event date and divide all
    adj_* history before it by the factor, making the series continuous.

    Also rescales volume and volume_adjusted by the same factor (a 1:4 split
    divides prices by 4 and multiplies volume by 4 — same economic activity,
    more shares). Used by amihud_illiquidity and turnover_ratio features.

    ponytail: events with |ln(1/factor)| < 0.3 can't be told apart from
    market moves and are left alone.

    Events are keyed under each company's ticker at the time of the split.
    Rekey through the continuity map to translate old-name events to new names,
    so that splits recorded under VVAR3 still match BHIA3 rows (after rename chains).
    """
    if not CORPORATE_EVENTS_PATH.exists():
        print("corporate_events.parquet missing — skipping split repair")
        return prices

    ev = pd.read_parquet(CORPORATE_EVENTS_PATH)
    ev = ev[ev["factor"] > 0].copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev[np.abs(np.log(1.0 / ev["factor"])) >= MIN_DETECTABLE_JUMP]

    # Rekey events through continuity map: if a split is recorded under an old ticker
    # (e.g. VVAR3 has a split), add a copy keyed under the new ticker (BHIA3, eventually)
    # so the repair logic can match rows regardless of which name they're under in prices.
    if CONTINUITY_PATH.exists():
        events_map = json.loads(CONTINUITY_PATH.read_text()).get("events", [])
        # Build a ticker-to-all-descendants map: VVAR3 -> [VVAR3, VIIA3, BHIA3]
        # (resolve chains via repeated application)
        descendants = {}
        for e in events_map:
            if e.get("type") not in ("tender", "keep_separate"):
                old, new = e.get("old"), e.get("new")
                if old and new:
                    # VVAR3 -> VIIA3: if VVAR3 had descendants, they now belong to VIIA3
                    if old in descendants:
                        descendants[new] = descendants[old] | {new}
                        del descendants[old]
                    else:
                        descendants[old] = {old}
                    descendants[new] = descendants.get(new, {new}) | {new, old}
        # Duplicate each event keyed under old names to new names
        new_rows = []
        for _, e in ev.iterrows():
            ticker = e.get("ticker")
            if ticker and ticker in descendants:
                for desc_ticker in descendants[ticker]:
                    if desc_ticker != ticker:
                        e_copy = e.copy()
                        e_copy["ticker"] = desc_ticker
                        new_rows.append(e_copy)
        if new_rows:
            ev = pd.concat([ev, pd.DataFrame(new_rows)], ignore_index=True)

    print()
    print("=" * 80)
    print("REPAIRING UNADJUSTED SPLITS IN adj_* PRICES")
    print("=" * 80)

    # Cast volume columns to float up front: the in-place rescale below
    # multiplies a SLICE of these (generally non-integer factor) while the
    # column is still int64, which pandas already warns is deprecated
    # (silently upcasting only that slice) and will be a hard error in a
    # future version. Casting the whole column once here avoids the warning;
    # the final .round().astype("int64") below still converts back after all
    # rescaling is done (share counts round-trip exactly).
    for c in VOLUME_COLS:
        if c in prices.columns:
            prices[c] = prices[c].astype("float64")

    n_fixed = 0
    for ticker, g_ev in ev.groupby("ticker"):
        mask = prices["ticker"] == ticker
        if not mask.any():
            continue
        g_idx = prices.index[mask]  # trade_date-sorted (load_prices sorts)
        adj = prices.loc[g_idx, "adj_close"].to_numpy(dtype=float)
        dates = prices.loc[g_idx, "trade_date"].to_numpy()
        price_cols = ADJ_PRICE_COLS + RAW_PRICE_COLS if ticker in RAW_OHLC_ALSO_UNADJUSTED \
            else ADJ_PRICE_COLS

        # The audit log's factor direction is inconsistent (SBSP3 records 0.2
        # where the observed basis change is x5, ETER3 records 100 for /100),
        # and one event can manifest as several re-anchoring steps days apart
        # (TIMS3's /10000 arrives as two /100 jumps). So: match the jump in
        # BOTH directions, always repair the EARLIEST unrepaired jump first,
        # and rescan until the ticker's windows are clean.
        applied = set()
        for _ in range(2 * len(g_ev) + 2):  # bound: each pass fixes a new day
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.log(adj[1:] / adj[:-1])
            best = None  # (jump_row, factor)
            for _, e in g_ev.iterrows():
                lo = np.datetime64(e["date"] + pd.Timedelta(days=EVENT_WINDOW_DAYS[0]))
                hi = np.datetime64(e["date"] + pd.Timedelta(days=EVENT_WINDOW_DAYS[1]))
                win = (dates[1:] >= lo) & (dates[1:] <= hi)
                for factor in (e["factor"], 1.0 / e["factor"]):
                    expected = np.log(1.0 / factor)
                    cand = np.where(win & (np.abs(lr - expected) < JUMP_MATCH_TOL))[0]
                    for c in cand:
                        jump = c + 1  # first row already on the post-event scale
                        if dates[jump] in applied:
                            continue
                        if best is None or jump < best[0]:
                            best = (jump, factor)
                        break
            if best is None:
                break  # all windows clean — the normal case is zero passes
            jump, factor = best
            applied.add(dates[jump])
            prices.loc[g_idx[:jump], price_cols] /= factor
            # volume scales OPPOSITE to price: 1:4 split divides price by 4,
            # multiplies volume by 4 (same economic activity, more shares
            # outstanding trading it) so that volume*price (dollar volume)
            # stays invariant across the splice -- same invariant
            # continuity.py's merger-ratio volume scaling preserves.
            vol_cols_present = [c for c in VOLUME_COLS if c in prices.columns]
            if vol_cols_present:
                prices.loc[g_idx[:jump], vol_cols_present] *= factor
            adj[:jump] /= factor
            n_fixed += 1
            print(f"  {ticker} {pd.Timestamp(dates[jump]).date()}: rescaled "
                  f"{jump} rows before factor-{factor:g} basis change")

    # volume is a share count -- round back to int so the /factor divisions
    # above don't silently upcast the whole column to float (share counts
    # round-trip exactly; the /factor is always a clean split ratio).
    for c in VOLUME_COLS:
        if c in prices.columns:
            prices[c] = prices[c].round().astype("int64")

    print(f"Repaired {n_fixed} unadjusted events")
    return prices


def repair_isolated_adj_close_glitches(prices):
    """Snap back a single-day adj_close value that's inconsistent with both of
    its neighbors, while raw close is fine.

    Root-caused 2026-08-21 (docs/DATA_LAYER_FOLLOWUP_FINDINGS.md, and
    TOP50_ML_READINESS_AUDIT.md's PETR4/ITUB4/SBSP3/BBDC4/BBAS3/ITSA4/GGBR4/
    GOAU4/VIVT3 findings): a vendor batch defect corrupts adj_close for one
    day on two specific calendar dates (2012-02-22, 2020-11-20) -- confirmed
    dataset-wide (92 rows / 58 tickers, unrelated sectors, no matching
    corporate_events entry), not a per-company corporate action.

    Detection works on ratio = adj_close/close instead of the raw return:
    on an ordinary day (even a huge real price move) ratio is flat, because
    close and adj_close move together. A REAL split changes ratio once and
    it STAYS changed. Only a data glitch makes ratio jump on day D and land
    back near its day-(D-1) level by day D+1.

    That's not quite airtight on its own, though (found 2026-08-21, AFLT3):
    if raw `close` itself has an isolated one-day error (unrelated to any
    split) sitting right after `repair_unadjusted_splits` has ALREADY made
    adj_close continuous, the ratio pulses and reverts for exactly the same
    shape -- but here `close` is the broken side, and "fixing" adj_close
    would overwrite an already-correct value with a wrong one, reintroducing
    the very discontinuity `repair_unadjusted_splits` just removed. So also
    require `close` itself to stay flat across the window: a real glitch
    only ever shows up in one side of the ratio, never both.

    Repair: hold the prior day's (confirmed-good) ratio through the bad row.
    """
    print()
    print("=" * 80)
    print("REPAIRING ISOLATED SINGLE-DAY adj_close GLITCHES")
    print("=" * 80)

    n_fixed = 0
    for ticker, g in prices.groupby("ticker"):
        if len(g) < 3:
            continue
        idx = g.index.to_numpy()
        close = g["close"].to_numpy(dtype=float)
        adj_close = g["adj_close"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(close > 0, adj_close / close, np.nan)
            lr_in = np.log(ratio[1:-1] / ratio[:-2])
            lr_revert = np.log(ratio[2:] / ratio[:-2])
            close_lr_in = np.log(close[1:-1] / close[:-2])
            close_lr_revert = np.log(close[2:] / close[:-2])
        glitch = (np.isfinite(lr_in) & np.isfinite(lr_revert)
                  & np.isfinite(close_lr_in) & np.isfinite(close_lr_revert)
                  & (np.abs(lr_in) >= MIN_DETECTABLE_JUMP) & (np.abs(lr_revert) < JUMP_MATCH_TOL)
                  & (np.abs(close_lr_in) < MIN_DETECTABLE_JUMP) & (np.abs(close_lr_revert) < MIN_DETECTABLE_JUMP))
        for r in np.where(glitch)[0] + 1:  # +1: back to the middle (glitch) row
            good_ratio = ratio[r - 1]
            for raw_col, adj_col in (("open", "adj_open"), ("high", "adj_high"),
                                      ("low", "adj_low"), ("close", "adj_close")):
                prices.loc[idx[r], adj_col] = g[raw_col].iloc[r] * good_ratio
            n_fixed += 1
            print(f"  {ticker} {pd.Timestamp(g['trade_date'].iloc[r]).date()}: "
                  f"repaired isolated adj_close glitch")

    print(f"Repaired {n_fixed} isolated adj_close glitches")
    return prices
