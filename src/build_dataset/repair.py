"""repair.py — rescale adj_* price history where a split/inplit was left unadjusted."""

import json
import numpy as np
import pandas as pd

from .paths import CORPORATE_EVENTS_PATH, CONTINUITY_PATH

ADJ_PRICE_COLS = ["adj_open", "adj_high", "adj_low", "adj_close"]
VOLUME_COLS = ["volume", "volume_adjusted"]

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
            prices.loc[g_idx[:jump], ADJ_PRICE_COLS] /= factor
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
