"""continuity.py — splice renamed/merged tickers into their surviving series."""

import json

import pandas as pd

from .paths import CONTINUITY_PATH


def apply_ticker_continuity(prices, fundamentals, path=CONTINUITY_PATH):
    """Splice renamed/merged tickers into their surviving series.

    Event types (data/raw/reference/ticker_continuity.json, hand-maintained):
      rename: same legal entity under a new ticker — splice prices AND
              fundamentals (history is genuinely continuous).
      merger: the old ticker's entity ceased to exist — splice prices only
              (shareholder-return continuity via the exchange ratio); its
              fundamentals are dropped, so pre-boundary rows show
              has_fundamentals=0 downstream like any early-history gap.
      tender: cash-out, nothing to splice (terminal-event payoff handling is
              a separate planned task — see DELISTED_UNIVERSE.md).

    Events apply in date order so chains (VVAR3->VIIA3->BHIA3) resolve. The
    splice boundary is the NEW ticker's actual first trade date (ground
    truth), not the documented event date — approximate dates in the map are
    harmless.
    """
    if not path.exists():
        return prices, fundamentals

    events = sorted(json.loads(path.read_text())["events"], key=lambda e: e["date"])

    print()
    print("=" * 80)
    print("TICKER CONTINUITY (renames / mergers)")
    print("=" * 80)

    prices = prices.copy()
    fundamentals = fundamentals.copy()
    price_cols = [c for c in ("open", "high", "low", "close",
                              "adj_open", "adj_high", "adj_low", "adj_close")
                  if c in prices.columns]

    # boundaries from the PRISTINE input: a leg spliced by an earlier event must
    # not shift a later event's boundary (two legs onto one ticker would silently
    # swallow the second leg instead of tripping the duplicate guard below)
    first_trade = prices.groupby("ticker")["trade_date"].min()
    first_fund = fundamentals.groupby("ticker")["reference_date"].min()

    for ev in events:
        old, new, kind = ev["old"], ev["new"], ev["type"]
        ratio = float(ev.get("ratio", 1.0))
        if kind == "tender" or not (prices["ticker"] == old).any():
            continue

        boundary = first_trade.get(new)
        if boundary is not None:
            # no date overlap: past the boundary the new ticker is the record
            prices = prices[~((prices["ticker"] == old) & (prices["trade_date"] >= boundary))]
        old_rows = prices["ticker"] == old
        prices.loc[old_rows, price_cols] = prices.loc[old_rows, price_cols] * ratio
        prices.loc[old_rows, "ticker"] = new
        # volume stays unscaled: share counts change meaning across an exchange
        # ratio, and all volume features downstream are per-ticker relative

        f_old = fundamentals["ticker"] == old
        if kind == "rename":
            f_boundary = first_fund.get(new)
            if f_boundary is not None and pd.notna(f_boundary):
                fundamentals = fundamentals[~(f_old & (fundamentals["reference_date"] >= f_boundary))]
                f_old = fundamentals["ticker"] == old
            fundamentals.loc[f_old, "ticker"] = new
        else:  # merger: the acquired entity's books are not the survivor's
            fundamentals = fundamentals[~f_old]
        print(f"  {kind}: {old} -> {new}" + (f" (ratio {ratio})" if ratio != 1.0 else ""))

    dup = prices.duplicated(subset=["ticker", "trade_date"]).sum()
    if dup:
        raise ValueError(f"ticker continuity produced {dup} duplicate ticker+date "
                         f"price rows — two legs mapped to the same ticker? Check the map.")
    prices = prices.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
    fundamentals = fundamentals.sort_values(["ticker", "reference_date"]).reset_index(drop=True)
    return prices, fundamentals
