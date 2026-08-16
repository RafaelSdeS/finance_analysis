"""test_terminal_events.py -- synthetic checks for terminal_events.py's
taxonomy/payoff logic and its wiring into labels.forward_excess_return.

Fast group (synthetic only). Run: python tests/portfolio/test_terminal_events.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.build_dataset.terminal_events import build_terminal_events, find_rename_candidates  # noqa: E402
from src.portfolio.labels import forward_excess_return  # noqa: E402
from tests.test_utils import print_check, print_header, print_section_end  # noqa: E402

H = 3
N_DEAD = 20   # rows for each ticker that stops trading early
N_LIVE = 900  # rows for the ticker that defines the panel's end date


def _panel():
    """LIVE trades the whole span and defines the panel end. FAIL/ACQD/GONE
    all stop after N_DEAD rows -- far enough before LIVE's last date to
    clear STALE_TICKER_DAYS (730), so all three count as 'dead inside the
    panel'. GONE shares LIVE's cnpj and is registered ATIVO (an unspliced
    rename, not a real delisting)."""
    dates = pd.bdate_range("2015-01-01", periods=N_LIVE)

    def _rows(ticker, n, base_price):
        d = dates[:n]
        return pd.DataFrame({
            "ticker": ticker, "trade_date": d,
            "adj_close": [base_price + i for i in range(n)],
            "cdi": 0.01, "adj_close_precision_degraded": 0,
        })

    live = _rows("LIVE", N_LIVE, 100.0)
    fail = _rows("FAIL", N_DEAD, 50.0)   # goes to 0 -- failure
    acqd = _rows("ACQD", N_DEAD, 80.0)   # taken out at its own last price
    gone = _rows("GONE", N_DEAD, 30.0)   # unspliced rename, no payoff

    return pd.concat([live, fail, acqd, gone], ignore_index=True)


def _delist_events():
    return pd.DataFrame({
        "ticker": ["FAIL", "ACQD", "GONE"],
        "cnpj": ["111", "222", "333"],
        "delist_date": pd.to_datetime(["2015-02-01", "2015-02-01", "2015-02-01"]),
        "motivo_cancel": ["LIQUIDAÇÃO EXTRAJUDICIAL", "CANCELAMENTO VOLUNTÁRIO", None],
        "sit": ["CANCELADA", "CANCELADA", "ATIVO"],
    })


def main():
    print_header("test_terminal_events")
    passed = failed = 0

    df = _panel()
    delist_events = _delist_events()
    # LIVE shares GONE's cnpj -- the rename candidate this section checks for.
    delist_events_with_live = pd.concat([
        delist_events,
        pd.DataFrame({"ticker": ["LIVE"], "cnpj": ["333"], "delist_date": [pd.NaT],
                      "motivo_cancel": [None], "sit": ["ATIVO"]}),
    ], ignore_index=True)

    # --- build_terminal_events --------------------------------------------
    events = build_terminal_events(df, delist_events_with_live)

    ok = set(events["ticker"]) == {"FAIL", "ACQD"}
    print_check("only resolvable, non-ATIVO dead tickers get a terminal event", ok,
                f"got {sorted(events['ticker'])}")
    passed, failed = passed + ok, failed + (not ok)

    fail_row = events[events["ticker"] == "FAIL"].iloc[0]
    ok = fail_row["event_type"] == "failure" and fail_row["terminal_payoff"] == 0.0
    print_check("bankruptcy/liquidation reason -> event_type=failure, payoff=0.0", ok)
    passed, failed = passed + ok, failed + (not ok)

    acqd_last_close = 80.0 + N_DEAD - 1
    acqd_row = events[events["ticker"] == "ACQD"].iloc[0]
    ok = (acqd_row["event_type"] == "acquired"
          and np.isclose(acqd_row["terminal_payoff"], acqd_last_close))
    print_check("other cancellation reason -> event_type=acquired, payoff=last adj_close", ok,
                f"got {acqd_row['terminal_payoff']}, expected {acqd_last_close}")
    passed, failed = passed + ok, failed + (not ok)

    # --- find_rename_candidates ---------------------------------------------
    candidates = find_rename_candidates(df, delist_events_with_live)
    ok = (len(candidates) == 1 and candidates.iloc[0]["ticker_old"] == "GONE"
          and candidates.iloc[0]["ticker_new"] == "LIVE")
    print_check("ATIVO-but-dead ticker surfaces as a rename candidate against its live sibling",
                ok, f"got {candidates.to_dict('records')}")
    passed, failed = passed + ok, failed + (not ok)

    # --- forward_excess_return wiring ---------------------------------------
    label_before = forward_excess_return(df, horizon_td=H)
    label_after = forward_excess_return(df, horizon_td=H, terminal_events=events)

    def _tail_idx(ticker, n):
        return df.index[(df["ticker"] == ticker)][-H:]

    cdi_gross = 1.0001  # (1 + 0.01/100)

    # FAIL: last H rows filled, payoff=0 -> fwd_ret=-1 exactly, cdi compounds
    # only to FAIL's own last row (not a full H days).
    fail_idx = _tail_idx("FAIL", N_DEAD)
    days_to_end = np.array([N_DEAD - 1 - pos for pos in range(N_DEAD - H, N_DEAD)])
    expected_fail = -1.0 - (cdi_gross ** days_to_end - 1.0)
    got_fail = label_after.loc[fail_idx].to_numpy()
    ok = np.allclose(got_fail, expected_fail, atol=1e-8)
    print_check("FAIL tail label = -1 - CDI compounded only to its own last date", ok,
                f"got {got_fail}, expected {expected_fail}")
    passed, failed = passed + ok, failed + (not ok)

    # ACQD: last row's label collapses to exactly 0 (payoff == its own last
    # close, zero days of forward CDI).
    acqd_idx = _tail_idx("ACQD", N_DEAD)
    ok = np.isclose(label_after.loc[acqd_idx[-1]], 0.0, atol=1e-8)
    print_check("ACQD's own last row -> label = 0 (payoff = last observed price)", ok,
                f"got {label_after.loc[acqd_idx[-1]]}")
    passed, failed = passed + ok, failed + (not ok)

    # GONE: not in the terminal_events table at all -> completely unaffected.
    gone_idx = df.index[df["ticker"] == "GONE"]
    ok = label_after.loc[gone_idx].equals(label_before.loc[gone_idx])
    print_check("GONE (no terminal payoff, still ATIVO) is untouched by terminal_events", ok)
    passed, failed = passed + ok, failed + (not ok)

    # LIVE: no terminal event either -> untouched, including its own natural
    # tail NaNs (real leakage boundary, must never be imputed).
    live_idx = df.index[df["ticker"] == "LIVE"]
    ok = label_after.loc[live_idx].equals(label_before.loc[live_idx])
    print_check("LIVE ticker is completely unaffected by terminal_events", ok)
    passed, failed = passed + ok, failed + (not ok)

    # Non-tail rows for a dead ticker must be untouched: with H=3 and 20 rows,
    # rows 0..16 have a real forward window inside the ticker's own history.
    non_tail_idx = df.index[(df["ticker"] == "FAIL")][:N_DEAD - H]
    ok = label_after.loc[non_tail_idx].equals(label_before.loc[non_tail_idx])
    print_check("non-tail rows of a dead ticker are unaffected (only the true tail changes)", ok)
    passed, failed = passed + ok, failed + (not ok)

    print_section_end(passed, failed)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
