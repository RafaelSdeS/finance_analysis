"""
Test: h_series/milestone_h1.py's sector-neutral IC screen and kill-gate --
the direct regression test for the user's addendum #1 (MEDIUM_HORIZON_
RESEARCH_PLAN.md sec H1): a characteristic that is PURELY a sector tilt
must be killed by sector-neutralization even if its raw IC is strong, and
a characteristic carrying real stock-specific signal must survive it even
when a dominant sector tilt buries its RAW correlation with the target.

Synthetic panel, deterministic seed. No parquet IO -- CHARACTERISTIC_COLUMNS
is monkeypatched to two fabricated columns for the duration of the test.

Run from project root:
    python tests/h_series/test_milestone_h1.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import src.h_series.milestone_h1 as h1  # noqa: E402
from src.h_series.stats import rank_normalize, sector_demean  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def _synthetic_panel() -> pd.DataFrame:
    """3 sectors x 12 tickers, 24 monthly dates.

    target = stock_effect (real, per-ticker) + sector_offset (a large,
             purely sector-level tilt) + small noise
    real_char = stock_effect + small independent noise -- correlated with
             target ONLY through the stock-specific part, which is a small
             fraction of target's total variance (sector_offset dominates)
    sector_char = sector_offset EXACTLY (zero within-sector variance) --
             correlated with target ONLY through the sector tilt

    Expected: real_char's RAW IC is weak (buried under the sector tilt)
    but its SECTOR-NEUTRAL IC is strong (~1.0, real signal revealed).
    sector_char's RAW IC is strong (rides the same sector tilt as target)
    but its SECTOR-NEUTRAL IC is undefined/zero (a within-sector-constant
    value has zero variance left after demeaning -- exactly the "sector
    timing bet in disguise" the gate must kill).
    """
    rng = np.random.default_rng(42)
    sectors = ["S1", "S2", "S3"]
    n_per_sector = 12
    sector_offset = {"S1": 3.0, "S2": 0.0, "S3": -3.0}
    stock_effect = np.linspace(-1.0, 1.0, n_per_sector)

    ticker_sector, ticker_effect = {}, {}
    tickers = []
    for s in sectors:
        for i in range(n_per_sector):
            tk = f"{s}_{i}"
            tickers.append(tk)
            ticker_sector[tk] = s
            ticker_effect[tk] = stock_effect[i]

    decision_dates = pd.date_range("2015-01-31", periods=24, freq="ME")
    rows = []
    for d in decision_dates:
        for tk in tickers:
            s = ticker_sector[tk]
            target = ticker_effect[tk] + sector_offset[s] + rng.normal(0, 0.02)
            real_char = ticker_effect[tk] + rng.normal(0, 0.02)
            sector_char = sector_offset[s]
            rows.append({"decision_date": d, "ticker": tk, "sector": s,
                         "target": target, "real_char": real_char, "sector_char": sector_char})
    panel = pd.DataFrame(rows)

    panel["real_char_sector_neutral"] = sector_demean(panel["real_char"], panel["decision_date"], panel["sector"])
    panel["sector_char_sector_neutral"] = sector_demean(panel["sector_char"], panel["decision_date"], panel["sector"])
    panel["fwd_rel_return_k21"] = panel["target"]
    panel["target_rank_k21"] = rank_normalize(panel["target"], panel["decision_date"])
    # Mirrors features.build_monthly_panel's fix: the sector-neutral variant's target
    # must ALSO be sector-demeaned, or a purely stock-specific characteristic is measured
    # against a target that still carries the sector tilt (structural attenuation).
    panel["fwd_rel_return_sector_neutral_k21"] = sector_demean(
        panel["target"], panel["decision_date"], panel["sector"])
    panel["target_rank_sector_neutral_k21"] = rank_normalize(
        panel["fwd_rel_return_sector_neutral_k21"], panel["decision_date"])
    return panel


def _screen_and_gate(panel: pd.DataFrame) -> tuple:
    original = h1.CHARACTERISTIC_COLUMNS
    h1.CHARACTERISTIC_COLUMNS = ("real_char", "sector_char")
    try:
        screen = h1.screen_characteristics(panel, k_horizons=(21,))
        gated = h1.apply_gate(screen)
    finally:
        h1.CHARACTERISTIC_COLUMNS = original
    return screen, gated


def test_sector_char_constant_within_sector_is_fully_demeaned(passed, failed):
    panel = _synthetic_panel()
    ok = bool((panel["sector_char_sector_neutral"].abs() < 1e-9).all())
    print_check("synthetic fixture sanity: sector_char is EXACTLY 0 after sector-demeaning "
                "(no within-sector variance to begin with)", ok)
    return passed + ok, failed + (not ok)


def test_real_signal_survives_sector_neutralization(passed, failed):
    panel = _synthetic_panel()
    screen, gated = _screen_and_gate(panel)
    real_sn = screen[(screen.characteristic == "real_char") & (screen.variant == "sector_neutral")].iloc[0]
    real_passes = bool(gated.loc[gated.characteristic == "real_char", "passes_gate"].any())
    ok = real_passes and abs(real_sn["tstat"]) >= 2.0 and abs(real_sn["mean_ic"]) > 0.5
    print_check("gate: real stock-specific signal (buried under a dominant sector tilt in RAW form) "
                "PASSES after sector-neutralization",
                ok, f"sector_neutral tstat={real_sn['tstat']:.2f}, mean_ic={real_sn['mean_ic']:.3f}")
    return passed + ok, failed + (not ok)


def test_pure_sector_tilt_is_killed_by_sector_neutralization(passed, failed):
    panel = _synthetic_panel()
    screen, gated = _screen_and_gate(panel)
    sector_raw = screen[(screen.characteristic == "sector_char") & (screen.variant == "raw")].iloc[0]
    sector_sn = screen[(screen.characteristic == "sector_char") & (screen.variant == "sector_neutral")].iloc[0]
    sector_passes = bool(gated.loc[gated.characteristic == "sector_char", "passes_gate"].any())

    raw_is_strong = abs(sector_raw["tstat"]) >= 2.0 or np.isnan(sector_raw["tstat"])  # rides the sector tilt
    sn_is_dead = np.isnan(sector_sn["tstat"]) or abs(sector_sn["tstat"]) < 2.0

    ok = (not sector_passes) and sn_is_dead
    print_check("gate: a PURE sector tilt (significant RAW IC) is KILLED once sector-neutralized "
                "-- never reaches the stock-selection composite",
                ok, f"raw tstat={sector_raw['tstat']}, sector_neutral tstat={sector_sn['tstat']}, "
                    f"raw_is_strong={raw_is_strong}")
    return passed + ok, failed + (not ok)


def test_gate_never_uses_raw_variant_to_pass(passed, failed):
    panel = _synthetic_panel()
    screen, gated = _screen_and_gate(panel)
    ok = bool((gated["variant"] == "sector_neutral").all())
    print_check("apply_gate: gate table contains ONLY the sector_neutral variant, "
                "raw is never eligible to pass regardless of significance", ok)
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("h_series/milestone_h1.py (sector-neutral gate regression)")
    passed = failed = 0
    for test_fn in [
        test_sector_char_constant_within_sector_is_fully_demeaned,
        test_real_signal_survives_sector_neutralization,
        test_pure_sector_tilt_is_killed_by_sector_neutralization,
        test_gate_never_uses_raw_variant_to_pass,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
