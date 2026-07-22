"""
Test: conviction_model/run_diagnostics.py's GATES dict (Phase 1's quantitative gate
table) -- pure lambda functions over plain numbers, no data dependency, so these are
tested directly here despite run_diagnostics.py itself being the real-checkpoint
report script (no other diagN_* function has a test file -- see
docs/conviction_model/REVIEW_REMEDIATION_PLAN.md Phase 1 for why).

Focuses on gates 2 and 6, which got an added practical effect-size floor (review
finding 10: a permutation-null/p-value test alone can't tell "significant" apart from
"significant but negligible" -- the real Stage 1A run cleared both gates on
magnitudes that read as near-zero: MI=0.0006 vs null_p95=0.0002, corr=0.14).

Run from project root:
    python tests/conviction_model/test_run_diagnostics.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.conviction_model.run_diagnostics import GATES, MIN_REGIME_MI, MIN_SMOOTHNESS_CORR  # noqa: E402
from test_utils import print_check, print_header, print_section_end  # noqa: E402


def test_gate2_fails_below_magnitude_floor_even_if_significant(passed, failed):
    thresh = 0.0002
    mi_below = MIN_REGIME_MI - 0.005  # clears significance (mi > thresh) but not the floor
    ok = not GATES[2](mi_below, thresh)
    print_check("GATES[2]: fails when MI clears the permutation-null threshold but is below "
                "the practical-magnitude floor", ok, f"mi={mi_below}, thresh={thresh}, floor={MIN_REGIME_MI}")
    return passed + ok, failed + (not ok)


def test_gate2_passes_above_both_bars(passed, failed):
    thresh = 0.0002
    mi_above = MIN_REGIME_MI + 0.01
    ok = GATES[2](mi_above, thresh)
    print_check("GATES[2]: passes when MI clears BOTH the null threshold and the magnitude floor",
                ok, f"mi={mi_above}, thresh={thresh}, floor={MIN_REGIME_MI}")
    return passed + ok, failed + (not ok)


def test_gate2_fails_when_not_significant_even_above_floor(passed, failed):
    # Magnitude alone isn't enough either -- must still clear the null threshold.
    mi = MIN_REGIME_MI + 0.01
    thresh = mi + 0.005  # null threshold higher than the observed MI -- not significant
    ok = not GATES[2](mi, thresh)
    print_check("GATES[2]: fails when MI is above the magnitude floor but does not clear "
                "the null threshold", ok, f"mi={mi}, thresh={thresh}")
    return passed + ok, failed + (not ok)


def test_gate6_fails_below_correlation_floor_even_if_significant(passed, failed):
    corr_below = MIN_SMOOTHNESS_CORR - 0.02
    ok = not GATES[6](corr_below, 0.001)  # p well below 0.05
    print_check("GATES[6]: fails when corr is significant (p<0.05) but below the practical "
                "effect-size floor", ok, f"corr={corr_below}, p=0.001, floor={MIN_SMOOTHNESS_CORR}")
    return passed + ok, failed + (not ok)


def test_gate6_passes_above_both_bars(passed, failed):
    corr_above = MIN_SMOOTHNESS_CORR + 0.05
    ok = GATES[6](corr_above, 0.001)
    print_check("GATES[6]: passes when corr clears BOTH significance and the magnitude floor",
                ok, f"corr={corr_above}, p=0.001, floor={MIN_SMOOTHNESS_CORR}")
    return passed + ok, failed + (not ok)


def test_gate6_fails_when_not_significant_even_above_floor(passed, failed):
    corr_above = MIN_SMOOTHNESS_CORR + 0.05
    ok = not GATES[6](corr_above, 0.2)  # p above 0.05 -- not significant
    print_check("GATES[6]: fails when corr is above the magnitude floor but not significant (p>=0.05)",
                ok, f"corr={corr_above}, p=0.2")
    return passed + ok, failed + (not ok)


def main() -> int:
    print_header("conviction_model/run_diagnostics.py (GATES)")
    passed = failed = 0
    for test_fn in [
        test_gate2_fails_below_magnitude_floor_even_if_significant,
        test_gate2_passes_above_both_bars,
        test_gate2_fails_when_not_significant_even_above_floor,
        test_gate6_fails_below_correlation_floor_even_if_significant,
        test_gate6_passes_above_both_bars,
        test_gate6_fails_when_not_significant_even_above_floor,
    ]:
        passed, failed = test_fn(passed, failed)
    print_section_end(passed, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
