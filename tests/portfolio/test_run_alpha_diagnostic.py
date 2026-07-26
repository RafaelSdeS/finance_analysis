"""
test_run_alpha_diagnostic.py -- checks for run_alpha_diagnostic.make_alpha_weighted_fn's
no-trade band (plan Phase 1.1): a name already held should survive a rank
dip that a fresh buy wouldn't clear.

Fast group (synthetic only). Run: python tests/portfolio/test_run_alpha_diagnostic.py
"""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.portfolio import run_alpha_diagnostic  # noqa: E402
from src.portfolio.run_alpha_diagnostic import make_alpha_weighted_fn, window_bounds  # noqa: E402
from tests.test_utils import print_check, print_header, print_section_end  # noqa: E402


def _check_window_bounds(passed, failed):
    """window_bounds (plan Phase V.0c) against a fabricated split_config.json --
    monkeypatches the module's SPLIT_CONFIG_PATH so this stays fast/synthetic,
    not a dependency on the real (gitignored, locally-built) split file."""
    tmp = Path(tempfile.mkdtemp()) / "split_config.json"
    tmp.write_text(json.dumps({"train_end": "2018-07-30", "val_end": "2022-07-26"}))
    orig_path = run_alpha_diagnostic.SPLIT_CONFIG_PATH
    run_alpha_diagnostic.SPLIT_CONFIG_PATH = tmp
    try:
        full_ok = window_bounds("full") == (None, None)
        print_check("window='full' returns (None, None) -- unrestricted, today's original behavior",
                    bool(full_ok), f"got {window_bounds('full')}")
        passed, failed = passed + full_ok, failed + (not full_ok)

        train_ok = window_bounds("train") == (pd.Timestamp("2018-07-30"), None)
        print_check("window='train' truncates input at train_end, doesn't restrict eval",
                    bool(train_ok), f"got {window_bounds('train')}")
        passed, failed = passed + train_ok, failed + (not train_ok)

        trainval_ok = window_bounds("trainval") == (pd.Timestamp("2022-07-26"), None)
        print_check("window='trainval' truncates input at val_end",
                    bool(trainval_ok), f"got {window_bounds('trainval')}")
        passed, failed = passed + trainval_ok, failed + (not trainval_ok)

        test_ok = window_bounds("test") == (None, pd.Timestamp("2022-07-26"))
        print_check("window='test' does NOT truncate input, only restricts eval to > val_end",
                    bool(test_ok), f"got {window_bounds('test')}")
        passed, failed = passed + test_ok, failed + (not test_ok)

        raised = False
        try:
            window_bounds("bogus")
        except ValueError:
            raised = True
        print_check("an unknown window name raises, doesn't silently fall through to full", raised)
        passed, failed = passed + raised, failed + (not raised)
    finally:
        run_alpha_diagnostic.SPLIT_CONFIG_PATH = orig_path
    return passed, failed


def main():
    print_header("test_run_alpha_diagnostic")
    passed = failed = 0

    # 10 tickers ranked T0 (best) .. T9 (worst) by alpha.
    uni = {f"T{i}" for i in range(10)}
    preds = {f"T{i}": 1.0 - i * 0.1 for i in range(10)}

    # top_frac=0.5 -> buy top 5 (T0-T4); hold_frac=0.7 -> once held, stay in top 7 (T0-T6).
    fn = make_alpha_weighted_fn({"d1": preds, "d2": preds}, top_frac=0.5, hold_frac=0.7)

    w1 = fn("d1", uni, {"prev_weights": {}})
    entry_ok = set(w1) == {f"T{i}" for i in range(5)}
    print_check("first rebalance buys exactly the top_frac cut (no prior holdings)",
                bool(entry_ok), f"got {sorted(w1)}")
    passed, failed = passed + entry_ok, failed + (not entry_ok)

    # T4 (rank 4, at the buy edge) is already held; T5/T6 are NOT held and
    # rank just outside the buy cut but inside the hold band -- the band
    # only protects names ALREADY in the book, it doesn't admit new ones.
    w2 = fn("d2", uni, {"prev_weights": w1})
    band_ok = set(w2) == {f"T{i}" for i in range(5)}
    print_check("unchanged ranks: no-trade band doesn't add T5/T6 just because they're inside hold_frac",
                bool(band_ok), f"got {sorted(w2)}")
    passed, failed = passed + band_ok, failed + (not band_ok)

    # T4 slips to rank 6 (still inside hold_frac=0.7 -> top 7) -- a held name
    # must SURVIVE this, where a hard top_frac-only cutoff would sell it.
    preds_dip = {**preds, "T4": preds["T6"] - 0.001}  # T4 now ranks just below T6
    fn_dip = make_alpha_weighted_fn({"d1": preds, "d2": preds_dip}, top_frac=0.5, hold_frac=0.7)
    w1_dip = fn_dip("d1", uni, {"prev_weights": {}})
    w2_dip = fn_dip("d2", uni, {"prev_weights": w1_dip})
    survives_ok = "T4" in w2_dip
    print_check("a held name that dips to rank 6/10 (inside hold_frac) is NOT sold",
                bool(survives_ok), f"got {sorted(w2_dip)}")
    passed, failed = passed + survives_ok, failed + (not survives_ok)

    # A held name that dips PAST hold_frac (e.g. to rank 8/10, outside top 7)
    # must be sold -- the band is a slack, not infinite tolerance.
    preds_crash = {**preds, "T4": -5.0}  # T4 collapses to worst rank
    fn_crash = make_alpha_weighted_fn({"d1": preds, "d2": preds_crash}, top_frac=0.5, hold_frac=0.7)
    w1_crash = fn_crash("d1", uni, {"prev_weights": {}})
    w2_crash = fn_crash("d2", uni, {"prev_weights": w1_crash})
    sold_ok = "T4" not in w2_crash
    print_check("a held name that drops outside hold_frac IS sold", bool(sold_ok),
                f"got {sorted(w2_crash)}")
    passed, failed = passed + sold_ok, failed + (not sold_ok)

    # hold_frac=None reproduces the pre-band (top_frac-only) behavior exactly.
    fn_noband = make_alpha_weighted_fn({"d1": preds, "d2": preds_dip}, top_frac=0.5)
    w1_nb = fn_noband("d1", uni, {"prev_weights": {}})
    w2_nb = fn_noband("d2", uni, {"prev_weights": w1_nb})
    noband_ok = "T4" not in w2_nb  # T4 dipped below top_frac and there's no slack to save it
    print_check("hold_frac=None (default) has no slack -- a dip below top_frac sells immediately",
                bool(noband_ok), f"got {sorted(w2_nb)}")
    passed, failed = passed + noband_ok, failed + (not noband_ok)

    # Weights still sum to 1 and are equal-weighted among the chosen set.
    equal_w_ok = all(abs(v - 1 / len(w2_dip)) < 1e-9 for v in w2_dip.values())
    print_check("chosen names are equal-weighted", bool(equal_w_ok))
    passed, failed = passed + equal_w_ok, failed + (not equal_w_ok)

    # exposure_by_date (plan Phase 3.1c): scales the chosen set down, residual
    # falls through to cash (weights sum to exposure, not 1.0).
    fn_exposed = make_alpha_weighted_fn({"d1": preds}, top_frac=0.5, exposure_by_date={"d1": 0.7})
    w_exposed = fn_exposed("d1", uni, {"prev_weights": {}})
    exposure_scaled_ok = abs(sum(w_exposed.values()) - 0.7) < 1e-9
    print_check("exposure_by_date scales the chosen set's weights to sum to the cap, not 1.0",
                bool(exposure_scaled_ok), f"sum={sum(w_exposed.values()):.4f}")
    passed, failed = passed + exposure_scaled_ok, failed + (not exposure_scaled_ok)

    # A date missing from exposure_by_date defaults to full exposure (1.0) --
    # same "no data yet" convention as the rest of the pipeline, not a silent
    # forced-to-cash surprise.
    fn_exposed_missing = make_alpha_weighted_fn({"d1": preds}, top_frac=0.5, exposure_by_date={"d2": 0.7})
    w_missing = fn_exposed_missing("d1", uni, {"prev_weights": {}})
    default_full_ok = abs(sum(w_missing.values()) - 1.0) < 1e-9
    print_check("a date missing from exposure_by_date defaults to full (1.0) exposure",
                bool(default_full_ok), f"sum={sum(w_missing.values()):.4f}")
    passed, failed = passed + default_full_ok, failed + (not default_full_ok)

    passed, failed = _check_window_bounds(passed, failed)

    print_section_end(passed, failed)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
