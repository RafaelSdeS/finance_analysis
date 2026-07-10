"""
Unit tests for data_pipeline.cross_sectional_transform (synthetic, no data files).

DIAGNOSIS_PLAN.md Phase 2.3: the existing scaler is a per-window GLOBAL mean/std, fit once
on a 10+ year train span and applied unchanged to every date in it. A feature's typical
level can drift over a decade of regime change, and the decision the policy needs is
cross-sectional ("which stock TODAY"), so global scaling lets that drift swamp the
cross-sectional signal. These tests demonstrate the problem on a synthetic panel (a global
scaler's output for one date changes when the WHOLE MARKET shifts level that day, even
though relative rankings didn't change at all) and verify the solution's properties
(within-date only, active-names only, no leakage, exactly shift-invariant).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agent.data_pipeline import cross_sectional_transform


def _global_zscore(feat_tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The OLD approach this transform is contrasted against: one mean/std per feature,
    computed across ALL dates/tickers (mimicking data_pipeline.fit_train_scaler), applied
    uniformly everywhere -- reused here only to demonstrate the problem in Test 3."""
    valid = mask[:, :, None] & ~np.isnan(feat_tensor)
    mean = np.nanmean(np.where(valid, feat_tensor, np.nan), axis=(0, 1))
    std = np.nanstd(np.where(valid, feat_tensor, np.nan), axis=(0, 1))
    std = np.where(std < 1e-8, 1.0, std)
    return (feat_tensor - mean) / std


def main() -> None:
    print("Test 1: z-score correctness (hand-verifiable)")
    D, N, F = 3, 5, 1
    feat = np.zeros((D, N, F), dtype=np.float32)
    feat[0, :, 0] = [1.0, 2.0, 3.0, 4.0, 5.0]  # mean=3, std=sqrt(2)
    mask = np.ones((D, N), dtype=bool)
    out = cross_sectional_transform(feat, mask, method="zscore")
    expected = (feat[0, :, 0] - 3.0) / np.std([1, 2, 3, 4, 5])
    assert np.allclose(out[0, :, 0], expected, atol=1e-5), f"expected {expected}, got {out[0, :, 0]}"
    assert abs(out[0, :, 0].mean()) < 1e-5, "cross-sectional z-score must have ~0 mean within the date"
    print(f"  ✓ z-score matches hand-computed values: {out[0, :, 0].round(3)}")

    print("\nTest 2: rank correctness (hand-verifiable)")
    out_rank = cross_sectional_transform(feat, mask, method="rank")
    # 5 values, strictly increasing -> ranks 0..4 -> percentiles [-1, -0.5, 0, 0.5, 1]
    expected_rank = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert np.allclose(out_rank[0, :, 0], expected_rank, atol=1e-5), f"expected {expected_rank}, got {out_rank[0, :, 0]}"
    print(f"  ✓ rank matches hand-computed percentiles: {out_rank[0, :, 0]}")

    print("\nTest 3 (problem vs solution): a GLOBAL feature shift changes a global scaler's "
          "output but leaves the cross-sectional transform exactly unchanged")
    D2, N2, F2 = 10, 8, 1
    rng = np.random.default_rng(0)
    feat2 = rng.normal(scale=1.0, size=(D2, N2, F2)).astype(np.float32)
    mask2 = np.ones((D2, N2), dtype=bool)

    shifted = feat2.copy()
    SHOCK_DATE = 5
    shifted[SHOCK_DATE, :, 0] += 1000.0  # a market-wide regime shock on this one date -- every name moves together, relative order unchanged

    global_before = _global_zscore(feat2, mask2)[SHOCK_DATE, :, 0]
    global_after = _global_zscore(shifted, mask2)[SHOCK_DATE, :, 0]
    global_diff = np.abs(global_after - global_before).max()
    assert global_diff > 1.0, (
        f"expected the OLD global-scaler approach to be meaningfully distorted (>1 z-score "
        f"unit) by a market-wide level shift on one date, got max diff {global_diff}"
    )
    print(f"  ✓ (problem) global z-score on the shocked date changes by {global_diff:.2f} "
          f"z-score units (a 1000-unit market-wide shift swamps the global scaler)")

    cs_before = cross_sectional_transform(feat2, mask2, method="zscore")[SHOCK_DATE, :, 0]
    cs_after = cross_sectional_transform(shifted, mask2, method="zscore")[SHOCK_DATE, :, 0]
    cs_diff = np.abs(cs_after - cs_before).max()
    assert cs_diff < 1e-4, (
        f"expected the cross-sectional transform to be EXACTLY invariant to a uniform "
        f"level shift within a date (shift cancels in mean-subtraction), got diff {cs_diff}"
    )
    assert global_diff > 1000 * cs_diff, (
        f"the whole point: global_diff ({global_diff}) should be orders of magnitude "
        f"larger than cs_diff ({cs_diff}) for the identical shock"
    )
    print(f"  ✓ (solution) cross-sectional z-score on the SAME shocked date changes by only "
          f"{cs_diff:.2e} -- {global_diff / max(cs_diff, 1e-12):.0f}x smaller than the global "
          f"scaler's distortion (relative ranking is what matters, and it didn't change)")

    print("\nTest 4: other dates are completely unaffected by the shock (no cross-date leakage)")
    other_dates = [t for t in range(D2) if t != SHOCK_DATE]
    cs_before_all = cross_sectional_transform(feat2, mask2, method="zscore")
    cs_after_all = cross_sectional_transform(shifted, mask2, method="zscore")
    max_other_diff = np.abs(cs_after_all[other_dates] - cs_before_all[other_dates]).max()
    assert max_other_diff < 1e-6, f"shocking one date leaked into other dates' output: max diff {max_other_diff}"
    print(f"  ✓ shocking date {SHOCK_DATE} left every other date's output identical (max diff {max_other_diff:.2e})")

    print("\nTest 5: inactive/NaN names are excluded from the cross-section, not leaked in")
    feat3 = np.array([[[1.0], [2.0], [3.0], [999.0]]], dtype=np.float32)  # 1 date, 4 names
    mask3 = np.array([[True, True, True, False]])  # name 3 (value 999) is INACTIVE
    out3 = cross_sectional_transform(feat3, mask3, method="zscore")
    assert out3[0, 3, 0] == 0.0, "inactive name must get exactly 0, regardless of its raw value"
    expected3 = (np.array([1.0, 2.0, 3.0]) - 2.0) / np.std([1.0, 2.0, 3.0])
    assert np.allclose(out3[0, :3, 0], expected3, atol=1e-5), (
        f"the inactive name's value (999) must not pollute the active names' mean/std, "
        f"expected {expected3}, got {out3[0, :3, 0]}"
    )
    feat3_nan = feat3.copy()
    feat3_nan[0, 3, 0] = np.nan  # also check a NaN (not just inactive) doesn't crash/leak
    mask3_allactive = np.array([[True, True, True, True]])
    out3_nan = cross_sectional_transform(feat3_nan, mask3_allactive, method="zscore")
    assert out3_nan[0, 3, 0] == 0.0, "a NaN value must map to exactly 0, not propagate NaN"
    assert np.allclose(out3_nan[0, :3, 0], expected3, atol=1e-5), "a NaN active name must not pollute the others' stats"
    print("  ✓ inactive names and NaN values are excluded from the cross-section, mapped to exactly 0")

    print("\nTest 6: degenerate case (<2 active names) maps to 0, not NaN/crash")
    feat4 = np.array([[[5.0], [7.0]]], dtype=np.float32)
    mask4 = np.array([[True, False]])  # only 1 active name
    out4 = cross_sectional_transform(feat4, mask4, method="zscore")
    assert np.all(out4 == 0.0), f"expected all-zero for <2 active names, got {out4}"
    print("  ✓ <2 active names -> exactly 0, no NaN/crash")

    print("\nTest 7: invalid method raises")
    try:
        cross_sectional_transform(feat, mask, method="bogus")
        raise AssertionError("expected ValueError for an unknown method")
    except ValueError:
        pass
    print("  ✓ ValueError raised as expected")

    print("\n✓ All cross_sectional_features tests passed")


if __name__ == "__main__":
    main()
