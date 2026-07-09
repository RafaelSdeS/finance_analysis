#!/usr/bin/env python3
"""
Early-stopping / checkpoint-saving decoupling, replayed against REAL observed
val_sharpe_excess trajectories (2026-07-09 run, window_1 and window_3) to
confirm the fix actually changes behavior on the data that motivated it.

Mirrors ValSharpeCallback's two conditionals in isolation (not a full SB3
integration test -- that needs a live PPO model/env) so this runs in
milliseconds and doesn't touch training infra.

Run from project root: python tests/agent/test_early_stopping.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Real val_sharpe_excess sequences from artifacts/logs/agent/runs/20260709-132721/
WINDOW_3_TRAJECTORY = [0.2830, 0.2811, 0.2874, 0.2907, 0.2901, 0.2923, 0.2930, 0.2981, 0.3146]
WINDOW_1_TRAJECTORY = [
    -0.7286, -0.7259, -0.6954, -0.6541, -0.5845, -0.5715, -0.5500, -0.5171,
    -0.5019, -0.4732, -0.4360, -0.3964, -0.3030, -0.2947, -0.2807, -0.2742,
    -0.2645, -0.2679, -0.2520, -0.2508, -0.2290, -0.2107, -0.1732, -0.1399,
]


def replay(trajectory: list[float], threshold: float, patience: int, decouple_checkpoint: bool):
    """Replay ValSharpeCallback's checkpoint/patience logic against a val-Sharpe
    sequence. Returns (n_checkpoints_saved, stopped_at_eval_index or None)."""
    best_excess = float("-inf")
    best_saved = float("-inf")
    degrade_count = 0
    n_saved = 0

    for i, val in enumerate(trajectory):
        if decouple_checkpoint:
            if val > best_saved:
                best_saved = val
                n_saved += 1
        # else: old behavior folds checkpoint-saving into the same gate as the reset below

        if val > best_excess + threshold:
            best_excess = val
            degrade_count = 0
            if not decouple_checkpoint:
                n_saved += 1
        else:
            degrade_count += 1
            if degrade_count >= patience:
                return n_saved, i

    return n_saved, None


def main() -> None:
    patience = 8

    print("=== window_3: monotonic but slow improvement (0.283 -> 0.315 over 8 evals) ===")
    old_saved, old_stop = replay(WINDOW_3_TRAJECTORY, threshold=0.05, patience=patience, decouple_checkpoint=False)
    new_saved, new_stop = replay(WINDOW_3_TRAJECTORY, threshold=0.02, patience=patience, decouple_checkpoint=True)
    print(f"  OLD (threshold=0.05, coupled):   checkpoints_saved={old_saved}  stopped_at_eval={old_stop}")
    print(f"  NEW (threshold=0.02, decoupled): checkpoints_saved={new_saved}  stopped_at_eval={new_stop}")

    assert old_saved == 1, f"old logic should only ever save the first eval's checkpoint, got {old_saved}"
    assert old_stop == 8, f"old logic should exhaust patience at eval 8 (as actually observed), got {old_stop}"
    assert new_saved > old_saved, (
        f"decoupled checkpoint saving should capture more of this monotonic improvement, "
        f"got new={new_saved} vs old={old_saved}"
    )
    print("  ✓ old logic reproduces the actual bug (1 checkpoint, cut off at eval 8);"
          " new logic saves more checkpoints along the real improving trend\n")

    print("=== window_1: large monotonic improvement, still climbing at the old 1M ceiling ===")
    old_saved_w1, old_stop_w1 = replay(WINDOW_1_TRAJECTORY, threshold=0.05, patience=patience, decouple_checkpoint=False)
    new_saved_w1, new_stop_w1 = replay(WINDOW_1_TRAJECTORY, threshold=0.02, patience=patience, decouple_checkpoint=True)
    print(f"  OLD: checkpoints_saved={old_saved_w1}  stopped_at_eval={old_stop_w1}")
    print(f"  NEW: checkpoints_saved={new_saved_w1}  stopped_at_eval={new_stop_w1}")
    assert new_saved_w1 >= old_saved_w1, "decoupled saving should never save fewer checkpoints than the old logic"
    assert new_stop_w1 is None and old_stop_w1 is None, (
        "window_1's improvements were large enough to clear both thresholds every eval -- "
        "neither version should trigger early stopping on this trajectory"
    )
    print("  ✓ both versions correctly let window_1 run its full course (large jumps clear either threshold);"
          " this window's fix is the raised total_timesteps ceiling (config.py), not this logic\n")

    print("ALL EARLY-STOPPING REGRESSION TESTS PASSED ✓")


if __name__ == "__main__":
    main()
