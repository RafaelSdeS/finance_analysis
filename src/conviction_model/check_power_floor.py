"""
check_power_floor.py -- Phase 0, Step 0 (docs/conviction_model/CONVICTION_MODEL_PLAN.md):
before building anything else, check whether this plan's sample size (50-name
universe, monthly decisions) is even large enough to detect a real signal at
each of the 5 target horizons.

Deliberately NOT invoking h_series/milestone_h0.py directly -- that script is
hardcoded to K_HORIZONS=(21,63) with no CLI, not built for external
parameters. This calls the actual reusable primitive it's built on
(stats.py::min_detectable_ic) with this plan's own horizon set instead.
N_ASSETS=50 already matches milestone_h0.py's own constant -- same universe
size, not a new number.

Run from project root:
    python -m src.conviction_model.check_power_floor
    python -m src.conviction_model.check_power_floor --n-assets 150 \
        --membership-path data/processed/top150_universe_membership.parquet \
        --out PHASE0_POWER_FLOOR_v2.json
"""

import argparse
import json

import pandas as pd

from ..h_series.spine import hac_lag_for_horizon, monthly_decision_dates
from ..h_series.stats import min_detectable_ic
from .labels import HORIZONS
from .paths import DOCS_DIR, MEMBERSHIP_PATH

N_ASSETS = 50          # matches milestone_h0.py's own constant -- same universe, not re-derived
T_THRESHOLD = 2.0       # matches milestone_h0.py's own convention

# H1's actual survivor IC range (docs/conviction_model/CONVICTION_MODEL_PLAN.md, Phase 0) --
# the realistic effect-size band this plan's power floor is compared against, not a magic number.
H1_SURVIVOR_IC_RANGE = (0.035, 0.088)


def _n_monthly_obs(membership_path=MEMBERSHIP_PATH) -> int:
    """Number of monthly decision dates spanned by the membership history --
    the actual walk-forward sample size this plan will have to work with,
    not an assumed/round number. Universe width (top-50 vs top-150) doesn't
    change this -- same rebalance calendar, more names per period."""
    membership = pd.read_parquet(membership_path, columns=["start", "end"])
    calendar = pd.bdate_range(membership["start"].min(), membership["end"].max())
    return len(monthly_decision_dates(calendar))


def compute_power_floors(n_obs: int, n_assets: int = N_ASSETS) -> dict:
    floors = {"n_monthly_obs": n_obs, "n_assets": n_assets, "t_threshold": T_THRESHOLD}
    for k in HORIZONS:
        lag = hac_lag_for_horizon(k)
        floors[str(k)] = {
            "hac_lag_months": lag,
            "min_detectable_ic": min_detectable_ic(n_obs, n_assets, lag, T_THRESHOLD),
        }
    return floors


def _decision(floors: dict) -> str:
    """Go/underpowered call: every horizon's floor must fall at or below the
    realistic effect-size band (H1_SURVIVOR_IC_RANGE's lower bound) to be
    worth proceeding -- a floor above the range where real signals have
    actually been observed elsewhere in this project means a null result at
    that horizon would be uninterpretable (underpowered, not "no signal")."""
    lower_bound = H1_SURVIVOR_IC_RANGE[0]
    underpowered = [k for k in HORIZONS if floors[str(k)]["min_detectable_ic"] > lower_bound]
    if not underpowered:
        return "GO -- every horizon's power floor is at or below H1's realistic effect-size range."
    return ("UNDERPOWERED at horizon(s) " + ", ".join(str(k) for k in underpowered) +
            f" -- floor exceeds H1's lower bound ({lower_bound}). A null result at "
            "these horizons in Phase 4 would mean underpowered, not no signal.")


def main(n_assets: int = N_ASSETS, membership_path=MEMBERSHIP_PATH, out_name: str = "PHASE0_POWER_FLOOR.json") -> None:
    n_obs = _n_monthly_obs(membership_path)
    floors = compute_power_floors(n_obs, n_assets)
    decision = _decision(floors)

    print(f"n_monthly_obs={n_obs}, n_assets={n_assets}, t_threshold={T_THRESHOLD}")
    for k in HORIZONS:
        f = floors[str(k)]
        print(f"  k={k:>4}  hac_lag_months={f['hac_lag_months']}  "
              f"min_detectable_ic={f['min_detectable_ic']:.4f}")
    print(f"\nDecision: {decision}")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / out_name
    with open(out_path, "w") as fh:
        json.dump({**floors, "decision": decision}, fh, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-assets", type=int, default=N_ASSETS)
    parser.add_argument("--membership-path", type=str, default=str(MEMBERSHIP_PATH))
    parser.add_argument("--out", type=str, default="PHASE0_POWER_FLOOR.json")
    args = parser.parse_args()
    main(n_assets=args.n_assets, membership_path=args.membership_path, out_name=args.out)
