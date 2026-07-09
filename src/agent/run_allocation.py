"""
Daily allocation entry point.

Loads the trained agent, predicts portfolio weights for the requested date
(default: latest available), enriches with sector info, and writes CSV or
JSON to artifacts/allocations/.

Usage:
    python -m src.agent.run_allocation
    python -m src.agent.run_allocation --date 2026-06-29 --format json
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.agent.config import DEFAULT_CONFIG
from src.agent.infer import DEFAULT_MODEL_PATH, predict_weights

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "artifacts/allocations"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict daily portfolio allocation")
    parser.add_argument("--date", type=str, default=None, help="Target date (default: latest)")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--strict", action="store_true",
                         help="Fail instead of warning if --date is beyond the dataset's coverage")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    weights = predict_weights(date=args.date, model_path=args.model, config=config, strict=args.strict)

    # Enrich with sector (latest known per ticker)
    sectors = (
        pd.read_parquet(config.dataset_path, columns=["ticker", "sector"])
        .drop_duplicates("ticker")
        .set_index("ticker")["sector"]
    )
    weights["sector"] = weights["ticker"].map(sectors)

    date_str = weights.attrs["date"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.format == "csv":
        out_path = OUTPUT_DIR / f"allocation_{date_str}.csv"
        weights.to_csv(out_path, index=False)
    else:
        out_path = OUTPUT_DIR / f"allocation_{date_str}.json"
        payload = {
            "date": date_str,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": weights.attrs["source"],
            "positions": weights.to_dict(orient="records"),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    stale_days = weights.attrs.get("stale_days", 0)
    if stale_days > 0:
        print(f"\n⚠ WARNING: allocation is {stale_days} days stale (requested {args.date}, "
              f"data only covers through {date_str}). Rebuild the dataset for a current allocation.")
    print(f"\nAllocation for {date_str} ({weights.attrs['source']}):")
    print(weights.head(15).to_string(index=False))
    if len(weights) > 15:
        print(f"... and {len(weights) - 15} more positions")
    print(f"\nTotal weight: {weights['weight'].sum():.6f}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
