"""PAGE 2 — Data Quality: automated anomaly detection and integrity gates."""

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.agent.config import DEFAULT_CONFIG  # noqa: E402
from tools.explorer import data_access as da  # noqa: E402
from tools.explorer import ui  # noqa: E402

st.title("Data Quality")
st.markdown(
    """
This page answers one question: **is anything in the data broken?** It runs three layers
of automated checks, from cheap slice-level scans to a full-dataset gate:

1. **Anomaly scanner** — vectorized checks on the tickers/range you selected.
2. **V1–V7 gate** — the same full-dataset verification that must pass before training.
3. **Tensor cross-check** — verifies the dense tensors the agent trains on match the
   tidy dataset on disk, cell by cell.
"""
)

filters = ui.sidebar_filters()
tickers = filters["tickers"]
date_range = filters["date_range"]

# ------------------------------------------------------------------ Anomaly scanner
st.header("Anomaly scanner")
st.markdown(
    """
Each check targets a specific, historically common failure mode:

| Check | What it catches |
|---|---|
| `duplicate_row` | same (ticker, date) twice — double-weights that day in training |
| `date_gap` | >10 business days missing inside a ticker's active span — collection hole |
| `return_spike` | \\|log return\\| > 35% — the fingerprint of an **unadjusted stock split** (this project's past returns bug) |
| `stale_price` | ≥5 identical consecutive closes *with* volume — frozen/duplicated feed |
| `nan_critical` | NaN in ticker/date/close/volume/sector — columns that must never be missing |
| `zero_variance` | a feature that is constant — carries no signal, often a broken computation |
| `outlier:<col>` | robust z-score > 8 — data-entry scale errors (e.g. price in centavos) |
| `lookahead` | fundamental dated after the price date — future information leaking into the past |

Findings are worth reading top-down: a handful of `return_spike`s on one date usually
means one bad collection day; hundreds spread across tickers means a systematic bug.
"""
)
stage_name = st.selectbox("Stage", list(da.STAGES.keys()), key="dq_stage")
df = ui.stage_df(stage_name, tuple(tickers), date_range)
if len(df) > 0:
    numeric = list(df.select_dtypes(include=["number"]).columns)
    feature_cols = [c for c in DEFAULT_CONFIG.state_features if c in df.columns] or numeric[:10]
    findings = da.run_all_checks(df, feature_cols)
    st.write(f"**{len(findings)} findings** for {len(df):,} rows scanned")
    st.dataframe(findings, use_container_width=True)

    if len(findings) > 0:
        st.subheader("Drill-down")
        st.markdown("Pick a finding to see the raw rows ±5 days around it — the context "
                    "usually makes the cause obvious (gap? split? one corrupted row?).")
        idx = st.number_input("Finding row index", 0, len(findings) - 1, 0)
        f = findings.iloc[int(idx)]
        if pd.notna(f["date"]) and f["ticker"] in df["ticker"].values:
            center = pd.Timestamp(f["date"])
            around = df[(df["ticker"] == f["ticker"]) &
                        (df["date"] >= center - pd.Timedelta(days=5)) &
                        (df["date"] <= center + pd.Timedelta(days=5))]
            st.dataframe(around, use_container_width=True)
else:
    st.info("No data for the current filters.")

# ------------------------------------------------------------------ V1–V7 gate
st.header("Full V1–V7 dataset gate")
st.markdown(
    """
Runs `tests/agent/verify_dataset_for_training.py` on the **entire** processed dataset
(all tickers, all dates — not just your current selection). These are the pass/fail
gates that must all be green before a training run is meaningful: date coverage (V1),
ticker coverage (V2), feature completeness (V3), NaN rates (V4), distributions (V5),
lookahead bias (V6), sector diversity (V7). Takes ~30s — it loads the full 190 MB parquet.
"""
)
if st.button("Run V1–V7 gate"):
    sys.path.insert(0, str(ROOT / "tests" / "agent"))
    from verify_dataset_for_training import verify_dataset_for_training
    buf = io.StringIO()
    with st.spinner("Loading full dataset and running gates..."):
        with contextlib.redirect_stdout(buf):
            verify_dataset_for_training(str(da.PROCESSED_PATH))
    st.text(buf.getvalue())

# ------------------------------------------------------------------ Tensor cross-check
st.header("Tensor cross-check")
st.markdown(
    """
The agent never reads the parquet directly — it trains on dense numpy tensors
(`agent_tensors.npz`) built by pivoting the dataset and applying a train-only scaler.
A misalignment in that pivot (rows shifted by one date, tickers in the wrong column)
would silently feed the agent *another ticker's* features while all other checks pass.
This check samples random (ticker, date) cells and asserts
`npz value == scaler(parquet value)` exactly. Any mismatch is a serious bug.
"""
)
if st.button("Run tensor cross-check"):
    with st.spinner("Sampling cells..."):
        checked, mismatches = da.tensor_cross_check()
    if mismatches == 0:
        st.success(f"✓ {checked} cells checked, 0 mismatches — tensors match the dataset.")
    else:
        st.error(f"✗ {mismatches}/{checked} cells mismatch — the tensor pivot is misaligned. "
                 "Re-run `python -m src.agent.data_pipeline` and investigate.")
