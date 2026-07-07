"""
ML Pipeline Explorer — entry point.

Run from project root:
    streamlit run tools/explorer/app.py

Pages (each an independent module under pages/):
  Data Explorer      inspect raw/processed/training data, distributions, stage diffs
  Data Quality       anomaly scanner, V1-V7 gate, tensor cross-check
  Model Explorer     model inputs, portfolio allocations, live inference
  Training Analysis  learning curves, backtest performance vs baselines
"""

from pathlib import Path

import streamlit as st

_PAGES_DIR = Path(__file__).resolve().parent / "pages"

st.set_page_config(page_title="ML Pipeline Explorer", layout="wide")

nav = st.navigation([
    st.Page(_PAGES_DIR / "data_explorer.py", title="Data Explorer", icon=":material/dataset:"),
    st.Page(_PAGES_DIR / "data_quality.py", title="Data Quality", icon=":material/fact_check:"),
    st.Page(_PAGES_DIR / "model_explorer.py", title="Model Explorer", icon=":material/psychology:"),
    st.Page(_PAGES_DIR / "training_analysis.py", title="Training Analysis", icon=":material/monitoring:"),
])
nav.run()
