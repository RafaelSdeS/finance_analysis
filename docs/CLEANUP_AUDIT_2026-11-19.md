# Cleanup Audit — 2026-11-19

What could be deleted from the repo, the venv and git history, and what was. Branch at audit
time: `data-spec-tests` @ `d1cb371`. Disk before: 21 GB free of 165 GB.

Rule applied: delete only what nothing uses **and** loses nothing (owner: "não quero perder nada").

---

## 0. `data/` deleted by the owner (intentional)

`data/` was removed from disk on purpose. 2,301 tracked `data/raw/br/**` deletions are left
**unstaged** — committing them is a separate decision (BR raw data is git-tracked by design;
HEAD `d1cb371` still has it, `git restore data/` brings it back). `data/raw/us/` and
`data/processed/` were gitignored and are gone; `processed` is rebuildable, US needs a full re-collect.

- [x] Owner confirmed the deletion was intentional
- [x] `artifacts/checkpoints/*` deleted — every checkpoint described files that no longer exist.
      Kept, they were a silent-corruption trap: `yf/prices.py`'s tail-only path trusts
      `cp[ticker]["last_date"]` and either skips the ticker or fetches only the tail, writing a
      partial file with no error. Recreated automatically (`checkpoint.save` mkdirs).

---

## 1. `.venv` — 5.9 GB → 899 MB ✅

112 packages uninstalled: everything not in the transitive closure of `requirements.txt` + the
packages the code actually needs. Verified after: `pip check` clean, every `src.*` module imports,
`ruff` clean, fast suite unchanged (60/61 before and after — see §4).

| Removed | Size | Why it was there |
|---|---|---|
| `stable-baselines3` → `torch`, `triton`, `nvidia-*`, `cuda-*`, `sympy`, `networkx`, `gymnasium` | ~4.5 GB | Old PPO/EIIE RL agent (deleted 2026-07-23) |
| `graphifyy`, `tree-sitter-*`, `google-genai` | ~100 MB | graphify — runs from its own uv tool, not this venv (`graphify-out/.graphify_python`) |
| `bolsai-mcp`, `mcp`, `starlette`, `uvicorn`, ... | small | BolsAI MCP — launched via `uvx bolsai-mcp` (`~/.claude.json`) |
| `streamlit`, `pydeck`, `altair`, `matplotlib`, `openai`, `tiktoken`, `fundamentos`, `openpyxl`, `xlrd`, `html5lib`, `rich`, `uv` | ~250 MB | Unused |

## 2. `requirements.txt` ✅

Verified by building a throwaway venv from `requirements.txt` alone (what CI does) → same 60/61,
all `src.*` imports, parquet round-trip and `pd.read_html` work.

- [x] Added `pyarrow==21.0.0` — every `read/to_parquet`; nothing listed pulled it in, so a clean install couldn't read the dataset
- [x] Added `requests==2.32.5` — imported directly, was only arriving via yfinance
- [x] Added `lxml==6.0.2` — `pd.read_html` parser for `sec/tenq.py` + `sec/selected_financial_data.py`; without it (or html5lib) those tiers crash
- [x] Added `ipykernel`, `nbformat` — notebook kernel + plotly `fig.show()` inside a notebook
- [x] `cvxpy` 1.7.4 → 1.9.2, `lightgbm` 4.6.0 → 4.7.0 — pins now match what actually ran the recorded backtests
- [ ] Optional, not done: `pytz`, `python-dateutil`, `certifi`, `charset-normalizer`, `idna`, `urllib3` are transitive pins never imported directly. Harmless.

---

## 3. Git

### Done
- [x] Local-only branches pushed to GitHub, verified SHA-equal on `origin`: `diagnosis` (20 unique RL-era commits), `fix/build-memory-budget`
- [x] Local branches deleted after confirming every commit in each one's **reflog** is on a remote: `diagnosis`, `fix/build-memory-budget`, `agents/handle-missing-cagr-values`, `build_dataset`, `data_validation`, `refactor`, `ml_agent`. Kept locally: `data-spec-tests` (current), `main`.

### Not done — would lose something or rewrite shared history
- [ ] `.git` is 2.8 GB; ~2.4 GB is RL-era blobs **in `data-spec-tests`' own history** (`data/models/*.zip` 2.0 GB, `experiments/` 208 MB, `backups/` 84 MB, `data/backtest/` 61 MB, `data/processed/` 13 MB). Only `git filter-repo --invert-paths --path ...` reclaims it: rewrites every SHA, force-push of all branches, re-clone elsewhere. Do `git bundle create ~/rl-era.bundle --all` first if ever done.
- [ ] `refs/sessions/*` (7) + `refs/codex/*` (1): tool checkpoint refs from May 2026. Their commits are on **no branch** — deleting them loses those snapshots. Negligible size; kept.
- [ ] `git gc`: would expire reflog entries >90 days and prune unreachable objects — the same "lose something" risk. Only ~80 MB of loose objects to gain. Skipped.
- [ ] Remote `copilot/*` (3): two hold an empty "Initial plan" commit, one is merged. Frees nothing; deleting on GitHub may orphan Copilot PRs. Kept.

---

## 4. Working tree

- [x] `git rm` `src/visualizations/{agent_performance,agent_vs_benchmarks,rolling_eval_results}.ipynb` (19 MB) — import the deleted `src.agent.*`; already flagged orphaned in `PORTFOLIO_IMPLEMENTATION_PLAN.md` P9
- [x] `git rm` `src/data_collection/one_off/compress_archive_cache.py` — one-time migration (2026-08-15), applied, referenced nowhere
- [x] `git rm -r` `graphify-out/2026-07-16/` — stale graph snapshot, nothing references it
- [ ] Kept: `artifacts/reports/portfolio_dashboard_*.html` (75 MB) — no duplicates, and most of their backtest runs no longer exist, so each is the only record of that run
- [ ] Kept: `artifacts/logs/` (20 MB) — only record of the July collection runs
- [ ] Optional: `exploration.ipynb` is 7.2 MB, 6.3 MB of it cell outputs

**Pre-existing, unrelated:** `tests/data_collection/test_sec_unit_currency.py` fails 3 of 5
(foreign-currency facts admitted as USD in `companyfacts._facts_to_frame`). Same before and after
this cleanup, same in a clean venv — a real bug, not an environment issue.
