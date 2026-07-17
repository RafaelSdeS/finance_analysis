# Training Speedup Plan (quality-neutral only)

Diagnosis: the EIIE CNN is tiny (3 convs, ~3k params), so per-step cost is dominated by
CPU-side data prep in `_batch_tensors` (`train.py:60`) — numpy fancy-gathers in float64
(`window_tensor_batch`, `data.py:221`), fresh `torch.tensor(...)` casts, and blocking
host→GPU copies — repeated ~117k+ times (100k pretrain steps + `rolling_steps=30` per
online-backtest day). The GPU is starved. Everything below changes wall-clock only,
never results.

## Ranked items

- [x] **S1 — Precompute all training tensors once, keep them GPU-resident.** *(implemented:
  `_PanelStore`/`_get_store` in `train.py`, cached lazily on the panel — no signature changes)*
  X_t is a deterministic function of the panel: precompute `window_tensor_batch` over
  every in-window t once at load (compute in float64, store float32 → bit-identical to
  today's per-step cast), upload once. In-window T ≈ 3,850 → `(T, 3, 50, 50)` float32
  ≈ 115 MB on GPU. Same for `y` `(T, 172)`, `slot_gidx`, `valid` (tiny). `train_step`
  then just indexes GPU tensors — zero numpy, zero H2D per step. This is the big one;
  expect the largest single win by far.

- [ ] **S2 — Stop syncing every step.** *(deferred: post-S1 the sync bubble is ~launch
  latency → seconds per run; only the pbar postfix was throttled to every 100 steps.
  Revisit only if a profile disagrees.)* `float(loss.item())` (`train.py:112`) forces a
  CPU↔GPU sync per step. Append the 0-d GPU tensor to a list, `.item()` in bulk every
  ~100 steps (or at the end) for the loss curve. Identical recorded values. Only pays
  off after S1 removes the CPU prep, but then lets the CPU queue ahead of the GPU.

- [x] **S3 — `torch.compile(model, mode="reduce-overhead")`.** *(implemented behind
  `train.compile` config flag, default **off** — and measured NOT worth enabling on this
  machine: 2.50 → 2.22 ms/step (1.13×), because only the tiny model forward/backward is
  compiled while the eager ops around it (mu solve, PVM gather/scatter, Adam, clip)
  dominate the remaining step time. Also NOT bit-identical to eager (loss max rel diff
  ~1.3% after 220 steps as float rounding compounds), so enabling it forfeits exact
  reproducibility. Ceiling if ever needed: compile/graph-capture the whole train_step,
  not just the model.)*

- [ ] **S4 — O(1) recency sampler.** *(skipped: changes the RNG draw stream, breaking
  same-seed comparability with in-flight Phase-2 runs, for a win of seconds)* `sample_batch_starts` (`train.py:46`) builds and
  normalizes an O(k_max) weight vector and runs `rng.choice(p=...)` (cumsum + search)
  per call × ~117k calls. A truncated geometric has a closed-form inverse CDF:
  `k = floor(log1p(-u * (1 - (1-beta)**(k_max+1))) / log(1-beta))` — exact same
  distribution, O(1). Small win (~seconds–tens of seconds total), one-line change.

- [x] **S5 — Run seed-ensemble members concurrently.** *(implemented:
  `python -m src.rl_agent.sweep --config <cfg...> [--seeds 1 2 3] [-j 4]` — bounded
  subprocess pool, per-job logs in `experiments/sweep_logs/{ts}/`; experiment run-dir
  timestamps now carry microseconds so same-second parallel launches can't collide.
  ~0.5 GB GPU per job; default -j 4.)*

- [x] **S6 — Vectorize PVM uniform init** (`pvm.py`): the `for t in range(T)` loop is now
  `w_slots = valid_t.to(dtype) * uniform_weight.unsqueeze(1)` — startup-time only.

## Deliberately NOT proposed (would touch quality)

- AMP / fp16 / bf16 — no benefit at this size, changes numerics.
- Enabling TF32 or `cudnn.benchmark` — can break the bit-for-bit determinism the
  sanity gate checks.
- Cutting `pretrain_steps` / `rolling_steps`, bigger lr, bigger batch — those are
  hyperparameters, not optimizations.

## Verify

- [ ] Before/after: time a fixed 200-step pretrain profile (same seed) to confirm the
  CPU-bound diagnosis and measure each item's win.
- [ ] After S1–S3: fixed-seed short run must reproduce today's loss trajectory exactly
  (S1 is bit-identical by construction; S3 is the only item needing an empirical check).
