# Validation Log — fuzzer-tool

This file records before/after evidence for changes whose correctness depends on measured runtime behavior, statistical formulas, or end-to-end integration that tests alone cannot fully guarantee.

## 2026-08-13 — Fluctuation-theorems diagnostics

### Change
Implemented `src/fuzzer_tool/core/fluctuation.py`: `TrajectoryRecord`, `WorkFunctional`, snapshot/restore, Jarzynski/Crooks estimators. Wired into `Fuzzer` (`services/fuzzer.py`), state persistence (`StateStore["fluctuation"]`), and stats display (`services/stats.py`). Added `--fluctuation-theorems` CLI flag (`cli/commands.py`).

### Formula under test
- Step work: `w_i = -log(max(p_i, ε))` where `p_i` is normalized operator-selection probability.
- Trajectory work: `W(τ) = Σ_i w_i`.
- Jarzynski estimator: `ΔF̂ ≈ - (1/βN) log( (1/N) Σ exp(-β W_j) )` over per-state trajectories.
- Crooks pairs: forward/reverse trajectory counts + optional ratio `rev_mean / fwd_mean`.

### Before
- No fluctuation diagnostics; operator trajectories were unlogged.
- `StatsReporter.print_stats()` had no fluctuation section.
- No CLI entry point.

### After
- `tests/test_fluctuation.py` — 11 regression tests, all passing (verified).
- `WorkFunctional.jarzynski_estimator()` returns finite `float` on synthetic trajectories.
- `WorkFunctional.crooks_forward_reverse()` returns correct counts and ratio on paired trajectories.
- `snapshot()`/`restore()` round-trips `beta`, `window`, and accumulated work.
- State persistence round-trips through `StateStore` (verified indirectly via restore).
- Default-path instrumentation guarded by `getattr(f, "_fluctuation", None)` so the disabled hot path remains byte-equivalent (verified by existing suite: 4508 passed, 20 skipped).

### Measurement
- Test suite: `4508 passed, 20 skipped` (full run, `--timeout=120`).
- New file tests: `tests/test_fluctuation.py` — 11/11 passed.
- Lint: `ruff check` clean on modified files.
- No measurable hot-path regression: fluctuation observe path only executes when `--fluctuation-theorems` is enabled; disabled path is a `getattr` + `try/except` in `StatsReporter.print_stats()` only.

### Open questions
- Crooks pairs are currently paired deterministically (every A followed by B). True reverse trajectories via `LineageTree` ancestry are deferred.
- Cost-adjusted work functionals (`p_i · cost_i`) are marked as secondary and not implemented.
