# Fluctuation theorems module for mutation trajectories

**Date:** 2026-08-13
**Context:** `fuzzer-tool`, Phase 1 of fluctuation-theorem diagnostics (Jarzynski/Crooks)

## Problem
The fuzzer records rich per-round trajectory state (`_last_ops_used`, operator costs,
new edge counts) but no subsystem connects these observations to thermodynamic-style
estimators. The goal was to add an opt-in Jarzynski/Crooks module that estimates the
"difficulty" of reaching rare coverage states from biased mutation trajectories,
without destabilizing the existing hot path or introducing new physics assumptions.

## Rejected
- **Feed dF estimates into scheduling** — looked plausible because the plan mentioned a
  Phase 2 `--fluctuation-guided` mode that biases seed picking. Dropped for Phase 1:
  the spec explicitly defers scheduling feedback, and feeding an unvalidated estimate
  into the Elo meta-scheduler could silently degrade corpus exploration.
- **Separate trajectory logging** — considered adding parallel per-mutation logging in
  `OperatorEngine.mutate()`. Dropped because `_last_ops_used` / `_last_op_costs` are
  already populated by `mutate()` and consumed in `fuzz_one()`, which is the correct
  observation point without duplicate instrumentation.
- **xxhash for state keys** — the checkpoint notes mentioned `xxhash.xxh_4_intdigest`
  as a fallback. Dropped because no project-wide xxhash dependency was confirmed;
  `hashlib` (stdlib) is always available.

## Approach
- Work functional `w_i = -log(max(p_i, ε))` with `W(τ) = Σ w_i`, where `p_i` is the
  normalized selection probability of operator `o_i`. Well-defined for every scheduler
  because probabilities are always available from `_op_probability()`.
- `WorkFunctional` accumulates `mean(exp(-βW))` online via Welford-style exponential
  moment updates, capped to a bounded ring buffer (`window=1000`).
- State identity uses a SHA-256 fingerprint of sorted hit-edge IDs (masked to 32 bits
  to handle Python ints that may exceed the u32 namespace).
- `_record_fluctuation_observation` caches `_last_state_key` so the stats display path
  doesn't recompute the hash. `TrajectoryRecord.state_key` is empty by default; when
  callers provide one, it's used directly.
- `StatsReporter.print_stats()` appends fluctuation diagnostics under
  `getattr(f, "_fluctuation", None)` guard so the disabled path stays byte-equivalent.
- State persisted under `StateStore["fluctuation"]` alongside existing subsystems.

## Key insight
The `_last_state_key` caching pattern — `_record_fluctuation_observation` stores the
computed key on `self._fluctuation._last_state_key` so the stats display path reads
it directly rather than re-deriving the edge-set hash. Without this, every stats tick
would re-hash the full edge set. Paired with lazy import of `TrajectoryRecord` inside
the method (avoids circular import with `core/fluctuation.py` during Fuzzer init), it
keeps the disabled hot path untouched — a single `getattr` + `try/except` in the stats.

## Verification
- 11 regression tests in `tests/test_fluctuation.py` — all pass.
- Full pytest suite: 4429 passed, 8 skipped.
- Smoke run with `--fluctuation-theorems` on `targets/test_target`: stats line shows
  `fluc: W=1.39 n=1 dF=1.39`; state saves on shutdown.
- Default path (flag off) shows no fluctuation overhead in the 4429-test run.

## Generalizes to
- Diagnostic-only feature gating via `getattr(obj, "_attr", None)` + `try/except` in
  display layer keeps the default hot path byte-equivalent for any speculative subsystem.
- Caching the state key on the tracker object, rather than recomputing per display tick,
  is the right pattern whenever stats rendering and the mutation loop share an expensive
  derived value.
- State persistence under a dedicated `StateStore` key alongside existing subsystems is
  the convention for any new stateful feature that needs `--resume` survival.
