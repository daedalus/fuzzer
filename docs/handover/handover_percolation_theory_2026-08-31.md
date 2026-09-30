# Handover — Percolation Theory Applied to Coverage-Guided Fuzzing

**Original:** 2026-08-31 (updated 2026-09-01). Pruned 2026-09-26 to open items;
full original: `git show 1c689e8a^:docs/handover/handover_percolation_theory_2026-08-31.md`.
**Verified against:** `4c021daa`. Live: Module 1 (`core/percolation.py::bootstrap_minimize_corpus`,
`--bootstrap`), Module 2 (`core/analyzers/analyzer_coverage_regime.py::CoverageRegimeDetector`),
Module 4 (`services/seed_picker.py::invasion_select`, `--invasion` Elo arm).
Module 3 (`core/target_difficulty.py`) closed as diagnostic-only by decision (P2-1).

---

## Module 5. First-passage percolation for time budgeting — implemented (unwired)

**What.** Estimate executions to the next uncovered region; allocate time per
target in multi-target fuzzing.

**Status (2026-09-30).** `estimate_time_to_next_discovery(edge_tracker,
operator_stats, coverage_regime, *, phi_profile=..., target_delta=..., c=...,
max_n=...)` lives in `core/percolation.py`. Inverts Diskin et al.
(arXiv:2603.03257) Thm 3 via stepwise integration of `1/(c·Φ(t))` from current
`|cumulative_edges|` to a target size. Regime scales and operator success-rate
stretch are applied after the integral. When `phi_profile` is omitted, falls
back to a pure regime prior (`_REGIME_FALLBACK_EXEC_PER_EDGE`). Unit tests in
`tests/test_percolation.py::TestEstimateTimeToNextDiscovery` (10 cases),
including the handover cross-check that `Φ(x) ≳ x` + SUPERCRITICAL is cheaper
than CRITICAL.

**Production (gated, 2026-09-30).** Opt-in via `--target-schedule phi`.
`Fuzzer._phi_weights` calls `estimate_time_to_next_discovery` per target;
`Fuzzer.set_phi_profile(target, profile)` seeds Φ from
`target_difficulty.estimate_isoperimetric_profile`. Default schedules are
unchanged. Without a seeded profile the estimator uses its regime prior.
P2-1 is reopened only under the explicit `phi` gate — not for the default path.

## Module 6. Universality → strategy transfer — absent

**What.** Pre-initialise operator weights from a structurally similar target
(`core/strategy_transfer.py::transfer_strategy(source_profile, target_profile,
source_weights)`). Depends on Module 3 profiles and Module 4. Low priority;
blocks nothing.

## Open falsifier checks

1. **Φ vs chokepoints.** Does the approximated `Φ` profile correlate with
   observed discovery slowdowns at checksum/length gates? The graph is finite and
   non-transitive, so `Φ` is a heuristic, not the theorem.
2. **Bootstrap vs diversity.** Does the k-rigid core lose crash-finding ability
   vs the full corpus? `bench_paired.py` arm `bootstrap` is registered, never run.
3. **Regime noise.** CV of inter-discovery times; CV ≫ 1 breaks the Poisson
   assumption behind SUBCRITICAL/SUPERCRITICAL labels. Not measured.
