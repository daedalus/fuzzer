# Handover — Percolation Theory Applied to Coverage-Guided Fuzzing

**Original:** 2026-08-31 (updated 2026-09-01). Pruned 2026-09-26 to open items;
full original: `git show 1c689e8a^:docs/handover/handover_percolation_theory_2026-08-31.md`.
**Verified against:** `4c021daa`. Live: Module 1 (`core/percolation.py::bootstrap_minimize_corpus`,
`--bootstrap`), Module 2 (`core/analyzers/analyzer_coverage_regime.py::CoverageRegimeDetector`),
Module 4 (`services/seed_picker.py::invasion_select`, `--invasion` Elo arm).
Module 3 (`core/target_difficulty.py`) closed as diagnostic-only by decision (P2-1).

---

## Module 5. First-passage percolation for time budgeting — absent

**What.** Estimate executions to the next uncovered region; allocate time per
target in multi-target fuzzing.

**Design.** Add `estimate_time_to_next_discovery(edge_tracker, operator_stats,
coverage_regime)` to `core/percolation.py`. Prior from Diskin et al.
(arXiv:2603.03257) Thm 3: invert `∫[|S|→v_n] 1/(c·Φ(t)) dt = n` using the `Φ`
profile from `target_difficulty.estimate_isoperimetric_profile`. Cross-check:
`Φ(x) ≳ x` should predict compounding growth, matching SUPERCRITICAL.

**Blocker.** Needs a production consumer for `Φ`; `target_difficulty` is
diagnostic-only, so wiring Module 5 means reopening that decision.

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
