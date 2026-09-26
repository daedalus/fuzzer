# Handover — Navier–Stokes Continuum Extension of Coverage Percolation

**Original:** 2026-09-05. Pruned 2026-09-26 to open items; full original:
`git show 1c689e8a^:docs/handover/handover_navier_stokes_coverage_flow_2026-09-05.md`.
**Verified against:** `4c021daa`. Modules 3.1–3.4 are live as steady diagnostics
(`core/analyzers/analyzer_navier_stokes.py::ContinuumField`, `--continuum`,
also in `--hail-mary`): pressure, gradient, viscosity, Reynolds ratio, flux
ranking via `seed_picker.invasion_select(flux_map=...)`. Time-stepping / LBM
(3.5) deliberately not built (Tao blowup argument, module docstring).
`--continuum-reward` A/B done: off, out of `--hail-mary`.

---

## 1. Does Re track the regime label? (plan §6 step 2)

`CoverageRegimeDetector` records `(regime, Re)` pairs and exposes
`continuum_correlation()` (`core/analyzers/analyzer_coverage_regime.py`), but
nothing in production calls it and no measurement is recorded.

**To do.** Run `--continuum` campaigns on several targets, read
`continuum_correlation()`, record in `docs/learnings/`.

**Falsifiers.** Re never separates labels ⇒ diagnostic carries no information,
retire it. Sustained high Re under SUPERCRITICAL ⇒ estimator or analogy broken
for that target class.

## 2. A/B flux ranking (plan §6 step 4)

`tools/lib/bench_paired.py` arm `continuum` (baseline `invasion`, since the flux
map only reaches `invasion_select`) is registered, never run. Measure unique
edges and exec/s (Hard Rule 41) vs `invasion`.

**Acceptance.** Per-target W/L, median Δ, power statement. On a null or loss,
drop `continuum` from `_HAIL_MARY_FLAGS` (`cli/commands.py`).

## 3. Pressure-gradient energy term — not built

Plan §3/§4 step 4: energy ∝ |∇p| in `core/schedules.py`. Absent. Build only if
§1–2 show signal. Note from the original audit: the transferable idea is
"saturate a scale, then dump" (delay before leaking energy to the next
frontier); the crude form exists in the SUBCRITICAL branch of the main loop
(`services/fuzzer.py`, `ops._havoc_energy_scale` ×1.5, cap 5).
