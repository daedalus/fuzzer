# Handover — Navier–Stokes Continuum Extension of Coverage Percolation

**Date:** 2026-09-05
**Base:** `6b3b7c3` (`refactor(fractal_voronoi): hold the geometry caches on the instance`)
**Status: PLAN ONLY. NOTHING IMPLEMENTED.** Builds directly on the percolation
framing in `handover_percolation_theory_2026-08-31.md` (Modules 1–4 live).
Every proposal below is unwritten code. The existing discrete invasion /
regime detectors remain the ground truth; this document only sketches how to
lift them into a continuum fluid model.

Companion: `docs/handover/handover_percolation_theory_2026-08-31.md`.

---

## 0. Rule 1 — placement

| Object | Home | Why |
|---|---|---|
| Continuum field primitives (density, velocity, pressure on contracted graph) | `core/navier_stokes.py` (new) | Parallel to `core/percolation.py`; pure math, no scheduler interface |
| Regime / diagnostic signals (local Re, dissipation, divergence) | feed `CoverageRegimeDetector` in `core/coverage_regime.py` | Already the single phase classifier; do not invent a second one |
| Operator ranking that uses continuum resistance / flux | extend `invasion_select` in `services/seed_picker.py` or a new Elo ballot arm | Invasion is already the discrete fluid-invasion selector; continuum version is its natural generalisation |
| Power-schedule term (energy ∝ flux or pressure gradient) | `core/schedules.py` | Same place every other continuous energy allocation lives |
| Graph discretisation helpers (finite-volume faces, LBM lattices on horizon graph) | `core/navier_stokes.py` or `core/horizon.py` if already graph-adjacent | Keep graph geometry next to the existing ICFG / horizon machinery |

Hard Rule 1 still applies: any new scheduler arm must register in
`_OPERATOR_STRATEGY_NAMES` and implement `select_op` / `record` / `bandit_stats`.
A pure continuum diagnostic that only *modulates* an existing arm does not
need its own strategy name.

---

## 1. Framing: from discrete invasion to continuum flow

Percolation already models coverage as fluid invasion of a porous medium
(the state-space graph). Invasion percolation is the quasi-static, capillary-
dominated limit. Navier–Stokes supplies the continuum PDE that governs the
same physics when inertia and viscosity matter:

\[
\frac{\partial\mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}
= -\frac{1}{\rho}\nabla p + \nu\nabla^{2}\mathbf{u} + \mathbf{f},
\qquad
\nabla\cdot\mathbf{u}=0
\]

(incompressible form). Map:

| Continuum quantity | Fuzzer quantity |
|---|---|
| density / occupancy field \(\rho\) | coverage density or frontier occupation on a coarse-grained graph |
| velocity \(\mathbf{u}\) | directed discovery flux (new edges per unit “time” along graph directions) |
| pressure \(p\) | inverse coverage, rarity, AFLGo distance, or information-theoretic score |
| viscosity \(\nu\) | damping of unsuccessful operators / lineages |
| body force \(\mathbf{f}\) | external bias (directed targets, format priors, dictionary hits) |
| Reynolds number \(\mathrm{Re}=UL/\nu\) | diagnostic of laminar vs turbulent exploration regime |

The discrete invasion selector is the \(\mathrm{Re}\to 0\), quasi-static limit.
The continuum model becomes useful precisely when the fuzzer is near criticality
(where critical slowing already fires) or when operator momentum and cross-scale
coupling matter.

---

## 2. What already exists (reuse, do not re-implement)

| Component | Location | Role in continuum lift |
|---|---|---|
| `CoverageRegime` + bootstrap | `core/percolation.py` | Phase labels remain; continuum signals only refine the classifier |
| `CoverageRegimeDetector` | `core/coverage_regime.py` | Single place that emits SUBCRITICAL / CRITICAL / SUPERCRITICAL |
| `CriticalSlowingDown` | `core/critical_slowing.py` | Variance / autocorrelation already detect approach to transition; NS dissipation / enstrophy are additional observables |
| Invasion select | `services/seed_picker.py` | Discrete lowest-resistance rule; continuum version ranks by predicted flux |
| Horizon / ICFG / Katz | `core/horizon.py`, `core/icfg.py`, `core/schedulers/katz.py` | Supply the discrete graph on which any finite-volume or LBM discretisation lives |
| Elo meta-scheduler | `services/operators.py` + `core/elo.py` | Ballot already accepts new arms; continuum ranking can be one more candidate |
| Power schedules | `core/schedules.py` | Natural home for continuous energy allocation driven by pressure gradient |

Do not create a parallel regime detector or a second invasion function outside
the existing call sites.

---

## 3. Module proposals (all PLAN)

### 3.1 Pressure field from existing scores

Define \(p(v)\) on vertices (or contracted supernodes) of the horizon graph as
a monotone transform of:

- inverse local coverage density,
- rarity (Chao2 / rare-edge ownership),
- AFLGo harmonic distance,
- or a linear combination already computed by seed quality / distance modules.

Gradient \(\nabla p\) supplies the driving term. No new instrumentation required;
the numbers already flow through the stats and seed-picker paths.

### 3.2 Velocity / flux estimator

Maintain a sparse velocity field by finite differences (or simple upwind) of
recent edge-discovery events on the contracted graph. Update cost must stay
\(O(\#\text{frontier edges})\) per observation window; full NS solve every
iteration is forbidden by Hard Rule 41 (speed).

First practical surrogate: treat operator success rate × frontier adjacency
as a discrete flux and only later replace it with a real finite-volume step.

### 3.3 Viscosity and Reynolds diagnostic

- Viscosity \(\nu\): function of recent operator failure rate or of the CSD
  variance signal. High \(\nu\) damps the Elo / bandit arms that are losing.
- Local \(\mathrm{Re}\): computed on the same window used by
  `CriticalSlowingDown`. High \(\mathrm{Re}\) near criticality is expected;
  sustained high \(\mathrm{Re}\) in a claimed SUPERCRITICAL regime is a
  falsifier (see §5).

Expose the diagnostic through the existing regime detector’s observe path;
do not invent a new CLI flag unless the signal proves actionable.

### 3.4 Continuum ranking for operator / seed selection

Replace pure resistance \(1/\text{success}\) in `invasion_select` with a
predicted flux that includes a pressure-gradient term and a viscous drag
term. Keep the same function signature and the same “return None when stuck”
contract so the Elo ballot wiring does not change.

### 3.5 Optional: lattice Boltzmann on the horizon graph

Lattice Boltzmann recovers NS in the continuum limit and is graph-friendly.
Only worth implementing after the pressure + flux surrogates above have shown
a measurable coverage or time-to-bug improvement. Place the LBM step in
`core/navier_stokes.py`; keep the collision / streaming kernels pure and
vectorised.

---

## 4. Integration sketch (no code yet)

1. `CoverageRegimeDetector.observe(...)` gains optional continuum diagnostics
   (local Re, mean dissipation). Classification precedence stays the same;
   continuum signals only break ties or raise early CRITICAL.
2. `invasion_select` (or a thin wrapper) accepts an optional continuum flux
   map; when present it ranks by flux instead of raw resistance.
3. Elo ballot continues to own the final choice; continuum ranking is just
   another candidate that can win or lose on the same reward signal.
4. Any new power-schedule term that allocates energy proportional to
   \(|\nabla p|\) lives in `core/schedules.py` and is selected by the existing
   `--schedule` machinery.

---

## 5. What could falsify this framing

- Continuum signals never change the regime label relative to pure CSD +
  stall + homogeneity → the lift adds no information; abandon.
- Adding viscous damping or pressure ranking increases wall-clock time per
  exec without a compensating rise in unique edges or crash rate
  (Hard Rule 41).
- Local Re stays high in a regime the discrete detector calls SUPERCRITICAL
  for long stretches → either the Re estimator is wrong or the continuum
  analogy is broken for that target class.
- Any implementation that requires a full sparse linear solve or dense
  adjacency matrix on the raw edge set will be too slow; the design is
  falsified if the only correct discretisation needs that cost.

---

## 6. Recommended first steps (order)

1. Instrument only: compute a cheap pressure field from already-available
   rarity / distance numbers and log its gradient magnitude next to the
   existing CSD line. No behaviour change.
2. Add the Reynolds diagnostic to the regime detector’s observe path;
   confirm it correlates with the existing CRITICAL label on a few targets.
3. Only then touch `invasion_select` to optionally rank by a flux that
   includes \(\nabla p\).
4. Measure unique-edge rate and time-to-first-interesting under the new
   ranking versus pure invasion and versus the current Elo winner.
5. Lattice Boltzmann or any real NS discretisation only after steps 1–4
   show a clear win.

---

## 7. Files that would be touched (when implementation starts)

| File | Change |
|---|---|
| `src/fuzzer_tool/core/navier_stokes.py` | **NEW** — pressure, flux, Re, optional LBM kernels |
| `src/fuzzer_tool/core/coverage_regime.py` | optional continuum diagnostics in `observe` |
| `src/fuzzer_tool/services/seed_picker.py` | continuum-aware ranking path inside or beside `invasion_select` |
| `src/fuzzer_tool/core/schedules.py` | optional pressure-gradient energy term |
| `tests/test_navier_stokes.py` | **NEW** — unit tests for field construction, Re, flux ranking |
| `tests/test_invasion_select.py` | extend with continuum ranking cases |
| `docs/handover/handover_percolation_theory_2026-08-31.md` | cross-link once any module lands |

No new CLI surface until a signal has proved actionable.

---

## 8. References

- Continuum limit of invasion percolation / porous-media flow: standard
  Darcy → Brinkman → NS progression in the porous-media literature.
- Lattice Boltzmann methods on irregular graphs: existing CFD literature;
  only relevant after the cheap surrogates succeed.
- Existing discrete foundation: `handover_percolation_theory_2026-08-31.md`
  (Modules 1–4) and the live code in `core/percolation.py`,
  `core/coverage_regime.py`, `services/seed_picker.py`.
