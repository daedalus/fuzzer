# Action Plan: Compositional Stability Theory for daedalus/fuzzer

**Basis:** consolidation of `daedalus_fuzzer_integration_analysis.md` and
`HANDOVER_daedalus_fuzzer_compositional_self_stabilization.md`, re-verified
against a fresh full clone of `daedalus/fuzzer` at HEAD `0455cff0`
(2026-09-21). The two source docs were written from an incomplete/partial
clone; this plan corrects the parts that are now stale and narrows the
work to what is actually missing.

---

## 1. What changed since the source docs were written

Both uploaded docs treat "spectral-radius / small-gain monitoring of the
scheduler dynamics" as **entirely absent** — the handover doc's negative
finding explicitly lists "no spectral-radius monitoring of the
scheduler/edge-tracker state."

**That finding is now stale.** `core/kuramoto.py` (added 2026-09-19,
`docs/handover/handover_kuramoto_oscillators_2026-09-19.md`) already:

- treats the operator transition graph (`op_katz.build_transition_matrix`)
  as a Kuramoto coupling matrix,
- computes `spectral_radius(coupling)` = Λ₁ via `eigvals` — the exact
  quantity a small-gain / linearized-stability test needs,
- computes `critical_coupling()` = K_c = K₀/Λ₁, a genuine phase-transition
  threshold on that same operator graph.

It is explicitly a **standalone, unwired diagnostic** (same status as
`centrality.py`) — not used by any scheduler or the CLI. So the primitive
the blog post is searching for (a working spectral test on a system with
memory and coupling) already exists in the codebase; it's just disconnected
from the parts of the fuzzer that would give it a *contract* to check.

A repo-wide grep for `assume-guarantee|small-gain|Lyapunov|metastab|
spectral-radius` confirms: the *term* "spectral radius" only appears in
`kuramoto.py`/`op_katz.py`/`analyzer_sensitivity.py` and their handover —
nothing about self-stabilization, metastability, or assume-guarantee
contracts by name. So the conceptual gap (no formal contracts, no
discharge of a compositional theorem) still holds; only the "no spectral
primitive exists" claim needed correcting.

Also stale: the "≥16 bandit schedulers" count — `core/schedulers/` now has
**49** modules.

---

## 2. Revised gap table

| Blog concept | Fuzzer artefact | Status |
|---|---|---|
| Λ₁ / spectral radius of coupling matrix | `kuramoto.spectral_radius()`, `op_katz.classical_katz_scores()` | **Exists**, unwired |
| Critical coupling K_c (phase-transition threshold) | `kuramoto.critical_coupling()` | **Exists**, unwired, untested against live scheduler state |
| Critical slowing down as early warning | `analyzer_critical_slowing.py` | Exists, used for coverage-rate series only |
| Parametric assume-guarantee contract (badness-indexed family) | None | **Missing** — no contract language anywhere in `core/` or `docs/` |
| Multi-dimensional (matrix, not scalar) linearization of a stateful loop | None | **Missing** |
| Formal discharge / proof that a monitor doesn't destabilize under composition | None | **Missing** |

---

## 3. Recommended action plan, in priority order

### P0 — Wire `kuramoto.critical_coupling` into a live diagnostic (small, self-contained)

Feed it real data instead of leaving it a pure utility:
- On each scheduler-selection tick, pull the current operator transition
  matrix from `op_katz.build_transition_matrix` (already computed) and the
  per-operator reward rate as a stand-in for `ω_i`.
- Compute `r(t)` (order parameter) over a sliding window and log
  `spectral_radius` / `critical_coupling` alongside existing scheduler
  telemetry (no behavior change — observation only).
- Cross-check against `analyzer_critical_slowing.py`: does rising variance
  in reward rates precede a rise in `r`? This is the empirical test the
  Kuramoto handover already flagged as future work ("sanity-checking a
  `critical_coupling` estimate against the real ODE" was done in tests
  with synthetic data, not live fuzzer state).
- Scope: read-only instrumentation + a new analyzer module
  (`analyzer_kuramoto_sync.py`, following the existing `analyzer_*`
  naming/interface convention) + tests. No scheduler behavior changes.

### P1 — Express one scheduler's contract in badness-indexed form (proof of concept)

Pick the scheduler most directly implicated by the lock-in bug fixed this
week (`op_kuramoto.py` / `op_katz.py`, both just had the floor-probability
lock-in fix applied): write its `select_op` guarantee as a parametric
family indexed by corpus/coverage "badness" (e.g. Chao2 richness estimate
or stall duration), not a single fixed `explore_floor`:

    λ(badness) = explore_floor + f(badness)   # currently: flat constant

This directly operationalizes the blog's "if queue < 6, 0 retries" →
"whatever L turns out to be, ≤ λ(L) retries" move, applied to
"whatever the corpus state turns out to be, floor ≥ λ(badness)." Given the
lock-in bug was empirically found via a convergence harness (not
analytically), this is also the natural place to test whether a
badness-indexed floor out-performs the current flat 0.06 default across
the seeds that got stuck.

### P2 — Two-dimensional linearization of a real coupled pair

The blog's core contribution isn't the small-gain theorem (which the
fuzzer's Λ₁ machinery already covers for one matrix) — it's the
**multi-queue** linearization when small-gain's scalar/memoryless
assumptions break. The fuzzer's closest analogue to "two queues" is
**operator reward state × corpus diversity state**, which the source
docs already sketched (Part 3.4 §2, "Four-Slope Stability Analysis for
Scheduler Interactions"):

|              | effect on operator reward | effect on corpus diversity |
|---|---|---|
| operator succeeds | + (reinforcement) | − (exploitation, near-duplicate accepted) |
| operator fails | − (reward decay) | + (exploration forced) |

Fit this 2×2 empirically from logged (reward, diversity) time series
around a stall/recovery event (the stall short-circuit + Metropolis
recovery heuristics already produce exactly this kind of shock), take its
eigenvalues, and check whether the *existing* recovery heuristics are the
fuzzer's implicit "retry budget" / "fresh-first service" stabilizers
(zeroing an off-diagonal term) — i.e., reverse-engineer what the current
ad hoc safeguards are formally doing, rather than designing new ones.

### P3 — Not recommended now: full compositional-contract framework

Both source docs converge on the same conclusion the blog itself reaches:
a general compositional theory for stateful/coupled components is an open
research problem, not an engineering task. Building a general
"contracts" DSL for all 148 operators / 49 schedulers before P0–P2
produce empirical signal would be solving the unsolved part of the blog
post rather than the part the fuzzer is positioned to contribute
(concrete linearizations + working spectral/critical-slowing monitors
feeding a proof, not the proof itself).

---

## 4. Non-goals / explicitly out of scope

- Reproducing distributed-systems primitives (queues, retries, TLA+
  models) inside the fuzzer — there's no retry-storm analogue worth
  building; the value is in reusing the fuzzer's *detectors* as monitors
  and using its *bandit dynamics* as a second worked example for the
  blog's open problem, not in porting the blog's system.
- Renaming/merging `kuramoto.spectral_radius` and
  `op_katz.classical_katz_scores` into one shared helper — the Kuramoto
  handover deliberately kept them separate (diagnostic utility vs.
  scheduler, no cross-dependency) and flagged the duplication
  intentionally; P0 doesn't require touching that decision.

---

## 5. Suggested next concrete step

Start with **P0**: it's additive, read-only, matches the repo's existing
`analyzer_*` module convention, has an existing test harness pattern to
copy (`tests/test_kuramoto.py`, 24 cases), and produces the empirical data
P1/P2 both need before they're worth attempting. Say the word and I'll
scope and implement it against a fresh clone, with the usual
`git am`-verified patch + handover doc.
