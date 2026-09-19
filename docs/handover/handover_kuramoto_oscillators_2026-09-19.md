# Kuramoto phase-oscillator diagnostics over the operator transition graph

**Status:** implemented, standalone diagnostic utility -- same status as
`centrality.py`. Not wired into any scheduler, `distance.py`, or the CLI.
HEAD at time of writing: `2939095`.

## Trigger

User question: does the fuzzer relate to Kuramoto oscillators at all? A
survey of the existing dynamical-systems apparatus turned up several
real structural matches before any new code was written:

- `core/qea.py`'s `rotation_gate()` is already a single (uncoupled) phase
  oscillator: `α ← cos(Δθ)·α ∓ sin(Δθ)·β` is exactly Kuramoto's
  `dθ/dt = ω` term for one bit, with no coupling to any other bit's phase.
- `core/schedulers/op_katz.py` already builds the exact object a
  networked Kuramoto model needs as its coupling matrix: a weighted,
  directed operator transition graph (`build_transition_matrix`), and
  already computes that matrix's spectral radius (`classical_katz_scores`,
  via `np.linalg.eigvals`) -- for a different reason (bounding a Neumann
  series), but it is the same quantity the Kuramoto synchronization
  literature needs to estimate a critical coupling strength.
- `core/analyzers/analyzer_critical_slowing.py`'s rising-variance/
  autocorrelation precursor is the same critical-slowing-down signature
  expected approaching *any* second-order phase transition, including
  the Kuramoto incoherent-to-synchronized transition -- already used
  here for the percolation-style coverage-transition framing.
- `core/corpus_flux.py`'s "quiet vs. balanced" distinction (both look
  flat in the net, only the gross rate tells them apart) is structurally
  the same problem Kuramoto's order parameter r solves (incoherent vs.
  synchronized populations can look similar in a naive aggregate).

None of this was previously connected to Kuramoto by name anywhere in
the repo (confirmed via `grep -ri kuramoto` — zero hits before this
patch). This module provides the actual Kuramoto machinery so those
structural matches can be tested empirically instead of staying
analogies.

## What was built

### `core/kuramoto.py` (new module)

- `order_parameter(phases) -> (r, ψ)` -- r=0 incoherent, r=1 fully
  synchronized. Empty input returns `(0.0, 0.0)` rather than raising.
- `kuramoto_step(phases, omega, coupling, k, dt=0.05)` -- one explicit-
  Euler step of `dθ_i/dt = ω_i + (K/N)·Σ_j coupling[i,j]·sin(θ_j−θ_i)`.
  `coupling` is used exactly as given (not symmetrized, not row-
  normalized again), so `op_katz.build_transition_matrix`'s output plugs
  in directly.
- `simulate(phases0, omega, coupling, k, steps, dt) -> r_trace` --
  convenience wrapper returning r(t) for a whole run, for sanity-checking
  a `critical_coupling` estimate against the real ODE.
- `spectral_radius(coupling)` -- Λ₁, the same `eigvals`+`max(abs)`
  computation `op_katz.classical_katz_scores` already runs on this exact
  matrix shape. Deliberately not imported from `op_katz` (no dependency
  either direction between a diagnostic utility and a scheduler); flagged
  in both docstrings as the one place to hoist into a shared helper if
  the two ever need to guarantee identical results.
- `frequency_density_at_zero(omega)` -- Gaussian-KDE estimate of g(0)
  (Silverman bandwidth), needed by the K_c formula below. Returns `0.0`
  for degenerate input (fewer than 2 points, or zero variance) rather
  than raising.
- `critical_coupling(coupling, omega)` -- Restrepo, Ott & Hunt's
  eigenvalue approximation (Phys. Rev. E 71, 036151, 2005):
  `K_c = K_0/Λ_1`, `K_0 = 2/(π·g(0))`. Returns `float("inf")` when either
  factor is unavailable (Λ_1=0: no cycles, can never synchronize at any
  K; g(0)=0: degenerate frequency density, the approximation's own
  precondition doesn't hold) -- both cases mean "doesn't fit this
  approximation," not "K_c is small," so they're kept distinct from 0.0.

24 tests in `tests/test_kuramoto.py`: order-parameter identities
(identical phases → r=1; evenly-spaced phases on the circle → r=0;
antipodal pair cancels), step correctness (zero coupling / zero K both
reduce exactly to free rotation `θ + dt·ω`), a full `simulate()` run
showing all-to-all identical-frequency oscillators actually lock
(r: <0.9 → >0.99 over 300 steps) and zero-coupling heterogeneous
oscillators never do (r stays below 0.9 throughout, not just at one
tick — a single-tick check is too noisy at n=6), `spectral_radius`
sanity checks (diagonal matrix, linear scaling), a KDE check against
the closed-form Gaussian density at 0 (loose tolerance — this is a
finite-sample KDE, not exact), and `critical_coupling`'s degenerate
cases plus one test that builds a transition matrix with
`op_katz.build_transition_matrix` directly and checks the K_c formula
against a hand-computed expected value on that exact object — the test
this module exists to make possible.

`ruff` clean; `mypy` on the new file alone attributes zero errors to it
(the 108 errors mypy reports come from unrelated pre-existing modules
pulled in transitively — none in `kuramoto.py` or `test_kuramoto.py`).
Directed sweep (`test_kuramoto.py` + `test_op_katz.py` + `test_katz.py`
+ `test_centrality.py`, the modules this one reads from or parallels):
59/59 passing, no regressions.

## Design decisions

- **Diagnostic utility, not a scheduler.** Same posture as
  `centrality.py`: wiring this into an actual `OpKuramotoScheduler` means
  deciding what a "natural frequency" ω_i even *is* for an operator
  (success rate? inverse mean-time-between-discoveries? something from
  the Elo tracker?) and what firing an operator should do to its
  phase — design questions with no empirical answer yet, not a math
  question this patch can settle by guessing.
- **The K_c formula is flagged as an approximation with real
  preconditions, not a verified threshold.** Restrepo-Ott-Hunt's own
  paper notes the eigenvalue approximation degrades for networks with
  small minimum degree — likely true of an operator pool of a few dozen
  arms, several with few recorded transitions. `critical_coupling`'s
  docstring says this explicitly rather than presenting a number that
  invites over-trusting it.
- **`spectral_radius` duplicated rather than imported from `op_katz`.**
  A one-line function; importing it would create a coupling between a
  standalone diagnostic and a scheduler for a single `eigvals` call.
  Noted in both docstrings for whoever eventually wants to hoist it.
- **Explicit Euler, not an adaptive integrator.** This is a diagnostic
  ("does the graph predict synchronization"), not a physically precise
  simulator; `dt` is exposed for a caller to shrink if a specific
  coupling matrix's fastest mode needs it.

## What this does *not* claim

No claim that fuzzer operators actually behave like phase oscillators.
That's an empirical question this module makes testable, not one it
answers: assign each operator a phase advancing at some rate tied to its
own behavior, couple through `op_katz`'s discovery-transition graph, and
check whether the *measured* order parameter r(t) tracks anything the
fuzzer already cares about (e.g. does r rise before a stall, the way
`analyzer_critical_slowing.py`'s variance/autocorrelation precursor
does? does a real campaign's transition graph's `critical_coupling(...)`
estimate land anywhere near a coupling strength that would occur
naturally?). Nobody has run that experiment; this patch only builds the
instrument.

## Suggested next steps (not done here)

1. Pick a concrete definition of "operator phase" and "operator natural
   frequency" from data the fuzzer already tracks (Elo rating change
   rate? time-since-last-success?), then run `simulate()` against a real
   campaign's `op_katz.build_transition_matrix` output and see whether
   r(t) correlates with anything currently used for stall detection.
2. If step 1 finds something, that's the point to design an actual
   `OpKuramotoScheduler` — not before, per the "diagnostic first" posture
   used for `centrality.py` and (per its own handover) left open for
   `betweenness_centrality` too.
3. Compare `critical_coupling`'s eigenvalue estimate against a real
   `simulate()` sweep over K for an actual campaign's transition matrix,
   to see how far the approximation's known degradation (small minimum
   degree) actually is from the true numerically-found threshold on this
   kind of sparse, small operator graph.

## Files changed

- `src/fuzzer_tool/core/kuramoto.py` (new, ~185 lines)
- `tests/test_kuramoto.py` (new, ~180 lines, 24 tests)
