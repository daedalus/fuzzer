# QEA: literature-faithful rotation gate (Han & Kim Q-gate)

## Background

`handover_qea_hilbert_space_analysis_2026-08-31.md` (§5 of `handover_FINDINGS.md`)
established that `core/qea.py`'s representation is a real-valued, per-bit
Bernoulli field (`P(bit=0)=α²`), not a point in a `2ⁿ`-dimensional Hilbert
space — a mean-field product distribution, matching what Han & Kim's own
combinatorial-optimization QEA uses in practice. A prior fix (`2edbce1`)
already corrected `rotation_gate()` from a linear `α ± delta` walk to a true
trigonometric rotation on `(α, β)`.

This handover closes the remaining fidelity gap in that same function: the
*rule* deciding rotation direction and magnitude did not match the published
algorithm.

## The gap

Han & Kim's Q-gate compares, per bit, the current collapsed solution's bit
`x_i` against the bit `b_i` of the best solution found so far for that
individual's lineage, together with whether `x`'s fitness is at least as
good as `b`'s. The standard lookup table used across the literature (e.g.
the version reproduced in arXiv:1101.0362, itself citing Han & Kim's
original combinatorial QEA) reduces, once amplitudes are kept non-negative
(the convention this codebase already uses), to:

- `f(x) >= f(b)`: `Δθ=0` for every bit, unconditionally.
- `f(x) < f(b)`, `x_i == b_i`: `Δθ=0` — the bit already agrees with best.
- `f(x) < f(b)`, `x_i != b_i`: rotate toward `b_i`.

The previous implementation had no notion of `b` at all. It rotated every
bit, every call, using only `x_i` and a global `improved` flag — so a bit
that already agreed with the best-known solution kept getting perturbed
away from it whenever the *overall* result wasn't an improvement, and a bit
that was actually wrong got reinforced whenever the overall result *was*
an improvement, regardless of whether that specific bit had anything to do
with it.

`QEAIndividual` already carried the state this needed — `best_collapsed`
(`b`) and `edge_count` (a cheap available proxy for `f`) — but
`best_collapsed` was write-once at construction and never updated, and
`edge_count` was never compared against the current call's `edge_count`.

## What changed

`rotation_gate()` gained an opt-in `best: bytes | None = None` parameter:

- `best=None` (default at the function level): unchanged legacy behavior,
  byte-for-byte identical to before this patch. All pre-existing unit tests
  of the bare function pass unmodified.
- `best=<bytes>`: the literature-faithful table above, vectorized the same
  way the existing trig rotation already was (no per-bit Python loop).

`QEALifecycle.on_fuzz_result()` now uses the faithful mode by default:

- `is_improved = edge_count >= parent.edge_count` when the parent has a
  tracked best (falls back to `new_coverage` for a freshly-constructed
  individual with no tracked best yet, i.e. `best_collapsed == b""` —
  routing through `best=None`, not an all-zero `best`, since an all-zero
  default would be an arbitrary tie-break disguised as a real target).
- On improvement, `parent.best_collapsed`/`parent.edge_count` are promoted
  to the current call's result (elitist replacement), and no rotation is
  applied — matching the table's `Δθ=0` rows.
- On non-improvement, `rotation_gate` rotates only the bits that disagree
  with `best_collapsed`, toward it.

The intra-byte coupling extension (`update_couplings`, a classical
Ising/Gibbs-sampling layer, unrelated to Han & Kim's Q-gate) is untouched
and still keyed off `new_coverage` directly.

## Tests

- `tests/test_qea.py::TestRotationGateFaithfulMode` (new): the four table
  rows in isolation, per-bit independence within a mixed byte, and the
  one case the two modes provably disagree on (`x_i == b_i`, not
  improved) as an explicit differential test between `best=None` and a
  matching `best`.
- Three existing integration tests in `tests/test_qea.py` and
  `tests/test_regression_qea_fixes.py` asserted the old self-referential
  direction at the `QEALifecycle.on_fuzz_result` level; updated to assert
  the new (correct) behavior — improvement promotes `best_collapsed`
  without moving amplitudes, non-improvement rotates only disagreeing
  bits toward the tracked best. Diffs kept minimal; no assertions removed,
  only redirected at what the faithful table actually specifies.
- Full `test_qea*`/`test_regression_qea*` suite: 168/168 passing, including
  5 repeated runs to rule out flakiness from `collapse()`'s stochastic
  sampling (the empty-`best_collapsed` fallback path was flaky before it
  was routed to `best=None` instead of an implicit all-zero `best` — see
  commit for the specific failure).

Full-repo suite not re-run for this patch (single-module change, isolated
call sites, verified no other file calls `rotation_gate`).
