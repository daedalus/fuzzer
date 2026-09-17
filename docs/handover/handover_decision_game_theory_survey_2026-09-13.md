# Handover — decision theory / game theory / marginal cost integration survey

**Date:** 2026-09-13
**Base (fuzzer):** `937e9c4` (HEAD at time of analysis)
**Sources:** five Wikipedia articles brought in for integration analysis —
[Decision theory](https://en.wikipedia.org/wiki/Decision_theory),
[Game theory](https://en.wikipedia.org/wiki/Game_theory),
[Best response](https://en.wikipedia.org/wiki/Best_response),
[Marginal cost](https://en.wikipedia.org/wiki/Marginal_cost),
[Combinatorial game theory](https://en.wikipedia.org/wiki/Combinatorial_game_theory).

Analysis only. No patch in this handover.

**Negative result first:** most of what these five pages would normally
motivate is already in the tree, under other names. `grep -rniE
'(shapley|replicator|nash|elo|alpha.?beta|minimax|ucb|exp3)'` across `core/`
and `core/schedulers/` turns up Shapley attribution (`core/shapley.py`),
replicator dynamics / ESS (`core/schedulers/op_replicator.py`), Elo with a
softmax selector (`core/elo.py`), and alpha-beta minimax in two places
(`core/cond_stmt.py`, `core/smt_solver.py`). A survey that just re-proposed
those would be re-deriving work already merged. The two things actually worth
reporting are (1) a concrete gap — marginal cost has no representation
anywhere, only total/average cost — and (2) a bug found by taking the
"minimax game" framing in `smt_solver.py` at face value and checking whether
it computes what its own docstring claims.

---

## 1. Gap: no marginal-cost signal anywhere (Marginal Cost page)

`grep -rniE 'marginal' src/ tests/ docs/` returns nothing except this
document. Every cost-aware mechanism in the tree reasons about **total** or
**average** cost, never the first difference:

- `core/schedulers/op_replicator.py` fitness is `_fitness_sum[op] /
  _fitness_count[op]` (`:134`) — an average over a fixed `window_size`
  (default 200), reset to zero every window (`:189-191`). It cannot see
  whether an operator's cost-per-edge is rising or falling within a window,
  only the window's mean.
- `core/elo.py`'s K-factor and rating decay are exponential smoothers over
  match outcomes, not a cost/output derivative.
- `core/parallel_cost_partition.py` calls `core.job_scheduling.multifit` to
  balance **total** load across `-j N` workers (its own docstring, `:5-9`:
  "roughly equal total load"). Multifit is a makespan bin-packer; it has no
  notion of the *rate of change* of cost as more work is assigned to a
  worker, only the sum.
- `core/cost_ledger.py` exposes `seed_exec_time`, `effective_fuzz_count` —
  point measurements and EWMAs, never a difference between consecutive
  measurements.

The Marginal Cost page's central identity, MC = ΔTotalCost/ΔQuantity, and
its central result — a rational allocator keeps investing in an activity
only while MC is below the marginal value of what it buys, and stops (or
reallocates) once MC exceeds it — has no analogue here. What exists instead
is uniformly windowed or EWMA-smoothed *average* cost, which is blind to the
shape of the cost curve within the window.

**Concrete proposal.** Add a marginal-cost estimator next to the existing
per-operator/per-worker fitness trackers: for operator *i*, maintain the last
two window-boundary values of `(execs_i, edges_i)` and compute
`MC_i = Δexecs_i / Δedges_i` (cost per marginal edge, the natural unit here
since "quantity produced" for a fuzzer is discovered coverage, not time).
Two consumers, in order of how directly they map onto the wiki page:

1. **`parallel_cost_partition.py`**: today workers are balanced so that
   Σcost is roughly equal per worker (this is what `multifit` does). The
   economically efficient allocation instead equalizes **marginal** cost
   across workers — the standard result that at an efficient allocation, no
   reassignment of one more unit of work between two workers lowers total
   cost, which holds exactly when their marginal costs are equal. This is a
   different objective than makespan balancing and could be checked
   empirically: does equalizing `MC_i` across workers converge total-corpus
   coverage faster than Multifit's total-cost balance, on a target with
   heterogeneous seed costs (the existing `docs/handover/handover_pending
   _2026-09-06.md` §P3-3 EWMA-drift caveat still applies to whichever cost
   signal feeds this).
2. **`op_replicator.py`**: replace (or add alongside) the fixed `window_size`
   cutoff with a marginal-cost stopping rule — stop investing further
   executions in an operator once its `MC_i` (executions per new edge, over
   a short trailing window) exceeds some multiple of the population-average
   `MC`, rather than waiting for a fixed 200-execution window to elapse
   regardless of whether the operator's cost curve is still falling
   (increasing returns) or has started rising (diminishing returns, the
   U-shaped MC curve from the wiki page). This is a more direct fit than the
   worker-partition case because `op_replicator.py` already tracks per-operator
   `_fitness_sum`/`_fitness_count`; a marginal-cost variant is a small
   extension, not a new subsystem.

Not measured yet — this section is the proposal, not a result. It would need
a synthetic or real-target harness comparing "fixed window" vs "marginal-cost
stop" allocation the same way `handover_control_theory_loops_2026-09-12.md`
§2 measured L1's dwell sweep, before writing a patch.

**2026-09-14 update — consumer #2 implemented, plus two more sites beyond
the original proposal's scope; consumer #1 (worker partitioning) still
open.**
`core/marginal_cost.py` now provides the estimator proposed above:
`MarginalCostTracker.record_snapshot(key, cumulative_cost, cumulative_output)`
taken at each window boundary, `.marginal_cost(key)` returning
`Δcost/Δoutput` between the last two snapshots, `.population_average_mc()`,
and `.should_stop(key, multiplier)` implementing the "MC_i exceeds a multiple
of the population average" rule (including the edge case a naive port would
miss: a window where an operator spent cost but produced zero output reads
as `None` from `marginal_cost()`, but `should_stop()` still flags it as
worse than any finite MC rather than silently treating undefined-ratio as
"no signal, keep going").

`op_replicator.py` (consumer #2) is wired: constructor gained an optional
`marginal_cost_stop_multiplier` (default `None`, so `--replicator` alone is
byte-for-byte unaffected); `record()` now also feeds cumulative
`(execs, discoveries)` counters per operator, `_replicator_update()` snapshots
them into the tracker at every window boundary regardless of whether the
multiplier is set, and — only when it *is* set — applies `min(growth,
1 - eta)` to any operator `should_stop()` flags, on top of (never instead
of) the ordinary fitness-relative-to-mean update. `operator_marginal_costs()`
exposes the current per-operator MC for diagnostics/stats display. CLI flag
`--replicator-mc-stop-multiplier` threads it through. 13 tests in
`tests/core/test_marginal_cost.py` (estimator in isolation) + 6 in
`tests/test_regression_marginal_cost_stop.py` (disabled-by-default parity,
snapshot timing, and a same-seed baseline-vs-gated comparison showing the
flagged operator's population share can only shrink further, never less,
under the stop rule).

**Two more sites turned out to have the identical shape** and got the same
treatment (found by grepping the tree for other "cumulative reward / cost"
ratios computed once per decision with no cross-window memory):

- `services/seed_picker.py::_pick_ecofuzz_seed` — EcoFuzz's
  `energy = reward_prob / cost` is itself a *lifetime* average (both terms
  cumulative since the seed was first seen), so a seed with a strong early
  streak keeps healthy energy long after it has actually gone cold. Optional
  `f._ecofuzz_mc_penalty_multiplier` (default `None`): every pick snapshots
  each seed's cumulative `(cost, coverage_edges)` unconditionally, and when
  set, divides the weight of any seed `should_stop()` flags by the
  multiplier — same "additional, only ever shrinks" contract as the
  replicator case. CLI flag `--ecofuzz-mc-penalty-multiplier`. 6 tests in
  `tests/test_regression_ecofuzz_mc_penalty.py`, including one that
  hand-derives the exact weight ratio and picks a fixed rng fraction that
  provably crosses the cheap/expensive selection boundary only once the
  penalty is applied.
- `core/schedulers/op_mopt.py::_pso_update` — particle fitness
  (`disc/execs_in_window`, `_update_fitness`) is a within-window mean with
  no memory of the *previous* window, same pre-fix shape as `op_replicator.py`.
  Optional `marginal_cost_stop_multiplier` (default `None`) on
  `MOptScheduler`; each particle now also tracks cumulative
  `(cum_execs, cum_discoveries)`, snapshotted at every PSO update, and
  `min(fitness, fitness / multiplier)` is applied before the pbest/gbest
  comparison for any particle `should_stop()` flags. New
  `particle_marginal_costs()` diagnostic. CLI flag
  `--mopt-mc-stop-multiplier`. 6 tests in
  `tests/test_regression_mopt_mc_stop.py`.

Both follow the same pattern as `op_replicator.py`: additive, off by default,
and the penalty can only ever shrink a flagged key's weight/fitness relative
to the baseline, never grow it, for any multiplier value — asserted directly
in both new test files.

Consumer #1 (`parallel_cost_partition.py` — equalize `MC_i` across workers

instead of balancing total cost via Multifit) is **not** implemented. It's a
different objective from what Multifit computes today, not a small
extension like the replicator case, and the doc above is explicit that it
needs an empirical A/B before a patch is warranted — that measurement
hasn't been run. Left as the open item.

---

## 2. Bug found: `smt_solver.py`'s "minimax game" carries no signal

### 2.1 What the docstring claims

`Z3Solver._alpha_beta_wall` (`core/smt_solver.py:655-736`), reached via
`solve_comparison_wall` (`:614-653`), frames ordering a comparison wall as:

> Fuzzer (maximizer): chooses which comparison to solve next.
> Target (minimizer): the wall's structure resists progress.

with `evaluate()` (`:674-690`) scoring a subset by unique vs. overlapping
tainted byte offsets — non-overlapping offsets score `+2` each, overlapping
ones cost `-1`. That is a reasonable *objective* (it is exactly the same
overlap signal `core/path_constraints.py::_overlapping()` computes for a
different purpose, §3 below). The bug is in how the recursion applies it.

### 2.2 What it actually computes

At each recursive step the incremental term is `evaluate([cond])` —
`evaluate()` called on a **singleton** list (`:709`, `:724`). Reading
`evaluate()`'s own body: `covered_offsets` starts empty on every call, so a
one-element list can never have `overlap > 0`. The overlap penalty the
docstring advertises can only ever fire on the *leftover* list scored once
at the depth cutoff (`:700-701`) — never on the sequence actually being
built. So the search is, by construction, blind to overlap for every
condition it schedules, and only accidentally sees overlap in whatever
remains unscheduled when `depth` runs out.

Traced by hand and confirmed by re-running the exact recursion standalone
(`evaluate`/`minimax` copied verbatim from `:674-731`, not reimplemented) at
`937e9c4`, three symmetric fixtures of 4 identical-shaped conditions each —
fully disjoint offsets, all four sharing one offset (maximum overlap), and
two overlapping pairs — all three produce **identical value = 0.0** and the
identity order:

| fixture | value |
|---|---|
| 4 disjoint singletons | 0.0 |
| 4 conditions sharing one offset (max overlap) | 0.0 |
| 2 overlapping pairs | 0.0 |

Overlap structure that spans the full spread from "no conflict" to
"complete conflict" is invisible to the returned value. Sweeping `n`
identical singleton conditions from 2 to 6 makes the reason clear:

| n | value |
|---|---|
| 2 | 0.0 |
| 3 | 2.0 |
| 4 | 0.0 |
| 5 | 2.0 |
| 6 | 0.0 |

Value depends only on the **parity of n** (one leftover `+evaluate`-term
when the maximizer gets one more move than the minimizer, zero otherwise —
the `+`/`-` sign the maximizer/minimizer roles apply to the *same* per-item
term cancels pairwise). It does not depend on offsets, overlap, or content
at all in the symmetric case. For an asymmetric fixture (6 conditions,
varying widths, mixed overlap) the search does produce a non-trivial order,
but that order matches neither a plain greedy sort by taint width
(`sorted(conds, key=len(offsets), reverse=True)`) nor any ordering that
visibly respects the overlap graph — it is an artifact of which
depth-parity slot each condition lands in, not of the objective the
docstring describes.

### 2.3 Why this is a combinatorial-game-theory framing error, specifically

The Best Response and Combinatorial Game Theory pages both make the same
point from different directions: a minimax/alternating-turn model is only
the right tool when there is a second party whose moves genuinely oppose the
first. Here there isn't one. The "target" doesn't choose a move in response
to which comparison the fuzzer solves next — the wall's structure (which
byte offsets each comparison touches) is fixed at trace-capture time,
before the search runs. Modeling a single-agent sequencing problem as a
two-player zero-sum game, with the second "player" implemented by negating
the same objective on alternate recursion levels, is exactly the
degenerate case the sign-cancellation above exposes.

The `evaluate()` function's own logic already gestures at the right
non-adversarial framing: this is a **disjunctive sum** (Combinatorial Game
Theory page, §"sum of two games") of independent overlap-connected
components. Conditions whose taint offsets never intersect any other
condition's offsets don't interact — solving them in any order or in
parallel gives the same total value, by the same reasoning
`core/path_constraints.py::_overlapping()` already uses to decide, per
*single* branch, whether it needs z3 at all (`negate()`, `:276-289`: no
overlap → closed-form `_direct_solve`; overlap → `_solve_z3` with only the
overlapping records as constraints). `path_constraints.py` gets this right
for one branch at a time. `smt_solver.py`'s wall-ordering search is solving
a version of the same problem for a *set* of branches and gets it wrong,
because it reaches for alternating-turn minimax instead of the disjoint-sum
decomposition the rest of the codebase already relies on.

**Concrete proposal**, cheaper than the current search and provably correct
on the symmetric fixtures above (value becomes strictly overlap-sensitive
instead of parity-only):

1. Build the overlap graph over `conditions` by pairwise offset intersection
   (the same primitive as `path_constraints.py::_overlapping()`, or reuse
   the union-by-size `_UnionFind` already added to
   `core/overlap_density.py` for exactly this kind of connected-component
   problem).
2. Within each connected component (typically small — most walls are mostly
   non-interacting per the taint data `evaluate()` already reads), keep
   `_alpha_beta_wall`'s search — real coupling justifies real search there.
3. Between components, order is provably free (disjunctive sum): concatenate
   components' solved sequences in any order, e.g. widest-taint-first to
   match `path_constraints.py::frontier()`'s existing convention (`:147`).

This also shrinks the search space from over the whole wall to over its
largest connected component, which for a mostly-disjoint wall is
exponentially cheaper than searching all of it — a genuine win, not just a
correctness fix, once the wall has more than a handful of conditions.

**Status: implemented (2026-09-13, same day).** `_connected_components_by_offset`
added to `core/smt_solver.py` (module-level, reuses `_UnionFind` from
`core/overlap_density.py` per the proposal above). `_alpha_beta_wall` now
partitions first, runs the unchanged minimax only within each component,
and concatenates components widest-first (by summed `len(offsets)`, singleton
components skip the search entirely). Re-running the exact three symmetric
fixtures from §2.2 now shows the partitioning actually distinguishing the
overlap structure — 4 singleton components for the disjoint fixture vs. 1
four-element component for the full-overlap fixture, where before both were
searched as one undifferentiated 4-condition wall. Note this does **not**
fix the residual parity artifact *within* a genuinely coupled component
(§2.2's `evaluate([cond])`-on-a-singleton issue still applies to the
minimax call inside `solve_component`) — that's now bounded to small,
truly-interacting components instead of silently affecting the whole wall,
which was the actual bug, but it's a known remaining limitation, not
claimed as fixed. Regression tests in
`tests/test_regression_alpha_beta_wall_disjunctive.py` (13 cases: component
partitioning, empty/singleton/no-offset edge cases, widest-first ordering,
independent-pair separation). `tests/test_smt_solver.py` and
`tests/test_game_theory.py` still pass unchanged (84 passed, 53 skipped for
missing `z3-solver` in this sandbox — pre-existing, unrelated to this
change).

Also noted in passing while implementing this, out of scope for the fix:
`core/cond_stmt.py::solve_comparison_wall_minimax` (`:317-410`) is a second,
independent minimax implementation of the same "comparison wall" idea, with
its own bug (the minimizing branch's loop variable `_cond` is unused, so
every branch of that loop is computed against identical unmutated state —
the loop does `branching_limit` redundant recursions of the same call
instead of exploring different target responses). It has no callers and no
tests anywhere in the tree (`grep -rn "solve_comparison_wall_minimax"` only
finds its own definition) — looks like an earlier, abandoned attempt at the
same idea `Z3Solver.solve_comparison_wall` implements. Worth a follow-up
decision (delete it, or fix and wire it in) but not addressed here.

---

## 3. What's already right and shouldn't be touched

- `core/path_constraints.py::negate()`/`_overlapping()` — this **is** the
  disjunctive-sum decomposition done correctly, at the single-branch level:
  no-overlap → cheap closed form, overlap → z3 with only the genuinely
  coupled records. §2's proposal is this same pattern applied one level up,
  to ordering a *set* of branches, not a new idea.
- `core/elo.py::_softmax_select` (`:37-56`) is the Best Response page's
  "smoothed best response" function verbatim — softmax over expected payoff
  with a temperature/γ parameter, used specifically because pure best
  response "jumps" discontinuously between actions on the slightest payoff
  difference, which the wiki page notes is both psychologically unrealistic
  and numerically brittle near indifference. No change needed; worth a
  one-line docstring cross-reference so a future reader doesn't reinvent
  softmax selection while reading the Best Response page.
- `core/schedulers/op_replicator.py`'s ESS/Nash-equilibrium claim in its
  docstring (`:19-25`) is *not* verified anywhere in this survey — whether
  the operator-scheduling game is actually a potential game (the condition
  under which the Best Response page's convergence theorem applies) was out
  of scope here and is worth a follow-up, but is a separate, smaller
  question than §§1-2.

---

## 4. Open questions gating further work

- **§1**: is `Δexecs/Δedges` the right marginal quantity, or should it be
  wall-clock time per edge (execs and wall-clock diverge under `--hail-mary`
  and job-scheduler-gated maintenance passes, per prior handovers)? Needs
  deciding before the estimator is written, not after.
- **§2**: *(resolved by implementation above — kept here for the record)*
  how large are connected components on real walls in practice? The fix
  is committed regardless since it's strictly more correct than before
  even in the worst case (one fully-connected component degrades gracefully
  to today's full search, just no longer silently blind to overlap
  elsewhere in the wall), but the *performance* win depends on component
  size — still worth measuring the distribution on a real target (e.g. the
  same ffmpeg/sqlite corpora used in prior `_swap_pair` validation work) to
  know how much it actually saves in practice.
- Not investigated this turn: whether `op_katz.py`/`op_tang.py`'s
  Elo-arbitrated scheduling has the same false-minimax pattern as §2 — they
  weren't in scope for this survey but use adjacent machinery and are worth
  a similar sanity check.
