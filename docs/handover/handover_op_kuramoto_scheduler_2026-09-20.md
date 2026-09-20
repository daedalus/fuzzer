# OpKuramotoScheduler: wiring the Kuramoto diagnostic into a real scheduler

## Background

`core/kuramoto.py` (see `handover_kuramoto_oscillators_2026-09-19.md`) is a
standalone diagnostic: the order parameter, the stepping ODE, and the
Restrepo-Ott-Hunt critical-coupling estimate, with no operator-selection
opinion attached. Two follow-on commits since then (`desync.py`,
`circular_stats.py`) reused the same order-parameter math for worker-sync
timing and record-offset phase-locking respectively -- neither is a
scheduler. This handover is the actual "build `OpKuramotoScheduler`"
follow-up the original diagnostic's suggested next steps pointed at.

## What was unresolved

The diagnostic module deliberately took no position on two questions:

1. What is an operator's natural frequency?
2. What should firing an operator do to its phase?

Both are now answered, in `core/schedulers/op_kuramoto.py`:

1. **omega_i = omega_scale * success_rate_i.** Reuses `op_katz.py`'s own
   beta convention (raw success rate, not its complement -- operator
   selection wants to exploit, unlike seed selection's frontier-seeking
   bias) rather than inventing a new one.
2. **Firing an op does not perturb its phase directly.** Phases only move
   through the shared ODE (`kuramoto_step`), batched every
   `recompute_batch` calls -- same reasoning `WhittleIndexScheduler`/
   `op_tang` already use for their own expensive recomputation.

## Design choices worth flagging for the next person

- **Coupling** is `op_katz.build_transition_matrix` applied directly to
  this scheduler's own discovery-linked `transition_counts` -- reused, not
  reimplemented. This is literally what the original handover's step 1
  suggested running.
- **Selection score**: `rate_i * (1 + r * cos(theta_i - psi))`. Own rate
  amplified by phase-alignment with the coherent cluster, mirroring
  `OpKatzScheduler.scores`'s "own rate amplified by a structural signal"
  frame (centrality there, phase coherence here). The `r *` factor is load
  bearing, not cosmetic: `core/kuramoto.py`'s own `order_parameter`
  docstring says `psi` is meaningless when `r` is near 0, so multiplying by
  `r` makes the alignment term vanish exactly when the diagnostic module
  says it should, rather than chasing a meaningless angle on a cold or
  incoherent population.
- **`k` (coupling strength) is a plain constructor argument**, not
  auto-derived from `critical_coupling`. That estimate is exposed only via
  `diagnostics()`. `core/kuramoto.py` already flags the eigenvalue
  approximation as degrading on small/sparse graphs -- likely true of an
  operator pool of a few dozen active arms -- so silently driving selection
  off a number the diagnostic module itself distrusts would repeat the
  `kl_ducb` paper-fidelity mistake (`handover_kl_ducb_paper_fidelity_2026-09-14.md`).

## What this is not

Not a claim that operators behave like phase oscillators, or that phase
alignment beats plain rate as an exploitation signal. That's exactly the
empirical question the original diagnostic handover left open. This makes
the question answerable (`tools/bench_paired.py`,
`tests/support/bandit_env.py`'s convergence harness) rather than answering
it here.

## Status

Off by default, Elo-only, absent from `_FALLBACK_PRECEDENCE` (see the
comment block there) -- same posture `op_katz`/`op_tang`/
`WhittleIndexScheduler` had before their own harness runs. Wired into:
ballot (`operator_strategy_pool`), `select_op` dispatch, `_register_arms`,
the shared `record()` fan-out, CLI (`--op-kuramoto` + five tuning flags),
`_HAIL_MARY_FLAGS`, and the two test-scaffolding lists
(`tests/support/operator_env.py`'s `BALLOT_SCHEDULERS`,
`test_regression_scheduler_fallback_precedence.py`'s `_SCHEDULER_ATTRS`).
Deliberately **not** added to `core/schedulers/__init__.py`'s `__all__` or
`test_regression_scheduler_operator_reach.py`'s exhaustive list --
`op_katz`/`op_tang`/`op_kruskal_count`/`op_credit` are excluded from both
for the same reason (see that test file's own `test_every_exported_...`
docstring), and this arm's cold-start reachability profile (an unattempted
op scores exactly 0 until its first pull, reached only via the shift-and-
sample draw's epsilon floor) is the same one `op_katz` already has and is
not held to that exhaustive bar for.

`tests/test_op_kuramoto.py`: 21 tests, all passing, covering construction
validation, the discovery-linked transition semantics, batched-vs-immediate
phase advance (including a zero-coupling/zero-omega free-rotation identity
check mirroring `core/kuramoto.py`'s own test), the `r * cos(...)`
degrade-to-plain-rate behavior, cold-start reachability, `diagnostics()`,
and a qualitative all-to-all synchronization sanity check against strong
coupling.

## Suggested next steps (unchanged from the original diagnostic's, now
actually runnable against this scheduler)

1. `tests/support/bandit_env.py`'s convergence harness (RELIABLE/STUCK/
   RECOVERS regimes) against `OpKuramotoScheduler`.
2. A real `tools/bench_paired.py` A/B against the current Elo arbiter,
   the same benchmark `op_tang`'s and `op_katz`'s own negative/positive
   results came from.
3. If (1)/(2) show nothing, document a negative result the same way
   `op_tang`'s seed-side and operator-side runs were documented, rather
   than leaving the question open indefinitely.

## Rule 52 caveat

AGENTS.md rule 52 (added upstream mid-session, commit `4195c3b3`) asks for
calibration against the fuzzgoat target built with clang. Neither clang nor
a fuzzgoat checkout is available in the sandbox this patch was built in
(`clang: not found`; no `fuzzgoat` directory anywhere reachable) -- the same
constraint the ffmpeg/ASAN validation work hit earlier this project
(documented then as "sin clang disponible"). All testing here is the unit
suite above plus the synthetic scaling/profiling runs in the companion
`handover_bo_gp_ucb_hang_fix_2026-09-20.md`. A real fuzzgoat+clang
calibration run is still owed before either this scheduler or the
`bo_gp_ucb` fixes are trusted on a real campaign.
