# BOGPUCBScheduler: fixing the operator-reachability-test hang

## How this was found

While building `OpKuramotoScheduler` (see the companion handover),
`tests/test_regression_scheduler_operator_reach.py::TestAllSchedulersReachAllOperators::test_every_registered_operator_is_selected[BOGPUCBScheduler]`
was found hanging (not merely slow -- exceeded every timeout tried, up to
several minutes). Bisected by running each scheduler's parametrization
individually with a hard `timeout`; `BOGPUCBScheduler` was the only one that
didn't finish. Confirmed pre-existing and unrelated to the Kuramoto work via
`git stash` against a clean checkout before touching anything.

Four independent, compounding bugs were found and fixed, all in
`core/schedulers/op_bo_gp_ucb.py`. None of them are about this project's
own logic being wrong in principle -- the GP/EI math was always correct --
they're implementation defects that only became visible at this project's
current operator-registry size (218 as of this fix; it was 197 not long
ago and keeps growing).

## Bug 1: O(n^3) refit was pure-Python, not BLAS

`_cholesky`/`_solve_triangular`/`_solve_triangular_T` were hand-rolled over
`list[list[float]]` with Python-level loops and generator-expression
`sum()` calls. At n=218 that's genuinely ~218^3 ~= 10.3M scalar Python
operations per refit. Measured: 200 rounds at 47.6s wall time before any
fix.

Fix: rewrote `_build_kernel_matrix`/`_update_posterior` onto
`numpy.linalg.solve` (LAPACK-backed) -- the same tier of fix `op_katz.py`
and `core/kuramoto.py` already use for their own matrix work, not a new
dependency (numpy is already a project dependency; scipy/sympy/gmpy are
excluded by AGENTS.md rule 51, not used here). A small diagonal nugget
(`_JITTER = 1e-10`) replaces the old Cholesky's implicit
`max(diagonal_term, 1e-12)` regularization: many operators share a
category and therefore an identical one-hot feature vector, making `K`
exactly rank-deficient whenever `noise=0.0` (exercised directly by
`TestNoisyGP::test_zero_noise_is_deterministic`), and `numpy.linalg.solve`
raises on an exactly singular matrix where the old loop-based Cholesky
degraded gracefully instead.

Also fixed in the same pass: `_predict_mean`/`_posterior_variance` did an
O(n) `list.index(op)` lookup per call. With every candidate in a 218-op
ballot queried twice a round (mean + variance), that's an O(n^2)-per-round
cost sitting on top of the refit. Replaced with an O(1) `_op_index` dict
built alongside `_op_list` at refit time.

## Bug 2: `record()` defeated the `refit_interval` cadence

`select_op` already tracks `_pulls_since_refit` against `refit_interval`
(default 100) specifically to bound refit cost -- but `record()` set
`self._needs_refit = True` unconditionally on every single call, so a full
refit ran on nearly every round regardless. Even after bug 1 made each
individual refit cheap (~3ms measured at n=218), 20,000 refits at 3ms each
is still a minute.

Fix: removed the forced set from `record()`. The posterior now refreshes
only via the existing cadence, or immediately the first time it's needed
(`_kernel_matrix is None`, still forced regardless of cadence). Checked
every existing test in `tests/test_regression_bo_gp_ucb.py` before making
this change: none interleaves `record()` calls with posterior reads in a
way that depends on immediate refresh -- each test's first read is always
that always-forced initial refit. All 14 still pass unchanged.

## Bug 3: `_best_mean()` was an O(n) scan called O(n) times per round

`_expected_improvement` calls `_best_mean()` -- `max(m.mean for m in
self._moments.values())` -- once per candidate. With 218 candidates
evaluated every round, that's an O(n^2)-per-round cost. Profiled directly:
this was the single largest remaining cost after bugs 1-2 (~28s of a
42.6s/3000-round profile).

Fix: incremental cache (`_note_mean`, wired into both `record()` and
`init_arm`'s prior-pseudo-observation path), maintaining `(_best_op,
_best_mean_value)` in O(1) for the common case. `RunningMoments.mean` here
is a sliding-window mean, not monotonic, so the one case that can't be
O(1) -- the *current* champion's own mean decreasing -- falls back to a
full O(n) rescan, but only fires on that specific operator's own `record()`
calls, not on every operator's.

## Bug 4: tie-break systematically starved alphabetically-early operators

Independent of the three performance bugs above, and only visible once the
test could actually finish: `select_op`'s tie-break,
`max(scores.items(), key=lambda kv: (kv[1], kv[0]))`, picks the
lexicographically **largest** name on an exact score tie. Every
never-observed operator gets the identical default EI (mu=0, sigma=1 --
maximum uncertainty), so the entire never-observed group ties exactly
(verified directly: `phi(0) = 0.3989...` for every one of them at the
start of a run). The old tie-break deterministically and permanently
favored whichever name sorts last among however many operators are
currently tied, starving early names once a later-sorting operator entered
that tied group. Reproduced directly: 16 operators (`bit_flip`,
`arithmetic`, `ascii_num`, ...) never selected once in 20,000 rounds
against the full 218-op ballot.

Fix: `BOGPUCBScheduler` gained a `rng: RandPool | None = None` constructor
parameter (defaulting to `get_default_rand_pool()`), the same
rng-or-default idiom `UCBBase`/`DUCBScheduler` already use (Hard Rule 16).
`select_op` now breaks ties by uniform-random choice among the full argmax
set (`self._rng.choice(tied)`), the identical pattern `UCBBase.select_op`
already uses for its own unpulled-arm case
(`self._rng.choice(unpulled)`) -- not a new idiom invented for this fix.
`services/fuzzer.py`'s construction site now passes `rng=self._rng` so a
given `--seed` still reproduces the campaign.

`bo_gp_ucb` is a graduated scheduler (present in `_FALLBACK_PRECEDENCE`,
reachable without `--elo`), unlike the exploratory arms `op_katz`/
`op_kuramoto`/`op_tang` -- this bug was live on every campaign using it,
not gated behind an experimental flag.

## Net result

| Scenario | Before | After |
|---|---|---|
| 200 rounds, n=218, full ballot | 47.6s | <1s |
| 20,000 rounds (the actual reach-test scenario) | hung indefinitely | 5.2s |
| `test_every_registered_operator_is_selected[BOGPUCBScheduler]` | hung | passes in 8.65s |
| `tests/test_regression_bo_gp_ucb.py` (14 tests) | n/a | still 14/14, unchanged values |

## What wasn't touched

`test_regression_scheduler_operator_reach.py::test_every_exported_scheduler_is_covered`
and `test_regression_track_op_effect_coverage.py::test_every_ballot_name_is_mapped`
both fail independently of this fix (`CorralScheduler`/`FEWAScheduler`/
`SoftmaxScheduler`/`TopKScheduler`/etc. missing from unrelated coverage
lists, and `fewa`/`softmax`/`topk` missing from a kwargs map) -- confirmed
pre-existing via `git stash` against a clean checkout, untouched by this
patch, out of scope here.

## Rule 52 caveat

Same caveat as the companion Kuramoto handover: no clang or fuzzgoat
available in this sandbox, so the numbers above are synthetic
(`REGISTRY.names()` driven directly, not a real campaign) rather than a
fuzzgoat calibration run. Owed before trusting this on real fuzzing.
