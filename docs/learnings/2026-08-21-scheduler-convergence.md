# Scheduler convergence: four defects, none of which any test noticed

Date: 2026-08-21
Added: `tests/support/bandit_env.py`, `tests/test_scheduler_convergence.py`,
`random_seed` fixture in `tests/conftest.py`.

## Why

Ten bandits live in `core/schedulers/`. Every existing test checks structure —
the import graph (`test_regression_scheduler_independence`), fallback ordering
(`test_regression_scheduler_fallback_precedence`), per-scheduler internals
(`test_cmaes`, `test_montecarlo`, `test_contextual`). All of them pass for a
scheduler that ignores feedback entirely.

Method is from TigerBeetle's ARR fuzzer (matklad, *A Tale Of Four Fuzzers*):
an idealized environment where the best arm is known **by construction**, so
the acceptance criterion can be *optimality* rather than *doesn't crash*. The
harness is deliberately unrealistic — stationary Bernoulli arms, no operator
interaction — because ambiguity about what "best" means is exactly what makes
a live campaign useless as an oracle.

Arms are drawn from the real `OPERATOR_CATEGORIES` taxonomy, not synthetic
names. `HierarchicalBanditScheduler` silently drops `init_arm`/`record` for
names it cannot map to a category, and `GPUCBScheduler` builds its features
from the same table; a harness using `"op0".."op11"` would pass while feeding
two schedulers nothing at all. The best arm is also placed in a category
alongside weak arms, or the hierarchy hands Hierarchical the answer for free
at the top level.

## Measurements

12 arms, best p=0.30, runner-up p=0.18, base p=0.05. Uniform baseline 0.083.
Tail share = fraction of the final 20% of the campaign spent on the best arm.

**Stationary, 20k rounds, median over 25 seeds:**

| scheduler | tail share | regret slope | verdict |
|---|---:|---:|---|
| GPUCB | 1.000 | 0.00 | converges — but see defect 3 |
| ContextualLinUCB | 1.000 | 0.08 | converges |
| Hierarchical | 0.998 | 0.11 | converges — but see defect 4 |
| MonteCarlo | 0.996 | 0.35 | converges |
| EpsilonGreedy | 0.990 | 0.07 | converges |
| Exp3 | 0.904 | 0.58 | converges |
| Replicator | 0.400 | 0.81 | weak; correct arm, poor concentration |
| **MOpt** | **0.080** | **1.02** | **fails — equals uniform** |
| **CMAES** | **0.005** | **1.01** | **fails — worse than uniform** |

**Non-stationary** (best arm's yield collapses at round 10k, runner-up becomes
best) — tail share on the *new* best arm:

| scheduler | new best | stuck on dead arm |
|---|---:|---:|
| Exp3 | 0.839 | 0.078 |
| ContextualLinUCB | 0.601 | 0.374 |
| MonteCarlo | 0.306 | 0.673 |
| Replicator | 0.195 | 0.184 |
| Hierarchical | 0.139 | 0.829 |
| EpsilonGreedy | 0.002 | 0.990 |
| GPUCB | 0.000 | 0.998 |

Note the inversion: **GP-UCB is the best stationary scheduler and the worst
non-stationary one.** Coverage saturation — an operator exhausting the edges
it can reach — is the regime a real campaign spends most of its time in, so
the stationary ranking is close to the wrong way round for production use.

## Defect 1 — MOptScheduler never converges

Tail share 0.080 against a 0.083 uniform baseline; regret slope 1.02, i.e.
linear. It is statistically indistinguishable from random selection.

`_normalize_to_simplex` applies a **softmax to a vector that is already a
probability distribution**. Particle positions live on the simplex, so entries
are around 1/12 ≈ 0.083 and the spread between largest and smallest is at most
`exp(max - min)` ≈ `exp(0.3)` ≈ 1.35. Velocities are clamped to `max_vel`, so
positions can never leave that narrow band, and the softmax crushes whatever
signal PSO accumulates back toward uniform. Measured after 20k rounds:
`global_best_pos` entries all lie in [0.077, 0.120].

The particle-selection half works — fitness-proportional selection does pick
the one particle with non-zero fitness ~96% of the time. The distribution it
then samples from is simply uniform.

**Fix:** project onto the simplex by clipping to ≥0 and dividing by the sum,
*or* keep particle positions as unconstrained logits and softmax only at
sampling time — which is what `CMAESScheduler` does correctly.

Note the misleading comment in `record()`: "This is the core fix: each
particle's fitness reflects only the outcomes of operators IT chose, enabling
PSO to differentiate." That fix is real and correct; it is downstream of a
projection step that discards the differentiation.

## Defect 2 — CMAESScheduler diverges numerically

Tail share 0.005, *worse* than uniform: it commits fully to one arbitrary arm.
After 20k rounds, σ has grown from 0.3 to ~72 and the mean vector has entries
of magnitude ~1e14, so `_softmax` produces a hard one-hot on whichever logit
happened to be largest — `dict_compound`, a base-rate arm, in the seed-92 run.

Three compounding causes in `_update_cmaes`:

1. **σ has no upper clamp** — only `max(self._sigma, 1e-12)`. The CSA update
   `sigma *= exp(c_s/(1-c_s) * (||ps||²/n - 1))` is a positive feedback loop
   once `||ps||` is large, and the mean update uses `step = lr_mean * sigma`,
   so a growing σ grows `delta`, which grows `ps`.
2. **The `ps` normalisation divides by the wrong quantity.** Standard CSA uses
   `(m_new - m_old)/sigma`. Here the mean step is `lr_mean * sigma * delta`,
   so `(m_new - m_old)/sigma` is `lr_mean * delta` — but the code writes
   `delta / (self._sigma + 1e-12)`, inflating `ps` by `1/sigma²` relative to
   intent.
3. **The `hsigma` damping heuristic reads `self._eval_count` after it has been
   zeroed** at the top of the same function, so
   `(1 - c_s)**(2 * (self._eval_count or 1))` evaluates to the constant
   `(1-0.3)**2` on every generation. `hsigma` is therefore effectively always
   1 and never damps.

**Fix:** at minimum (3) is a clear bug and (1) is a cheap safety net. (2)
needs care — it is the actual CSA formula that is wrong.

## Defect 3 — GPUCBScheduler starves the best arm on ~42% of seeds

Measured: 42/100 seeds end with tail share < 0.5 at 6k rounds. In **every**
failure the best arm has `count == min_samples == 3` and `mean == 0.0`.

In `select_op`, an arm past `min_samples` scores
`mean + beta * max(stddev, 1e-6)`. An arm whose observations are all zero has
`mean == 0` *and* `stddev == 0`, so it scores `2e-6` forever and is never
pulled again. There is **no count-based exploration bonus** — a UCB algorithm
is treating an under-sampled arm as a *certain* one. With p=0.30 and
`min_samples=3`, the best arm draws three zeros with probability 0.7³ = 0.34,
which is the floor of the observed rate.

Second, separate finding: **`select_op` never calls `_rbf` or `_kernel_row`.**
The RBF kernel, `_features`, `_kernel_cache`, and `kernel_matrix()` are all
dead on the selection path. The class docstring describes sharing statistical
strength between correlated operators; the implementation is plain
empirical-stddev UCB with per-arm independence. The inline comment
("Simplified: use the empirical stddev scaled by correlated ops") describes
scaling that does not happen.

**Fix:** replace the confidence term with the standard `beta * sqrt(log(t)/n)`,
which also removes the starvation, and then either wire the kernel in or
delete it and rename the class.

## Defect 4 — Hierarchical starves the best arm's category (~1.5% of seeds)

3/200 seeds. Same shape as defect 3 one level up: the top-level Thompson
sample over categories can drive the posterior of the category containing the
best operator to ~0.02 before the bottom level ever discovers it, after which
the category is never selected again.

1.5% is squarely in the range that produces intermittent, hard-to-reproduce CI
failures. This is why the strict thresholds in the test module run only at the
fixed seed, with the random-seed run asserting a much weaker floor.

## Reproducibility findings

**`CMAESScheduler` is not seedable as constructed.** `__init__` falls back to
`RandPool()` when `rng is None`, and `services/fuzzer.py:1375` constructs it
without an `rng`. `RandPool(None)` seeds from OS entropy, so `--seed` does not
determine CMA-ES behaviour and a crash found under CMA-ES scheduling cannot be
replayed. Every other scheduler draws from the module-level `random` and is
covered by a global `random.seed()`. One-line fix at the construction site.

**Six schedulers use the module-level `random` rather than an injected
generator** (`epsilon_greedy`, `exp3`, `replicator`, `hierarchical`,
`monte_carlo`, `mopt`). That is reproducible via `random.seed()` but couples
every scheduler's stream to every other consumer in the process, so parallel
workers cannot have independent scheduler streams. Not fixed here.

## Unrelated pre-existing flake, found in passing

`tests/test_bloom_exec_dedup.py::TestUpdateBytes::test_reset_on_full_wipes_at_capacity`
fails at a measured rate of **66/20000 = 0.33%**. It inserts 64 `os.urandom(8)`
keys into a capacity-64 filter and asserts `n_added == 64`; a bloom false
positive makes it 63. Because the keys come from `os.urandom`, the failure
cannot be reproduced from any seed — the exact failure mode the session-seed
work is meant to eliminate. Not fixed in this patch.

## Test design notes

- **Two-run seeding**: strict thresholds at fixed seed 92, weak invariant
  (beat uniform 3x) at a random per-session seed. A strict assertion driven by
  a random seed is a debugging session five years from now.
- **The seed is generated in `pytest_configure` and printed in
  `pytest_report_header`**, before collection — so it survives the segfault and
  hang this suite has produced before.
- **Failure rates are pinned as assertions with documented bands** rather than
  hidden behind xfail. A fix to GP-UCB will turn
  `test_gpucb_starvation_rate_is_pinned` red and tell the reader what to do.
- **`test_uniform_selection_does_not_pass`** drives a feedback-ignoring
  scheduler through the same harness and asserts it fails every threshold.
  Without it the thresholds could be satisfied by chance and the module would
  prove nothing — matklad's counting pattern applied to the harness itself.

Runtime: 1.2s for the default suite, 16s including `-m slow`.

## What this exercise was actually for

Two of the ten schedulers do not work, and a third fails 42% of the time. None
of that was visible from 4,700 passing tests, because every one of them asked a
structural question. The finding that changed my model of the subsystem was not
any individual bug but the stationary/non-stationary inversion: the ranking the
Elo arbitration layer is implicitly optimizing for is close to the reverse of
the one that matters once operators saturate.
