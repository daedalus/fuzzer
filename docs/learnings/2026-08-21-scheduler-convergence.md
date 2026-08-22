# Scheduler convergence: five defects found, four fixed

Date: 2026-08-21

Added: `tests/support/bandit_env.py`, `tests/test_scheduler_convergence.py`,
`--fuzz-seed` / `random_seed` in `tests/conftest.py`.
Changed: `core/schedulers/{mopt,cmaes,gp_ucb,hierarchical}.py`,
`services/fuzzer.py`, and three existing tests.

## Why

Ten bandits live in `core/schedulers/`. Every existing test checks structure —
the import graph (`test_regression_scheduler_independence`), fallback ordering
(`test_regression_scheduler_fallback_precedence`), per-scheduler internals
(`test_cmaes`, `test_montecarlo`, `test_contextual`). All of them pass for a
scheduler that ignores feedback entirely, and two of them did.

Method is from TigerBeetle's ARR fuzzer (matklad, *A Tale Of Four Fuzzers*):
an idealized environment where the best arm is known **by construction**, so
the acceptance criterion can be *optimality* rather than *doesn't crash*. The
harness is deliberately unrealistic — stationary Bernoulli arms, no operator
interaction — because ambiguity about what "best" means is exactly what makes
a live campaign useless as an oracle.

Two harness details that matter. Arms are drawn from the real
`OPERATOR_CATEGORIES` taxonomy, because `HierarchicalBanditScheduler` silently
drops `init_arm`/`record` for names it cannot map to a category and
`GPUCBScheduler` builds its features from the same table — synthetic `"op0"`
names would have produced a green test that fed two schedulers nothing. And
the best arm is placed in a category alongside weak arms, or the hierarchy
hands Hierarchical the answer for free at the top level.

## Results

12 arms, best p=0.30, runner-up p=0.18, base p=0.05. Uniform baseline 0.083.
Tail share = fraction of the final 20% of the campaign spent on the best arm.
Stationary at 6k rounds, min/median over 40 seeds; decay at 20k with the
switch at 10k, median over 10 seeds.

| scheduler | before (med) | after (min / med) | decay: new arm | decay: dead arm |
|---|---:|---:|---:|---:|
| Hierarchical | 0.996 | 0.970 / **0.994** | 0.230 → **0.993** | 0.740 → **0.001** |
| ContextualLinUCB | 1.000 | 0.983 / 1.000 | 0.630 | 0.348 |
| MonteCarlo | 0.996 | 0.978 / 0.990 | 0.037 | 0.937 |
| EpsilonGreedy | 0.990 | 0.920 / 0.934 | 0.502 | 0.490 |
| Exp3 | 0.904 | 0.873 / 0.903 | 0.625 | 0.287 |
| GPUCB | 0.000 (42% fail) | 0.720 / **0.828** | 0.001 → **0.422** | 0.998 → **0.303** |
| MOpt | **0.080** | 0.258 / **0.382** | 0.091 → **0.284** | 0.081 → 0.201 |
| Replicator | 0.400 | 0.400 (20k) | 0.188 | 0.194 |
| CMAES | **0.005** | 0.080 (bimodal) | 0.075 | 0.076 |

Note the stationary/non-stationary inversion, which survives the fixes:
MonteCarlo is near-perfect stationary (0.990) and nearly helpless under decay
(0.037). Coverage saturation — an operator exhausting the edges it can reach —
is the regime a real campaign spends most of its time in, so the stationary
ranking is close to the wrong way round for production use. Hierarchical is
now the only scheduler strong in both.

---

## Defect 1 — MOptScheduler was `random.choice`

Tail share 0.080 against a 0.083 uniform baseline; regret slope 1.02.
Statistically indistinguishable from random selection.

**Five stacked bugs, each hidden by the one above it.** Fixing any one alone
changed nothing measurable, which is why no amount of unit testing would have
surfaced them.

1. **`_normalize_to_simplex` softmaxed an already-normalized vector.**
   Positions live on the simplex, so entries are ~1/12 and the post-softmax
   spread is at most `exp(max−min)` ≈ 1.35. With `max_vel` clamping the step,
   no particle could ever concentrate.
   *Fix:* clip negatives and divide by the sum. Softmax is the right
   projection for unconstrained logits — which is what CMA-ES keeps — and the
   wrong one for a vector that is already a distribution.

2. **Index-0 compounding in `_rebuild_particles`.** `init_arm` registers
   operators one at a time and rebuilds on every call, giving each new
   operator a flat 0.01. After twelve registrations the first operator held
   ~0.90 of the mass before a single execution ran; the swarm "converged" on
   whatever was registered first. Fixing (1) alone made this visible as
   convergence to `bit_flip` in 20/20 seeds regardless of reward.
   *Fix:* new entries get the uniform share, jittered.

3. **The swarm was frozen.** All five particles start at exactly `1/n` with
   zero velocity, so `pbest == gbest == pos` and both PSO force terms are
   identically zero. Measured after 6000 executions: all five particles
   bit-identical, every velocity component exactly 0.0. A swarm needs
   positional diversity to generate a gradient; that is what makes it a swarm.
   *Fix:* randomized initial positions, jittered around uniform.

4. **Self-reinforcing particle starvation.** `_update_fitness` zeroed the
   fitness of any particle whose window was empty; fitness-proportional
   selection then starved it permanently. Three of five particles pinned at
   exactly 0.0 for whole campaigns.
   *Fix:* keep the previous fitness on an empty window, and floor the
   selection weight relative to the best particle rather than at an absolute
   0.001 (which stops being a floor the moment any particle exceeds ~0.1).

5. **No per-operator signal existed at all.** Fitness was measured per
   *particle*, so PSO was searching a 12-dimensional simplex using differences
   that are sampling noise on ~40 draws. MOpt (Lyu et al., USENIX Sec '19)
   steers the swarm by each operator's *measured efficiency*; that vector was
   never computed.
   *Fix:* track per-operator window counters, normalize to the simplex, and
   add it as a third attractor (`c3`) alongside pbest and gbest.

Also fixed: pbest/gbest were recorded at the bottom of the update loop, pairing
the old fitness with the already-moved position, so each particle steered
toward a point it had never evaluated. And `global_best_fitness` was
monotonic, so the swarm's attractor was whatever ever scored highest — now
decayed per window (`gbest_decay`).

**Result:** identifies the correct best arm on **100/100 seeds** (was 0/100),
prefers the live arm over a decayed one on **39/40**. Tail share plateaus near
0.38 because the efficiency attractor is proportional to reward rate
(`p_best / Σp ≈ 0.31`) — MOpt is a soft-allocation scheduler like Replicator,
not an argmax bandit, and is asserted as such.

**Production-scale bug found on the way:** the exploration floor was a
hardcoded 0.01 per operator. At the 12 operators a unit test uses that is
harmless; at the 135 in the live registry, `135 × 0.01 = 1.35` exceeds the
whole simplex, so flooring-then-renormalizing returned *exactly uniform* and
silently disabled PSO in production. Now a fraction of uniform
(`min_prob_frac`), so total floored mass is constant in `n`. The same
absolute-vs-relative bug was present in `cmaes._softmax` and is fixed there
too.

---

## Defect 2 — GPUCBScheduler starved the best arm on 42% of seeds

In **every** failure the best arm had `count == min_samples == 3` and
`mean == 0.0`.

`select_op` scored an observed arm `mean + beta * max(stddev, 1e-6)`. An arm
whose observations are all zero has `mean == 0` *and* `stddev == 0`, so it
scored `2e-6` forever. **The empirical stddev measures the opposite of what a
confidence bound needs**: it reads a deterministically-zero arm as *certain*
rather than as *under-sampled*. With p=0.30 and `min_samples=3` the best arm
draws three zeros with probability 0.7³ = 0.34, the floor of the observed rate.

*Fix:* `mean + beta * sqrt(2 * log(t) / n)` — UCB1, where the width is
governed by how little the arm has been sampled. Unobserved arms score `inf`,
so every operator is tried before any is judged.

**Second finding:** `select_op` never called `_rbf` or `_kernel_row`. The RBF
kernel, `_features`, `_kernel_cache` and `kernel_matrix()` were all dead on
the selection path — the class was plain per-arm UCB with an RBF kernel bolted
to the side and described in the docstring. Now wired in: an arm below
`min_samples` borrows a kernel-weighted posterior mean from correlated
operators.

**A trap worth recording.** My first attempt also let neighbours' counts into
the *width* (`n_eff`). That took the failure rate to **100%**: a well-sampled
category suppresses the exploration bonus of an untried operator inside it,
starving it exactly as the stddev did. The kernel may smooth the mean; the
width must be the arm's own count.

`beta`'s default moved 2.0 → 1.0, because what it multiplies changed. 2.0 was
calibrated against an empirical stddev (magnitude ~0.4); against a count-based
width it is 2× over-exploration and costs about half the achievable tail share
(measured: 0.434 at beta=2.0, 0.828 at beta=1.0). 1.0 is UCB1 for rewards in
[0,1] — a derived value, not one tuned against this benchmark.

**Result:** **0/100 seeds** starve (was 42/100). Recovers from arm decay
(0.422 vs 0.303 on the dead arm) where it previously locked on permanently
(0.001 vs 0.998).

---

## Defect 3 — HierarchicalBanditScheduler: category starvation and no recency

Top-level Thompson sampling could drive the posterior of the category holding
the best operator to ~0.02 before the bottom level ever saw it, after which
the category was never selected again. ~1.5% of seeds. Separately, it could
not leave a decayed arm (0.829 on the dead arm).

Both are the same root cause: `Beta(a, b)` has variance ~`1/(a+b)`, so an
uncapped posterior becomes arbitrarily confident and its Thompson samples
collapse onto the mean. An early wrong verdict becomes permanent.

*Fix:* cap `alpha + beta` at `max_pseudocount` by rescaling both, which
preserves the posterior mean while stopping its variance from shrinking —
sliding-window Thompson sampling. This is what `arm_decay` was reaching for,
but at 0.999 per 100 pulls it removes ~6% of the mass over a 6000-pull
campaign, far too slow to matter.

**Result:** starvation 1.5% → **0.25%** (1/400 seeds). Decay recovery
**0.230 → 0.993**, with the dead arm down to 0.001. Hierarchical is now the
strongest scheduler in the non-stationary regime by a wide margin.

---

## Defect 4 — `--seed` did not determine Hierarchical's behaviour at all

Found because the fixed-seed test was still flaky *after* the fix above.

`OPERATOR_CATEGORIES` maps categories to **sets**. Both `hierarchical.py` and
`gp_ucb.py` iterated them directly, so the order of `random.betavariate` calls
depended on `PYTHONHASHSEED`. With the RNG seed pinned at 92, tail share on
the best arm ranged from **0.001 to 0.998** across hash seeds, with 4 of 26
producing total starvation.

A crash found under hierarchical scheduling could not be replayed from the
seed alone.

*Fix:* sorted iteration when building `_op_to_cat`, and sample categories from
the insertion-ordered `cat_ops` dict rather than the `avail_cats` set.
Verified stable across 16 hash seeds.

`test_reproducible_across_hash_seeds` runs the scheduler in a subprocess under
two different `PYTHONHASHSEED` values and compares pick counts — checked
behaviourally rather than by inspecting the code, since new set iteration can
appear anywhere on the selection path. Confirmed to fail when the bug is
reintroduced.

---

## Defect 5 — CMAESScheduler diverged (fixed) but remains unsuitable (not fixed)

Tail share 0.005, *worse* than uniform: σ grew 0.3 → ~72, the mean vector
reached ~1e14, and the softmax became a hard one-hot on whichever logit
happened to be largest.

**Seven numerical errors**, found by following the divergence down:

1. `hsigma` read `self._eval_count` *after* it was zeroed four lines earlier
   in the same function, so the Hansen damping heuristic evaluated the
   constant `(1−c_σ)²` every generation and never damped.
2. `_new_generation` sampled at scale `σ·√n`, so mutation vectors
   `y = (x−mean)/σ` had norm ~n where CSA assumes ~√n. `‖p_σ‖²/n` sat far
   above 1 on every generation and drove σ up exponentially.
3. No clamp on σ in either direction.
4. The rank-μ covariance term used raw displacements instead of `y = Δ/σ`,
   multiplying C by an extra σ² (0.09) per generation until it hit
   `cov_diag_min`. A collapsed C makes every candidate identical, which
   removes the variation being ranked and drags σ down — the same runaway in
   reverse. Same bug in the `pc` evolution path.
5. The mean step applied `lr_mean · σ · delta`, but `delta` already carries a
   factor of σ, so the step was `σ²·⟨y⟩_w` — a 3× overshoot at the clamped
   step size, walking the mean into the logit clip every generation.
6. `mu_eff` was computed as `mu²/mu`, i.e. just `mu`: it counted the elites
   instead of using the weights it was meant to summarize.
7. The rank-μ term measured spread from the **already-stepped** mean, folding
   the mean update into the covariance and inflating it (measured C diagonal:
   5.6e4).

**Correction to an earlier claim.** I previously reported `delta / sigma` in
the CSA path as a bug. It is not: `delta = σ·⟨y⟩_w`, so dividing by σ yields
exactly the `⟨y⟩_w` that CSA wants. That change was reverted. The real cause
of the blowup was (2).

Also fixed: each candidate received exactly **one** Bernoulli observation, so
the rank-μ update was sorting eight candidates by a single coin flip each —
and because a fresh population was drawn every `pop_size` selections while the
update ran every `generation_size` records, 24 of every 25 populations were
sampled and discarded unevaluated. Candidates now get
`generation_size // pop_size` observations each.

**Result:** no longer diverges — σ and the mean stay bounded, and it reaches
~0.95 tail share on good seeds given ~100k executions. But it stays **bimodal**
(0.004 on bad seeds) and needs orders of magnitude more executions than any
other scheduler here.

**Left as-is deliberately.** CMA-ES is a continuous black-box optimizer being
asked to optimize a 12-dimensional categorical distribution from Bernoulli
feedback: each generation spends `generation_size` executions to buy one very
noisy ranking of `pop_size` near-identical candidates. Constants exist that
would make this benchmark go green; none of them are derived from anything,
and tuning until the test passes is the failure mode this whole exercise
exists to avoid. The convergence *rate* over 40 seeds is pinned as an
assertion instead, with a documented band — an xfail would XPASS or fail
depending on which seed it drew.

---

## Reproducibility fixes

**`CMAESScheduler` was not seedable as constructed.** `services/fuzzer.py`
built it with no `rng`, so `__init__` fell back to `RandPool()`, which seeds
from OS entropy. `--seed` did not determine CMA-ES behaviour. Now threaded
through from `self._rand_pool`.

**Six schedulers use the module-level `random`** (`epsilon_greedy`, `exp3`,
`replicator`, `hierarchical`, `monte_carlo`, `mopt`). Reproducible via
`random.seed()`, but it couples every scheduler's stream to every other
consumer in the process, so parallel workers cannot have independent
scheduler streams. Not fixed here.

## Two pre-existing flakes, fixed in passing

Both surfaced during full-suite runs and were confirmed on a clean tree, so
neither was caused by this work. Both are the same bug: an unseeded RNG in a
test that asserts a statistical property.

- `test_bloom_exec_dedup.py::test_reset_on_full_wipes_at_capacity` — 66/20000
  (0.33%). Inserts 64 `os.urandom(8)` keys into a capacity-64 filter and
  asserts `n_added == 64`; a bloom false positive makes it 63.
- `test_mb_cbh_reanchor.py::test_reanchoring_does_not_lose_solvable_cases` —
  2/200 (1.0%). Seeds its 80 cases with `random.Random(3)` but drives
  `climb_hill` with an unseeded `RandPool()`, so the tolerance band absorbs
  case-selection noise but not mutation noise.

Neither could be reproduced from any seed — the exact failure mode the session
seed work exists to eliminate.

## Test design notes

- **Two-run seeding**: strict thresholds at fixed seed 92, weak invariant
  (beat uniform 3×) at a random per-session seed. A strict assertion driven by
  a random seed is a debugging session five years from now.
- **The seed is generated in `pytest_configure` and printed in
  `pytest_report_header`**, before collection — so it survives the segfault
  and hang this suite has produced before.
- **Failure rates are pinned as assertions with documented bands** rather than
  hidden behind xfail, wherever behaviour is bimodal. A fix turns the test red
  and the message says what to do about it.
- **`test_uniform_selection_does_not_pass`** drives a feedback-ignoring
  scheduler through the same harness and asserts it fails every threshold.
  Without it the thresholds could be satisfied by chance.
- **`test_mopt_swarm_is_not_degenerate`** asserts the three structural
  properties whose absence made PSO a no-op — identical positions, zero
  velocities, collapsed fitnesses. Each passed silently before.

Runtime: 2.2s default, 25s including `-m slow`. Full suite 4813 passed.

## What this was actually for

Two of ten schedulers did not work, a third failed 42% of the time, and a
fourth was not controlled by `--seed` at all. None of it was visible from 4,767
passing tests, because every one of them asked a structural question.

The finding that changed my model of the subsystem was not any individual bug.
It is that the stationary and non-stationary rankings are nearly inverted, and
the Elo arbitration layer is implicitly optimizing for the stationary one —
which is the regime a campaign leaves within its first few minutes.
