# Handover — non-UCB scheduler families: what is missing, and what is already answered

**Date:** 2026-09-13
**Base:** `77d3ab8` (`docs(handover): analysis of thermal/stochastic-process concepts vs current tree`)
**Status: ANALYSIS ONLY — nothing implemented.**

Question asked: with ten UCB variants in `core/schedulers/`, which *non-UCB*
scheduler families are still missing and worth implementing?

The useful answer turned out not to be a list of algorithms. It is one
structural finding (§2) that changes which algorithms are even candidates,
plus two corrections to the first pass of this analysis (§0) — one of which
retracts what looked like the strongest proposal.

---

## 0. Corrections to the first pass, recorded rather than edited away

Both mistakes are the kind that is easy to repeat, so the reasoning stays
visible.

**Correction 1 — the wrong Elo class.** The first pass asserted that the
meta-arbiter is `softmax(rating / 400)` with `k_factor=16`, and concluded from
the 400-point scale that it is near-uniform by construction. That formula is
`EloTracker.select_strategy` (`core/elo.py:494-525`), which is **not what
runs**. `_activate_elo` (`core/analyzer_registry.py:483-496`) builds a
`BayesianEloTracker(initial_mu=1500, initial_sigma=350, beta=200, tau=5.0,
min_matches=10)`, whose `select_strategy` (`core/elo.py:896-919`) is Thompson
sampling from a Gaussian posterior per strategy. There is no temperature. The
`temperature` parameter is accepted and ignored.

**Correction 2 — the conclusion drawn from it is already in the tree, and
already measured.** The first pass proposed a principled model-selection
master (Corral / EXP3-over-schedulers) as a *new* opportunity, on the argument
that the arbiter cannot concentrate. That argument is right, is not new, and is
recorded verbatim in `core/schedulers/consolidated.py:1-19`, with numbers:

- the selected strategy loses ~97% of its games whatever its quality, because
  the score it is credited with is its own round outcome and fuzzing success
  rates are low;
- a strategy three times as productive as its rival (3% vs 1%) gets **51%** of
  the picks; ten times (10% vs 1%) gets **54%**;
- seventeen schedulers under `--elo` each got **5.6–6.4%** of the picks;
- the portfolio found **13% fewer** discoveries than its best member alone.

So the project has already answered this by removing the arbitration rather
than improving it. What remains open is narrower and is an A/B question, not a
discovery — see T2-2.

The mechanism is worth stating precisely because it constrains any future
master. `Fuzzer._record_operator_strategy_matches`
(`services/fuzzer.py:6102-6118`) plays the selected strategy against every
other member of `operator_strategy_pool` with `score = surprisal_weight if
success else 0.0`. The opponents were never run. Their posteriors move by
`var_b * k * ((1 - score) - eb)` on an outcome they did not produce. There is
no counterfactual and no importance weighting: it is one arm's reward fanned
out as K−1 pseudo-matches. Any master proposed later has to supply the thing
this lacks (an unbiased estimate for the arms that were not played), not merely
a better ranking rule.

---

## 1. Inventory — what families are already here

Verified by reading `core/schedulers/` at `77d3ab8` (27 modules, 20 on the
operator ballot at `services/operators.py:467-539`).

| Family | Present as | Non-UCB? |
|---|---|---|
| UCB confidence-width | `ucb_common`, `ducb`, `swucb`, `kl_ducb`, `kl_swucb`, `cucb`, `c2ucb`, `moss`, `gp_ucb`, `cusum_ucb`, `contextual` (LinUCB), `bayes_ucb` (WIP, uncabled) | no |
| Posterior sampling | `monte_carlo` (Beta-Bernoulli Thompson), `consolidated` (Thompson + category prior + capped evidence), `seed_picker._pick_bayesian_seed` | yes |
| Exponential weights / adversarial | `exp3`, `fpl` | yes |
| Evolutionary / population | `cmaes`, `mopt` (PSO), `monte_carlo` CEM, `replicator`, plus `ga.py`, `qea.py` | yes |
| Tree search | `mcts` (UCT), `AlphaBetaMCTSSeedScheduler` | UCT is UCB-shaped |
| Naive baselines | `epsilon_greedy` (decayed), `round_robin` | yes |
| Domain seed pickers | `boltzmann`, `ecofuzz`, `aflgo`, `katz`, `tang`, `markov`, `pareto`, `format` | yes |
| Meta-arbitration | `BayesianEloTracker` (see §0) | yes |

**Absent entirely**, confirmed by grep over `src/`: Gittins index (already
open as **P3-2** in `handover_FINDINGS.md` — not new), Whittle index, FTRL /
Tsallis-INF, Corral or any model-selection master with a regret guarantee,
gradient bandit / softmax policy, successive elimination or sequential
halving, and rotting bandits.

---

## 2. The structural finding: every forgetting mechanism runs on the global clock

Ten of the schedulers exist to handle non-stationarity. Each of them indexes
"forget the past" by **rounds elapsed**, never by an arm's own pull count:

| Mechanism | Where | Clock |
|---|---|---|
| EXP3 window decay | `exp3.py:104-105` — one `_decay_factor` multiply per `record()`, applied to all arms through the shared factor | global |
| Thompson arm decay | `monte_carlo.py:349-358` — every 100 `record()` calls, `arm_alpha[k] *= 0.999` for **every** `k` | global |
| D-UCB / KL-D-UCB | `ducb.py:7-8` — `gamma^(t-s)` over all rounds `s <= t` | global |
| SW-UCB / KL-SW-UCB | `swucb.py:5, 81` — hard window of the last 4000 pulls, any arm | global |
| CUSUM-UCB | `cusum_ucb.py:11-18` — the CUSUM statistic is per arm, but *"a detection in any arm resets every arm's"* state | per-arm detect, global reset |
| Consolidated cap | `consolidated.py:105-109` — ceiling on an arm's own `alpha + beta` | **per-arm, but bounds confidence only** |

The last row is the near miss and the reason this finding needs care. The
capped pseudocount *is* indexed on the arm's own evidence, and the docstring is
explicit that rescaling "keeps the mean and stops the variance shrinking". That
is per-arm forgetting of *strength*. It is not what a saturating operator
needs, which is for the estimated **mean itself** to fall as the arm is pulled,
because the yield really did drop and the arm really is worse now.

**Why the distinction is not academic.** The decay mechanism measured in the
NN-over-metrics work is fatigue: an operator stops finding edges because it
already found what it reaches. That process is indexed by *that operator's own
pull count*. Under a global clock two wrong things follow:

1. An arm not pulled for 10k execs is forgotten although nothing about it
   changed. D-UCB at `gamma=0.9999` discounts it to 37% of its weight over
   10k rounds regardless of whether it was sampled once or never.
2. An arm pulled hard keeps its mean until the *global* discount catches up,
   which it does at the same rate for every arm — so the scheduler cannot tell
   exhaustion from bad luck.

**And the harness shares the error.** `tests/support/bandit_env.py`'s
`DecayingBest` (`:135-175`) is the environment that validates every recency
mechanism above, and its docstring names the right process — *"an operator that
was finding new edges runs out of new edges to find"* — while `p(arm, t)`
switches on `t < self.switch_at`, the **global** round counter. A pull-indexed
forgetter and a round-indexed forgetter are indistinguishable on this
environment. Every recency scheduler in the tree was selected under a
simulation that agrees with its own clock.

This is the same shape as the lesson already recorded about the forkserver
gate: a suite that cannot detect the defect it was written for. Here the
environment cannot express the hypothesis the schedulers are supposed to be
answering.

**Related gap, recorded so it is not re-derived.** `consolidated.py:24-26`
credits its design partly to "a 150-arm environment with rare heavy-tailed
yields, **fatigue on success** and periodic unlocks". `grep -rn fatigue` over
the whole tree returns only `core/allan_variance.py:134-136` (an unrelated
slope threshold) and that docstring line. The environment that motivated the
winning scheduler is **not in the repository** — same pattern as the PNG/JPEG
sweep TSVs that were cited but never committed. Reconstructing it is a
prerequisite for T1 and cheap, and it is the only place in the tree where
pull-indexed fatigue was ever modelled at all.

---

## 3. Proposals, ranked by what they answer

### T1. Rotting bandits — pull-indexed decay

The missing *model*, not just a missing algorithm. Rested rotting bandits
(Levine–Crammer–Mannor 2017; Seznec et al. FEWA/RAW 2019–2020) assume each
arm's mean is a non-increasing function of **its own** pull count, which is
exactly the fatigue process §2 describes and no scheduler here represents.

Honest caveat on the "non-UCB" framing: `RAW-UCB` is UCB-shaped, so the
family does not escape UCB by construction. The genuinely different member is
the **adaptive-window / filtering estimator** (FEWA): for each arm, use the
longest recent window of *that arm's own* pulls whose mean is statistically
consistent with the shortest one, and act on that. The novelty is the
per-arm window, not the index formula.

**Before any code, the gating measurement.** Per-operator reward as a function
of that operator's own cumulative pull count, on a real target. If those curves
are flat in own-pulls and only move on wall-clock, the whole family is rejected
and D-UCB keeps the job. The measurement is currently **impossible from the
logs**: the only per-execution record is the `--schedule-ablation` CSV
(`services/fuzzer.py:1659-1663`), whose columns are

```
iter,seed_idx,seed_hash,fuzz_count,coverage_edges,age_s,temperature,
base_w,burst,penalty,subsumption,diversity,spatial,mdl,final_w,
new_coverage,new_crash
```

— the seed's row, with **no operator column**. This is the same plumbing gap
filed as commit 2 of the `handover_nn_over_metrics` sequence, and it blocks
both items. That commit is the prerequisite and is worth landing on its own
account.

Note on that citation: `handover_nn_over_metrics` is **not in the tree** —
`ls docs/handover/` does not list it. `handover_done_2026-09-06.md:1295` and
`:1367` both cite it by bare name, and `:1367` is the only in-tree place its
sequence is described (append-mode `dump_stats`, chosen operator in the
ablation log, raw inputs rather than derived multipliers, an IPS estimator with
a falsification test that `final_w` *is* the propensity). Cite that line, not
a filename, until the document is committed.

Second prerequisite: a `RottingArms` environment in `tests/support/bandit_env.py`
whose `p(arm, t)` depends on the arm's pull count rather than `t`, so the
convergence harness can tell the two clocks apart. Cheap, and it retroactively
strengthens or falsifies the existing recency results.

**Kill criterion:** if `bench_paired` on the new environment shows the
per-arm-window estimator inside noise of D-UCB *and* the real-target
reward-vs-own-pulls curves are flat, record it in the Rejected section of
`docs/port-backlog.md` and stop.

### T2-1. Gradient bandit / softmax policy with a baseline

Sutton–Barto §2.8. One preference parameter per arm, constant step size,
reward-baseline subtraction, no confidence width, O(K) per update. Absent from
the tree; the closest thing is the softmax *inside* `cmaes.py:46`, which is
over CMA-ES logits and not a policy learned from reward.

Two reasons it is the cheapest real candidate. First, a constant step size
tracks drift natively without any window or discount, so it sidesteps §2's
clock question instead of answering it — a useful control arm against T1.
Second, the NN-over-metrics result was that *low*-capacity trackers survive
drift while a ~4000-parameter MLP falls to uniform; a gradient bandit is the
lowest-capacity tracker that exists, and it is the exact shape the frozen
offline distillation of `_compute_weights` (that handover's Plan A) would
deploy.

~50 lines, `supports_priors = False`, Hard Rule 16 for all randomness, testable
immediately on the existing three environments.

**Gating question:** does it beat `consolidated` on `bandit_env.py`? If it
loses on all three environments it does not get wired, since the portfolio
result in §0 says adding a weak arm costs rather than hedges.

### T2-2. A master with an unbiased estimate for unplayed arms (Corral)

Demoted from "strongest proposal" to an A/B question by §0. The critique of the
Elo arbiter is already measured and already acted on. The remaining question is
specific: **does log-barrier OMD with importance weighting over the base
schedulers beat the single consolidated learner?** Corral's guarantee is regret
relative to the best base algorithm, which is precisely the property the Elo
fan-out lacks — but the consolidated scheduler does not merely arbitrate
better, it removes the arbitration, and it already won by 13% against the
portfolio.

Do not start this before T1 or T2-1. It is more code than either, its payoff is
bounded above by "recover what the portfolio lost", and it needs the
propensity plumbing that T1 also needs.

### T3-1. Whittle index — the honest completion of P3-2

`P3-2` already scopes the Gittins index and already records the caveat that
Gittins is exactly optimal for the **discounted rested** FABP, which is not our
problem. The reason it is not our problem has a name: these arms are
**restless** (a seed's remaining novelty shrinks when *other* seeds find
edges) and **rotting** (§2). Whittle is the index for the restless case.

Cost is real: a per-arm MDP model, plus an indexability proof or an empirical
check, before the index means anything. Ranked last on purpose. Recorded here
so it is treated as the known extension of an open item rather than
re-proposed as a fresh idea.

### T3-2. Two near-free baselines the tournament never had

- **Annealed softmax over operator means.** Reuses the existing
  `self._temperature` (`DEEP_DIVE.md`: linear 1.0 → 0.1 over
  `--anneal-budget`). The tournament's only baseline today is uniform, and a
  softmax baseline is the standard thing a Thompson result should be compared
  against. Twenty lines, mostly to say whether any of this beats the simplest
  annealed greedy.
- **Successive elimination / sequential halving.** Attractive because fuzzing's
  objective is closer to pure exploration than to cumulative regret. But the
  catalogue survey (`handover_algorithm_catalogue_survey_2026-09-06.md:297`)
  already records the failure mode of halving schemes — an evicted item
  restarts at zero — and under rotting an arm's rank is *supposed* to change
  after elimination, which is the worst possible pairing. Proposed only as a
  bounded warm-up phase (choose this target's top-k once, then hand off), never
  as the campaign-long policy.

---

## 4. Considered and rejected

Recorded because "absent" and "rejected" are different states.

**Q-learning / SARSA over (coverage regime, operator).** The NN-over-metrics
measurements already settle it: the shared context vector has **3.27 effective
dimensions of 13** (participation ratio, measured with `_build_shared_context`
over 800 real seeds — 6 dims constant by construction, `fmt:png`/`fmt:other`
correlated −1.000), rewards are sparse, and the surface drifts *because* you
exploited it. In a matched 120k-step simulation with fatigue on, LinUCB held
1.07–1.14× uniform while a tuned MLP fell to 0.98–1.03× — at parity with
uniform — and with fatigue off every learner reached ~4×. A value function over
a state space is strictly more capacity than that MLP. No new measurement
needed.

Those three numbers are quoted at length on purpose: they live only in the
uncommitted `handover_nn_over_metrics`, so until that document lands this file
is their only in-tree record. Same reason the K-Scheduler calibration notes
were moved into `DEEP_DIVE.md` before its plan doc was deleted.

**Pareto / multi-objective bandits (Drugan–Nowé).** The operator reward is
already a deliberate scalarisation: bounded to `[0, 1]` with cost folded in
including `t_exec` (p50), per the scheduler-fixes work. Re-splitting it into a
vector to compute a Pareto front would undo a decision that was made on
purpose, and the seed side already has a real `pareto` arm.

**EXP3.S / EXP3-IX as separate modules.** Variants of a scheduler already
present, and both discount on the global clock — they inherit §2's error rather
than addressing it. If the non-stationary exponential-weights slot is worth
filling, fill it once with Tsallis-INF (below), not twice with EXP3 variants.

**Tsallis-INF, this round.** Genuinely absent, and the pitch is real: optimal
in both the stochastic and adversarial regimes with one algorithm and no regime
detection, which is the question `--elo` was built to dodge. It is rejected for
*now* on a specific point rather than on cost: our regime is neither of the two
it covers. Fatigue is a stochastic process with drift that the fuzzer itself
causes, which is not adversarial (no-one is choosing losses against us) and not
stationary-stochastic. A best-of-both-worlds guarantee over two regimes we are
not in buys a bound, not a behaviour. Revisit **after** T1 settles what the
reward process actually is — the rejection is contingent on that, not
permanent.

---

## 5. Proposed sequence

1. **Operator column in the ablation CSV** — already filed, see
   `handover_done_2026-09-06.md:1367`; blocks everything here. Log the selected
   operator and its reward per execution.
2. **`RottingArms` environment** in `tests/support/bandit_env.py`, pull-indexed,
   plus reconstruction of the missing 150-arm fatigue environment (§2).
3. **Measure** reward vs own-pull-count per operator on png and zlib. This is
   the gate: a flat result kills T1 and is worth knowing either way.
4. **Gradient bandit** (T2-1) — independent of 1–3, lands or dies on the
   existing harness.
5. **FEWA-style per-arm adaptive window** (T1), only if 3 says yes.
6. **`bench_paired`** with a pre-registered threshold before anything becomes
   a default, per the Boltzmann A/B discipline: bounded null, per-target noise
   floor from `tools/cost_dispersion.py`, replicates not seeds.

Steps 1–3 are worth doing even if every algorithm here is eventually rejected:
they close a logging gap, fix an environment that cannot express its own
docstring, and answer an empirical question about the reward process that ten
shipped schedulers currently assume without evidence.

---

## 6. Open questions

- Is fatigue *rested* (a function of the arm's own pulls) or *restless* (also
  moved by other arms' discoveries)? §2 assumes rested for tractability. The
  edge-ownership data could answer it: if an operator's yield falls in step with
  total corpus coverage rather than its own use, it is restless and T1's model
  is the wrong one too.
- Does the seed side have the same clock error? `_compute_weights` uses `age =
  now - added_at` (wall clock) and a staleness term of
  `fuzz_count/(coverage+1)`, which is productivity. Neither is "pulls of this
  seed since its last discovery". The `last_picked` field that P3-3 needs is
  the same missing field.
- `bayes_ucb.py` shipped uncabled with three of its own tests failing upstream.
  Is it meant to reach the ballot, or be retired? It is in neither
  `core/schedulers/__init__.py`'s `__all__` nor `operator_strategy_pool`; the
  same is true of `tang` and `katz`, which are *seed* strategies and correctly
  absent — `bayes_ucb` is the only unexplained one.

---

## 7. One line

The interesting gap is not a missing algorithm but a missing *clock*: ten
schedulers forget on the global round counter while the process they model —
an operator exhausting what it can reach — is indexed by that operator's own
pulls, and the synthetic environment that validates them shares the same error.
