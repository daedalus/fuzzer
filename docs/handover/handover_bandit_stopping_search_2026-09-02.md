# Handover — Index Policies, Optimal Stopping and Search Allocation

**Date:** 2026-09-02
**Base:** `dc30854` (`docs(handover): plan for classical job-scheduling algorithms`)
**Status: PLAN ONLY. NOTHING IMPLEMENTED.** Every measurement below was taken
on this tree; every proposal below is unwritten code.

Sources surveyed:

- Multi-armed bandit — the general `K`-arm formulation
- Gittins index — the exact optimal index for the discounted FABP
- Optimal stopping — secretary / house-selling / retirement-value formulations
- Search theory — Koopman's optimal search-effort allocation
- Stochastic scheduling — SEPT/WSEPT and the preemptive Gittins scheduling rule
- Explore-then-commit — the `m`-round uniform-exploration baseline

Companion document: `handover_job_scheduling_2026-09-02.md`. That one covers
the *deterministic* sequencing problems (`1|prec|f_max`, `P||C_max`). This one
covers the *stochastic* half. Where they touch, this document says so.

---

## 0. Rule 1 — where each piece goes

`AGENTS.md` Hard Rule 1 is explicit: *"new schedulers register in
`_OPERATOR_STRATEGY_NAMES` and follow the `select_op`/`record`/`bandit_stats`
interface."* Two of the six sources here produce objects that satisfy that
interface and two produce objects that do not. The split is not cosmetic.

| Source | Object it produces | Home |
|---|---|---|
| Gittins index | an arm index — a scalar per `(α, β)` | `core/gittins.py` (pure table) + consumers |
| Explore-then-commit | an arm-selection policy | `core/schedulers/etc.py`, Rule 1 shape |
| Optimal stopping | a retirement value — a scalar threshold | *same table as Gittins* (see §4) |
| Search theory | a continuous **allocation** over seeds | `core/schedules.py` — a power schedule |
| Stochastic scheduling | a sequencing rule over jobs | `core/job_scheduling.py` (companion doc) |
| Multi-armed bandit | — already implemented 14 ways | nothing new; see §7 |

The important placement decision is the fourth row. Koopman's rule does not
answer *"which seed next"*; it answers *"how many seconds does each seed get"*.
That is what `SeedScorer` produces and nothing else in the tree does. Putting it
behind `select_op` would require discretising a continuous allocation into a
sampling distribution and would throw away the only property that makes it worth
porting.

The third row is the one that saves work: the Gittins index is *defined* by an
optimal-stopping problem, so the same table answers both questions. One module,
two consumers.

---

## 1. Inventory — what this tree already has

Before proposing an index policy it is worth being precise that this repo is not
short of bandits.

- **14 operator schedulers** in `core/schedulers/` (5,273 lines), balloted in
  `services/operators.py::select_op:3373` and arbitrated by Elo:
  `replicator, bandit, mopt, cem, exp3, eps_greedy, hierarchical, gp_ucb,
  contextual, cmaes, ducb, swucb, cucb, invasion`.
- **13 seed-selection arms** balloted in `services/seed_picker.py::_pick_seed_elo:139`:
  `ga, qea, weighted, pareto, format, bayesian, markov, boltzmann, ecofuzz,
  aflgo, mcts, alphabeta, katz`.
- **11 power schedules** in `core/schedules.py::SeedScorer.SCHEDULES:68`:
  `base, fast, coe, rare, mopt, lin, quad, go, aflgo, entropic, katz`.
- **Beta-Bernoulli posteriors on both layers**: `MonteCarloScheduler.arm_alpha/
  arm_beta` (operators, geometric decay `0.999` every 100 records) and
  `core/seed_quality.py::BayesianSeedQuality` (seeds, `Beta(1,1)` prior,
  optional decay and hierarchical pooling).
- **An optimal-stopping module**: `core/secretary.py`, behind `--secretary`,
  with three instances — `_seed_secretary`, `_op_secretary`, `_corpus_secretary`.

So the honest question is not *"should this fuzzer have a bandit"*. It is
*"do these six sources contain anything the fourteen do not already do"*.
Sections 2–4 answer that with measurements rather than with taxonomy.

---

## 2. Finding 1 (measured) — `core/secretary.py` is not a stopping rule

This is the largest finding in the document and it comes first because it
changes what §4 should propose.

### The rank term cannot bind

`should_stop()` (`core/secretary.py:100`) has two clauses:

```python
threshold = max(1, int(n * self.exploration_frac))       # n = window length
if self._steps_since_improvement < threshold: return False, "exploration phase"
if rank <= threshold:                          return True,  "..."
```

`rank` is `_rank_of_best()`: `sum(decay**age for each record in the window)`
with `decay = 0.95`. Its supremum over *any* stream is the geometric sum
`1/(1-0.95) = 20.0`. `threshold` is `int(n/e)`, and `n` grows to
`window_size = 500`, giving `threshold = 183`.

Measured on this tree:

| stream | n | rank | threshold |
|---|---|---|---|
| strictly increasing — every observation a record, the best case that exists | 500 | **20.000** | 183 |

So for `n ≥ 58` the second clause is unconditionally true and the rule reduces to
its first clause alone:

> stop iff no **all-time** record in the last `floor(n/e)` observations.

The `rank` computation, the `decay` parameter, the `_record_count` bookkeeping
and the `_record_flags` deque are all decorative. `_record_count` and
`_best_idx` are written, persisted by `save()`, and never read by any decision.

### The surviving clause is a clock, not a productivity test

`_best_value` is the all-time maximum and is *never* recomputed when the window
slides, so the "sliding window for non-stationary quality adaptation" the module
docstring claims does not exist on the record path. In an i.i.d. stream the
expected number of records in `n` draws is `H_n ≈ ln n`, so record gaps grow
geometrically — `steps_since_improvement ≥ 183` arrives with near-certainty for
*any* stream, productive or not.

The observed stream makes this much worse. `services/fuzzer.py:4033` feeds:

```python
fuzz_count = max(meta["fuzz_count"], 1)
discovery_rate = len(new) / fuzz_count
self._seed_secretary[seed_key].observe(discovery_rate)
```

`fuzz_count` is the seed's **cumulative** count, so the series carries a `1/t`
envelope. A later discovery can only set a record by finding more than `t` times
as many edges as the first one did. Replaying that exact stream for a seed that
finds new coverage forever at a constant rate:

| P(new edge per exec) | first `should_stop() == True` | steps since "improvement" at exec 3000 |
|---|---|---|
| 0.50 | **exec 20** | 2999 |
| 0.20 | **exec 20** | 2999 |
| 0.05 | **exec 20** | 2990 |
| 0.01 | exec 21 | 2986 |

Exec 20 is `min_observations`. A seed producing new coverage on **half** of its
executions, forever, is declared stopped at the first instant the code is
allowed to say so, and never recovers.

### What `--secretary` actually does

`seed_picker.py:638` applies `w *= 0.01` to any stopped seed. Since every seed
stops at its ~20th observation regardless of yield, the multiplier is applied to
essentially the whole corpus. A uniform multiplier does not change a normalised
weight vector — so the live effect of `--secretary` is a hard **100× preference
for seeds with fewer than `min_observations` recorded executions**. It is a
recency bias pointed the opposite way from a stopping rule, and it is worth
naming as such rather than tuning.

### Finding 2 — `_op_secretary` has no consumer

`services/fuzzer.py:4180` calls `observe(a/(a+b))` for every operator used on
every iteration. The only read anywhere in the tree is
`services/stats.py:860`, which takes `len()` of the dict for a counter. There is
no `should_stop()` call on the operator instances. This is per-operator,
per-iteration work — deque append, record comparison, counter update — feeding
nothing.

`_corpus_secretary` *is* consumed (`corpus_manager.py:693` → `_defer_minimize()`)
and inherits the same defect: `_stats.discovery_rate()` is a globally decaying
quantity, so it stops permanently once past `min_observations`.

### Consequence for this plan

Do not port a *second* secretary variant (Bruss' odds algorithm, Gilbert–Mosteller
thresholds) alongside a broken first one. The fuzzer never faces the problem
`core/secretary.py` names — irrevocable, no-recall, one-shot selection from a
known-length sequence. It faces *"how long do I keep paying for this seed"*,
which is a retirement-value problem, which is the Gittins problem. §4 replaces
the module rather than joining it.

---

## 3. Finding 3 (measured) — where a Gittins index would and would not change a pick

The Gittins index theorem: for a family of alternative bandit processes with
geometric discounting, independent arms, and only the engaged arm changing state,
the optimal policy is to play the arm of largest index

```
ν(x) = sup_{τ>0}  E[ Σ_{t<τ} γ^t r(x_t) ]  /  E[ Σ_{t<τ} γ^t ]
```

— the best achievable discounted reward *rate* over stopping rules. Equivalently
`ν(x)` is the retirement charge that makes continuing and quitting equally good.

### The table is cheap and correct

Prototyped on this tree with calibration on a scalar-`λ` grid: for each fixed
retirement value `λ`, backward induction over the truncated `(α, β)` triangle,

```
W(a,b) = max( λ/(1-γ),  μ + γ[ μ·W(a+1,b) + (1-μ)·W(a,b+1) ] ),   μ = a/(a+b)
```

and `continue-optimal at (a,b) ⟺ ν(a,b) ≥ λ`, which one pass decides for every
state at once. Carrying `λ` as a vector axis runs all `K` dynamic programs in a
single sweep with only two levels of `W` live, so memory is `O(K·N)`.

Validation against the published Gittins–Jones values for the `Beta(1,1)` state:

| γ | computed `ν(1,1)` | published |
|---|---|---|
| 0.9 | **0.7029** | 0.7029 |
| 0.99 | **0.8694** | ≈0.8699 |

Cost, `K = 2048` grid points:

| N (max `α+β`) | build | stored (float32) |
|---|---|---|
| 80 | 64 ms | 25 KiB |
| 300 | ~1.0 s | 350 KiB |
| 400 | 2.1 s | 625 KiB |

Monotonicity in `α` and `β` and `ν ≥ μ` hold everywhere with `α+β ≤ N`; they
fail *only* on the truncation edge `α+β > N`, where the finite horizon depresses
the index. That is the usable-region caveat: outside the table, fall back to
`μ`, whose error at `α+β = 200, γ = 0.99` is ≈0.014.

### But does it pick differently?

This is the §1-identity discipline from `handover_boltzmann_ab_2026-08-30.md`:
establish before spending benchmark cells whether the two arms are the same
computation. Argmax agreement with the Gittins index (`γ = 0.99`), 4,000 draws
per cell:

| arms | regime | Thompson | greedy (posterior mean) | D-UCB (coef 2.0) |
|---|---|---|---|---|
| 8 | cold (n≈5) | 43.4% | 67.1% | 25.7% |
| 8 | hot (n≈600) | 86.7% | **99.7%** | 81.2% |
| 40 | cold | 21.4% | 59.5% | 0.3% |
| 40 | hot | 78.1% | **98.7%** | 72.0% |
| 155 | cold | 10.5% | 63.6% | 0.0% |
| 155 | warm (n≈60) | 51.6% | 82.6% | 22.3% |
| 155 | hot (n≈600) | 74.7% | **97.5%** | 64.6% |
| 155 | heavy-tailed mix | 48.2% | 75.5% | 0.0% |

Two conclusions, and the first is the one that scopes the whole port:

**(a) On the operator layer, Gittins is nearly a no-op in steady state.** At 155
arms well sampled, Gittins and the posterior mean agree 97.5% of the time — the
exploration bonus has decayed to nothing. And the operator arms *are* well
sampled in steady state: with `arm_decay = 0.999` applied every 100 records, an
arm pulled at frequency `f` settles at `α+β ≈ 100f/(1-0.999) = 10^5·f`, i.e.
≈645 for `f = 1/155`. The operator ballot lives in the "hot" row.

**(b) On the seed layer it is not.** Seeds arrive continuously, are pruned by
`_maybe_prune`, and turn over; most seeds hold few observations at any moment,
so the seed posteriors live in the cold and warm rows, where Gittins agrees with
Thompson only 10–52% of the time. `_pick_bayesian_seed`
(`seed_picker.py:483`) is exactly a Thompson draw over `BayesianSeedQuality`.

So the recommendation inverts the obvious one: **land Gittins on the seed
picker first, and treat the operator arm as optional.**

A third observation falls out for free: D-UCB at 155 arms in the cold regime
agrees with Gittins 0.0% of the time, because `sqrt(log t / n)` dominates
`μ` whenever `n` is small and `K` is large — it deterministically picks the
least-sampled arm. That is the textbook UCB cold-start pathology, present in a
shipped arm of this tree, and it is worth a look independent of anything here.

---

## 4. Landing zone A — `core/gittins.py`, and the retirement value that replaces the secretary

### The module

```
core/gittins.py
    build_table(n_max=300, gamma=0.99, grid=2048) -> np.ndarray   # (n_max, n_max)
    index(alpha, beta, *, gamma) -> float                          # table lookup + mu fallback
    retirement_value(alpha, beta, *, gamma) -> float               # == index(), named for the caller
```

Pure functions, no fuzzer imports, table memoised per `gamma` on first use and
optionally cached under the workdir. Non-integer `α, β` — `MonteCarloScheduler`
adds `weight` on success and `1` on failure, so the pair is not an integer
lattice point — are handled by bilinear interpolation, not rounding; rounding a
weighted pseudo-count silently discretises the cost signal.

### Consumer 1 — the seed picker

`BayesianSeedQuality` grows a sibling to `select_seed()`:

```python
def gittins_select(self, seed_ids, *, gamma):
    return max(seed_ids, key=lambda s: gittins.index(self._alpha[s], self._beta[s], gamma=gamma))
```

and `seed_picker.py` grows a `gittins` arm next to `bayesian` in the Elo ballot
plus `_SEED_STRATEGY_NAMES`. This is the methodologically cleanest A/B available
in this repo: **both arms read the identical posteriors**, so the two arms differ
in policy and in nothing else — no state divergence, unlike the Boltzmann A/B
where the two arms computed different quantities and could not be reproduced at
a fixed seed.

### Consumer 2 — the stopping rule, replacing `core/secretary.py`

The retirement value *is* the stopping rule. A seed should stop receiving energy
when its index falls below the index of the best alternative — which for a
Beta-Bernoulli arm is a threshold on `(α, β)` read from the same table. This
gives, for free, everything `should_stop()` was supposed to give and does not:

- it is monotone in evidence, so a productive seed cannot be stopped;
- it is comparative, so "stop" means "someone else is worth more", not "a clock
  ran out";
- it is reversible — an arm whose posterior improves comes back;
- it needs no `min_observations`, no `window_size`, no `decay`, no
  `exploration_frac`. Four tuning knobs retire with it.

Do **not** leave `--secretary` in place alongside. Keeping a flag whose live
behaviour is a 100× recency bias, next to a flag that does the thing its
docstring claims, is precisely the "a gate that looks authoritative and is not"
lesson already recorded in `docs/learnings/`.

### The theoretical caveat, stated plainly

The Gittins index theorem does not strictly apply here, and the port should say
so in the module docstring rather than in a commit message:

1. **Arms are not independent.** Reward is *new* edges. An edge found from seed A
   removes it from B's future reward. The bandit is restless; Gittins is optimal
   only for a frozen-arm FABP. The correct object for restless arms is the
   Whittle index, which requires indexability and is not free.
2. **Arms arrive, and arrivals are policy-dependent.** Whittle (1981) showed the
   index policy survives arm arrivals when arrivals are i.i.d. and independent of
   the policy. In a fuzzer the seeds you fuzz are what produce new seeds, so that
   condition fails.
3. **Rewards are not Bernoulli.** `len(new)` is a count, not a success bit. The
   Beta-Bernoulli posteriors already in the tree make the same approximation, so
   this is inherited, not introduced.

What survives all three is the narrow claim worth defending: the Gittins index is
a *better-calibrated exploration bonus than `sqrt(log t/n)` or a Thompson draw
for the discounted objective*, and it supplies a retirement value that the tree
currently fakes. That is the claim the A/B should test, and it should be written
in the handover for the next session so nobody upgrades it later into optimality.

---

## 5. Landing zone B — Koopman search allocation as a power schedule

### The mapping

Koopman's problem: a target sits in one of `n` boxes with prior `p_i`; effort
`t_i` in box `i` detects it with probability `1 - exp(-λ_i t_i)`; maximise total
detection probability subject to `Σt_i = T`. The Lagrangian solution is
water-filling,

```
t_i = max(0, (1/λ_i) · ln(p_i λ_i / θ)),      θ chosen so Σt_i = T
```

The mapping onto this tree is unusually direct, and every input already exists:

| Koopman | fuzzer | source in tree |
|---|---|---|
| box `i` | seed `i` | `f.corpus` |
| prior `p_i` | P(a new edge is reachable from seed `i`) | *does not exist per-seed* — see below |
| detection rate `λ_i` | new edges per **second** of target time | `meta["coverage_edges"] / meta["total_time"]` |
| effort `t_i` | seconds of target time | `core/cost_ledger.py` |
| budget `T` | one maintenance interval | `fuzzer.py` tick |

The `λ_i` denominator being *seconds* rather than *executions* is the whole
point, and it is the reason this belongs on the cost ledger rather than on
`fuzz_count`: a seed whose executions cost 10× more correctly receives 10× fewer
of them for the same allocation.

### Validated closed form

Bisection on `θ` reproduces both degenerate cases exactly: equal `p` and equal
`λ` gives a uniform split; equal `λ` alone gives an allocation ordered by
`log p`. On a 200-seed synthetic corpus (`p ~ Beta(0.6, 3)`, `λ` lognormal, which
matches the cost dispersion `tools/cost_dispersion.py` measured on grep — CV
0.922, p90/p10 7.18×), expected detections against the model's own objective:

| budget `T` (s) | Koopman | uniform | greedy | seeds funded |
|---|---|---|---|---|
| 0.1 | **0.250** | 0.020 | 0.132 | 3 |
| 1 | **1.534** | 0.197 | 0.147 | 10 |
| 10 | **6.076** | 1.802 | 0.147 | 29 |
| 100 | **15.998** | 10.939 | 0.147 | 89 |
| 1,000 | **29.912** | 26.425 | 0.147 | 167 |
| 10,000 | 32.522 | 32.350 | 0.147 | 200 |
| 100,000 | 32.522 | 32.522 | 0.147 | 200 |

### The falsification condition — read this before spending cells

The table above is the *falsifier*, not the sales pitch. Koopman's rule has two
degenerate limits and they bracket the useful region on both sides:

- **Large budget** (`λ_i T ≫ 1` for all `i`): everything saturates and the
  allocation converges to uniform. At `T = 10,000` the gain is 0.5%.
- **Small budget** (`λ_i t_i ≪ 1` for all `i`): `1 - exp(-λt) ≈ λt`, the
  objective becomes linear in `t`, and the optimum degenerates to putting the
  entire budget on `argmax(p_i λ_i)` — i.e. pure greedy.

So the schedule is only distinguishable from schedules already in
`SeedScorer.SCHEDULES` in the middle regime, where the funded support is a
strict subset of the corpus. **Measure `λ_i T` on a real campaign per target
before running an A/B.** This is the same identity check that disqualified the
`locked` target set in `handover_boltzmann_ab_2026-08-30.md` §1, applied to a
different quantity, and it is cheap: one campaign, read the ledger, histogram
`λ_i · T`.

### The missing input, and the loop it closes

`p_i` — the per-seed prior that new coverage is reachable — does not exist. What
*does* exist is its corpus-level analogue: `good_turing_estimate()`
(`edge_tracker.py:1997`) already returns `discovery_probability = 1 - Ĉ` from the
Chao2 incidence estimator, and that key is read in exactly three places
(`report.py:900`, `stats.py:210`, and nowhere else) — all display. The only
scheduling consumer of the whole estimator is `saturation`, used by the
`_saturation_gate` at `seed_picker.py:1046`.

The per-seed version is Chao2 restricted to one seed's own edge incidence, which
is computable from `seed_edges` and `_edge_owner_count` — both already
maintained, both already correct after the `_maybe_prune` owner-count fix. That
gives `discovery_probability` its first scheduling consumer and answers one of
the open questions carried in `docs/port-backlog.md`: Entropic and our Chao2 come
from the same STADS framework and have never been reasoned about together. A
Koopman schedule is the place where they meet — Entropic *is* an allocation rule
over rare-feature counts, and Koopman is the allocation rule that makes explicit
what objective is being maximised.

---

## 6. Landing zone C — stochastic scheduling, which mostly retires a question

The companion job-scheduling handover proposes deterministic sequencers over
`(p_j, d_j, prec)`. Stochastic scheduling asks what changes when `p_j` is a
random variable. For this tree the answer is: less than it looks, and that is a
useful result.

**`1||E[Σw_j C_j]` is solved by WSEPT** — Smith's ratio rule with `E[p_j]` in
place of `p_j` — and `1||E[ΣC_j]` by SEPT. Both are exactly optimal, and both
need only the **first moment**, which `core/cost_ledger.py` already stores
(`total_time / cost_samples`). So the maintenance-tick proposal in the companion
document does not need a distribution: the mean is sufficient, and the theorem
says so. Recording that retires a question rather than opening one.

Two things genuinely change:

1. **`P||E[C_max]`**: for exponential processing times on two machines, LEPT is
   optimal — *longest* expected first, the reverse of the deterministic LPT
   intuition. This matters to the Multifit item (companion doc §5) and is a
   reason to prefer LEPT-style balancing over FFD if worker partitioning ever
   lands. It is also the second independent argument against a cost-recomputed
   partition, on top of the non-monotonicity already recorded there.
2. **Preemptive `1|r_j, pmtn|E[ΣC_j]` with partially-observed jobs** is solved by
   a *scheduling* Gittins index — a different index from §3 but computed the same
   way, on the hazard rate of the remaining processing time given elapsed
   service. That one *does* need a second moment, and the ledger does not have
   one. Adding `total_time_sq` is one float per seed and yields Var and CV; it is
   cheap and it is a prerequisite for anything genuinely stochastic here. Note
   this is a different quantity from what `tools/cost_dispersion.py` measures:
   that tool reports dispersion of the per-seed *mean* across seeds; this needs
   dispersion *within* one seed.

---

## 7. Landing zone D — explore-then-commit as a control arm

ETC has no chance of beating fourteen tuned schedulers and should not be
proposed as if it might. Its value here is entirely methodological: it is the
one policy in the family with a closed-form regret bound and no tuning surface,
which makes it the **null arm the benchmark harness currently lacks**.

`tools/bench_replicated.py` compares two arms of interest. It has no floor. An
ETC arm with `m` uniform rounds per operator then permanent commit gives one, and
answers a question no current cell answers: *how much of the measured coverage is
attributable to scheduling at all, versus to the mutation operators being good?*
The Boltzmann result — a bounded null of ≈5 edges on png/jpeg — is much easier to
interpret against a known-bad floor than against nothing.

Roughly 60 lines in `core/schedulers/etc.py` with `supports_priors = False`
(Rule 40) and a one-line reason: ETC's exploration phase is uniform by
construction and cannot consume an informative prior.

Regarding the multi-armed-bandit source generally: after the inventory in §1 the
honest finding is that nothing in the general formulation is missing. The gap is
not a missing algorithm, it is that the fourteen have never been compared against
a floor.

---

## 8. Commit sequence

Each commit is independently revertible and each states its own falsifier.

1. **`core/gittins.py` + tests.** Pure table, no wiring. Tests assert
   `ν(1,1) = 0.7029` at `γ = 0.9` against the published value, monotonicity in
   both parameters over `α+β ≤ N`, `ν ≥ μ`, and `ν → μ` as `γ → 0`. Falsifier:
   perturb the backward-induction recursion and the published-value test fails.
2. **`--secretary` removal.** Delete `core/secretary.py`, the three instances,
   the `w *= 0.01` at `seed_picker.py:638`, the `_defer_minimize` trigger, the
   CLI flags and the report/stats lines. Carry the measurements of §2 into
   `docs/learnings/` — the module's value is entirely the falsified hypothesis,
   and "absent from the tree" and "considered and rejected" are different states.
   Bug-with-evidence; no A/B.
3. **`gittins` seed arm.** `BayesianSeedQuality.gittins_select()`, the ballot
   entry, `_SEED_STRATEGY_NAMES`, persistence. A/B against `bayesian` on the same
   posteriors. This is where the retirement value lands as a real stopping rule.
4. **Cost-ledger second moment.** `total_time_sq` written in lockstep with
   `total_time` and `cost_samples`, persisted, with the same
   no-samples-is-not-free discipline the module docstring already sets out.
   Instrumentation only.
5. **`core/schedulers/etc.py`** plus a `--baseline-etc` arm in the bench harness.
   No campaign default changes.
6. **Per-seed Chao2 `p_i`**, then the `koopman` power schedule behind
   `--schedule koopman`. Gated on the `λ_i·T` measurement of §5 coming back in
   the middle regime; if it comes back saturated, stop at commit 6a and record
   why.

Commits 1–2 and 4–5 are unconditional. Commits 3 and 6 are the ones that need
benchmark time.

---

## 9. Falsifiers

- **§2 (secretary).** Reproduce with `tests/support/scripted_rng.py`-style
  determinism: feed `SecretaryStopping` a strictly increasing 500-point stream
  and assert `rank == 20.0` while `threshold == 183`; feed the real
  `len(new)/fuzz_count` stream at `p = 0.5` and assert `should_stop()` is true at
  observation 20. Both are exact, neither uses a retry-until-hit loop (Rule 39).
- **§3 (Gittins ≈ greedy when hot).** If the seed posteriors turn out to be *hot*
  rather than cold in a real campaign, commit 3 is a no-op and should not be
  benchmarked. Measure first: dump the `(α, β)` distribution of
  `BayesianSeedQuality` after a 100k-exec campaign per target. If the median
  `α+β` exceeds ~200, stop.
- **§5 (Koopman).** If `λ_i·T` is above ~5 for the bulk of the corpus at the
  live maintenance interval, the allocation is uniform and the schedule is a
  no-op. If it is below ~0.01 for the bulk, it is greedy and equally uninformative.
- **A/B design.** Reuse `tools/bench_replicated.py` with `--lock-single-thread`
  and time-paired arms. The measured noise floor stands: sd ≈4.6 edges on png,
  4.7 on jpeg, 12.3 on grep. The design resolves ~5 edges with 3 replicates on
  png/jpeg and needs ~2× the seeds on grep for the same absolute resolution.
  What buys power here is **replicates, not seeds** — the dispersion is
  intra-cell.

---

## 10. Considered and rejected

Recorded as its own section on purpose: "absent from the document" and
"considered and rejected" are different states, and only one of them should be
re-proposable.

- **A second secretary variant** (Bruss' odds algorithm, Gilbert–Mosteller
  house-selling thresholds). Correct solutions to a problem the fuzzer does not
  have: there is no irrevocable, no-recall, one-shot choice anywhere in the loop.
  Every "stop" decision here is reversible and comparative, which is the
  retirement-value formulation, which §4 already covers with the same table.
- **Whittle index for the restless formulation.** Theoretically the right object
  given §4's caveat 1, but it requires establishing indexability for this
  particular arm process, which is a research result and not a port. Revisit only
  if commit 3 shows a real effect and the independence caveat is the obvious
  explanation for its size.
- **Gittins on the operator ballot as a default.** Measured at 97.5% argmax
  agreement with the posterior mean in the regime the operator arms actually
  occupy (§3). Land it as a selectable arm if commit 1 is already there, but do
  not spend benchmark cells on it and do not make it a default.
- **Replacing D-UCB with Gittins.** Tempting given the 0.0% cold-regime
  agreement, but that is an argument for *investigating the D-UCB coefficient*,
  not for deleting a shipped arm. Different commit, different evidence.
- **Koopman as a seed *selector*.** It produces an allocation over all seeds, not
  a choice among them. Sampling from a normalised allocation to make it a
  selector discards the property that makes it worth porting and reinvents
  `weighted_pick_seed`.
- **Optimal stopping for the campaign as a whole** (`--stop-when-saturated`).
  The retirement value is defined relative to the best alternative arm; for the
  campaign there is no alternative arm, so the formulation has no content. The
  Chao2 saturation gate already covers this and covers it correctly.

---

## 11. Open questions

1. Does `BayesianSeedQuality` run cold or hot in a real campaign? Everything in
   §4 hangs on this and it is one measurement (§9), not an argument.
2. `MonteCarloScheduler.record()` adds `weight` on success but `1` on failure.
   That makes `(α, β)` a weighted pseudo-count, not a Beta-Bernoulli posterior.
   Thompson sampling tolerates it; a table lookup indexed by `(α, β)` inherits
   whatever distortion it introduces. Worth quantifying before commit 3 extends
   the same treatment to seeds.
3. What discount `γ` corresponds to this fuzzer's horizon? `arm_decay = 0.999`
   per 100 records implies an effective horizon; `BayesianSeedQuality` defaults
   to `decay = 1.0`, i.e. no forgetting at all, which is *undiscounted* and for
   which the Gittins index is not defined. Commit 3 must either set a decay or
   pick `γ` from the campaign length, and should say which and why.
4. Does the `_maybe_prune` eviction interact with the retirement value? A seed
   evicted for memory is not the same event as a seed retired for low index, and
   conflating them would make the stopping rule look effective for the wrong
   reason.

---

## 12. Reproducing the measurements

All measurements in §2, §3 and §5 were taken on `dc30854` with no patches
applied. The secretary replays import `fuzzer_tool.core.secretary` directly with
`src` on `sys.path` and need no build. The Gittins table and the agreement matrix
need only numpy. The Koopman sweep is self-contained. None of them run a target,
so none of them are subject to the machine-load contamination described in
`handover_boltzmann_ab_2026-08-30.md` — but anything in §9 that runs a campaign
is, and must go through `--lock-single-thread`.
