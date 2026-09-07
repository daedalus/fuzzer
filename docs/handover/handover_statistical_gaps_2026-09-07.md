# Handover: statistical analysis gaps in bench_paired, coverage_regime, edge_tracker, and scheduler bandits

**Date:** 2026-09-07
**Base:** `b80c6d7` (perf(cmplog): Aho-Corasick multi-pattern operand scan)
**Status:** ANALYSIS ONLY — no code changes

---

## 0. One-line summary

Four separate gaps where the code already has the raw material for a strictly better
analysis but throws it away: paired deltas in `bench_paired.compare()` are binarised
for McNemar and their magnitude is only reported as a median; coverage-drop cells are
pooled into one count with no per-arm diagnostic; `CoverageRegimeDetector.continuum_correlation()`
computes bucket means instead of a Spearman rank correlation; and `EdgeTracker._cdf_walk()`
computes KS for free before `_wasserstein_vs_aggregate()` discards it. A fifth section
maps where FPL and KL-UCB would land if those analyses ever move to implementation.

---

## 1. Wilcoxon signed-rank on deltas — `tools/bench_paired.py`, `compare()`

`compare()` already accumulates every cell's signed delta (`deltas.append(d)` at
line 346), then throws away the magnitudes when it builds the McNemar table
(wins/losses/ties from the sign only). McNemar is the right test for a *binary*
outcome, but edge-count deltas are a *paired continuous measurement* per
(target, seed) cell — the same pairing the harness was built to preserve.

A Wilcoxon signed-rank test on the raw deltas would use the rank of each delta's
magnitude, giving more power at the same cell count. The analysis output itself
warns:

> "~10-point difference in per-cell win rate needs roughly 100 paired cells to resolve"

Wilcoxon is the natural complement, not a replacement: report both, the way Fisher
is currently reported as a contrast to McNemar.

### Where to land it

`compare()` already returns `median_delta` and `iqr`. A `_wilcoxon_signed_rank(deltas)`
helper (exact for small n, normal approximation for n > ~20) fits alongside
`_mcnemar_exact` and `_fisher_exact`. The per-target path at line 414 already splits
the pairing — Wilcoxon needs the same treatment, not a pooled run.

### Falsification

The existing `tests/test_bench_paired_stats.py` exercises `_mcnemar_exact` and
`_fisher_exact` against hand-computed values. A Wilcoxon test should be added there
with the same shape: known deltas, expected statistic and p-value.

### Caveat

Edge deltas across heterogeneous targets are not on a common scale (the code says
this explicitly, hence median/IQR instead of mean). Wilcoxon assumes a symmetric
difference distribution — needs a per-target run, same as `--by-target` already does
for McNemar, not a pooled one.

---

## 2. MCAR / MNAR diagnostic for `dropped_no_coverage`

`compare()` silently drops any cell where `coverage_attached` is False on either
arm (line 343–345: `dropped += 1; continue`). The comment at line 313–316 already
reasons carefully about not letting a build failure "masquerade as an arm difference" —
i.e. the code has MCAR/MNAR instincts without the vocabulary.

But there is no test of the assumption for the coverage-drop case:

- **MCAR** (missing completely at random): coverage failure is independent of which
  arm is being tested. Dropping is safe.
- **MNAR** (missing not at random): coverage failure correlates with the arm itself —
  e.g. a mutation-heavy arm is more likely to produce a malformed input that crashes
  the harness before coverage attaches. Then dropping cells biases `median_delta`
  toward whichever arm survives to completion more often.

### Where to land it

Log `dropped_no_coverage` **per arm**, not just pooled. In `compare()`, replace the
single `dropped` counter with `dropped_base` / `dropped_test` and add them to the
return dict. `cmd_analyse` already prints `dropped_no_coverage` per arm when non-zero
(line 411–412) — that path needs to distinguish which arm dropped.

### Falsification

A test that constructs base and test rows with asymmetric `coverage_attached=False`
distributions and asserts the per-arm counts land on the correct side.

### What the diagnostic tells you

If drop rate differs between base and test arms, that is evidence against MCAR and
the reported win rate is optimistic for whichever arm drops less. The McNemar/Wilcoxon
p-values are computed only on the surviving cells, so a skewed drop rate changes the
population being tested.

---

## 3. Spearman on (regime, Re) — `src/fuzzer_tool/core/coverage_regime.py`, `continuum_correlation()`

Despite the name, this method isn't computing a correlation at all. At line 232–242:

```python
def continuum_correlation(self) -> dict:
    buckets: dict[CoverageRegime, list[float]] = {}
    for regime, re_value in self._continuum_history:
        buckets.setdefault(regime, []).append(re_value)
    return {k: sum(v) / len(v) for k, v in buckets.items()}
```

It buckets (regime, Re) history and reports `sum(v)/len(v)` per regime — i.e. mean
Reynolds number per discrete bucket. The docstring's actual question — "does Re
separate the discrete regime labels the detector already produces" — is exactly a
Spearman rank-correlation question: are `CoverageRegime` (ordinal — presumably ordered
like SUBCRITICAL < CRITICAL < SUPERCRITICAL) and Re monotonically associated across
the recorded window?

Spearman on (regime_rank, Re) pairs gives a single ρ and a p-value instead of
eyeballing whether bucket means look separated. It is the right tool specifically
because Re is continuous and possibly non-linear in its relationship to regime while
the regime labels are ordinal, not interval — Pearson would be the wrong tool even
if someone reached for a coefficient.

### Where to land it

`CoverageRegime` is an enum, so `regime_rank` is a deterministic integer mapping.
Return `(rho, p_value)` alongside the bucket means, or replace the bucket means
entirely if the diagnostic is only ever used to answer the correlation question.

### What does not fit

`seed_picker.py`'s `_rank_by_*` functions rank by a single scalar score, not two
paired series — no natural Spearman application there without inventing a second series.

---

## 4. KS distance is computed and discarded — `src/fuzzer_tool/core/edge_tracker.py`

`_cdf_walk()` (line 1610) does a single pass over the merged CDF support and returns
all three norms: `(wasserstein, ks, crps)` — L¹, L∞, and L² of the same CDF difference.

The one live production consumer, `_wasserstein_vs_aggregate()` (line 1523), calls
`_cdf_walk` and explicitly discards the other two:

```python
wasserstein, _ks, _crps = self._cdf_walk(self._hitcount_profile(hc), corpus)
return wasserstein
```

KS is already computed on every call — it is being thrown away, not skipped for cost
reasons.

### Why it matters

Wasserstein (L¹) integrates the CDF difference across the whole hit-count axis, so
it measures aggregate/global divergence and is comparatively insensitive to one
localized spike if the rest of the profile lines up. KS (L∞) is exactly the opposite —
it is the single largest CDF gap, so it flags a seed with one sharply anomalous
hit-count bucket even if the rest of its profile matches the corpus closely.

Given this file's own stated goal for the metric family — the falsification note a
few lines up says the weight exists to catch "three edges hit 500 times" against a
flat-profile seed, i.e. localized intensity anomalies, not breadth (breadth is
explicitly routed to `compute_subsumption_weight` instead) — KS is arguably closer
to what the docstring says this weight is for than Wasserstein is. A profile with
one edge 100x hotter than the corpus but otherwise ordinary could register as only
moderately far in L¹ while being maximal in L∞.

### Where to land it

Pull `ks` out of the existing `_cdf_walk()` call in `_wasserstein_vs_aggregate`
(already computed, zero extra cost), and measure its correlation against
loopiness/edge-count on the same 30–60-seed corpus the JS-divergence comment used,
to check whether it tracks a genuinely different signal than Wasserstein does or is
just redundant with it on real corpora before wiring it into the diversity weight.

### What exists but is dead

`compute_ks_distance()` (line 1669) is unit-tested (`test_edge_tracker.py`,
`test_edge_ground_metric.py`) but has no production caller. `_ks_p_from_cdf_diff()`
has zero production callers — only the tests exercise it.

---

## 5. Scheduler framing: MDP structure, FPL, and KL-UCB

### 5.1 MCTS is an optimal-stopping MDP, not a tree search with a known model

`mcts.py` is the one place a real MDP structure shows up, over `LineageTree`:

- **S** = lineage-tree nodes (seeds), keyed by hash.
- **A** at each node = {descend into a specific eligible child, stop and fuzz this node}.
- **P**: the scheduler does not control transitions. Children are added by
  `LineageTree.insert` whenever the fuzzer (not the scheduler) mutates a seed and
  the mutation lands in the corpus. So `P_a(s,s')` is a graph that grows exogenously
  between calls to `select()` — closer to a restless/growing bandit over a graph than
  to tree search with a fixed, known transition model.
- **R**: `_squash(new_edges)`, a saturating map into [0,1] — deliberately bounded,
  matching the wiki's note that reward is a random variable and that UCT's exploration
  term is only calibrated when it is bounded.

The genuinely good match is **optimal stopping**. The `select()` loop at line 182–186:

```python
if node_key in eligible and self._self_uct(node_key, parent_visits) >= self._uct(
    best_child, parent_visits
):
    best_eligible = node_key
    break
```

is `V(s) = max(stop-value(s), continue-value(s))` — an optimal-stopping MDP, the same
family the wiki links under "Odds algorithm."

Where the analogy breaks: there is no discount factor. MCTS never decays old visits/
values — γ is implicitly 1, undiscounted infinite-horizon. That is fine for a
stationary reward process, but `docs/learnings/2026-08-21-scheduler-convergence.md`
measured that reward stationarity is exactly the assumption that fails in a real
campaign (coverage saturates; a productive subtree goes sterile). Every other
scheduler tested there showed a stationary/non-stationary inversion. Nothing in
`mcts.py` forgets old accumulated reward, so a subtree that was fruitful early and
has since gone sterile keeps an inflated `_value()` for a long time.

### 5.2 AlphaBetaMCTS is the wrong model class

`AlphaBetaMCTSSeedScheduler` (line 305) treats "the fuzzer as maximizer and target
response as minimizer" and runs full alpha-beta minimax. But an MDP is explicitly the
single-player case: "A Markov decision process is a stochastic game with only one
player." Minimax is for the two-player zero-sum case. The target program under fuzzing
is not an adversary; it does not observe your action and pick a state transition to
hurt you. The alternating maximizer/minimizer alternation is structurally meaningless
here, not just a suboptimal heuristic.

### 5.3 FPL — fits alongside `exp3.py`, not as a KL-UCB variant

`Exp3Scheduler` is the repo's only adversarial/non-stochastic-bandit scheduler.
Exp3 carries an explicit overflow guard — `_max_relative`, the 1e9 blowup check, and
a renormalization block (lines 41, 114–119) — because multiplicative weight updates
diverge without periodic rescaling. FPL's additive perturbation has no such failure
mode; there is nothing to renormalize.

Under bandit feedback (only the chosen arm's reward is observed, same constraint Exp3
faces here), FPL needs geometric-resampling FPL (Neu & Bartlett 2013) rather than
Exp3's importance weighting. The "no importance weights" simplicity only holds in
full-information FPL, which does not match this repo's per-round semantics.

Where FPL is a genuine win: empirical robustness under sudden regime shifts (FPL's
perturbation magnitude is a single tunable, Exp3 has both gamma and window_decay to
detune) plus removing the overflow-guard surface area.

### 5.4 KL-UCB — fits `ducb.py` / `swucb.py`, not `cucb.py`

`ducb.py` and `swucb.py` build their index as a literal Hoeffding-style bound on
genuine per-arm Bernoulli statistics. KL-UCB (Garivier & Cappé) replaces that radius
with the tightest upper confidence bound that inverts the Bernoulli KL divergence.
It is a strictly tighter version of the same confidence-interval construction, and
the payoff is largest exactly where a mean sits near 0 or 1 — directly relevant here:
`cucb.py`'s own measured table has a "weak p=0.05" arm, and CUCB's docstring already
documents the Hoeffding radius losing separation as stack size grows (0.25 true gap →
0.006 posterior separation at k=16).

KL-UCB does not fix the semi-bandit credit-assignment problem CUCB's contrast estimator
exists to solve, but for `ducb.py` / `swucb.py`, which have real per-pull binary
outcomes, it is a direct drop-in tightening of the existing index.

For `cucb.py` specifically: the contrast estimator's `mu_hat_i` is a bias-adjusted
quantity derived from round-level aggregates, not a running count of individual
Bernoulli draws of arm i alone — plugging it into a KL-inverted bound built for N_i
real draws would misstate the actual sample size backing the confidence claim.

### Where to start

`ducb.py`'s index (single formula swap, existing test coverage) is the lowest-risk
KL-UCB landing spot. A new `core/schedulers/fpl.py` sibling to `exp3.py` is the
separate second patch. Neither exists in the tree.

---

## 6. What does not fit

- `seed_picker.py`'s `_rank_by_*` functions rank by a single scalar score, not two
  paired series — no natural Spearman application without inventing a second series.
- No missing-data-imputation decision is actually made anywhere in the repo. MCAR/MNAR
  mostly matter for deciding whether to impute or reweight, not just whether to drop —
  the current code's answer is uniformly "skip, don't score," which sidesteps needing
  MCAR/MNAR machinery except as a diagnostic on whether skipping introduces bias.
