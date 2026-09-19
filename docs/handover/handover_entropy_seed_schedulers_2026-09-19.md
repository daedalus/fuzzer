# Entropy-aware seed schedulers — six standalone proposals (2026-09-19)

**Base:** `766fefc` (`corpus: cumulative Shannon entropy tracked at seed-read
time`), on top of upstream `4af8a07`.
**Status: §2 and §3 IMPLEMENTED (2026-09-19). §1, §4, §5, §6 still analysis
only.** See `docs/DEEP_DIVE.md` (Byte-entropy seed arms) for what shipped;
the corrections the build forced on this doc are recorded inline below.

Asked what seed schedulers could harness the two Shannon-entropy signals now
available (per-seed byte entropy, and the corpus-wide cumulative tracker from
`766fefc`), then asked to write each proposal up as its own self-contained
seed scheduler — i.e. a standalone `core/schedulers/seed_*.py` module with
its own dispatch entry in `seed_picker.py`, its own CLI flag, and its own
Elo-pool registration, the same shape as `seed_katz.py` / `seed_kruskal_count.py`
/ `seed_canary.py` — rather than folded into the existing default weighted
picker as one more `_weight_*` term.

## 0. What's already there, so a new scheduler doesn't collide with it

Two Shannon entropies already exist in the tree, both different from what
this doc proposes:

- **Edge-hit entropy** (`EdgeTracker.shannon_entropy_seed` /
  `shannon_entropy_global`) — entropy over a seed's *coverage profile* (which
  edges it hits, how often). This is what `seed_picker._weight_entropy_and_distance`
  (`seed_picker.py:1050-1101`) already uses for a deviation bonus, and what the
  live status line's `ent:`/`simp:` fields report. **Not touched by this doc.**
- **Byte-content entropy, per seed** (`byte_entropy_pct`, `core/byte_entropy.py`)
  — entropy over a seed's own bytes. Cached once per seed in
  `seed_meta["input_entropy"]` by `Fuzzer._seed_entropy_pct`
  (`services/fuzzer.py:3683-3699`). Currently consumed in exactly one place:
  `SeedScorer._energy_factor` (`core/schedules.py:388-396`), which scales
  *mutation energy* (how much budget a seed gets once already picked) via three
  fixed breakpoints — `ENTROPY_SPARSE_PCT=25`, `ENTROPY_STRUCTURED_PCT=62`,
  `ENTROPY_RANDOM_PCT=93` (`schedules.py:47-49`). It never influences *which*
  seed gets picked.
- **Byte-content entropy, corpus-wide (cumulative)** (`core/byte_entropy.CumulativeByteEntropy`,
  `766fefc`) — pooled entropy over every seed's bytes folded in at
  `load_corpus()` time, stored as `f._corpus_entropy`, O(1) to read. Currently
  surfaced only in the status line (`byte-ent:`) and the final report. Nothing
  reads it back into a decision yet.

So: individual byte entropy exists but only gates energy, and cumulative byte
entropy exists but only gets displayed. The six proposals below are ways to
make a seed *selection* decision from one or both.

All six are written as pure-function scorers over `(corpus, corpus_entropy)`
first — matching the project's own practice (see `handover_huffman_seed_scheduler`,
`handover_mincut`, `handover_centrality`) of keeping the scoring math testable
standalone before any wiring into `Fuzzer`/`seed_picker.py`. Wiring for all six
follows the same three-step shape `seed_katz`/`seed_kruskal_count` already
established: (1) `services/fuzzer.py` builds the strategy object at init and
registers it in the Elo pool, (2) `seed_picker.py` gets a `_pick_<name>_seed`
dispatch entry gated on `getattr(f, "_<name>", None) is not None`, (3)
`cli/commands.py` gets a `--<name>` flag. None of that is done here.

---

## 1. `seed_entropy_deviation.py` — byte-entropy deviation from the corpus mean

The direct byte-content analogue of the edge-hit deviation bonus already in
`_weight_entropy_and_distance`. Score:

```
score(seed) = |byte_entropy_pct(seed) - mean(byte_entropy_pct(s) for s in corpus)|
```

normalized into a weight the same way the existing edge-hit version does —
`w *= 1.0 + min(deviation / effective_mean, 1.0) * 0.5`. Picks seeds whose
*content statistics* stand out from the corpus average, independent of
whether their *coverage* does. On a corpus mixing, say, a text-protocol format
with an embedded compressed blob, this favors alternating attention between
the two regimes rather than letting whichever regime is more numerous dominate
sampling.

**Cost:** O(1) per seed once `input_entropy` is cached (already true), O(n) to
refresh the corpus mean — same amortization problem `_mean_seed_entropy`
already solves for the edge-hit version (`seed_picker.py:1074-1084`, cached and
invalidated on `len(seed_hit_counts)` change). Reuse that caching pattern
verbatim, keyed on corpus length instead.

**Open question before wiring:** whether "deviation from the mean" is the
right shape at all, or whether it should be deviation from the *cumulative
pooled* entropy (§2's game, without the KL machinery) — the pooled entropy is
not the mean of per-seed entropies (Jensen's inequality: pooling low-entropy
and high-entropy seeds together generally raises the pooled figure above
either one, let alone their average). Worth computing both against a real
corpus before picking one; this proposal deliberately keeps the *mean* version
distinct from §2 so they can be A/B'd independently rather than shipped as one
conflated metric.

---

## 2. `seed_entropy_kl.py` — KL-divergence novelty scheduler

A proper relative-entropy version of §1: score a seed by how much its own byte
distribution diverges from the corpus's *pooled* distribution, not just by a
scalar gap between two entropy numbers.

```
score(seed) = KL(P_seed || P_corpus)
            = sum_b P_seed(b) * log2(P_seed(b) / P_corpus(b))
```

`CumulativeByteEntropy` already maintains the 256-bin corpus-wide frequency
table internally (`_freq`, `core/byte_entropy.py`) but doesn't expose it; this
needs one new accessor, `freq_dist() -> tuple[float, ...]` (256 normalized
probabilities), no change to the O(1) `add`/`bits` path. Per-seed cost is
O(min(len(seed), cap)) to build that seed's own histogram, i.e. the same pass
`byte_entropy_bits` already does — can be fused into one function that returns
both a seed's own entropy *and* its histogram, avoiding a second scan.

This is a genuinely different criterion from §1: two seeds can have identical
scalar entropy (say both at 50% of max) while one is uniform-over-two-symbols
and the other is uniform-over-a-different-two-symbols — §1 scores them
identically (same deviation from the mean/pooled scalar), §2 correctly scores
them as maximally different from the corpus if the corpus's pooled
distribution looks like neither. This is the sharper tool if what's wanted is
literally "which seed introduces a byte pattern the corpus doesn't have,"
matching the active-learning "expected information gain" seed-selection
framing rather than a coarser one-number proxy.

**IMPLEMENTED** as `core/schedulers/seed_entropy_kl.py` (`--entropy-kl`).

Two corrections this doc got wrong. (1) The accessor is `freq_dist()` on
`CumulativeByteEntropy` as proposed, but the arm does **not** read
`f._corpus_entropy`: that tracker is built once per `load_corpus()` and
never sees a seed discovered mid-campaign, and has no way to drop a pruned
one, so scoring against it compares every mid-run seed to a distribution
that predates it. The arm owns its own pool, folding on admission and
unfolding on eviction (`CumulativeByteEntropy.remove()`, added for this) so
`Q` is exactly the live corpus. (2) The fusion suggested here landed one
level lower, as `byte_histogram()` + `entropy_bits_from_counts()` in
`core/byte_entropy.py`, which `byte_entropy_bits` and the tracker both go
through — a 4 KiB fold went 167us -> 24us as a side effect.

**Open question before wiring:** `P_corpus` needs an epsilon floor
(Laplace/add-one smoothing) wherever `P_corpus(b) = 0` for a byte value the
seed uses but the corpus has never seen, or `log2(P_seed(b) / 0)` is
undefined. Standard fix, but the smoothing constant needs to be picked and
measured, not defaulted blindly — the KL-UCB paper-fidelity handover
(`handover_kl_ducb_paper_fidelity_2026-09-14.md`) is the standing reminder in
this repo that borrowing a formula without checking its assumptions against
the actual regime it's applied in has bitten this project before.

**RESOLVED.** `POOL_SMOOTHING = 1/256` (one pseudo-byte spread over the
alphabet), not Laplace's add-one, which injects 256 pseudo-bytes and would
dominate any pool under a few kilobytes. The question also turned out to be
narrower than it looked: every seed passed to `scores()` is folded into the
pool before it is scored, so its support is already in `Q` and the
undefined `log2(p/0)` term cannot arise for a corpus seed at all. The floor
guards the margin (a seed scored before folding), it does not tune the
ranking.

---

## 3. `seed_entropy_zscore.py` — corpus-relative adaptive regime scheduler

Not a modification of `SeedScorer`'s fixed 25/62/93 breakpoints in place —
per the reframing above, a fully standalone alternative picker that never
touches `SeedScorer`. It computes each seed's byte entropy as a z-score
against the corpus's own running mean and variance (via `RunningMoments`,
already in the tree at `core/running_stats.py`, already used elsewhere for
exactly this kind of streaming moment tracking) rather than a target-agnostic
fixed percentage scale, then picks with weight peaked at some target z (e.g.
z≈0, "typical for this corpus") or trough at |z| large ("outlier, likely
noise"), whichever direction the campaign wants.

This exists as its own scheduler because the general fixed-threshold problem
is real and independent of anything else in this doc: `ENTROPY_RANDOM_PCT=93`
penalizes any seed above 93% of the theoretical max-entropy scale as
"probably compressed/random noise" — correct for, say, a text-config-format
target, actively wrong for an image/audio/compressed-container target where
*every* well-formed seed legitimately sits above 90%. A z-score against the
corpus's own observed distribution self-calibrates to whatever the target's
native entropy regime actually is, with no target-specific constant to tune.

**Cost:** O(1) per seed given `RunningMoments`' incremental update, invoked
once per corpus-load and once per `save_to_corpus` event (mirrors where
`766fefc`'s `CumulativeByteEntropy.add()` is already invoked, so the two could
share a single hook point rather than two separate ones scanning the same
bytes).

**IMPLEMENTED** as `core/schedulers/seed_entropy_zscore.py`
(`--entropy-zscore`, `--entropy-zscore-target`). `target_z` is the only
knob: 0 favours seeds typical for this corpus, positive chases the
high-entropy tail, negative the sparse one, so "whichever direction the
campaign wants" is one parameter rather than two modes.

**Open question before wiring:** what to do in the first N seeds of a fresh
corpus, where the running variance estimate is noisy or degenerate (a corpus
of 1-2 seeds has no meaningful spread). `RunningMoments` callers elsewhere in
this tree already have a minimum-sample gate for this
(`analyzer_critical_slowing.py` requires a minimum window before trusting its
variance/autocorrelation signals) — reuse that precedent rather than inventing
a new one.

**RESOLVED, and the gate needed a second half this doc did not anticipate.**
`MIN_OBSERVATIONS = 20` is `CriticalSlowingDown.min_observations` as
suggested, but a count gate alone is not enough: `byte_entropy_bits` of a
single-symbol input returns 5.6e-15, not 0.0, so a corpus of constant seeds
passes any `stddev > 0` test with a 1.5e-15 spread and produces z-scores
that are rounding noise amplified to +-3. Readiness therefore also requires
`stddev > MIN_SPREAD_PCT = 1e-6`. Two further consequences:

- The arm is listed in the Elo pool **while still cold** and only afterwards
  while `ready`. Scoring is what observes seeds, so an arm excluded until
  warm can never warm up; and a corpus with genuinely no entropy spread
  would otherwise be a phantom opponent declining forever. On any corpus of
  at least `MIN_OBSERVATIONS` seeds this costs no declined pick at all —
  the first selection observes the whole corpus before its own readiness
  check.
- The moments are **not** persisted. A resume reloads the corpus whole and
  the first scoring pass rebuilds the calibration from the seeds that
  survived; restoring a window on top of that counts every one of them
  twice. `to_dict()` carries counters and knobs only.

---

## 4. `seed_entropy_gradient.py` — chase seeds behind recent entropy growth

Distinct from §1-3 (which score a seed by a *static* property of its bytes):
this scores a seed by its *track record* — an EWMA of how much the cumulative
corpus entropy (`f._corpus_entropy.bits()`) moved in the window immediately
after that seed was last picked and mutated. A seed credited with recent
entropy-increasing children gets picked again; one whose recent children
haven't moved the pooled distribution decays.

This is the byte-entropy analogue of what `analyzer_critical_slowing.py`
already does for discovery-rate variance (rising-variance/autocorrelation as
a precursor signal before a coverage phase transition) — same idea, different
observable. Corpus-wide byte diversity plateauing is a signal independent of
edge coverage plateauing: a target can still be gaining new byte-pattern
diversity (new record types, new field encodings) after coverage growth has
flattened, or vice versa, and neither existing detector sees the other's
signal.

**The hard part, flagged rather than glossed over:** credit assignment.
Between one pick of seed S and the next tick where `_corpus_entropy` is read,
other seeds get picked and mutated too, and the change in the pooled corpus
entropy is a shared effect of everything added to the corpus in that window,
not attributable to S alone. `handover_non_ucb_schedulers_2026-09-13.md` §0
already documents this exact failure mode in this codebase for a different
signal (Elo match credit fanned out to arms that weren't played, with no
counterfactual) — a naive version of this scheduler would repeat it. The
honest starting point is to credit S only for entropy contributed by S's
*direct children* specifically (children carry a parent pointer already, via
`core/lineage.py`'s `LineageTree`), i.e. `Δentropy` computed by folding just
the newly-accepted children of S into a scratch copy of the pooled histogram,
not the whole corpus's aggregate delta since the last tick. That's an O(1)
per-child computation against `CumulativeByteEntropy`'s existing running
totals (fold the child's bytes into a copy, diff `bits()` before/after,
discard the copy) rather than a global attribution problem.

---

## 5. `seed_entropy_shapley.py` — marginal entropy contribution scheduler

Score each seed by its Shapley-style marginal contribution to the corpus's
pooled entropy: how much would `_corpus_entropy.bits()` drop if this seed were
removed from the pool. Dual-purpose by construction — the same score that
tells `seed_picker` "this seed is uniquely valuable, pick it more" is exactly
the score corpus minimization needs to *not* retire that seed as a redundant
near-duplicate.

**This does not get the same closed form `core/shapley.py` already has for
edge attribution, and that's worth stating plainly rather than assuming it
carries over.** The edge-attribution game has a clean symmetry: credit for an
edge splits evenly among the operators that can produce it, independent of
permutation order, which is what makes `contribution(e, op) = credit(e,op)/k`
an exact one-pass answer instead of a Monte Carlo estimate. Shannon entropy of
a pooled multiset is not additively separable across members that way — it's
a submodular set function (diminishing returns: the tenth seed with a given
byte pattern adds less marginal entropy than the first), which gives Shapley
values nice approximation *properties* but not, in general, a comparable
closed form. Two honest options, cheaper first:

- **Leave-one-out marginal** (not exact Shapley): `Δ(s) = H(corpus) -
  H(corpus \ {s})`, one histogram-subtraction pass per seed, O(n) total using
  `CumulativeByteEntropy`'s running `_freq` table (subtract s's counts,
  recompute `bits()`, add back). Cheap, orders seeds correctly for "which is
  most redundant" in the common case, but is not permutation-averaged so it
  can misprice a pair of near-identical seeds (removing either alone looks
  cheap; removing both is expensive — the standard failure mode leave-one-out
  credit has on near-duplicates).
- **Full permutation-sampled Shapley**, same Monte Carlo shape the edge
  version *used to* use before the closed form was found
  (`handover_...shapley...` sessions this project has already run) — correct
  even for near-duplicate cliques, but back to O(n_samples · n) instead of
  O(n).

Per this project's own stated practice (`Cook-Mertz`, `EI vs UCB acquisition`,
the analytic transfer-entropy bias correction that got measured and rejected):
build the leave-one-out version first, measure how often it disagrees with a
Monte Carlo reference on a corpus with deliberately-planted near-duplicate
clusters, and only pay for full Shapley sampling if that measured disagreement
rate is actually large enough to matter for minimization decisions.

---

## 6. `seed_entropy_contribution.py` — EWMA parent-credit scheduler

The seed-scoped sibling of proposal §4, factored out as its own module because
it answers a different question with a different (and cheaper, and
unambiguous) credit rule: not "is the corpus's entropy still climbing"
(§4, a population-level trend) but "which *specific* seed has historically
been the better parent." Maintain a per-seed EWMA of `Δentropy` contributed by
that seed's direct children only — same child-level fold-and-diff computation
proposed in §4 as the fix for its credit-assignment problem, but here it *is*
the whole scheduler rather than a workaround for one. A seed's score decays
between successes (`ewma *= decay` each round it's not the parent of a newly
accepted child) and jumps on a genuinely diversity-increasing child, so
picking tracks "recently productive parents" the way `EdgeTracker`'s existing
per-seed reward-shaping already tracks recently-productive parents for
*coverage* — same mechanism, byte-diversity payoff instead of edge payoff.

Because the credit rule here is scoped to direct children only (no shared-tick
attribution across unrelated seeds), this one has no open credit-assignment
question left to resolve before it could be measured — unlike §4, which is
explicitly a trend detector over the whole corpus and inherits the
multi-seed-attribution problem that entails. §4 and §6 are proposed as
separate modules specifically so one isn't blocked on solving the other's
harder problem.

---

## Priority, if picking one to build first

1 and 2 are the cheapest and most directly testable against a synthetic
corpus (plant seeds with known KL/deviation properties, check the scheduler
orders them as expected) before touching `Fuzzer`/`seed_picker.py` at all — the
same feasibility-first move the Huffman-scheduler and mincut/centrality
handovers already used. 3 is a genuine, independently-motivated correctness
fix to a real blind spot (target-specific fixed thresholds) rather than a new
capability, and is a similarly small standalone build. 5 and 6 need a
`LineageTree` parent-pointer read that 1-3 don't. 4 is the one with an
unresolved design question (credit assignment) and should wait until 6 has
been measured, since 6 provides the child-fold-and-diff primitive 4 would
otherwise have to invent from scratch anyway.

§2 and §3 were built on 2026-09-19, in that order, against this priority.
§1 was deliberately skipped rather than shipped alongside §2: its open
question (mean vs pooled) is a question about which metric §2 already
answers properly, and shipping both would have conflated them exactly as
this doc warned. §4, §5 and §6 are untouched.
