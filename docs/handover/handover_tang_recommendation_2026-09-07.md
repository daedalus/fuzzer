# Handover: Tang's quantum-inspired recommendation algorithm as a seed scheduler

**Date:** 2026-09-07
**Base commit:** `b80c6d7` (perf(cmplog): Aho-Corasick multi-pattern operand scan)
**Paper:** Ewin Tang, *A quantum-inspired classical algorithm for recommendation systems*, arXiv:1807.04271v3
**Status:** IMPLEMENTED, OFF BY DEFAULT, MEASURED NEGATIVE, NO A/B RUN YET

---

## 0. One-line summary

The port works and the low-rank reconstruction is genuinely good; as a *scheduler
signal* it is a laundered row sum — controlling for `A.sum(1)` it adds nothing
(mean partial Spearman **+0.006** across ten campaigns, Wilcoxon **p = 1.0**).
It ships behind `--tang`, off, because a measured negative is worth more in-tree
than out, and because the paper's subroutines are independently useful.

**Do not enable it as a default without running `bench_paired.py` first.** No
interventional evidence exists. §7 lists exactly what was never tested.

---

## 1. What landed

| File | What |
|---|---|
| `src/fuzzer_tool/core/schedulers/tang.py` | `TangRecommendationScheduler`, plus faithful ports of Prop 4.2 (inner-product estimation), Prop 4.3 (rejection sampling from `Vw`) and `modfkv_sample_complexity` (Alg. 2's `q`) |
| `src/fuzzer_tool/services/fuzzer.py` | `"tang"` in `_SEED_STRATEGY_NAMES`; constructor kwargs; refit hook on the exec path |
| `src/fuzzer_tool/services/seed_picker.py` | `_pick_tang_seed`, the Elo arm, availability gate |
| `src/fuzzer_tool/cli/commands.py` | `--tang`, `--tang-rank`, `--tang-refit-interval` |
| `tests/test_tang_scheduler.py` | 31 tests |
| `docs/DEEP_DIVE.md` | feature entry (Hard Rule 11) |

### Mapping

```
users    -> corpus seeds        (EdgeTracker.seed_edges keys)
products -> coverage edges
A[i][j]  -> hit count of edge j under seed i  (EdgeTracker.seed_hit_counts)
```

Algorithm 3 samples an entry from row *i* of `D = A Vhat Vhat^T`. Read as a
recommender it answers "which edge should this seed reach next"; read as a
scheduler it yields a per-seed energy, which is what the arm consumes.

### What is deliberately *not* a faithful port

`refit()` runs a dense truncated SVD, not ModFKV. That is the **stronger**
estimator — no row/column subsampling noise — so every negative result below is
a negative on Tang's best case, not on a degraded stand-in. §2 is why ModFKV
itself cannot run here.

---

## 2. ModFKV cannot run at our dimensions

`q = Theta(K^4 / eps_bar^2)`, `K = ||A||_F^2 / sigma^2`, `eps_bar = eta*eps^2`.
`modfkv_sample_complexity()` keeps this checkable rather than folklore.

| corpus | shape | K at sigma_3 | q | rows available |
|---|---|---|---|---|
| png | 116 x 884 | 6.75 | 828,570 | **116** |
| png (binary form) | 116 x 884 | 65.3 | 7.3e9 | 116 |
| zlib | 49 x 229 | 55.3 | 3.7e9 | **49** |

Even at the most forgiving parameters anywhere (`K=4, eps=0.5, eta=0.2`),
`q ~ 1e5`. The subsample is larger than the input. The bound is independent of
`m, n` — which is the paper's entire point — but our `m, n` are *already*
10^2–10^4, so the constant is what governs.

---

## 3. The low-rank assumption: it depends on the cell value, and the two Tang assumptions conflict

`rho_k = ||A - A_k||_F / ||A||_F`. Tang wants `rho << 1` at constant `k`.

| matrix | form | rho@1 | rho@3 | rho@10 | rho@20 |
|---|---|---|---|---|---|
| png 116x884 | hit counts | 0.748 | 0.308 | 0.123 | 0.050 |
| png 116x884 | binary | 0.402 | 0.330 | 0.232 | 0.173 |
| zlib 49x229 | hit counts | 0.371 | 0.145 | **0.051** | 0.025 |
| zlib 49x229 | binary | 0.646 | 0.485 | 0.299 | 0.172 |

Hit counts fit much better. **But they destroy the other assumption.** The
`(gamma, zeta)`-typicality of §5.1 (row norms within a factor of the mean):

| matrix | typical fraction at gamma=1.0, binary | ... hit counts |
|---|---|---|
| png 116x884 | 91.4% | **17.2%** |
| zlib 49x229 | 91.8% | **6.1%** |

The value form that makes the matrix low-rank is the form that makes row norms
heavy-tailed. **Tang's two structural assumptions cannot be satisfied
simultaneously on this data.** This is the most transferable finding here and
applies to any future collaborative-filtering proposal over coverage.

---

## 4. The sampling direction is backwards for a fuzzer

Sampling proportional to `|D_ij|^2` samples by magnitude, and magnitude is
popularity. `E[owner_count of the drawn edge]` vs a uniform draw:

| corpus | rank 2 | rank 5 | rank 10 | uniform | singleton edges |
|---|---|---|---|---|---|
| png 116x884 | 63.0 | 67.5 | 70.3 | 46.96 | 11.9% |
| zlib 49x229 | 24.6 | 25.3 | 27.3 | 12.79 | 13.1% |

1.3–1.9x more crowded than uniform, on corpora where 10–19% of edges are
singletons. That fights `RARE_EDGE_OWNERS` and the crowding penalty in
`seed_picker` head-on — the exact signal the edge-distribution work installed.

`P(the draw is an edge the seed does not already cover)` is 36%/40% at rank 2
but 12.7%/5.0% at rank 10: as a "what next" oracle it is mostly self-recall.

### Inverting it does not help — and is not even a separate measurement

`frontier_mass = 1 - covered_mass` **exactly** (rows of `P` sum to 1; measured
`max|covered + frontier - 1| = 6.7e-16`). So the inverted score is the same
number sign-flipped, and it is anti-predictive in 10/10 campaigns. `mode="frontier"`
exists to keep this falsifiable, not as a candidate. `test_frontier_mode_is_the_complement_of_tang_mode`
asserts the identity so the docstring cannot drift from the code.

---

## 5. The decisive measurement

Prospective design: `edge_first_seen` gives a discovery clock. Split edges at
the 60% quantile, build `A` from early edges only, restrict to seeds present
early, label each seed with how many *late* edges it ends up owning.

Ten independent campaigns (png and zlib x `-s 5/17/29/41`, plus two earlier
runs), 19–87 seeds each after filtering. Spearman vs the label:

| score | range | consistent |
|---|---|---|
| degree | +0.27 … +0.77 | 10/10 positive |
| total hits | +0.16 … +0.80 | 10/10 positive |
| Tang l2 covered mass | +0.04 … +0.79 | 10/10 positive |
| frontier mass | -0.02 … -0.79 | 10/10 negative (see §4: same fact) |
| singletons owned | mixed | 6+/3- |

**Partial correlation of the Tang score controlling for total hit volume:**
mean **+0.006**, positive in 5/10, Wilcoxon signed-rank vs 0 **p = 1.0**. A
pure-noise control through the same path gives +0.01.

Every bit of the Tang score's predictive power is `A.sum(1)`.

---

## 6. What was *wrong* in the first pass, and why

Two claims made mid-analysis were later overturned. Both are recorded because
the failure modes recur.

### 6.1 A silent binarisation destroyed rounds 1–2

After the `to_dict` round-trip, `seed_edges` holds edge ids as `int` while
`seed_hit_counts` and `edge_owner_count` hold them as `str`. An analysis harness
reading the raw dict gets `hc.get(e, 1)` missing every time and a **silently
binary matrix**, with no error.

`EdgeTracker.from_dict` (edge_tracker.py:2765-2816) *does* re-cast with `int(e)`
on all six maps, so **this is not a fuzzer bug** — but anything reading the
payload directly must do the same. `test_string_keyed_hit_counts_do_not_silently_binarise`
pins it.

**The tell:** hit-count, binary and log1p forms returned identical spectra to
three decimals. When two different transforms of the data give the same number,
the problem is upstream of both — same signature as the truncated `random_list`
lesson.

### 6.2 "Low-rank only buys 7% over a column sum" was an artifact

The round-1 holdout masked entries MCAR. But missingness here is fuzz-driven:
`rho(fuzz_count, per-row observed density)` is +0.40 to +0.62 on zlib
(p down to 4e-10), +0.343 on png — so Tang's uniform-subsample assumption (♣)
is badly violated. Redone with correct hit counts *and* MNAR masking (biased
toward low-count entries, which is how coverage is actually missed):

| corpus | mask | rank 10 | popularity |
|---|---|---|---|
| png 116x884 | MNAR | 0.387 | 0.247 |
| png 116x884 | MCAR | 0.391 | 0.347 |
| zlib 49x229 | MNAR | **0.523** | 0.083 |
| zlib 49x229 | MCAR | 0.442 | 0.151 |

**Tang's reconstruction fidelity is real** — 1.6x to 6.3x over popularity, not
7%. The scheduler verdict is unchanged because §5 was already run on hit counts.

---

## 7. Methodology gaps — read this before trusting §5

1. **The label is downstream of the policy under test.** `seed_edges` grows on
   every execution, so "owns late edges" partly measures `fuzz_count`, which the
   *current* scheduler decides. Measured `rho(fuzz_count, label)` up to +0.60.
   Controlling for it the conclusion survives (Tang +0.368 with 9/10 positive,
   degree +0.407 with 10/10), but no IPS estimator and no propensity were used.
   This is the same plumbing gap `handover_nn_over_metrics` identified.
2. **Two clocks, never aligned.** The split uses `edge_first_seen` (a counter)
   while the seed filter uses `added_at` (wall time). Verified no contamination
   (0 kept seeds have their first edge on the late side), but the
   `added_at <= p75` filter is a near no-op; what actually filters is "owns an
   early edge", which is weaker than "existed early".
3. **No interventional evidence at all.** Everything is observational on a
   corpus the current scheduler produced. The project standard is
   `bench_paired.py` with a pre-registered threshold. It was not run.
4. **Two targets, both small and single-format** (png, zlib). ffmpeg — 8189
   edges, genuinely multimodal — is where the low-rank assumption would have its
   best shot, and it would not build in that container (vendored headers).
5. **Statistical power.** n runs 19–87 after filtering; no weighting, no
   intervals. At n=19 a rho of +0.30 has 95% CI [-0.18, +0.66]; it excludes zero
   only around n=55. The eight replication runs share seed files, target and
   mutator set — only `-s` differs — so effective independence is well under 10.
6. **Rank fixed at 10, no cross-validation.** ModFKV thresholds by `sigma`, not
   by `k`, so even the idealised estimator is not the paper's.
7. **Cost timings used dense random matrices**, not real sparse sets, and
   incremental rank-1 update was never considered — the matrix grows one row per
   accepted seed, so a streaming update would weaken the cost objection.

---

## 8. Cost

Randomized SVD (k=10, oversampling 8) and the matrix build from Python sets:

| shape | build | SVD | project + sample one seed |
|---|---|---|---|
| 116 x 884 | 6.0 ms | 1.18 ms | 0.019 ms |
| 500 x 2000 | 40.8 ms | 5.20 ms | 0.026 ms |
| 2000 x 8189 | 658 ms | 102 ms | 0.082 ms |
| 8000 x 8189 | 2407 ms | 481 ms | 0.083 ms |

The build dominates. Same shape as the NN-over-metrics finding: the cost is
feature extraction, not the model. This is why `maybe_refit` is on the exec path
behind an interval gate and never on the pick path.

**Lemma 3.1's BST is also a loss.** It looked like the one portable ingredient —
a real answer to `_cdf_pick` rebuilding `itertools.accumulate` every 200 execs.
In CPython a Fenwick point update costs 1.09 us at N=8000 while the full cumsum
rebuild amortized over 200 picks costs 0.88 us, and its draw is 5–6x slower than
`bisect`. `accumulate` and `bisect` run in C; the tree runs in Python. Do not
re-propose it without a C extension.

---

## 9. Rejected, with reasons (do not re-propose without new evidence)

| Idea | Why it died |
|---|---|
| Inverted frontier score (`mass on uncovered / owner_count`) | Algebraically `1 - covered_mass`; anti-predictive 10/10 |
| Residual after removing the rank-1 popularity component | +0.45 zlib, -0.19 png — sign flip, noise |
| Transpose (for a rare late edge, which seed reaches it) | Degenerate: `\|\|D_i\|\|` gives *exactly* degree's precision (0.067/0.067, 0.516/0.516) |
| Operator x edge matrix | Cannot be built from state — `corpus['op_edges']` is a scalar credit float per operator (119 entries), not a matrix. Would need new instrumentation |
| Degree-normalized label (to test the degree confound) | Degree still wins (+0.338, +0.498); frontier/owners flips sign across datasets |
| Lemma 3.1 Fenwick/BST sampler | §8 — loses to `accumulate`+`bisect` in CPython |

---

## 10. If someone picks this up

Ordered by what would actually change the answer:

1. **Run `bench_paired.py`** with `--tang` vs baseline, pre-registered threshold.
   Everything above is observational; this is the only test that settles it.
2. **Build ffmpeg and re-measure §3 and §5 there.** 8189 edges and real
   modality is the one condition under which the low-rank assumption plausibly
   holds. If `rho@10 < 0.05` *and* typicality survives, §3's conflict is
   target-specific rather than structural, and this reopens.
3. **Log the training set.** Gaps 1 and 3 both reduce to "there is no
   off-policy log". The sequence in `handover_nn_over_metrics` §5 (append-mode
   `dump_stats`, chosen operator in the ablation log, raw M3 inputs, an IPS
   estimator with a falsification test that `final_w` *is* the propensity) is
   worth doing on its own and would make this measurable properly.
4. **Use `recommend()` for edges, not seeds.** The edge answer is the paper's
   actual contribution and was never the thing that failed — a directed-fuzzing
   or frontier-targeting consumer is the honest use, and none exists yet.
