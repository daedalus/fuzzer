# Huffman tree as a seed scheduler — feasibility analysis (2026-09-16)

Pulled clean to HEAD `f6afef0` (`feat: port three FFmpeg-derived structural
mutators + wire up flac.py`). Asked to analyze adding a Huffman tree as a
seed scheduler. Note: `huffman_tree_mutate` already exists in this repo, but
as a *mutation operator* over DEFLATE-adjacent bitstreams — unrelated to
seed scheduling. No prior art for Huffman-as-scheduler anywhere in
`docs/handover` or `docs/`.

## Where "seed scheduler" weighted sampling already lives

`services/seed_picker.py::_cdf_pick` is already an O(log n) sampler: it
caches `itertools.accumulate(weights)` and does `bisect` against it,
invalidated only by weight-list object identity. `weighted_pick_seed`
deliberately keeps that weight vector **stale for up to 200 execs** (or
until the corpus grows by 20 seeds) because a full CDF rebuild is O(n) and
recomputing per-exec was an O(corpus_size) tax on every accepted seed
during growth phases (see the comment at `seed_picker.py:1786`).

That staleness window is the actual design tradeoff a "better tree" would
be buying its way out of — not sampling speed, which is already O(log n).

## Two different things "Huffman tree scheduler" could mean

**(1) Literal static Huffman tree, rebuilt periodically.** No better than
today: building a Huffman tree is O(n log n), strictly worse than the
existing O(n) CDF rebuild, and it still needs a full rebuild on every
reweight unless shape is maintained incrementally (Vitter/FGK dynamic
Huffman). Not worth building.

**(2) Entropy-optimal *sampling shape*.** A Huffman tree minimizes expected
leaf depth to Shannon entropy H(p) rather than log2(n). Under the skewed
"favored seed" energy distributions this project's own docs describe
(a handful of high-energy seeds dominate mass — the same regime the
EcoFuzz/MOpt marginal-cost work and the katz/tang seed arms are tuned for),
H(p) can be much less than log2(n).

## What the synthetic harness actually measured

`tools/bench_huffman_scheduler.py` (checked in, not wired into the fuzzer).
Three experiments, `paretovariate(alpha)` for skew (lower alpha = heavier
tail = more "favored seed" dominance):

**(A) Sampling depth.** Huffman expected depth tracks H(p) almost exactly,
confirming the theory — e.g. n=500, alpha=0.5: H(p)=0.18 bits vs
log2(n)=8.97. But `_cdf_pick`'s bisect is already O(log n) — a handful of
comparisons either way is not where the time goes.

**(B) Staleness error, the real cost.** `KL(true_weights || stale_weights)`
under the current 200-exec window, at realistic skew:

| n | skew(alpha) | recompute_every | mean KL (bits) | max KL (bits) |
|---|---|---|---|---|
| 100 | 0.6 (heavy tail) | 200 | 4.29 | 17.48 |
| 2000 | 0.6 | 200 | 1.29 | 20.01 |
| 100 | 4.0 (near-uniform) | 200 | 0.06 | 0.20 |

Staleness cost is **negligible when weights are close to uniform and huge
exactly when the corpus has the skewed, favored-seed structure this project
cares about** — a multi-bit KL gap between the distribution the scheduler
thinks it's sampling from and the one it should be.

**(C) Wall-clock, why the staleness window exists at all.** Full CDF
rebuild vs. an O(log n) Fenwick-tree (BIT) point-update, per reweight:

| n | CDF rebuild | Fenwick update | speedup |
|---|---|---|---|
| 100 | 3.2 us | 0.64 us | 5x |
| 1000 | 31.4 us | 0.81 us | 39x |
| 10000 | 289 us | 1.13 us | 256x |

## Recommendation

> **Superseded 2026-09-17** — built and measured; it loses. See the
> Follow-up section at the end before acting on this.

Don't build a literal Huffman tree. Build a **Fenwick-tree-backed exact
sampler** as a drop-in replacement for the `_cdf_pick` cache in the
`weighted_pick_seed` path: O(log n) point-update on every energy change
*and* O(log n) sample, so the 200-exec staleness window can go away
entirely at a cost (per §C) that's cheaper than today's batched rebuild
already pays, not more expensive. This captures ~all of the measured
benefit (§B) at a fraction of the complexity of maintaining an
entropy-optimal (Huffman) shape incrementally (Vitter/FGK).

Reserve genuine dynamic Huffman shaping as a follow-up only if profiling
ever shows raw per-sample comparison count — not staleness — as the
bottleneck; §A shows that's a few-comparisons-per-pick effect, dwarfed by
the ~us-vs-hundreds-of-us gap in §C. Not implemented against
`seed_picker.py` itself in this pass (would touch `weighted_pick_seed`,
`_compute_weights`, `_weight_cache`/`_cdf_cache` invalidation, and the
Elo-arm registration in `fuzzer.py` the way `katz`/`tang`/`kruskal_count`
are wired — real surface area, worth a dedicated pass once this recommendation
is confirmed) — this pass is feasibility-only, per the project's own practice
of measuring before integrating (see the katz/tang-for-operator harness).

## Follow-up (2026-09-17): the Fenwick wiring was built, measured, and rejected

Pulled to `03c80ca`. A `core/fenwick_sampler.py` (`FenwickSampler`:
`update`/`append`/`find`/`sample`) plus a `seed_picker._fenwick_pick` drop-in
for `_cdf_pick` were written against this recommendation, with 30 tests. They
are **not** landed. This is the second time the idea has been measured and
lost; the first is `handover_done_2026-09-06.md` §8/§9 (Tang Lemma 3.1), which
said not to re-propose it without a C extension. §C above did not cite that
row, and it compares the wrong two things.

**Why §C's speedup does not transfer.** It prices one Fenwick point update
against one full CDF rebuild. The picker never does a single point update: a
reweight is a full `_compute_weights` pass, and every seed's weight moves on
every pass because `age = now - meta["added_at"]` (`seed_picker.py:1399`)
feeds the exploit and burst terms. A reweight is therefore *n* updates
(≈1 us each in Python) against one `accumulate` (C). Likewise §B's KL gap is
the cost of *not recomputing weights*; a faster sampler does not recompute
them, so it cannot close that gap. The 200-exec window exists because of
`_compute_weights`, not because of the prefix sum.

**Measured on this container** (CPython, Pareto(0.6) weights, min of 7):

| n | `accumulate` | Fenwick build (O(n log n) as written) | O(n) linear build | `bisect` draw | Fenwick `find` |
|---|---|---|---|---|---|
| 1000 | 17 us | 509 us | — | 0.12 us | 1.01 us |
| 2000 | 34 us | 1149 us | 179 us | 0.17 us | 1.19 us |
| 8000 | 131 us | 5288 us | 735 us | 0.13 us | 1.24 us |

One steady 200-pick window at n=8000: `accumulate` + 200 bisects = 158 us;
Fenwick build + 200 finds = 5536 us (≈1 ms with the best-case linear build).
The growth path — the one case the tree was aimed at (≤19 padded-list CDF
rebuilds per window) — costs 3.5 ms today at n=8000 against 5.9 ms for build
plus 19 appends. Both lose.

The growth path is also the wrong place to save. Each growth event already
invalidates `_pareto_cache` (keyed on `len(f.corpus)`), which walks the whole
corpus in Python and rehashes every seed through `seed_key`. Synthetic
replica with 64 B–4 KiB seeds: 2.7 ms (xxh64) / 6.4 ms (sha256) at n=2000,
9.5 ms / 23.2 ms at n=8000, against 74 us / 347 us for pad + `accumulate`.
The CDF is ~1–3 % of a growth event. If growth-phase pick cost ever matters,
that pass is the lever, not the sampler.

**Two hazards found in the prototype, recorded so a rewrite avoids them:**

1. *In-place growth starves new seeds in `_cdf_pick`.* The `_fenwick_pick`
   tests assumed `weighted_pick_seed` switches from `cached + [1.0]*k` to
   `cached.extend(...)` so the tree can detect growth by identity. `_cdf_pick`
   validates by identity too, and it is still called for the `"corpus"` slot
   from `_pick_from_pareto_front` (corpus < 3, empty `seed_meta`, front < 2).
   Measured: after an in-place extend from 10 to 20 weights, 2000 draws never
   selected index ≥ 10 and raised nothing — the stale prefix sum clamps
   silently. The padded list being a *new* object is load-bearing.
2. *Cancellation residue in long-lived trees.* Internal nodes carry every
   historical delta. `FenwickSampler([1e17, 1, 1, 1]).update(0, 0.0)` leaves
   `total() == 0.0` (true total 3.0), so `sample` raises "must be greater than
   zero". At realistic magnitudes drift was benign (relative 4e-12 after 2e5
   updates on n=5000), but a tree that is never rebuilt — the whole point of
   removing the window — needs a rebuild policy for heavy-tailed energies.

Draw equivalence held on random `r` (0 mismatches in 100 000 draws against
`bisect` on `accumulate`) but not term for term: with `r` placed exactly on a
prefix sum, 4804 of 9941 draws disagreed, because the tree's partial sums
associate differently from `accumulate`. `_cdf_pick`'s docstring promises the
stronger property; a replacement could only promise the weaker one.

**Status:** rejected again. Keep `_cdf_pick`. Re-open only with a C-level tree
(or numpy-backed batch update) *and* an incremental `_compute_weights` that
actually emits sparse deltas — without the second, the first has nothing to
consume.

Where the shape does fit: `Exp3Scheduler.select_op` (one arm changes per
record, the decay factor cancels). The tree landed there on 2026-09-17 —
see `core/schedulers/op_exp3.py` and `tests/test_exp3_fenwick_select.py`.
