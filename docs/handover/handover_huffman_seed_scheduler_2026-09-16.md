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
