# Handover — seventeen-source survey: mazes, diff, SSSP, noise, LNS, sampling, compression

**Status: ANALYSIS ONLY.** Nothing here is implemented. Every claim below was
checked against live source at `fdbf237` and every number was measured in a
container on that tree. Where a source turned out to be worth nothing to us,
that is recorded too — the "Rejected" section exists so the same seventeen
links do not come back around as a fresh survey in a month. Same discipline as
`docs/port-backlog.md`.

The seventeen inputs collapse to **eleven distinct ideas**: two of the links
are the same artifact (see D2), and two are algorithm catalogues rather than
techniques (see R3).

---

## Ranking

Ordered by evidence, not by how interesting the source is.

| # | Source | Lands on | Evidence | Verdict |
|---|--------|----------|----------|---------|
| A1 | Diff algorithms (flo.znkr.io) | `core/similarity.py` | **measured, 840×, fixes an OOM** | do first |
| A2 | Diff algorithms (same) | `core/crash_metadata.py::cluster_crashes` | **measured, 59 s at n=400** | do first |
| B1 | Floyd's sampling | `core/rand_pool.py::sample` | **measured, 4–9×** | cheap, do |
| B2 | BED / lexicase (`bed-full.pdf`) | `core/ga.py` selection | structural, grounded | high value, real work |
| B3 | Maze algorithms (jamisbuck + astrolog) | seed & operator schedulers | structural, grounded | high value, design work |
| C1 | Perlin noise | new sibling to `fractal_voronoi` | gap confirmed | good fit, moderate |
| C2 | TSP | operator registry gap | **gap confirmed: 0 of 157** | cheap, do |
| C3 | Knuth Algorithm X | mutation candidate evaluation | gap confirmed | narrow but real |
| C4 | LEGO thesis (LNS) | `qea.py` / stall handling | structural | speculative |
| C5 | Compression walkthrough | `mutations/recompress.py` | gap confirmed | moderate |
| D1 | Expected-linear-time MST | crash clustering | **negative result** | reject as stated, see A2 |
| D2 | DMMSY-SSSP / arXiv 2504.17033 | `core/distance.py` | **negative result** | reject |
| R1 | Evolutionary algorithm (Wikipedia) | — | already covered | reject |
| R2 | Wilson's applet (cruzgodar) | — | duplicate of B3 | fold into B3 |
| R3 | keon/algorithms, TheAlgorithms/Python | — | catalogues | reject as dependency |

---

## A1 — The Levenshtein path allocates a full DP table and will OOM on a real crash

**Source:** <https://flo.znkr.io/diff/> — specifically the observation that
quadratic-memory diff implementations are a liability, and that the linear-space
Myers variant plus preprocessing is the standard answer.

`core/similarity.py::_levenshtein_align_numpy` allocates
`dp = np.empty((n + 1, m + 1), dtype=np.int32)` to support traceback. That is
`4·n·m` bytes for a single alignment.

Measured on this tree (near-identical inputs, one bit flip per 200 bytes, plus a
2-byte length change so the equal-length shortcut does not fire):

```
n=   512      1.1 ms   dp table =    1.1 MB   peak RSS =   38 MB
n=  2048     74.5 ms   dp table =   16.8 MB   peak RSS =   52 MB
n=  8192   1093.6 ms   dp table =  268.6 MB   peak RSS =  286 MB
n= 16384   3941.0 ms   dp table = 1074.0 MB   peak RSS = 1044 MB
```

A 64 KiB crash input against a 64 KiB corpus seed needs a **17 GB** table. It
does not get slow; it dies.

**Who calls this without a guard.** `adapters/filesystem.py:571` already knows —
it caps `compute_delta_v2` at 512 bytes with the comment *"v2 uses
levenshtein_align which is O(n\*m) — skip for large inputs."* The other two
callers have no cap at all:

- `core/root_cause.py:61` `build_edit_script(base, crash)` — crash triage, runs
  on the delta-debugging path.
- `core/crash_metadata.py:364` `levenshtein_diff_offsets(crash_data, nearest)`
  and `:370` `levenshtein_similarity` — the crash sidecar, on every novel crash.

So one caller has the size guard and two do not, and the two that do not are the
ones that see attacker-controlled input sizes. The 512-byte comment is the
evidence that the cost was already understood at one call site and never
generalised.

**The fix, and the honest caveat.** Myers' greedy `O(ND)` with the linear-space
refinement. Prototyped the forward pass in pure Python on the same inputs:

```
n=   512    0.04 ms  D=  6
n=  2048    0.18 ms  D= 22
n=  8192    1.57 ms  D= 82        vs 1093.6 ms  →  697×
n= 16384    4.69 ms  D=164        vs 3941.0 ms  →  840×
n= 65536   59.79 ms  D=652        vs (17 GB, would not run)
```

Now the caveat that decides the design. On *dissimilar* inputs Myers is much
worse than what we have, because `D` grows with the edit distance while the
numpy DP vectorises its inner loop in C:

```
two unrelated random buffers
n=  512  myers   118.2 ms (D= 920)   current  10.2 ms   0.09×
n= 1024  myers   828.9 ms (D=1824)   current  10.0 ms   0.01×
n= 2048  myers  2972.6 ms (D=3642)   current  23.9 ms   0.01×
n= 4096  myers 16080.8 ms (D=7224)   current  70.5 ms   0.00×
```

A hundred times worse. So this is **not** "swap the algorithm". It is the exact
structure the article describes: preprocessing plus heuristics plus a backstop.
Concretely:

1. Common prefix/suffix trimming — already present (`similarity.py:229-238`),
   keep it.
2. Myers forward pass with the article's **"Too Expensive"** heuristic: bail out
   once `D` exceeds a bound (the article quotes `O(N^1.5 log N)` for this).
3. On bail-out, fall through to the existing numpy DP **only if `n·m` fits a
   byte budget**; above that, return a coarse block-level diff rather than
   allocating a gigabyte.

The real workload favours Myers heavily: `crash_metadata` picks `nearest` by
4-gram Jaccard *before* aligning, so by construction the pair being aligned is
the most similar seed in the corpus. `D` is small in production and large only
in the adversarial case the backstop covers.

One heuristic from the article does **not** transfer: stripping elements unique
to one side. That needs a hash map and, more importantly, needs a large symbol
alphabet. At byte granularity there are 256 symbols and almost nothing is
unique. It would only pay off if we diffed at chunk/token granularity — which is
a separate idea, and a good one for `weizz_structural`, but not this fix.

## A2 — Crash clustering is quadratic in Levenshtein calls: 59 s at 400 crashes

`core/crash_metadata.py:380 cluster_crashes` loops
`for i in range(n): for j in range(i+1, n)` and computes a full similarity for
every pair before unioning. Measured, five well-separated synthetic crash
families:

```
n=100    3.7 s
n=200   16.1 s
n=400   59.3 s     (79,800 pairwise similarity computations)
```

Clustering quality is fine — it recovered 5/5 families at every size, so there is
no chaining defect to report. The cost is the finding. This is reachable from
`services/report.py`, so a campaign with a few hundred crashes pays a minute
every time a report is printed.

Two things to note about the implementation while it is open:

- The union-find at `:404-415` has path halving but **no union by rank** —
  `parent[px] = py` unconditionally. The astrolog page calls this out in its
  Kruskal description ("merging as well as lookup can be done in near constant
  time by using the union-find algorithm"). Minor, but free to fix.
- What this computes is single-linkage clustering at a fixed threshold, which is
  Kruskal on the complete similarity graph. See D1 for why the MST literature
  does *not* rescue it and what does.

## B1 — Floyd's sampling for `RandPool.sample`

**Source:** <https://buttondown.com/jaffray/archive/floyds-sampling-algorithm/>

`core/rand_pool.py:284 sample()` has hand-written fast paths for `k == 1` and
`k == 2` and then falls through to `self._rng.choice(n, size=k, replace=False)`
followed by a Python list comprehension. Floyd's algorithm does the whole thing
in exactly `k` draws and `O(k)` space, with no numpy round-trip.

Measured against the live `sample()`, mean over 300–2000 reps:

```
n=     256 k= 3   current  6.07 µs   floyd 0.70 µs   8.6×
n=    4096 k= 3   current  5.43 µs   floyd 0.64 µs   8.6×
n=   65536 k= 3   current  5.49 µs   floyd 0.60 µs   9.1×
n= 1048576 k= 3   current  5.63 µs   floyd 0.95 µs   5.9×
n=    4096 k= 8   current  5.97 µs   floyd 1.55 µs   3.9×
n=    4096 k=32   current  8.26 µs   floyd 4.88 µs   1.7×
```

**Be honest about what this is.** The win is constant-factor — eliminating the
numpy call and the list rebuild — not asymptotic. `Generator.choice` already
avoids materialising the population (the timings are flat in `n` out to a
million), so the earlier worry about `O(n)` memory is wrong. 4–9× on a
microsecond-scale call in a path called a few times per mutation.

A second, non-performance reason to prefer it: Floyd's draws from `_draw()`,
i.e. from the pool, whereas `_rng.choice` advances the bit generator directly
and out of band from the pool refills. Both are reproducible under `--seed`
today, but the two streams interleave, so any change in how often `sample()` is
called shifts every subsequent refill. Routing `sample` through `_draw()` puts
it on one stream.

Keep the `k == 1` and `k == 2` fast paths — Floyd's is not faster than them, and
changing them would change the byte-for-byte output of every seeded run for the
dozen mutators that call `rng.sample(range(len(x)), 2)`. Only replace the
`k >= 3` branch.

## B2 — Lexicase selection for `ga.py`

**Source:** `bed-full.pdf` = Schulte, Ruchti, Noonan, Ciarletta, Loginov,
*"Evolving Byte-Equivalent Decompilation from Big Code."* Evolutionary search
over a database of source snippets, fitness = byte-similarity of the recompiled
binary to a target binary.

The transferable part is **§III-E, lexicase selection**. In the paper: fitness
is a *vector* of independent test cases rather than a scalar sum, and selection
filters the population by best performance on a **random ordering** of those
tests until one candidate remains. The stated effect is that a candidate which
matches a region nobody else matches survives with high probability even if it
fails most other tests. The paper uses each machine-code instruction of the
target binary as a distinct test case.

`core/ga.py` is the opposite of this today. `FitnessFunction` at `:75-125`
computes a scalar:

```
fitness = w_novelty * novelty + w_diversity * diversity
```

and `select_parent` (`:313`) does rank-based tournament selection over that
scalar, with `_ensure_pool_sorted` sorting by `ind.fitness` descending.

**Why this is a good fit and not just an analogy.** The fuzzer already has the
per-test-case fitness vector the paper needs, and it is exactly the right
object: **a seed's edge set**. Each edge is an independent test the seed either
passes or fails. A seed that owns one rare edge is precisely the "matches a
region nobody else matches" case, and under the current scalar fitness it is
dominated by seeds with broad-but-redundant coverage.

We already have two hand-built approximations of what lexicase would do
structurally:

- The rare-edge bonus in `seed_picker` (`RARE_EDGE_OWNERS` / `RARE_EDGE_GAIN`,
  from the edge-distribution work).
- Speciation via MinHash LSH in `ga.py:132 Speciation`.

Both are ways of saying "do not lose the seed that is unique in some dimension."
Lexicase gets that for free from the selection rule instead of from a tuned
bonus with a calibrated constant.

**The cost, stated plainly.** Naive lexicase is `O(pop_size · n_tests)` per
selection, and `n_tests` here is the number of distinct edges — 8,189 for our
ffmpeg target. That is not viable per pick. The mitigations to evaluate before
committing: filter on a random *sample* of edges rather than all of them; or
restrict the test set to the rare edges only, since edges owned by most of the
population never discriminate anyway.

Three smaller things from the same paper, ranked:

- **Diff-targeted mutation (§III-B4).** Mutation targets are drawn from the
  regions that currently fail to match, with probability `TargetChance`,
  otherwise uniform. We have the machinery — `colorization.py` and the cmplog
  path both identify "interesting" byte regions — but nothing gates general
  havoc site selection on a per-seed failure map. Worth a look after B2.
- **Homologous crossover (§III-B3).** Parents are aligned before crossover
  points are chosen, and a point in one parent is mapped through the alignment
  to the other. Our splice/crossover picks offsets independently. Alignment is
  exactly `levenshtein_align`, so this is blocked on A1 — do not build it on a
  path that allocates a gigabyte.
- **Minimization after search (§III-F).** Delta debugging to strip bloat. We
  already have this as `tmin`. No action.

## B3 — Growing Tree / Growing Forest as the scheduler generalisation

**Sources:** <https://www.jamisbuck.org/mazes/> and
<https://www.astrolog.org/labyrnth/algrithm.htm>. (The astrolog reference
implementation is named *Daedalus*. Coincidence, but it will confuse a reader,
so say so.)

Both pages make the same structural point from different angles, and it is the
point worth taking:

> "Growing Forest is perhaps the most generalized algorithm possible, since
> setting its input parameters appropriately can duplicate Recursive
> Backtracking, Kruskal's, Prim's, and Growing Tree."

Growing Tree is one loop with one policy knob — how you pick the next cell from
the frontier list. Always newest ⇒ recursive backtracker (DFS). Always random ⇒
approximately Prim's. Always oldest ⇒ lowest "river" factor. Mostly-newest with
occasional random ⇒ high river but short solution. Jamis Buck's demo exposes
this as a literal string: `random:50, newest:30, oldest:75, middle:100`.

**The mapping.** Our corpus is a frontier and the seed picker is a policy over
it. We have nine operator schedulers plus several seed-selection arms, each a
separate class, and no shared parameterisation that says how any two of them
differ. Growing Tree says that a large family of these are one algorithm with
one knob. That is not a port; it is a way to describe the scheduler space we
already have, and it would let `bench_paired.py` sweep a *continuum* instead of
A/B-ing named implementations.

**The second idea, which is more immediately actionable: Houston's algorithm.**
Jamis Buck's page describes it as running Aldous-Broder until some minimum
number of cells have been visited, then switching to Wilson's — cheap and biased
early when nearly everything is unvisited, expensive and uniform late when the
frontier is sparse, at the cost of losing the uniformity guarantee.

That is exactly the shape of our saturation gate. The `>= 99%` gate cuts
subsumption/diversity/Wasserstein/proximity to neutral multipliers once coverage
saturates. Houston's is the same trade in the other direction and gives a
principled name for what we do ad hoc: the expensive selection signal is worth
paying for only when the frontier is sparse.

**Third, the astrolog characterisation table.** It scores every maze algorithm
on *Bias Free?*, *Uniform?*, *Memory*, *Time*, *Dead End %*, *Solution %*. We
have no equivalent table for our schedulers, and the two properties it separates
are ones we conflate:

- **Bias free** — treats all directions equally.
- **Uniform** — generates every outcome with *equal probability*.

The table notes only bias-free algorithms can be uniform, and that "no" (can
reach every maze, not equiprobably) and "never" (cannot reach some mazes at all)
are distinct failure modes. Applied to us: a scheduler that can never select
certain operator sequences is a different and worse defect than one that selects
them rarely — and that is precisely the class of bug found in the
scheduler→mutator reach audit, where Hierarchical silently dropped
runtime-registered operators. We had no vocabulary for it then. This gives one.

## C1 — Perlin noise as the smooth sibling of `fractal_voronoi`

**Source:** <https://blog.jaysmito.dev/blog/02-perlins-noise-algorithm/>

`core/mutations/fractal_voronoi.py` is a **spatial meta-operator**: it maps the
buffer to a 2D grid, partitions it into Voronoi cells, assigns each cell a
sub-operator from its root hash, and blends at cell boundaries. The cell
assignment is piecewise-constant — hard edges.

Perlin gives the same idea with a *continuous* field. Sample `noise(i/scale)`
per byte and use it to modulate mutation *intensity* rather than to select a
sub-operator: probability of touching byte `i`, or the magnitude of the
arithmetic delta. The correlation length becomes a tunable (frequency), and fBm
octaves give multi-scale bursts — long-range structure plus local jitter — from
one cheap function.

Why this fits our plumbing specifically:

- `MutatorBase` / `MutationContext` already exist and `fractal_voronoi`
  registers through `REGISTRY.register_mutator()`, so a sibling needs no new
  interface.
- The permutation table makes it deterministic from a seed, so `--seed`
  reproducibility (Hard Rule 16) holds without effort.
- It is **much cheaper than Voronoi**. `fractal_voronoi::mutate` was 36% of a
  whole campaign before the plan-cache work, and its cost was `_nearest_site`
  scanning a 5×5 neighbourhood per byte. Perlin is 4 dot products, 3 lerps and 3
  table lookups per sample, with no search.

Registry check confirms the gap: of 157 registered operators, `fractal` and
`voronoi` match only `fractal_voronoi`; `noise` and `wave` match nothing.

**Warning for whoever implements it: the article's code is wrong.** It prints

```c
int b00 = permutation[i + by0];
int b10 = permutation[j + by1];
int b01 = permutation[i + by0];
int b11 = permutation[j + by1];
```

`b00 == b01` and `b10 == b11`, so two of the four corners are duplicated and the
y-interpolation degenerates. Perlin's original is `b00 = p[i+by0]`,
`b10 = p[j+by0]`, `b01 = p[i+by1]`, `b11 = p[j+by1]`. Take the structure from
the article and the indices from the original. The prose in the article is
correct; only the snippet is transposed.

## C2 — TSP: two missing operators, confirmed absent

**Source:** <https://en.wikipedia.org/wiki/Travelling_salesman_problem>

Not the tour-construction problem — the **local-search neighbourhood over
permutations**. TSP's two canonical moves are 2-opt (reverse a contiguous
segment) and Or-opt (relocate a short segment elsewhere, preserving length).

Checked the registry directly:

```
total registered: 157
operators matching 'rev'  : []
operators matching 'move' : ['bit_transpose_8/16/32/64', 'transpose_16/32/64']
```

The `transpose_*` entries are bit-matrix transposes, unrelated.
`grep '\[::-1\]\|reversed('` over `services/operators.py` and
`core/mutations/generic.py` returns one hit and it is
`for raw in reversed(corpus)` — iteration order, not a mutation.

So: **157 operators and not one reverses a byte span, and not one relocates a
span without changing length.** We have `block_insert`, `block_delete`,
`block_duplicate`, `swap_regions`, `swap_bytes`, `byte_shuffle`,
`chunk_shuffle`, `region_shuffle`, `token_shuffle`, `block_shuffle_variable` —
insertions, deletions, swaps and shuffles, but neither of the two TSP moves.

Why it matters concretely: a reversed span is a distinct, plausible-but-wrong
encoding that shuffle essentially never produces — a uniform shuffle of `k`
bytes hits the exact reversal with probability `1/k!`, so 1 in 720 at `k = 6`.
For endian-sensitive multi-byte fields and length-prefixed records, byte-order
reversal of a field is a realistic corruption that we currently reach only by
accident. Or-opt is the length-preserving relocation that
`block_delete`+`block_insert` only approximates as a two-step composition the
bandit has to discover.

Two new operators, `span_reverse` and `span_relocate`, in the existing havoc
family. Cheap to write, easy to falsify.

## C3 — Algorithm X: take the undo discipline, not the exact cover

**Source:** <https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X>

**Reject the direct reading.** Corpus minimization is *set cover*, not *exact
cover* — seeds may overlap, and they must. `services/minimize.py:149` uses
greedy set cover, which is the right choice (ln n approximation on an NP-hard
problem) and `core/percolation.py::bootstrap_minimize_corpus` already extends it
with iterative k-rigid-core reduction for transitive redundancy. DLX on
65,536 edges × thousands of seeds would not finish, and it would be solving a
problem we do not have.

**Take the mechanism instead.** Dancing links' actual contribution is that
backtracking is `O(1)`: you *unlink* a node and restore it by relinking, rather
than copying state and restoring the copy. That maps onto a defect we have
already measured once. From the algorithm-perf audit,
`core/gradient_descent.py`'s probe loop did `bytearray(best)` plus
`bytes(candidate)` per trial — two full buffer copies to evaluate a one-byte
change — and was fixed by making the objective incremental (0.79 ms → 0.088 ms
at 8 KiB).

The dancing-links framing generalises that from one function to a policy: **any
operator that evaluates many candidate edits against one buffer should mutate in
place and undo, not copy.** The AST loop-nesting scan from that same audit
already named the remaining depth-4 candidates —
`build_tag_map_from_cmplog`, `_extract_parser_tokens`, `gradient_cmp`,
`_compute_bb_values`, `colorize_from_cmplog`. Those are where to check whether
the copy-per-candidate pattern repeats.

## C4 — LEGO thesis: critical destroy and the escalating neighbourhood

**Source:** Kollsker, *Mathematical Models and Algorithms for Optimisation of
the LEGO Construction Problem*, DTU 2020. 253 pages, almost all of it about
brick packing and static equilibrium. The transferable content is **§3.5**,
about four pages.

Large neighbourhood search: define a neighbourhood by a *destroy* method that
removes part of the solution and a *repair* method that rebuilds it. Three
specific refinements the thesis reports from the literature:

1. **Critical destroy.** Do not destroy at random — destroy the part of the
   solution that is failing. Luo et al. select the brick at the interface of two
   disconnected components, weighted toward bricks with the most distinct
   neighbouring component IDs.
2. **Escalating neighbourhood.** Destroy the critical element plus its *k-ring*
   neighbours; when repair fails 10 times, increase `k`. A widening blast radius
   driven by repeated failure.
3. **Switch to a random objective when the objective is blind.** Testuz et al.
   converge to a local optimum, find the critical area, and then switch to a
   *random* objective function specifically because the real objective did not
   capture that area — "in the hope of not reaching the same solution again."

(3) is the one to think hardest about. It is the honest version of what
`--reseed-on-stall` gestures at. The thesis's reasoning is that a stall is
evidence the objective is not measuring the thing that is stuck, so randomising
the objective is better than randomising the input. We currently reseed the
input. Different claim, testable against each other.

(2) maps onto havoc stack depth: `n_mutations` scales with `perf_score`
(`services/operators.py:4088-4093`) but not with *consecutive failure on this
seed*. An escalating radius on repeated failure is a small change with a clear
hypothesis.

The thesis also criticises Luo et al. for pairing a targeted destroy with a
random repair — "the repair method does not know how to repair it efficiently
... they rely on a guess-and-check strategy." That criticism applies to us
verbatim wherever cmplog identifies a comparison wall and havoc then mutates
without using that information.

**Do not read further than §3.5.** The structural-integrity chapters are
quadratic-programming static limit analysis for physical stability. There is
nothing there.

## C5 — DEFLATE structure-aware mutation

**Source:** <https://cefboud.com/posts/compression/> — a long walkthrough of
GZIP/DEFLATE, Snappy, LZ4 and ZSTD internals.

We have two positions on compressed streams and the gap is between them:

- `mutations/zlib.py`, `mutations/gzip.py` corrupt the *compressed* bytes in
  place. Its own docstring in `recompress.py` says this "reliably breaks the
  DEFLATE stream, so the target bails out in its decompression step and the
  parser behind the compression layer is never reached."
- `mutations/recompress.py` inflates, mutates the *plaintext*, re-deflates, and
  fixes the trailer so the payload parser runs.

Neither touches **DEFLATE's own structure**. Nothing in the tree mutates the
block-type field, the dynamic Huffman header (HLIT/HDIST/HCLEN), the code-length
code-order permutation, or back-reference distance/length pairs. Those are the
parts a decompressor's error paths live in, and they are reachable only by
producing a stream that is *structurally well-formed enough to be decoded* and
then wrong — which is exactly what neither current strategy does.

`recompress.py` already has the plumbing: bounded inflate via `decompressobj`
with explicit `max_length`, magic-byte sniffing before any work, a size cap, and
a memoized round-trip cache. A structure-aware DEFLATE mutator slots into the
same discipline.

Scope note: the article covers four schemes. We have zlib, gzip and lz4 targets;
zstd is not vendored. Do not scope zstd into this.

---

## Negative results

Recording these because "absent from the doc" and "considered and rejected" are
different states, and only one should be re-proposable.

### D1 — Expected-linear-time MST does not help the crash clustering

**Source:** <https://en.wikipedia.org/wiki/Expected_linear_time_MST_algorithm>
(Karger–Klein–Tarjan).

The connection is real: single-linkage clustering at threshold `t` is exactly the
connected components of the MST after deleting edges heavier than `t`, so A2's
`cluster_crashes` *is* an MST problem in disguise. But KKT is expected `O(m)` in
the number of **edges**, and our similarity graph is complete: `m = n(n-1)/2`.
Expected-linear in `m` is still quadratic in `n`. It buys nothing.

What actually fixes A2 is sparsifying the graph *before* building it, so the
expensive similarity is never computed for most pairs — and we already own the
sparsifier. `core/ga.py:132 Speciation` does MinHash LSH banding over seeds, and
`compute_signature` is already vectorised and banded. It is used for seed
speciation and not for crash clustering. Blocking on candidate pairs from LSH
turns `n²` similarity computations into `n²` cheap signature comparisons plus a
small number of real ones.

Second, unrelated benefit of the MST framing, worth keeping: the MST encodes
*every* threshold at once. Today `threshold=0.7` is baked in at
`crash_metadata.py:383` and changing it means recomputing everything. With the
MST, `report.py` could show the cluster hierarchy at several granularities from
one computation. That is a genuine capability gain — it just is not a KKT
argument.

### D2 — DMMSY-SSSP is inapplicable, and the two links are one artifact

**Sources:** <https://github.com/danalec/DMMSY-SSSP> and
<https://arxiv.org/html/2504.17033v2>. These are the same thing: the repo's
README cites arXiv:2504.17033 as its reference. Duan–Mao–Mao–Shu–Yin, "Breaking
the Sorting Barrier for Directed Single-Source Shortest Paths," STOC 2025;
the repo is an experimental C99 implementation claiming to break the
`O(m + n log n)` barrier via recursive subproblem decomposition instead of a
global priority queue.

Three independent reasons it does not apply to `core/distance.py`:

1. **Our graphs are unweighted.** `distance.py:12-13` and `:522-564` compute the
   AFLGo harmonic-mean call-graph distance with *unweighted reverse BFS*. BFS on
   an unweighted graph is already `O(m + n)`. There is no sorting barrier to
   break because there is no priority queue — you cannot beat BFS with an SSSP
   algorithm. `grep -i dijkstra` over `src/` returns zero hits, and that is
   correct, not an omission.
2. **Our graphs are too small.** The README states the speedup materialises at
   250k–1M+ nodes. Our call graphs and ICFGs are per-target function and
   basic-block graphs, orders of magnitude below that.
3. **The headline number is not an algorithmic comparison.** The README itself
   attributes the 20,000× to a combination of the algorithm, AVX-512
   auto-vectorization of DMMSY's loops, and 96 MB of V-Cache on an AMD 7950X3D,
   measured on sparse tree-like graphs. That is a hardware-and-compiler result
   wearing an algorithm's name. Same class of number as the PowerFuzz figures
   that `docs/port-backlog.md` already flags as unusable.

If the distance channel ever moves to *weighted* edges — profile-weighted call
graphs, or edge costs from execution time — this becomes worth re-reading. Until
then it is a no.

### R1 — Evolutionary algorithm (Wikipedia)

Overview article. Everything in it is already in the tree: `core/ga.py`
(population, tournament selection, crossover, speciation), `core/qea.py`
(quantum-inspired variant), and the CMA-ES / MOpt / replicator schedulers. The
specific technique worth having from the EA literature is lexicase, and that
comes from B2 with a concrete mechanism attached. No action.

### R2 — Wilson's algorithm applet (cruzgodar)

An interactive visualisation of loop-erased random walk. The algorithm is
described in prose on both maze pages already surveyed in B3, with the
comparative data (uniform, `N²` memory, ~5× faster than Aldous-Broder). The
applet adds intuition, not information. Folded into B3.

### R3 — keon/algorithms and TheAlgorithms/Python

Both are teaching-oriented reference collections of pure-Python algorithm
implementations. Neither is a candidate dependency: they are optimised for
readability, they carry no performance contract, and everything in them that we
need we either already have or would need to write against our own `RandPool`
and numpy conventions anyway (see the `--seed` reasoning in the
sampling-identities work — reproducibility, not speed, is the binding
constraint).

Their legitimate use is as a **checklist**: a way to notice a named technique we
have not considered, which is how C2 surfaced. Do not vendor them, do not cite
them as authority for an implementation, and do not open a port item against
"adopt these repos". If a specific entry looks useful, it gets its own item with
its own measurement, like everything else in `docs/port-backlog.md`.

---

## Suggested sequencing (unchanged rationale)

A1 first and alone. It is the only item that fixes a crash rather than improving
a number, it is measured, and B2's homologous crossover depends on it. A2 next,
since it is the same module and the LSH sparsifier already exists.

B1 and C2 are both small, independent, and easy to falsify — good filler.

B2 and B3 are the two items with real design content and neither should be
started without deciding the open questions above (lexicase test-set size for
B2; whether the Growing Tree parameterisation is a refactor or just a
description for B3).

C1, C3, C4, C5 are genuine but none of them is blocking anything.

---

## Implementation Plan — all new features opt-in gated

**Hard rule (from AGENTS.md + this survey):** every addition is behind an
explicit opt-in flag (CLI long option + matching config/env key where
applicable). Default behaviour of the fuzzer must remain byte-for-byte identical
to the tree at the commit this plan is based on. No new operator, scheduler
arm, selection policy, or distance path is active unless the user passes the
flag. Registration still happens so the feature appears in `--help` and
`bandit_stats`, but the hot path stays cold until enabled.

Conventions that every item below must obey:

- Surgical diffs only; match existing naming, error handling, logging, and
  comment style of the nearest sibling module.
- New mutators go through `REGISTRY.register_mutator()` (or the structured
  equivalent) and appear under an existing category or a new one that is itself
  gated.
- New scheduler policies register in `_OPERATOR_STRATEGY_NAMES` / the Elo /
  hierarchical tables exactly like the current ones.
- All random draws go through `RandPool` / `_draw()` so `--seed` remains
  deterministic.
- Tests: unit tests for the new path, plus a paired A/B (or multi-arm) entry in
  `tools/bench_paired.py` or the relevant sweep script so the feature can be
  falsified.
- Documentation: one paragraph in CHANGELOG under "Experimental / opt-in" and
  a short note in the relevant README section.

### Phase 0 — Safety / crash fixes (A1, A2) — still opt-in for the new path

These are correctness/robustness fixes, but the *new algorithm* must still be
selectable so we can A/B against the old path and roll back instantly.

#### A1 — Linear-space Myers + backstop for Levenshtein

- **Flag:** `--diff-myers` (bool, default false).  
  Env: `FUZZER_DIFF_MYERS=1`.  
  When false: existing numpy DP path unchanged (including the 512-byte guard in
  `adapters/filesystem.py`).
- **Modules:** `core/similarity.py` (new `myers_diff` / `levenshtein_align_myers`),
  call sites in `core/root_cause.py` and `core/crash_metadata.py` become
  conditional.
- **Behaviour when enabled:**
  1. Common prefix/suffix trim (already present).
  2. Myers forward pass; abort if `D` exceeds a tunable bound
     (`--diff-myers-max-d`, default derived from `O(N^1.5 log N)` heuristic).
  3. On abort or when `n·m` exceeds a byte budget (`--diff-myers-max-bytes`,
     default 64 MiB): fall back to the existing numpy DP *only if it fits*, else
     return a coarse block-level diff (documented contract change for
     `root_cause`).
- **Tests:** unit tests on the measured sizes from the survey; regression that
  seeded runs with the flag off produce identical edit scripts.
- **Risk:** `root_cause` currently consumes positional edit scripts. The coarse
  fallback must either emit a compatible script or be rejected by that path.

#### A2 — LSH-sparsified crash clustering

- **Flag:** `--crash-cluster-lsh` (bool, default false).  
  Companion: `--crash-cluster-lsh-threshold` (float, default 0.7, same as today).
- **Modules:** `core/crash_metadata.py::cluster_crashes`. Re-use the existing
  MinHash LSH banding from `core/ga.py` Speciation (do not re-implement).
- **Behaviour when enabled:** candidate pairs come from LSH buckets; only those
  pairs pay the full similarity cost. Union-find gains rank (free correctness
  fix, always on once the path is taken). Optional future: return the full MST
  so `report.py` can show hierarchy at multiple thresholds.
- **Tests:** recover the same 5 synthetic families; timing assertion that n=400
  drops well below 59 s.

### Phase 1 — Cheap, independent wins (B1, C2)

#### B1 — Floyd sampling in RandPool

- **Flag:** `--rand-floyd-sample` (bool, default false).  
  Only affects the `k >= 3` branch of `RandPool.sample`; k=1 and k=2 fast paths
  stay exactly as they are so seeded runs remain identical when the flag is off.
- **Module:** `core/rand_pool.py`.
- **Behaviour:** pure-Python Floyd (exactly k draws from `_draw()`). No numpy
  round-trip.
- **Tests:** statistical equivalence of the produced samples under the same
  seed; micro-benchmark confirming the 4–9× range on the sizes in the survey.

#### C2 — TSP neighbourhood operators

- **Flags (two independent operators):**
  - `--op-span-reverse` (bool, default false) → registers `span_reverse`.
  - `--op-span-relocate` (bool, default false) → registers `span_relocate`.
- **Module:** `core/mutations/generic.py` (or the havoc family) + registration
  in the operator registry under the existing `block` / `byte` category.
- **Behaviour:** classic 2-opt (contiguous reverse) and Or-opt (length-preserving
  relocate of a short span). Both respect the usual length / region constraints
  of the surrounding mutators.
- **Tests:** unit tests that the operators are reachable only when the flags are
  set; seeded determinism; a tiny corpus A/B that the new operators can discover
  a known endian-sensitive crash that pure shuffle misses.

### Phase 2 — Design-heavy items (B2, B3) — require answering open questions first

#### B2 — Lexicase selection for GA

- **Flag:** `--ga-lexicase` (bool, default false).  
  Tunables (only meaningful when the flag is on):
  - `--ga-lexicase-tests {rare,sample,all}` (default `rare`).
  - `--ga-lexicase-sample-size N` (when `sample`).
- **Module:** `core/ga.py` — new selection path beside the existing rank-based
  tournament; the scalar `FitnessFunction` remains the default.
- **Behaviour:** filter population by successive random (or rare-edge) tests
  until one individual remains. Edge sets are already available.
- **Open question gate:** do not land until an A/B design exists that can return
  “no difference” vs the current rare-edge bonus. Prefer the `rare` test set
  first because it is cheapest and closest to existing machinery.
- **Tests:** population diversity metrics; edge-ownership entropy; paired
  campaign on a target with known rare edges.

#### B3 — Growing-Tree / Houston parameterisation of seed & operator selection

- **Flag:** `--scheduler-growing-tree` (bool, default false).  
  Policy string (Jamis-style): `--growing-tree-policy "random:50,newest:30,oldest:20"`  
  Houston hybrid: `--houston-switch-frac 0.3` (run cheap biased policy until
  that fraction of the frontier has been visited, then switch to uniform).
- **Modules:** new thin wrapper around existing seed_picker / operator
  schedulers; does *not* replace the nine named schedulers unless the paper
  exercise shows real subsumption.
- **Open question gate:** answer on paper whether the parameterisation actually
  collapses any two existing schedulers before writing code. If it is only a
  description, document it and stop.
- **Tests:** continuum sweep in `bench_paired.py`; bias/uniformity diagnostics
  inspired by the astrolog table.

### Phase 3 — Moderate / speculative (C1, C3, C4, C5)

#### C1 — Perlin-noise intensity field (sibling of fractal_voronoi)

- **Flag:** `--op-perlin` (bool, default false) → registers a new mutator
  `perlin_intensity` (or similar) under the spatial/meta category.
- **Module:** new file `core/mutations/perlin.py` (or extend the fractal family).
- **Behaviour:** deterministic Perlin (correct indices — do not copy the buggy
  snippet from the article) samples a smooth field used as mutation probability
  or arithmetic magnitude. Frequency / octaves exposed as
  `--perlin-freq` / `--perlin-octaves`.
- **Tests:** visual / statistical smoothness; cost comparison vs Voronoi; seeded
  reproducibility.

#### C3 — In-place mutate + undo discipline (dancing-links style)

- **Flag:** `--mutate-in-place-undo` (bool, default false).  
  Applies to the candidate-evaluation loops identified in the survey
  (`gradient_cmp`, colourisation, tag-map builders, etc.).
- **Behaviour:** when enabled, those loops mutate the buffer in place and restore
  by relinking / memcpy of the changed span only. Default remains the current
  copy-per-candidate path.
- **Tests:** bit-identical results to the copy path; measured allocation drop.

#### C4 — Critical-destroy / escalating neighbourhood + objective randomisation on stall

- **Flags:**
  - `--stall-critical-destroy` (bool) — destroy the region indicated by the
    current comparison wall / colourisation map rather than a random span.
  - `--stall-escalate-k` (int, default 0) — on repeated failure, widen the
    destroy radius by this many rings.
  - `--stall-random-objective` (bool) — on stall, switch the fitness / reward
    signal to a random objective for a configurable number of iterations
    instead of (or in addition to) `--reseed-on-stall`.
- **Open question:** A/B the random-objective claim against the existing reseed
  claim on the same stall detector.
- **Modules:** stall handling in the main campaign loop + operator selection.

#### C5 — Structure-aware DEFLATE mutation

- **Flag:** `--op-deflate-structure` (bool, default false) → registers a new
  mutator that touches block type, dynamic Huffman header fields, code-length
  permutation, and back-reference (distance, length) pairs while keeping the
  stream decodable enough to reach the error paths.
- **Module:** extend `mutations/recompress.py` / new sibling; stay within the
  existing inflate size caps and memoisation.
- **Scope:** zlib / gzip / lz4 only; do not pull zstd.
- **Tests:** streams that survive the first-stage decoder but fail deeper;
  no size-budget violations.

### Rejected items stay rejected

D1, D2, R1–R3 remain negative results. Do not open implementation work against
them. If the distance channel ever becomes weighted, revisit D2 under a new
handover; until then it is closed.

### Global CLI / config surface

All flags above must appear in:

- `fuzzer-tool fuzz --help` under an “Experimental / opt-in” section.
- The JSON schema / config file loader (if present) with the same names.
- `docs/CHANGELOG.md` and a short entry in `docs/port-backlog.md` pointing back
  to this handover.

No flag may change default behaviour. A campaign started with zero of these
flags must produce the same edge map, crash set, and seed ranking (within RNG
noise) as the baseline tree.

### Validation requirements before merge

For every feature that lands:

1. Unit tests green under both the flag-off and flag-on paths.
2. At least one paired A/B (or multi-arm) run on a real target (png / jpeg /
   grep / ffmpeg subset) showing either a measurable win or an explicit “no
   difference” result that is recorded in the handover or VALIDATION_LOG.
3. `tools/bench_paired.py` (or the relevant sweep) entry so the experiment is
   reproducible.
4. Impactguard / pre-commit clean; no new warnings.

---

## Open questions (still open — gate the design-heavy items)

- **B2:** what is the right test set for lexicase? All edges (8,189 on ffmpeg,
  too slow), a random sample per selection, or rare edges only? The third is
  cheapest and closest to what the rare-edge bonus already approximates — but if
  it is equivalent to that bonus, lexicase buys nothing and should be rejected.
  Design the A/B so it can return "no difference".
- **A1:** what is the right bail-out bound for the Too Expensive heuristic, and
  what does the coarse fallback return? A block-level diff has a different
  contract than an edit script, and `root_cause` consumes the script positionally.
- **B3:** does the Growing Tree parameterisation actually subsume any two of our
  schedulers, or only look like it does? This should be answered on paper before
  any code moves.
- **C4:** randomise the objective on stall (thesis) vs. reseed the input
  (current `--reseed-on-stall`). These are different claims and can be tested
  against each other on the same stall condition.

---

*Updated 2026-09-06 — added full opt-in-gated implementation plan for every
accepted item from the seventeen-source survey. All new behaviour is behind
explicit flags; baseline remains unchanged.*
