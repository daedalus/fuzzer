# Handover — algorithm-catalogue survey: `keon/algorithms` and `TheAlgorithms/Python`

**Status: ANALYSIS ONLY.** Nothing here is implemented. Every claim was checked
against live source at `85fbf58` and every number was measured in a container on
that tree. Rebased onto `7cdf0ea` and re-verified there: the four FormatFuzzer
commits (`e445fdc`..`7cdf0ea`) touch no file this survey measures, and every line
anchor cited below still resolves. The one number they changed is the operator
count in K7 (see there).

## 0. Why this pass exists when R3 already rejected these repos

`docs/handover/handover_seventeen_source_survey_2026-09-06.md` §R3 already
settled the *dependency* question for both repos, and that verdict stands
unchanged: they are teaching collections in pure Python, optimised for
legibility, with no performance contract, and anything we needed would have to
be rewritten against `RandPool` and our numpy conventions anyway because the
binding constraint is `--seed` reproducibility (Hard Rule 16), not speed. **Do
not vendor them. Do not cite them as authority for an implementation. Do not
open a port item against "adopt these repos."**

R3 also said what they *are* good for: a checklist — a way to notice a named
technique we never considered. C2 (`span_reverse` / `span_relocate`, now
shipped as `12a49ca`) came out of exactly that. This document is that checklist
actually executed, one entry at a time, against the tree.

**Inventory.** 382 modules in `keon/algorithms`, 994 in `TheAlgorithms/Python`
outside `project_euler/` (a further 309 inside it, all skipped — competition
puzzles, not techniques). Of those ~1,376 modules, the overwhelming majority are
either already in the tree, structurally inapplicable, or textbook exercises
with no analogue here. **Eight** map onto something real. **Two of the eight are
bugs in live code**, not enhancements.

---

## 1. Ranking

| # | Catalogue entry | Lands on | Evidence | Verdict |
|---|---|---|---|---|
| K1 | `TAP strings/aho_corasick` | `colorizer.colorize_from_cmplog`, `weizz_tags.build_tag_map_from_cmplog` | measured 3.6×–10× | **Do it**, dispatch by affordability |
| K2 | `keon streaming/misra_gries` | `core/shapley.py::_prune_edges` | measured 5.3× credit distortion | **Bug.** Fix first |
| K3 | `TAP other/lru_cache`, `lfu_cache` | 9 wholesale-`clear()` sites | measured; one correctness aside | Real, cheap |
| K4 | `keon streaming/misra_gries` (again) | 4 ad-hoc eviction policies | argued, not measured | Vocabulary, then policy |
| K5 | `TAP machine_learning/frequent_pattern_growth`, `apriori_algorithm` | `edge_cooccurrence` | design only | Genuine, unproven |
| K6 | `TAP dynamic_programming/smith_waterman` | homologous crossover (B2b) | argued | Unblocks B2b differently from A1 |
| K7 | `TAP data_compression/burrows_wheeler` | new mutation domain | absent, verified | Low priority, genuine |
| K8 | — | `_deterministic_mutation_stream` | measured 2.3× | **Bug-adjacent.** Found while checking Gray code, which does *not* apply |

K8 is listed last because no catalogue entry produced it; it surfaced while
falsifying the Gray-code idea (see §10). It is the largest measured number in
the document, so read the ordering as "traceable to the catalogue" rather than
"most important."

---

## 2. K1 — Aho-Corasick for the cmplog operand scan

### What the tree does now

`core/colorizer.py:161 CmplogColorizer.colorize_from_cmplog` walks every cmplog
operand independently:

```
for op_a, op_b in cmplog_pairs:
    for token in (op_a, op_b):
        pos = 0
        while pos <= n - len(token):
            idx = input_data.find(token, pos)
            ...
            pos = idx + 1
```

That is `O(P · n)` with `P` = 2 × the pair count. `str.find` is memchr-backed,
so each scan is fast in C — but there are `P` of them and each traverses the
whole buffer.

`CMPLOG_PAIRS_MAX = 5_000` (`core/cmplog.py:159`), so `P` can reach **10,000
tokens** against a seed up to `max_len`.

`core/weizz_tags.py:580 build_tag_map_from_cmplog` has the same shape for the
same operand pool — step 1 of its own docstring is "locate occurrences of either
operand inside *data*." One automaton would serve both call sites.

### Where the cost actually is

Decomposed at n = 64 KiB, P = 512 pairs (1,024 tokens), synthetic buffer with
half the tokens drawn from the buffer so they really match:

| Component | Time |
|---|---|
| `find()` scans alone | 32.02 ms |
| scans + the Python span-marking loop | 31.78 ms |
| `color_mask()` genexpr | 2.05 ms |
| the same mask via numpy | 0.13 ms |

**The scans are the whole cost.** Only 854 match spans exist, so the marking
loop is free. This matters because the obvious reflex — "vectorise the inner
loop" — buys nothing here. `color_mask()` is a real 16× but on a 2 ms term.

### Measured, Aho-Corasick against the live shape

Automaton built as in `TheAlgorithms/Python/strings/aho_corasick.py` (goto trie,
BFS failure links, suffix-merged output sets), pure Python, no numpy:

| n | P (pairs) | current `find` loop | AC build | AC search | speedup (search) |
|---|---|---|---|---|---|
| 4,096 | 64 | 0.24 ms | 0.62 ms | 0.39 ms | **0.61×** |
| 4,096 | 512 | 1.90 ms | 4.29 ms | 0.73 ms | 2.61× |
| 16,384 | 512 | 6.36 ms | 4.01 ms | 2.40 ms | 2.65× |
| 65,536 | 512 | 30.96 ms | 4.05 ms | 8.51 ms | 3.64× |
| 16,384 | 2,048 | 32.65 ms | 24.53 ms | 3.42 ms | 9.55× |
| 65,536 | 2,048 | 123.25 ms | 16.72 ms | 12.26 ms | **10.05×** |

Span sets verified identical to the current implementation at every size.

### The caveat that decides the design

**At small `P` the automaton loses**, and loses for a structural reason, not a
tuning one: the AC search is a Python loop over every byte of the buffer
(~133 ns/byte measured), while `find` is a C scan. Below roughly 128–256 tokens
the C loop wins outright — 0.61× at P = 64.

So this is the same dispatch shape that `0315b3c` settled for A1: **not a flag,
not a fixed threshold on `n`, but a decision on the quantity that drives the
cost.** Here that quantity is the token count, because AC's cost is `O(n)`
independent of `P` and the current loop's cost is `O(P · n)`. Below the
crossover keep `find`; above it, build the automaton once per pair-pool version
and reuse it.

The build amortises for free: the automaton depends only on the operand set, and
that set is a per-target pool that turns over slowly. Keying the cached
automaton by pair-pool identity gives steady-state cost = the search column
alone.

### Two asides found in the same code

1. **Duplicate tokens are re-scanned.** At P = 512 the 1,024 tokens contain
   1,022 distinct values, and the baseline scanned all 1,024 — it emitted 854
   spans where the deduplicated automaton emitted 852, the difference being
   duplicate spans from duplicate tokens. Deduplicating the token set is
   independent of AC and correct on its own.

2. **`services/operators.py:943` keys the colorization cache on
   `id(cmplog_pairs)`.** CPython reuses an `id` after the object is freed, so a
   rebuilt pair list can land on a freed address and read a mask computed for a
   *different* operand set. The neighbouring memo at `operators.py:614` mixes in
   `len(cmplog_pairs)`; this one does not. Not observed in the wild, and the
   window is narrow, but it is free to close.

**Open question before implementing:** `build_tag_map_from_cmplog` processes
pairs in a deliberate order (`_pair_key`: shorter, more specific operands claim
bytes first). An automaton returns all matches in one pass with no such order.
The port has to collect matches first and *then* apply them in the existing
sorted order — verify byte-for-byte tag equivalence, do not assume it.

---

## 3. K2 — `shapley._prune_edges` drops the low half of the edge-ID space

This is a bug, and it is the reason `keon/algorithms/streaming/misra_gries.py`
earned a place in this document.

`core/shapley.py:74`:

```python
def _prune_edges(self):
    """Drop oldest half of tracked edges to bound memory."""
    edges = sorted(self._all_edges)
    drop = edges[: len(edges) // 2]
```

`_all_edges` is a `set[int]` of edge IDs. `sorted()` orders it **numerically**,
so `drop` is the numerically smallest half — the docstring's "oldest" is not
what the code does, and there is no insertion order to recover because a `set`
does not carry one.

Edge IDs are `prev_loc ^ cur_loc`. That is a hash, not a clock and not a metric
— the same fact already recorded in the edge-distribution work of Aug 2026 for
why Wasserstein over the edge index measured nothing. Low ID means nothing about
age, importance, or anything else.

### Measured

Two operators of **identical** productivity, each discovering 4 edges per
execution over 40,000 executions, differing only in which half of a 65,536-entry
map their edges land in:

```
tracked edges: 7080   low-half=1125   high-half=5955
shapley values: {'op_hi': 0.841, 'op_lo': 0.159}
```

**5.3× credit distortion between two operators that did exactly the same work.**
And it compounds: low-ID edges are evicted, rediscovered, re-added and evicted
again on every prune, so `op_lo` can never accumulate a stable footprint.

### Blast radius — stated honestly

`shapley_values()` is read only by `services/stats.py:362` and `:953`, both
display. It is gated behind `--shapley` (`cli/commands.py:1857`) and **does not
feed any scheduling decision.** So this is a wrong number in a report, not a
wrong fuzzing campaign. That is the honest scope and it should be in the commit
message; it is still worth fixing, because the number is presented as operator
attribution and it is not one.

### What Misra-Gries actually contributes

Misra-Gries is the principled answer to "bound a frequency map over a stream."
Its guarantee is the point: with `k` counters, **any item whose true frequency
exceeds `n/(k+1)` is guaranteed to still be in the summary**, and every retained
count underestimates by at most `n/(k+1)`. It achieves this by decrementing
*all* counters when a new item arrives at capacity, rather than deleting a
subset — so a frequent item that momentarily falls out is not permanently
disadvantaged the way it is under any drop-a-subset policy.

We already track exactly the right quantity: `self._edge_total[edge]` is the
per-edge occurrence count. The eviction that ignores it is the defect.

Two candidate fixes, both correct, pick one:

- **Minimal:** evict by `_edge_total` ascending instead of by edge ID. Fixes the
  bias, no new concept, no guarantee either.
- **Principled:** replace the map with Misra-Gries over `SHAPLEY_EDGES_MAX`
  counters. Gains the frequency guarantee and removes the halving cliff.

`TheAlgorithms/Python/other/majority_vote_algorithm.py` is the `k = 1` case of
the same algorithm, useful only as the shortest correct reference.

**Falsification for whichever is chosen:** the simulation above is the test. Two
operators, equal productivity, disjoint ID ranges; assert the Shapley values are
within noise of each other. It fails hard against today's code (0.84 / 0.16).

---

## 4. K3 — nine caches that flush entirely on overflow

`TheAlgorithms/Python/other/` carries `lru_cache.py`, `lfu_cache.py` and
`least_recently_used.py`. The technique is not news; the finding is that the
tree has nine places that reach for the crudest possible policy instead.

Sites (`if len(x) > CAP: x.clear()`):

| File:line | Structure | Cap |
|---|---|---|
| `services/fuzzer.py:4904` | cache | 512 |
| `services/parallel.py:289` | `seen_names` | `_SYNC_SEEN_MAX` |
| `services/operators.py:955` | colorization mask cache | 256 |
| `services/operators.py:3720` | `_region_cache` | — |
| `core/rq_encodings.py:399` | `_cache` | `_RQ_MUTATIONS_CACHE_MAX` |
| `core/mutations/fractal_voronoi.py:225` | `_plan_cache` | — |
| `core/corpus_compression.py:96` | `_seed_ratios` | — |
| `core/path_constraints.py:270` | `_attempted` | — |
| `adapters/filesystem.py:553,626,686,724` | `seen_hashes` | 200,000 |

For the true caches (`fractal_voronoi._plan_cache`, `rq_encodings._cache`,
`_region_cache`) a wholesale clear discards a working set that is about to be
rebuilt immediately. `functools.lru_cache` already exists in stdlib and the
fractal-voronoi plan cache was *deliberately* refactored to per-instance dicts
in commit 7 of `fuzzer-algorithm-perf-fixes` to escape `B019` — an LRU with a
bound is the shape that work was reaching for.

**The `filesystem.py` sites are different and worth reading carefully**, because
they are dedup sets, not caches, and they interact with the bloom filter:

```python
if bloom is not None:
    if not bloom.query(h):   bloom.add(h)
    elif h in seen_hashes:   return False
    else:                    bloom.add(h)
...
seen_hashes.add(h)
if len(seen_hashes) > SEEN_HASHES_MAX: seen_hashes.clear()
```

After a clear, the bloom still answers "seen" for every prior hash while
`seen_hashes` answers "no" — so the `elif` fails, control falls to the `else`,
and **every previously-seen seed is admitted exactly once more.** It is a
bounded re-admission, not an unbounded leak, and at a 200,000 cap it is rare.
But it is a silent correctness effect of a memory policy, and the fix (drop
half by insertion order, keeping a `dict` for order) costs nothing.

---

## 5. K4 — Misra-Gries as shared vocabulary for four eviction policies

Beyond the Shapley bug, the tree has four independent answers to the same
question, with four different shapes and no shared name:

| Site | Policy |
|---|---|
| `core/length_mi.py:51 _prune_lengths` | keep top 50% by total edge count |
| `core/length_mi.py:45` | per-length: keep top half of edges by count |
| `core/shapley.py:74 _prune_edges` | drop numerically smallest half (K2) |
| `core/mi.py:156 _evict_least_observed` | drop the single least-observed position |
| `services/fuzzer.py:3629` | dictionary: `dyn_cap`, keep `dyn_cap // 2` |

Three of the five are frequency-aware, which is the right instinct. None of them
has Misra-Gries' guarantee, and the halving ones share a specific failure: an
evicted item restarts at count zero, so an item that is genuinely frequent but
was briefly below the median gets evicted, restarts, and is evicted again. MG's
uniform decrement is precisely the fix for that.

This is a vocabulary item before it is a code item. The pattern to follow is the
one `docs/port-backlog.md` records for the maze-algorithm characterisation
table: naming a property we were confusing is worth more than porting an
implementation.

**Do not batch this with K2.** K2 is a bug with a falsifying test; K4 is a
refactor across five sites that changes behaviour in ways nothing currently
measures.

---

## 6. K5 — frequent itemset mining over edge co-occurrence

`TheAlgorithms/Python/machine_learning/frequent_pattern_growth.py` and
`apriori_algorithm.py`.

`EdgeTracker.edge_cooccurrence` already computes pairwise co-occurrence and, per
the Aug 2026 audit, reaches only `report.py`/`stats.py` — display, never
scheduling. Pairwise is the `k = 2` slice of a more useful question: **which
*sets* of edges always fire together?** A maximal frequent itemset over the
per-seed edge sets is a basic block group that no input has ever separated,
which is a statement about the target's structure, not about our corpus.

Two uses that would follow if the sets were computed:

- **Minimisation.** `services/minimize.py:149` greedy set cover treats every
  edge as an independent element. Edges that provably co-occur across the entire
  corpus are one element, and collapsing them shrinks the universe before the
  greedy runs.
- **Rarity.** `RARE_EDGE_OWNERS` counts owners per edge. An edge that is rare
  *only because its whole itemset is rare* is not independently informative;
  today it gets the full `RARE_EDGE_GAIN` bonus, and the fix in `edge-
  distribution-signal` (bonus applied once, log2) bounds the magnitude but does
  not address the redundancy.

**Honest status: design only, nothing measured.** Two questions gate it and both
should be answered on paper first. (a) FP-growth is `O(transactions × items)` to
build the tree and our transactions are per-seed edge sets of up to 8,189
elements on ffmpeg — feasibility is not established. (b) If the maximal itemsets
turn out to be almost entirely "the basic blocks inside one function," this
reproduces the ICFG we already build in `core/icfg.py` from the binary, more
expensively and less accurately, and should be rejected. **The A/B must be able
to return "this is just the ICFG."**

---

## 7. K6 — Smith-Waterman for homologous crossover

`TheAlgorithms/Python/dynamic_programming/smith_waterman.py`.

B2(b) in the seventeen-source survey — homologous crossover from the
decompilation paper (align the parents, map a cut point in one to the other) —
was recorded as **blocked by A1**, because the only alignment we had allocated a
gigabyte. A1 is now shipped (`93a5fa7`, `0315b3c`): `levenshtein_align`
dispatches DP → bounded Myers → coarse block diff by affordability, and never
allocates unboundedly.

So B2(b) is unblocked. But the catalogue raises a design question that the
original note did not: **global alignment may be the wrong tool.** Two seeds in
a corpus typically share *regions* — a header, a chunk, a table — inside
otherwise unrelated content. Smith-Waterman finds the best-scoring *local*
alignments and ignores the rest, which is exactly the "these two parents share
this span" relation crossover needs. Needleman-Wunsch/Levenshtein forces an
end-to-end correspondence and will happily align noise to noise to satisfy it.

Cost note before anyone starts: Smith-Waterman as written in the catalogue is
the same `O(n·m)` table A1 exists to avoid. Any port inherits A1's budget
discipline — affordability dispatch, byte budget, coarse fallback — or it
reintroduces the crash A1 fixed. Do not port the table; port the *scoring and
traceback rule* onto the dispatch we already have.

---

## 8. K7 — Burrows-Wheeler as a mutation domain

`TheAlgorithms/Python/data_compression/burrows_wheeler.py`. Verified absent:
`grep -rli "burrows\|bwt" src/ tools/ tests/ docs/` returns zero.

The idea is not compression. BWT is a **reversible permutation** that groups
bytes by their following context, and the operator plumbing for
"transform → mutate → invert" already exists in `core/mutations/recompress.py`
(inflate, mutate the plaintext, re-deflate, fix the trailer). A BWT operator
would be the same shape with a cheaper and always-invertible transform:
a byte edit in BWT space is a context-correlated edit in the original,
scattered across every position sharing that context. Nothing in the **163**
live registered operators produces that: our edits are positional (spans,
blocks, chunks) or value-based (arith, interesting, dictionary), never
context-grouped. That includes the four `ff_*` mutators added by `e445fdc`,
which are template-driven format mutators.

Counting note for anyone re-checking this: `REGISTRY.names()` returns 163 while
the static `_CATEGORIES` table returns 156. The difference is exactly the seven
self-registering `MutatorBase` operators — `ff_png`, `ff_zip`, `ff_isobmff`,
`ff_jpeg`, `fractal_voronoi`, `weizz_chunk_mutate`, `weizz_field_mutate` — and
`_CATEGORIES` is a strict subset of the registry (the difference the other way
is empty). That is the documented design, not drift; `REGISTRY.names()` is the
number to quote.

Genuine and cheap, but it does not unblock anything and there is no evidence it
finds anything. Rank it below everything above.

---

## 9. K8 — `_deterministic_mutation_stream` copies the seed twice per mutant

Not from the catalogue. Found while falsifying the Gray-code idea (§10), kept
because it is the largest measured number here.

`services/operators.py:270`, every pass, without exception:

```python
mutant = bytearray(data)      # full copy
mutant[byte_idx] = ...        # one byte
yield bytes(mutant)           # full copy again
```

The docstring is explicit that this is intentional — "each yielded mutant is a
fresh bytearray" — and the *contract* is right: the consumer gets an immutable
`bytes` one edit from the base. The `bytearray(data)` allocation is not.

The schedule costs **33 mutants per byte** (8 bitflip + 1 byteflip + 16 arith +
8 interesting, confirmed by counting the generator), so total copying is
`2 × 33 × n²` bytes for a seed of `n`.

Measured, whole-stream, uncapped:

| n | mutants | total | per mutant |
|---|---|---|---|
| 256 | 8,448 | 2.0 ms | 0.24 µs |
| 1,024 | 33,792 | 11.4 ms | 0.34 µs |
| 4,096 | 135,168 | 56.5 ms | 0.42 µs |
| 16,384 | 540,672 | 618.2 ms | 1.14 µs |

16× the seed size costs 54× the time. A prototype holding one persistent
scratch `bytearray`, editing one byte, yielding `bytes(scratch)`, and restoring,
ran at **0.49 µs/mutant at n = 16,384 against 1.14 — 2.3×** with the identical
contract. It is a constant factor, not an asymptotic win: `bytes(scratch)` is
still `O(n)` per mutant and that copy is required by the contract. Removing it
too would mean handing the consumer a `memoryview` into a buffer that changes
under it, which is a different and much more invasive change.

This is the third instance of one policy, and it should be written down as a
policy rather than fixed a third time in isolation: **an operator evaluating
many candidate edits against a buffer should mutate in place and undo, not
copy.** The first instance was `core/gradient_descent.py` (0.79 ms → 0.088 ms at
8 KiB, shipped). The second was the dancing-links framing in C3 of the
seventeen-source survey, which named the pattern and listed where to look next.
This is the largest remaining site on that list.

**Caveat before implementing:** the prototype above yielded a different mutant
count (25/byte vs 33) because I reconstructed the arithmetic pass rather than
copying it. Equivalence must be proved against the live generator
mutant-for-mutant, not just measured — the correct oracle is
`list(old_stream(d)) == list(new_stream(d))` over a sweep of seeds and caps,
including caps that land mid-pass, since `MAX_DET_MUTATIONS` truncation
interacts with buffer restore.

---

## 10. Rejected — with the reason, so these do not come back

The value of a catalogue survey is mostly here. "Absent from the tree" and
"considered and rejected" are different states and only one should be
re-proposable.

### Rejected because we already have it

| Catalogue entry | Where it already lives |
|---|---|
| `other/sliding_window_maximum`, `heap/sliding_window_max` | `seed_picker.py:1391` — an `O(N)` rolling-max sweep, already the monotonic-deque result |
| `other/fischer_yates_shuffle` | `byte_shuffle_bytes` (`operators.py:1476`), `exhaustive_pool.py:266` |
| `graphs/page_rank` | `core/lineage.py:302 pagerank_credit` |
| reservoir sampling (Algorithm R) | `core/grammar.py:533 SubtreePopulation` |
| `hashes/adler32`, `hashes/fletcher16` | integer-modulus checksum recovery, already ported |
| `strings/levenshtein_distance`, `edit_distance`, `damerau_levenshtein_distance` | `core/similarity.py`, rewritten by A1 this week — do not touch it with a textbook version |
| `dynamic_programming/longest_common_subsequence`, `longest_common_substring` | `core/similarity.py` |
| `searches/simulated_annealing`, `hill_climbing` | temperature `T` in `_compute_weights`; `climb_hill` operator |
| `machine_learning/gradient_descent` | `core/gradient_descent.py`, already optimised |
| `graphs/tarjans_scc`, `scc_kosaraju` | `core/horizon.py` Tarjan-SCC DAG conversion |
| `graphs/markov_chain` | `core/markov.py`, `schedulers/monte_carlo.py:103` |
| `set/set_covering` (greedy) | `services/minimize.py:149` |
| `data_structures` bloom filter | `core/bloom.py` |

### Rejected on structural grounds

- **Gray code** (`keon bit_manipulation/gray_code`, `TAP
  bit_manipulation/gray_code_sequence`). The intuition — "enumerate all values
  with one bit changing per step, so you can mutate in place" — is exactly right
  in general and **does not apply to our deterministic stage**, because each
  mutant there is one edit from the *base seed*, not from the *previous mutant*.
  Walking a Gray code would change what the schedule tests (multi-bit deltas
  from the base) rather than how it is computed. The in-place win is available
  without it: see K8. Gray code would only pay if we added an exhaustive
  enumeration of all `2^k` values of a `k`-bit field, which nothing does today.
  Note the term already appears in
  `handover_sjt_adjacent_transpositions_2026-09-04.md` in the *permutation*
  sense (adjacent-transposition Gray code); that is a different object.
- **All shortest-path algorithms** — `dijkstra` (7 variants across the two
  repos), `bellman_ford`, `johnson`, `a_star`, `bidirectional_*`,
  `floyd_warshall`. Already settled as D2 in the seventeen-source survey and the
  reasoning is unchanged: `core/distance.py:12-13,522-564` computes AFLGo
  harmonic distance with a reverse **unweighted** BFS, already `O(m+n)`;
  `grep -i dijkstra src/` is correctly zero. No weights, no priority queue, no
  barrier to break.
- **All MST algorithms** — `kruskal` (×2), `prims` (×2), `boruvka` (×2),
  `karger`. Settled as D1. Single-linkage clustering *is* an MST problem, but
  our similarity graph is complete (`m = n(n−1)/2`), so linear-in-`m` is still
  quadratic in `n`. A2 was solved instead by exact bounds that prune pairs before
  the graph is built (`bd33733`, 134× at n = 400, clusters identical).
- **`searches/median_of_medians`, `quick_select`.** Real technique, no site. All
  fifteen `sorted(...)[:k]` occurrences outside tests are display paths over
  small `n` (top-10 buckets in `report.py`, top-15 operators in `plotting.py`,
  50 edges in `edge_tracker.py:1075`). `length_mi.py:54,78` and
  `int_checksum_solver.py:390` are over caps of 500 and 128. `operators.py:3169`
  already uses `heapq.nlargest`. Nothing here is hot; the campaign's remaining
  6,614 `sorted` calls after `fuzzer-algorithm-perf-fixes` are not concentrated
  in these.
- **Every sorting algorithm in both repos** (47 modules). We sort with
  `sorted`/numpy. There is no site where a hand-written sort could win.
- **`other/linear_congruential_generator`.** Actively harmful: `--seed`
  reproducibility runs through `RandPool` and its refill accounting, and B1
  already established (this week) that even swapping `sample()`'s draw source
  reorders every downstream refill. A second RNG is the one change guaranteed to
  break the contract.
- **`other/davis_putnam_logemann_loveland`.** We have z3 behind the SMT extras.
  A pure-Python DPLL is strictly worse for path-constraint solving.
- **`linear_programming/simplex`, `knapsack/*`.** No constrained-optimisation
  objective in the tree. Energy assignment is a bandit problem, not an LP.
- **`scheduling/*` (9 modules).** Superseded by
  `handover_job_scheduling_2026-09-02.md`, which covers Lawler, EDF, LST,
  Multifit and MDD with an implementation plan and an explicit rejection list.
  Two entries there are *not* covered — `highest_response_ratio_next` and
  `multi_level_feedback_queue` — and both should be rejected for the reason that
  handover already gives for EDF-as-seed-picker: they optimise a latency
  objective (waiting time, response ratio) that nobody asked for, at the cost of
  the coverage signal in `_compute_weights` that we spent the whole Aug 2026
  edge-distribution campaign getting right. MLFQ's demote-on-budget-consumed is
  the only genuinely new idea and it is already approximated by
  `fuzz_count/(coverage+1)` staleness.
- **`ciphers/`, `blockchain/`, `financial/`, `physics/`, `electronics/`,
  `geodesy/`, `quantum/`, `project_euler/`, `audio_filters/`,
  `digital_image_processing/`, `computer_vision/`.** ~500 modules, no analogue.
- **`cellular_automata/`.** Plausible-sounding as a structured-fill operator
  alongside `de_bruijn_fill` / `monotone_fill` / `spectral_peak`, and rejected on
  purpose: those thirteen operators each invert a *specific named statistical
  test* from the dieharder battery in `core/randomness.py`. A CA fill inverts
  nothing in particular. If someone wants it, the entry price is naming the test
  it defeats.

### Rejected as already-answered questions

- `strings/z_function`, `prefix_function`, `knuth_morris_pratt`,
  `boyer_moore_search`, `naive_string_search`, `bitap_string_match`,
  `suffix_automaton` — all single-pattern. Our single-pattern search is
  `bytes.find`, which is memchr-backed C and beats any of them in Python.
  Aho-Corasick (K1) is the only one that wins, and it wins because it is
  *multi*-pattern.
- `strings/jaro_winkler`, `hamming_distance`, `ngram` — we already have Jaccard
  over 4-grams (`crash_metadata`), MinHash LSH (`ga.py:132`), popcount tables
  and an n-gram coverage channel.
- `graph/blossom`, `graphs/gale_shapley_bigraph` — matching. Speculatively
  applicable to seed-pair selection for splice; no evidence, and splice pairing
  is currently random by design. Not rejected on merit, rejected for lack of a
  question it answers.

---

## 11. Suggested sequencing

1. **K2** first and alone. It is a bug, the falsifying test is written above, and
   it is confined to one method.
2. **K8** next. Largest measured number, one file, contract unchanged — but the
   equivalence oracle in §9 is mandatory, not optional.
3. **K1** third. Real design work (affordability dispatch + automaton cache
   keying + preserving `weizz_tags`' pair ordering), two call sites, clear
   evidence.
4. **K3** any time; independent and small. Fix the `filesystem.py`
   bloom/`seen_hashes` interaction in its own commit — it is a correctness note,
   not a cache-policy note.
5. **K4** only after K2 lands, and as a deliberate refactor with its own
   before/after, not folded into anything.
6. **K6** when B2 is picked up; it changes what alignment B2(b) should use.
7. **K5** and **K7** are genuine and block nothing.

## 12. Environment notes

- `pip install -e . --break-system-packages` to import `fuzzer_tool` and measure
  against the live tree.
- `keon/algorithms` and `TheAlgorithms/Python` both clone shallow without
  incident (2.8 MB / 27 MB).
- `ShapleyAttribution` is the class name; `ShapleyAttributor` does not exist.
  `CmplogColorizer`, not `Colorizer`, owns `colorize_from_cmplog` —
  `Colorizer.__init__` takes `(data_length, checker)` and is a different
  (bisection-based) colorizer entirely.
