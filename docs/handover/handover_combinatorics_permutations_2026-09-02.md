# Handover — Combinatorics and Permutations Across `fuzzer-tool`

**Date:** 2026-09-02
**Base:** `be3533d` ("Lock crash-sidecar artifacts to read-only on disk")
**Status:** Analysis only. Every section states what combinatorics primitive
exists, where it lives, what it serves, and where it is duplicated or
under-exploited. No code changes proposed yet — the goal is to surface the
shape of the combinatorial surface area so follow-up implementations can be
scoped against it.

**Revision (2026-09-02, verification pass on §2 / §10a):** confirmed all 13
swap-pair call sites directly. The idiom is **not textually identical**
across all 13 — three distinct shapes exist, which changes the shape of the
`_swap_pair` helper proposed in §10a. See the updated §2 and §10a below.

**Revision (2026-09-02, implementation pass on §10a / §10i):**
- **§10a implemented.** `_swap_pair(domain, rng, *, start=0)` added to
  `core/mutations/generic.py`; all 15 call sites refactored (13 from the
  original table plus `zip.py`/`webp.py`, which use the identical idiom
  but were missing from the original survey — see §2). 20 new regression
  tests in `tests/test_swap_pair.py`, all passing; full existing suite
  re-run before/after with no new failures attributable to this change.
- **§10i downgraded from "gap" to "verified no-op."** The literal fix
  proposed there was checked by direct computation before implementing
  it: it does not change the scheduler's selection behavior for any
  blend weight below 1.0. See the revised §10i for the argument and the
  one real (narrower) edge case found instead.

**Revision (2026-09-03, C(n,2+k) generalization analysis + empirical
validation on §10a.1):** new §10a.1 added, analyzing the generalization of
`_swap_pair`'s C(n,2) primitive to C(n,2+k) and empirically validating the
"validity cliff" hypothesis against two real parsers (`libsqlite3` via the
Python `sqlite3` module, `libavformat` via both a standalone ASAN build and
a real coverage-guided `fuzzer-tool` campaign with clang instrumentation).
Also independently reproduced the parity-trap finding via a from-scratch
Cayley-graph BFS and found it recurs at every odd m under the rotation-only
generator, not only m=3 — a gap in the original recommendation. Analysis
and validation only; `_swap_tuple` itself is not implemented.

**Revision (2026-09-02, second implementation pass — §10f, and a bug
caught by §10a's own test suite):**
- **§10f implemented.** `core/debruijn_cache.py` — see the revised §10f
  for details.
- **A real regression was caught, not just a hypothetical one.**
  `_swap_pair`'s first version passed a bare `range(start, domain)` to
  `rng.sample(...)`. That's fine under stdlib `random` and `RandPool`,
  but `ExhaustivePool.sample()` only recognizes `list | tuple | bytes`
  as a sequence population and misreads a bare `range` as the
  population *size*, crashing with `TypeError` the first time an
  operator ran under exhaustive enumeration
  (`tests/test_exhaustive_pool.py` caught it: 4 failures, all in
  `x86._swap_insns`). This is exactly the failure mode 10 of the
  original 13 call sites avoided by wrapping in `list(range(...))` —
  the other 3 (`asf`, `riff`, `adts`) only got away with a bare `range`
  because nothing had exercised them under `ExhaustivePool` yet.
  Fixed by wrapping in `list(...)` inside the helper; 4 new regression
  tests drive `_swap_pair` directly through `ExhaustivePool.runs()` so
  this is pinned independent of which operator sweep happens to
  exercise it. Full suite: 21 failed → 17 failed (the removed 4 were
  all in `test_exhaustive_pool.py`; the remaining 17 match the
  pre-existing baseline, confirmed via `git stash`).

---

## 0. Rule

The combinatorial primitives in this codebase are **never abstract math**.
Every permutation, combination, Markov chain, exhaustive enumeration, and
graph aggregate exists to do exactly one of three jobs:

1. **Reach worst-case program paths** (adversarial input construction).
2. **Allocate execution budget to operators that have produced coverage**
   (bandit / Markov / PageRank over the operator and corpus forests).
3. **Falsify that a mutation operator is reachable in its parameter space**
   (`ExhaustivePool`).

If a proposed combinatorics addition does not serve one of those three jobs,
it does not belong in this fuzzer. This is the litmus test for the proposals
in §10.

---

## 1. The combinatorial surface area at a glance

| Layer | Combinatorial primitive | File | What it serves |
|---|---|---|---|
| Input permutation | C(n,2) chunk/element swap | `core/mutations/<format>.py` (13 files) | Worst-case reorder paths |
| Input permutation | n! Fisher-Yates over byte windows | `byte_shuffle` via `byte_shuffle_bytes` (`services/operators.py:1476`) | All-orderings coverage |
| Input permutation | n! token swap | `core/token_shuffle.py:46` | Delimiter-token reorder |
| Adversarial construction | n sorted-shape orderings | `core/mutations/structured.py:529` `perm_lock` | dieharder OPERM5 inverse |
| Adversarial construction | 2-cycle vs identity | `core/mutations/structured.py:562` `cycle_lock` | Index-chase worst cases |
| Adversarial construction | de Bruijn sequence B(k, n) | `core/mutations/structured.py:264` `_de_bruijn_symbols` | k-mer saturation |
| Adversarial construction | Arithmetic progressions, monotone runs, low-popcount streams, geometric degeneracies | `birthday_collide`, `monotone_fill`, `popcount_lock`, `degenerate_geometry` (`structured.py:722,224,940,835`) | Hash-flooding / BST-degeneration / ECC / geometry-degeneration |
| Operator scheduling | Thompson sampling over ~150 arms | `core/schedulers/monte_carlo.py:206` | Bandit arm selection |
| Operator scheduling | Pairwise operator Markov chain | `core/schedulers/monte_carlo.py:103` `transition_counts[prev][next]` | P(next \| prev) blend |
| Operator scheduling | Stationary distribution πP=π | `core/schedulers/monte_carlo.py:808` | Long-run operator mix |
| Operator scheduling | Spectral gap 1−λ₂ | `core/schedulers/monte_carlo.py:920` | Mixing-time / stagnation signal |
| Operator scheduling | Correlated Thompson via Cholesky | `core/schedulers/monte_carlo.py:961` | Covariance-aware selection |
| Operator scheduling | Matrix-UCB (covariance-penalized) | `core/schedulers/monte_carlo.py:1123` | Reduced exploration for correlated arms |
| Operator scheduling | CMA-ES / PSO / replicator dynamics / Elo / GP-UCB / SW-UCB / CUCB / DUCB / MCTS / hierarchical / contextual / katz | `core/schedulers/<name>.py` (12 schedulers) | Each is a different combinatorial strategy on the ~150-element arm space |
| Grammar mutation | Formal-language derivation tree | `core/grammar.py:256` `generate`, `Grammar.mutate` | Structured inputs |
| Grammar mutation | Per-rule reservoir sampling (Algorithm R) | `core/grammar.py:533` `SubtreePopulation` | Cross-corpus subtree splice |
| Grammar mutation | Tree-level swap / delete / duplicate / splice / rule-sub | `core/grammar.py:730` `TreeMutator.mutate_tree` | Structural mutation |
| Draw-space enumeration | Odometer over bounded draws | `core/exhaustive_pool.py:110` `ExhaustivePool` | Falsification: every reachable combination is visited, not sampled |
| Draw-space enumeration | n! Fisher-Yates over a `shuffle` call | `core/exhaustive_pool.py:266` | Enumerate every permutation reachable via the draw sequence |
| Draw-space enumeration | n!/(n−k)! ordered k-permutations | `core/exhaustive_pool.py:280` `sample` | Enumerate every ordered selection |
| Markov context | n-gram byte transition matrix | `core/markov.py` (referenced) | `markov_bytes` guided mutation |
| Corpus forest | Parent-pointer forest over corpus seeds | `core/lineage.py` `LineageTree` | Causal lineage replay, subtree pruning |
| Corpus forest | γ-discounted subtree weight | `core/lineage.py:83,175` | Branch productivity signal |
| Corpus forest | Damped PageRank over the forest | `core/lineage.py:302` `pagerank_credit` | Yield per mutation, not fan-out |
| Corpus forest | LCA / tree distance | `core/lineage.py:221,259` | Diversity term in seed scoring |
| Corpus forest | Per-operator γ-discounted credit | `core/lineage.py:371` `operator_credit` | Global operator yield signal |
| Corpus novelty | PPMD compression ratio | `core/corpus_compression.py:70` | Per-seed novelty / pruning signal |
| Format structure | Weizz P2/P3 chunk/field ops on parsed StructureMap | `core/mutations/weizz_structural.py`, gated by `_weizz_tags_available` (`core/operator_registry.py:517`) | C(n,2) swap over parsed chunks, not bytes |

The list is dense but every row traces to one of the three jobs in §0.

---

## 2. Layer 1 — Input permutation via C(n,2) swap

**Where:** `src/fuzzer_tool/core/mutations/<format>.py`. Every per-format
mutator has the same idiom: parse the input into a list of structural
elements (packets, chunks, boxes, pages, frames, instructions, fields,
segments), then `i, j = self._rng.sample(list(range(len(elements))), 2)` to
pick two and swap them.

**Files confirmed (13):**

| File:line | Format | Element |
|---|---|---|
| `mutations/mpegts.py:307` | MPEG-TS | packets |
| `mutations/asf.py:159` | ASF | objects |
| `mutations/riff.py:173` | RIFF | chunks |
| `mutations/avif.py:580` | AVIF | `meta_children` |
| `mutations/sqlite.py:484` | SQLite | `doc.pages` (B-tree pages) |
| `mutations/adts.py:189` | AAC ADTS | frames |
| `mutations/webm.py:371` | WebM/Matroska | elements |
| `mutations/nal.py:271` | H.264 NAL | units |
| `mutations/isobmff.py:388` | ISOBMFF/MP4 | boxes |
| `mutations/protobuf.py:354` | protobuf | fields |
| `mutations/pgs.py:187` | PGS | segments |
| `mutations/x86.py:515` | x86 | instructions |
| `mutations/arm.py:223` | ARM | instructions |

Plus the same idiom registered as a category in
`core/operator_registry.py:124-161` for: `png`, `jpeg`, `bmp`, `gzip`, `zlib`,
`gif`, `webp`, `zip`, `ogg`, `flv`, `mp3`, `elf`, `der`, plus `der_tlv_reorder`
which is the TLV-specific variant. The handler implementations live in
`services/operators.py` (`_op_*_chunk_mutate`).

**Combinatorial primitive:** `C(n, 2)` swap of two distinct indices. n is
the count of elements the parser recovered, typically 1–1024.

**What it serves:** worst-case reorder paths. Container formats often
assume header / index / payload ordering; reordering the elements exercises
the parser's reaction to the unexpected layout, and frequently surfaces
OOB / uninitialized-struct-member / wrong-page-type bugs in the indexing
code. The sqlite C(n,2) swap of B-tree pages is the most aggressive
example — it directly attacks the page lookup table.

**Gap 2a — duplication of the swap-pair primitive across 13 files, in
three distinct shapes.** Verified directly (not just grepped) at all 13
call sites. They are **not** textually identical:

| Shape | Files | Idiom |
|---|---|---|
| Plain range, listed | `avif`, `isobmff`, `mpegts`, `nal`, `pgs`, `protobuf`, `webm`, `webp`, `x86`, `zip` (10) | `i, j = self._rng.sample(list(range(len(x))), 2)` |
| Plain range, unlisted | `asf`, `riff`, `adts` (3) | `i, j = rng.sample(range(len(x)), 2)` — `Random.sample` accepts a `range` directly, so the `list(...)` wrap is a no-op difference, not a behavior difference |
| Offset range | `sqlite` (`_swap_pages`, line 484) | `i, j = self._rng.sample(range(1, len(doc.pages)), 2)` — **excludes index 0** because page 1 carries the file header; page 1 is deliberately never swapped |
| Filtered candidate list | `arm` (`_word_swap`, line 223) | `i, j = self._rng.sample(swapable, 2)` where `swapable = [i for i, w in enumerate(words) if w.kind != "raw"]` — swaps only over a **pre-filtered index subset**, not the full range |

The first two rows are one primitive (10 + 3 = 13 files use `sample` over
a contiguous 0-based range — the `list()` wrap doesn't matter). The
`sqlite` and `arm` cases are genuinely different: they sample over a
**restricted domain** (an offset range, or an arbitrary index subset),
not `range(len(x))`. A single `_swap_pair(seq, rng)` helper that only
takes a sequence and returns two random indices into it would not cover
`sqlite` or `arm` without also accepting either a `start` offset or an
explicit candidate list — see the revised fix in §10a.

The reason the duplication exists is the intentional per-format isolation
mandated by AGENTS.md rule 36 (don't punch through layers; each format's
mutator owns its own parser). But **the swap-pair primitive is a
combinatorics utility below the format-parser layer** — it doesn't read any
format-specific structure. A helper would not violate the rule; it would
move a duplicated primitive into the right layer.

---

## 3. Layer 2 — Token / chunk / byte Fisher-Yates permutations

Three operators walk all n! permutations of a region:

- **`token_shuffle`** (`core/token_shuffle.py:14`) — splits input on
  delimiter set `_DELIMS = b" \t\n\r,;:|/\\=&?"` (line 11), caps at 64
  tokens, picks two distinct token indices via `randint(0, n-2)` /
  `randint(idx1+1, n-1)`. Returns the original if either token is empty or
  >256 bytes. Comments describe it as a port of `honggfuzz mangle.c`
  `mangle_TokenShuffle`. Combinatorial primitive: **C(k, 2)** for k ≤ 64
  tokens.

- **`chunk_shuffle`** (`services/operators.py:1891`) — fixed-stride chunk
  shuffle. Reorders fixed-size byte regions. The combinatorial primitive is
  the **k!** permutations of k chunks.

- **`byte_shuffle`** (`services/operators.py:1476-1480`) — Fisher-Yates
  shuffle over the whole input buffer. **n!** permutations, where n is the
  byte length.

All three are registered in `core/operator_registry.py`:

| Operator | Category | file:line in `_CATEGORIES` |
|---|---|---|
| `token_shuffle` | structural | `operator_registry.py:99` |
| `chunk_shuffle` | structural | `operator_registry.py:100` |
| `byte_shuffle` | byte | `operator_registry.py:52` |

**What they serve:** all-orderings coverage of small structured regions.
`byte_shuffle` is the most expensive (n! grows fast) but the most
permutation-diverse; `token_shuffle` is the cheapest and the most
format-appropriate for text-style inputs.

**Gap 2b — the chunk / byte / token layers are not joint.** A single
operator that picks two regions of the corpus's typical token length and
swaps them would generalize all three. But see §0 — the current structure
deliberately exposes them as independent bandit arms so the scheduler can
learn which scale of permutation is productive on the current target. A
combined operator would hide that signal.

---

## 4. Layer 3 — Adversarial permutations in `core/mutations/structured.py`

This is the most combinatorial-rich file in the codebase. Every operator
here is the **constructive inverse** of a dieharder/diehard permutation or
regularity test — it builds a buffer whose statistic sits in the far tail
of the uniform null distribution, where random input never lands.

### 4.1 `perm_lock` (`structured.py:529`)

Overwrites a region with one of 5 orderings of `1..n`:
- **ascending** — `[1, 2, ..., n]`
- **descending** — `[n, n-1, ..., 1]`
- **organ-pipe** — `[1, 2, ..., k, k, ..., 2, 1]` (`structured.py:513`)
- **equal** — `[1, 1, ..., 1]` (line 515)
- **interleave** — low/high halves alternated (lines 519-526)

These are the **textbook worst-case inputs for comparison sorts**:
ascending / descending / organ-pipe defeat naive quicksort pivots, equal
defeats partition-based pivots, interleave is the canonical median-of-3
adversary. Together they cover the O(n²) quicksort paths in any sort
over parsed records.

### 4.2 `cycle_lock` (`structured.py:562`)

Two modes:
- **`single_cycle`** — `[(i+1) mod n for i in range(n)]`. One n-cycle.
  Worst case for index-chase: open-addressing hash probing, linked-list-as-
  array-indices, jump tables, union-find parent arrays. Following it
  visits every slot before repeating.
- **`fixed_points`** — `list(range(n))`. The identity. The opposite
  extreme — every chase terminates in one step.

These are at the two extremes of the permutation-cycle distribution. A
random permutation has expected cycle length O(log n); `single_cycle` is
n, `fixed_points` is 1.

### 4.3 `de_bruijn_fill` and `kmer_saturate_bits` (`structured.py:306, 416`)

`_de_bruijn_symbols(k, n)` (line 264) — iterative FKM construction of
`B(k, n)`, a de Bruijn sequence where every one of the `k^n` words of
length n over a k-symbol alphabet appears exactly once as a cyclic
substring. Used in:
- `de_bruijn_fill` — byte-aligned de Bruijn for byte-level lexicons.
- `kmer_saturate_bits` — **bit-packed** binary de Bruijn for sub-byte
  windows. Important because real bitfield parsers (H.264 RBSP Exp-Golomb,
  protobuf varint continuation bits, packed struct bitfields) read bits,
  not bytes — byte-aligned saturation never reaches the off-byte-offset
  windows those parsers actually use.

The de Bruijn construction is the **canonical solution to "visit every
n-permutation of an alphabet exactly once on a cycle"** — an Eulerian path
on the n-gram graph.

### 4.4 `birthday_collide` (`structured.py:722`)

Overwrites a region with arithmetic progression
`[base + i*delta for i in range(n)]`. The progression has **all spacings
identical** — the maximum-duplication tail of the birthday spacings
distribution. `_BIRTHDAY_DELTAS` is pre-weighted toward powers of two (line
698-719) because a progression with a power-of-two common difference
collides under any power-of-two bucket count, the common case for hash
tables.

With probability 1/4 the progression collapses to literal repeats of one
word — the degenerate limit at spacing zero.

### 4.5 `monotone_fill` (`structured.py:224`)

A strictly monotone run of fixed-width words. Defeats BST/interval-map/
sorted-index insertion: a uniform stream balances the tree, a monotone
run degenerates it into a linked list. Comment at line 232 names the
target structures: symbol tables, ZIP central directories, font cmaps.

### 4.6 `rank_deficient` (`structured.py:458`)

Builds rank-deficient GF(2) matrices — far left tail of the binary-rank
distribution. Targets Reed-Solomon / LDPC decoders, GF(2) checksum code,
linear-algebra fast paths whose "not invertible" branch is rarely reached.

### 4.7 `degenerate_geometry`, `spectral_peak`, `float_squeeze`, `popcount_lock`

All four follow the same pattern: name a regularity statistic whose far
tail is a degenerate construction, build a buffer that lands there.

- `degenerate_geometry` (line 835) — coincident or collinear coordinate
  tuples. Targets hull / triangulation / collision code that divides by
  a distance, determinant, or cross product.
- `spectral_peak` (line 645) — pure cosine / DC / Nyquist / impulse /
  max-AC blocks. Targets DCT-based codec saturation and clamping.
- `float_squeeze` (line 897) — IEEE-754 patterns that break convergence:
  one ulp from 1.0, denormals, infinities, NaN payloads.
- `popcount_lock` (line 940) — every byte pinned to a single Hamming
  weight. Targets bit-packed formats, UTF-8/Base64 validity classes,
  constant-weight codes, SIMD popcount scalar tails.

### 4.8 What `structured.py` deliberately is not

These operators are **length-preserving** — each overwrites a bounded
region of the input in place. They compose with the length-changing
operators rather than competing with them. They use the API shared by
`RandPool` and stdlib `random` (`randint`, `choice`, `random`, `sample`,
`randbytes`), so they stay usable with either (and importantly, with the
`ExhaustivePool` of §7).

The full taxonomy is at the module docstring (lines 13-32) — a 12-row
table mapping each dieharder test to its constructive inverse in this file.

---

## 5. Layer 4 — Operator scheduling over a ~150-element arm space

The operator registry (`core/operator_registry.py`) registers every
mutation operator once. The count by category, derived from
`_CATEGORIES`:

| Category | Count | `_CATEGORIES` line |
|---|---|---|
| bit | 11 | 31-43 |
| byte | 15 | 44-60 |
| block | 14 | 61-76 |
| dict | 8 | 77-86 |
| structural | 20 | 87-114 |
| radamsa | 7 | 115-123 |
| format | 36 | 124-161 |
| regularity | 15 | 162-183 |
| adaptive | 22 | 184-207 |

Total: ~150 operators. Each call to the fuzzer's main loop selects **one**
operator and applies it to **one** seed. The combinatorial space of
operator chains is `150^N` for N iterations.

### 5.1 The scheduler zoo

Each scheduler is a different strategy over the same arm set:

| Scheduler | File | Strategy |
|---|---|---|
| `MonteCarloScheduler` | `core/schedulers/monte_carlo.py` | Thompson sampling + CEM + pairwise Markov |
| `MCTSScheduler` | `core/schedulers/mcts.py` | Monte Carlo tree search over ops |
| `Exp3Scheduler` | `core/schedulers/exp3.py` | Adversarial bandit, EXP3-IX |
| `EpsilonGreedyScheduler` | `core/schedulers/epsilon_greedy.py` | Uniform random with prob ε |
| `GPUCBScheduler` | `core/schedulers/gp_ucb.py` | Gaussian-process UCB |
| `SlidingWindowUCB` | `core/schedulers/swucb.py` | UCB1 over a sliding window |
| `CUCBScheduler` | `core/schedulers/cucb.py` | Compressed UCB |
| `DUCBScheduler` | `core/schedulers/ducb.py` | Discounted UCB |
| `CMAScheduler` | `core/schedulers/cmaes.py` | CMA-ES over continuous op weights |
| `MOptScheduler` | `core/schedulers/mopt.py` | Particle-swarm on the operator simplex |
| `ReplicatorScheduler` | `core/schedulers/replicator.py` | Replicator dynamics on op frequencies |
| `HierarchicalScheduler` | `core/schedulers/hierarchical.py` | Multi-level bandit (ops grouped by category) |
| `ContextualScheduler` | `core/schedulers/contextual.py` | Context features drive selection |
| `KatzScheduler` | `core/schedulers/katz.py` | Katz centrality over transition matrix |
| `EloTracker` | `core/elo.py` | Elo ratings on op pairwise comparisons |

All implement `select_op(ops, prev_op=None) -> str` and
`record(name, success, weight=1.0)`, and declare `supports_priors: bool` per
AGENTS.md rule 40 (`monte_carlo.py:66` is the reference).

### 5.2 The pairwise Markov chain — `monte_carlo.py:103-105, 320-323`

```python
self.transition_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
self.transition_total: dict[str, int] = defaultdict(int)
self._prev_op: str | None = None
```

`transition_counts[prev][next]` is incremented **only on success** when
the recorded operator name differs from the previous recorded operator.
The `_prev_op` advances unconditionally at the end of `record()` so the
chain can bootstrap from the first recorded operator without depending on
a branch that itself requires the matrix to be populated.

The matrix is a **first-order Markov chain over the ~150-element operator
set**. The `stationary_distribution` (`monte_carlo.py:808-839`) computes
πP=π via power iteration. `spectral_gap` (line 920) and `correlated_select`
(line 961) exploit the chain structure.

### 5.3 Second-order Markov — `transition_counts[prev2][prev][next]`

A second-order extension would capture "A then B then C" patterns. Cost
is 150³ ≈ 3.4M counters, manageable. Whether second-order transitions add
signal is an empirical question; the architecture supports it but the
table is currently first-order only.

### 5.4 The Cholesky path — `correlated_select` and `matrix_ucb_select`

`operator_covariance(window=2000, segment_size=50)` (referenced line 980)
builds the empirical covariance matrix of operator success rates over
sliding segments. `_chol` (line 1023) Cholesky-decomposes it.

`correlated_select` (line 961): draws a multivariate normal `noise = L·z`
and adds it to each arm's Thompson score. **Correlated arms get correlated
score boosts**, so they're selected together rather than fighting each
other — the structural signal that "operators A and B tend to discover the
same edges" is used to schedule them as a coordinated block.

`matrix_ucb_select` (line 1123): adjusts UCB exploration bonuses by
`log(t) + quadratic_form(μ, inv_cov)`. Arms correlated with high-
performing arms get reduced exploration bonuses.

Both are first-class combinatorial schedulers — they treat the arm set
as a structured graph, not as independent arms.

### 5.5 Stationary distribution — `stationary_distribution()`

Power iteration to find π such that πP=π. Tells you which operator mix
the fuzzer **naturally settles into** under Thompson sampling — useful for
diagnostics, and a signal for stagnation (the stationary distribution
degenerating toward one or two arms).

The `should_explore(gap_threshold=0.1)` method (line 950) flags when the
spectral gap falls below the threshold — **slow mixing = the chain is
stuck in a narrow cycle of operators**, the analogue of the "stuck in a
local optimum" detection in single-objective optimization.

---

## 6. Layer 5 — Grammar as a formal language

`core/grammar.py` parses an S-expression grammar into
`rules: dict[str, list[alternatives]]`. The grammar is a context-free
language with quantifiers (`{N}`, `{N,M}`, `+`, `*`).

### 6.1 Generation — `Grammar.generate()`

`generate(rule, max_depth, max_len)` (line 256) recursively expands rules.
At each `repeat` token, `randint(lo, hi)` (line 292) picks a count. The
total derivation space is exponential in `(alts × repeats × depth)` — for
the shipped `dictionaries/jpeg.gram` (35 rules, 2–4 alts each) it's
astronomically larger than any fuzzer can visit.

`_MAX_REPEAT = 32` (line 163) caps the expansion at each quantifier so a
single runaway rule doesn't generate megabytes.

### 6.2 Tree-level mutation — `TreeMutator`

Parses inputs into `TreeNode` (line 461), mutates the tree directly. Five
operations:

| Op | Method | Combinatorial primitive |
|---|---|---|
| Subtree swap | `_tree_swap` (line 774) | C(k,1) target × generate-replacement |
| Subtree delete | `_tree_delete` (line 787) | C(k,1) target |
| Subtree duplicate | `_tree_duplicate` (line 799) | C(parents_with_≥2_children, 1) × C(children, 1) |
| Subtree splice | `_tree_splice` (line 813) | C(k,1) target × donor from `SubtreePopulation.sample(rule)` |
| Rule substitution | `_tree_rule_sub` (line 847) | C(rules_with_multi_alts, 1) target |

### 6.3 `SubtreePopulation` — Algorithm R reservoir sampling

`core/grammar.py:533-577`. Per-rule bounded reservoir of harvested
interior nodes, fed by every parsed corpus tree.

The `add` method (line 554) implements **Vitter's Algorithm R**:

```python
for node in tree.collect_interior():
    pool = self._pools.setdefault(node.rule, [])
    seen = self._seen.get(node.rule, 0)
    self._seen[node.rule] = seen + 1
    if len(pool) < self.max_per_rule:
        pool.append(node)
        continue
    j = rand.randint(0, seen)
    if j < self.max_per_rule:
        pool[j] = node
```

Every harvested node has an **equal chance of ending up in the pool**
regardless of corpus size — the canonical unbiased reservoir sample. The
`max_per_rule` bound keeps memory O(rules × max_per_rule), not O(corpus).

This is what makes cross-corpus subtree splice tractable. Without it, the
splice operator would either explode memory or undersample late-arriving
subtrees.

### 6.4 Grammar gap

The grammar's full derivation space is unreachable — every `generate()`
call walks one path through the exponential tree, capped at
`_MAX_REPEAT = 32` and `max_depth = 10`. The grammar operators (§6.2) do
the only practical thing: **mutate the existing tree** rather than
re-generate from scratch.

A precomputed set of "grammar skeletons" — one per derivation shape —
would give the bootstrap path a starting corpus that doesn't rely on a
single random walk. Combinatorially this is the **set of "minimum
representative derivations"**, one per equivalence class of the grammar's
derivation relation. Precomputing it requires the grammar to have a
finite canonical-form projection, which not all grammars do — so this
gap is open rather than closed.

---

## 7. Layer 6 — `ExhaustivePool`: enumerate the operator's draw space

`core/exhaustive_pool.py` is the most striking combinatorial artifact in
the codebase. It is a `RandPool`-shaped generator (same method names)
that **enumerates every reachable combination of bounded draws** instead
of sampling one. The data structure is an **odometer** of `[value, bound]`
positions in draw order:

```python
class ExhaustivePool:
    def __init__(self, max_depth=24, max_runs=1_000_000, allow_bulk=False):
        self._v: list[list[int]] = []   # odometer
        self._p = 0                     # current position
        ...
```

`_advance()` (line 158) increments the last incrementable position and
discards everything after, so the next run replays an identical prefix
and then diverges. `_bounded(bound, what)` (line 195) reads the current
value at the position, appending `[0, bound]` if this is the deepest the
enumeration has gone.

### 7.1 Combinatorial primitives implemented

- **`shuffle(seq)`** (line 266) — Fisher-Yates, **enumerates all `n!`
  permutations**. The comment is explicit: "numpy's shuffle draws entropy
  this pool does not intermediate, so delegating would silently produce
  one permutation per run instead of enumerating them."

- **`sample(population, k)`** (line 280) — sequential draws without
  replacement, tree of **`n! / (n-k)!`** ordered selections.

- **`randbytes(n)`** (line 317) — `256**n` paths. Gated by `allow_bulk`
  because "randbytes(4) alone is 2**32 paths."

- **`random`, `gauss`, `expovariate`, `betavariate`, `gammavariate`,
  `lognormvariate`** (lines 360-393) — refused, continuous draws cannot
  be enumerated.

### 7.2 Why this matters — the Hard Rule 39 contract

AGENTS.md rule 39 forbids `for _ in range(N): if cond: break/found=True`
loops in tests. The pool is the alternative: **inject a scripted RNG that
deterministically drives the exact call sequence and assert the exact
output**.

`tests/test_exhaustive_pool.py` (per the docs/refs/bug-classes.md §Testing
reference) uses the pool to assert `pool.exhausted` is True after walking
an operator's draw space. `ContinuousDrawError`, `BulkDrawError`,
`DepthExceededError`, `NondeterministicDrawError` (lines 87-107) are the
four failure modes — none silent, all reported.

The `NondeterministicDrawError` is the most subtle: a replayed prefix
requests a different bound than last run, which means **the code under
enumeration is reading entropy from outside the pool** — a module-level
`random`, a clock, a set iteration order, a hash seed. Tests that pass
against a sampled RNG and fail against the exhaustive pool are exactly
the class of bug the rule is designed to surface.

### 7.3 The `allow_bulk` gate — over-conservative for some operators

`randbytes(n)` raises `BulkDrawError` unless `allow_bulk=True`. The
rationale (line 58-61) is "randbytes(4) alone is 2**32 paths." But for
an operator that always draws `randbytes(4)` and does something
deterministic with the result (e.g. `seed_range_overwrite`,
`spectral_peak`, `de_bruijn_fill`), enumeration of the 2**32 paths **is**
the test — it's exactly the falsification property tests under Hard Rule
39 want.

The current gate forces the test author to either accept 2^32 paths
(set `allow_bulk=True`, max_runs=10^6, but `2^32 > 10^6` so the pool
times out) or hand-roll an exhaustive loop. A per-call budget (e.g.
`randbytes(n)` only enumerable when `n <= 2` by default, higher with
explicit opt-in) would let byte-level exhaustiveness be tested
selectively.

### 7.4 The coin-flip observation

The module docstring (line 65-66) notes: "Several byte-level operators
are unenumerable *only* because a fair coin is written
`rng.random() < 0.5` rather than `rng.randint(0, 1)`; see the census in
`tests/test_exhaustive_pool.py`." This is a known and documentable gap:
the operator's `rng.random() < p` idiom defeats enumeration when the
same idiom written as `rng.randint(0, 99) < p*100` would be enumerable.
The census in the test suite is the inventory of operators that need to
be re-expressed as bounded draws before they become enumerable.

---

## 8. Layer 7 — Markov context and the corpus forest

### 8.1 Markov n-grams (`core/markov.py`)

Referenced from `core/operator_registry.py:551` (`markov_trained` predicate)
and from `services/operators.py:1353` (`cem_sample` is related but
different). The Markov model is a row-stochastic transition matrix at
order n: `n × 256^(n-1)` distinct states, each with 256 outgoing edges.

The transition matrix is the same combinatorial object as the operator
pairwise transition matrix in §5.2, just at a different level — bytes
vs operators.

### 8.2 The lineage forest (`core/lineage.py`)

A parent-pointer forest over corpus seeds. Combinatorics here is
tree-structured.

#### `subtree_weight` (line 273)

The **γ-discounted sum of descendant weights** with `GAMMA = 0.9`
(line 35). Maintained incrementally on insert (line 175-186), O(depth)
with a short-circuit when the marginal delta drops below
`_PROPAGATE_EPS = 1e-6` (line 38). At γ=0.9, the depth cap is ≈ 131.

#### `pagerank_credit` (line 302)

Damped PageRank over the forest. The damping factor is γ (same as
subtree weight), the personalization vector is normalized node weights.
The critical combinatorial difference from `subtree_weight`: each child
hands its parent `credit / n_children[p]` rather than its full weight.
Comment at line 314 spells out the consequence:

> a parent that sprayed 13 children worth one edge each outranks a parent
> whose single child was worth twelve, even though the second is twelve
> times the yield per mutation.

`subtree_weight` measures **volume** (raw descendant yield);
`pagerank_credit` measures **volume per mutation** (yield normalized by
fan-out). Both are valid aggregate signals; they answer different
questions about branch productivity.

#### `lca` and `lca_distance` (line 221, 259)

The **lowest common ancestor** in the forest and the tree distance via
the LCA. Bounded by `len(self.nodes)` so a corrupt parent chain (cycle)
degrades to None instead of hanging. The docstring at line 224-226
flags a specific regression: "fresh-corpus --elo all hang" was the
LCA-blows-up failure mode.

`lca_distance` feeds a **diversity term in seed scoring** — two seeds
with a high LCA distance are more diverse than two seeds with a low
LCA distance, and the seed picker uses this to bias selection toward
underrepresented branches.

#### `chain_from` (line 453)

Walks root-ward through the parent chain from any node. Used by tmin
(`services/tmin.py`) to replay the exact mutation path instead of
delta-debugging from scratch — combinatorial compression of the test
input search space.

#### `operator_credit` (line 371)

Global γ-discounted new-edge credit attributed to one op. Sums
`w(c) × γ^depth(c)` over every node whose inbound edge applied that op.
This is the **per-operator yield** over the corpus tree — independent
of the per-arm bandit credit in the schedulers, and useful for
diagnostics ("which ops drove the most coverage?" without conditioning
on which arm the scheduler chose).

### 8.3 Corpus novelty — `core/corpus_compression.py`

PPMD (Prediction by Partial Matching) compression ratio as a novelty
signal. PPMD is itself a combinatorial-context model: at order n it
tracks 256^(n-1) contexts, each with a 256-symbol conditional
distribution.

The novelty score is `ratio = compressed_size / raw_size` over the
first `PPMD_SAMPLE_BYTES = 65536` (line 44). A low ratio means the
seed's n-gram statistics are predictable from the corpus context; a
high ratio means they aren't. Used for seed selection (boost novel)
and minimization (prune redundant).

The `PPMD_SAMPLE_BYTES` cap bounds cost per seed at ~30 ms instead of
letting it scale with the largest seed in the corpus. The
`PPMD_CACHE_MAX = 4096` (line 50) bounds cache size; overflow clears
the cache wholesale because recomputing is the same cost as a miss.

---

## 9. Layer 8 — Format-mutation via the Weizz StructureMap

`core/mutations/weizz_structural.py` and the `_weizz_tags_available`
predicate in `core/operator_registry.py:517-534`. The Weizz P2/P3
operators are gated on `--weizz-tags` being on and the parent seed
having a non-dirty StructureMap.

The combinatorics is **structure-aware**: each operator picks from the
parsed field/chunk list, not from raw bytes. `weizz_chunk_swap` is
the C(n,2) swap again, but now over the parsed-out chunk list rather
than byte positions. The `weizz_field_havoc` operator combines the
field-aware mutations (overwrite, set, repeat, scramble) — these are
the same combinatorial primitives as `token_shuffle` but lifted onto
the StructureMap's typed field slots.

This is the format-mutation layer that the per-format mutators in §2
are reaching toward but don't quite achieve — the per-format mutators
parse each format's specific structure, while the Weizz layer operates
on a generic structure map harvested from the corpus.

---

## 10. Gaps — where combinatorics is under-exploited

These are the surfaces where the analysis above surfaces a real gap.
Follow-up implementations should be scoped against §0's litmus test.

### 10a — The C(n,2) swap-pair primitive is duplicated 13 times, in 3 shapes

**Where:** every per-format mutator in `core/mutations/`. Verified at all
13 call sites (§2, revised) — not a single idiom, three:
1. sample over `range(len(x))` (13 files — the `list()` wrap in 10 of
   them is cosmetic, `Random.sample` accepts a bare `range`);
2. sample over an **offset** range `range(1, len(x))` (`sqlite`, to keep
   page 1 fixed);
3. sample over a **filtered candidate list** (`arm`, to exclude `raw`
   words from the swap).

**The fix:** a `_swap_pair(seq_or_len, rng, *, start=0, candidates=None)`
helper in `core/mutations/generic.py`:
- default call `_swap_pair(len(x), rng)` covers the 13-file plain-range
  case;
- `_swap_pair(len(doc.pages), rng, start=1)` covers `sqlite`;
- `_swap_pair(None, rng, candidates=swapable)` covers `arm`.

Returns `(i, j)` with `i != j`, or `None` when the effective domain has
fewer than 2 elements. One helper, one regression test suite covering
all three call shapes (non-degenerate pair, `k<2` degenerate case, offset
correctness, candidate-list correctness) — replaces 13 inlined variants,
not 13 identical copies of one variant.

**Why it doesn't violate AGENTS.md rule 36:** the swap-pair primitive
reads no format-specific structure — the format-specific part (which
indices are eligible: all of them, all but page 1, all but `raw` words)
stays in the caller and is passed in via `start`/`candidates`. Only the
"pick 2 distinct, non-degenerate" mechanics move into the shared helper.

### 10a.1 — Generalizing C(n,2) to C(n,2+k): analysis and empirical validation

**Status: analysis + empirical validation only, no implementation.** Follow-up
to §10a asking what happens if `_swap_pair`'s pick-2 primitive generalizes to
picking `m = 2+k` elements (`k >= 0`) and permuting them, instead of always
swapping exactly 2.

**Analysis (container inspection at HEAD `7796726`, the 3 `math-identities-round2`
commits, ascending):**

1. At k=0, C(n,2) does double duty — it counts the subset *and* the
   rearrangement, because 2 elements have exactly one non-identity
   permutation. For `m = 2+k`, a permutation must also be chosen: output
   space is `C(n,m)·!m` (subfactorial), not `C(n,m)`. n=64:
   2016 (m=2) → 8.3e4 (m=3) → 5.7e6 (m=4) → 3.4e8 (m=5).
2. **Parity trap at k=1.** Under a generator restricted to pure rotation
   (the cost-efficient form — see point 6 below), 3-cycles are even
   permutations; BFS on the Cayley graph reaches only `A_n`, half of `S_n`:
   360/720 (n=6), 2520/5040 (n=7), 20160/40320 (n=8). A pure-rotation m=3
   operator can never produce a single transposition — it is a
   **regression**, not an extension, relative to the current m=2 operator.
3. Hit rate on a specific pair does not collapse as naively feared:
   `P = [m(m−1)/(n(n−1))]·[!(m−2)/!m]`. Relative to m=2: m=3 → 0 (exact),
   m=4 → 0.667, m=5 → 0.455, m=6 → 0.509, m=7 → 0.498 — plateaus around 0.5,
   independent of n (checked n=16, 64, 256).
4. The real payoff is Cayley-graph diameter (BFS, unrestricted permutation
   generators): m=2 → n−1 (5,6,7 for n=6,7,8); m=3 → 3,3,4; m=4 → 2,3,3;
   m=5 → 2,2,2. Under corpus admission gating each intermediate step must be
   independently interesting to be saved, so diameter is the number of
   *accepted* corpus entries needed, not raw runs — the one non-cosmetic
   argument for k>0.
5. Hard cost: `ExhaustivePool` enumerates `n!/(n−k)!` exactly. n=16: k=2 →
   240, k=3 → 3,360, k=4 → 43,680, k=5 → 524,160 runs, against
   `DEFAULT_MAX_RUNS = 1,000,000`. At n=20, m=5 → 1,860,480, over budget —
   `exhausted=False`, `budget_exhausted=True`, silently degrading the
   guarantee for that operator-test class.
6. n varies wildly across formats — isobmff/webp top-level counts ~4,
   mpegts/NAL/x86 reach the thousands. A fixed k is wrong for all of them;
   m would need to scale with n.

**Untested at the time, flagged for measurement:** the "validity cliff"
hypothesis — that in offset-table formats (isobmff `stco`/`stsz`, zip's
central directory, sqlite's page pointers) a permutation of m elements
breaks m pointers instead of 2, so early parser rejection should be more
likely, leaving fewer deep paths reachable.

**Empirical validation (2026-09-03, HEAD still `7796726`):**

Independently reproduced finding 2 with a from-scratch BFS (no dependency
on the container's earlier run) and found a gap in the doc's own
recommendation:
- Generators = *all* non-identity permutations of the chosen m-subset
  (i.e. the `!m` used for the output-space count in finding 1) reach full
  `S_n` at m=3 — that generator set includes transpositions-fixing-one-
  element, which are odd. This does *not* reproduce finding 2's numbers.
- Generators = *pure rotation only* (the m-cycle and its inverse — the
  "cheap alternative: rotation of a contiguous window" named in the
  Recommended form below) reproduce finding 2 exactly: 360/720, 2520/5040,
  20160/40320 at m=3.
- **New finding: the parity trap is not specific to m=3.** Under the same
  rotation-only generator, m=5 also reaches only `A_n` (half of `S_n`) at
  n=6,7,8 — any odd-length cycle is an even permutation. Finding 2 names
  only k=1/m=3; the recommendation below needs the same exclusion (or an
  explicit fallback to arbitrary-permutation mode) for *every odd m*, not
  just m=3, if implemented as rotation.

Tested the validity-cliff hypothesis against two real parsers, not modeled:

- **sqlite**, via the real `sqlite3` C library. Built a real 206-page,
  400-row multi-table database. Random-position `_swap_tuple`-style swaps
  at m=2..8 (80 trials/m): zero `open_fail`/`query_fail` at any m — a
  full-table `COUNT(*)` is structurally insensitive to page order, since
  every page is visited once regardless of order. Forcing the swap to
  always include the b-tree root/interior page: still zero rejections;
  sqlite silently reads whatever content landed at that page number and
  reports a badly wrong row count (2 instead of 400) with no error at all.
  Only `PRAGMA integrity_check` reliably flags the corruption
  ("Rowid out of order"), at every m>=2 including the existing m=2 baseline.
- **ffmpeg**, via both a standalone ASAN build and the project's real
  coverage-instrumented `.so` harness (clang, `vendor_ffmpeg.sh --nosan
  --minimal` + `build_ffmpeg_ready.sh`, run through `fuzzer-tool fuzz
  --inprocess-direct -c --elo all --cmplog`, the actual tool this repo
  ships). Real mov/mkv/wav seeds via system ffmpeg. Top-level isobmff box
  swaps at m=2/3/4 (30 trials each, including `moov`↔`mdat`): zero
  rejections, zero ASAN crashes at any m — ISO-BMFF top-level box order is
  unconstrained by spec and the mov demuxer already linear-scans for
  `moov` regardless of position, so whole-box reordering can't produce the
  hypothesized cliff (it would need to disturb `stco`/`stsz` entries
  *inside* moov relative to mdat's shifted absolute offset, not just box
  order). A ~3,700-exec real coverage-guided campaign against the same
  target found 0 crashes/timeouts; the isobmff box mutator (`_swap_pair`
  lives inside it) was rarely offered on this corpus (10/~3,700 execs —
  corpus composition, not an operator fault) but changed the buffer 70% of
  the time and found new edges on 100% of those changes (4/4) — a small
  sample, but real signal that swap-family mutations are not inert on
  ffmpeg's coverage, contrary to what the crash-only ASAN result alone
  would suggest.

**Verdict on the validity-cliff hypothesis: not supported, in the opposite
direction of what was assumed.** Two independent, mature real-world
parsers (`libsqlite3`, `libavformat`) respond to offset-table corruption
from this swap family by *silently degrading output* rather than
rejecting early. The actual risk to the fuzzer is not "fewer deep paths
from early rejection" — it's that a crash/coverage-only harness gets close
to zero signal from many of these mutations regardless of m, which is a
corpus-scheduling problem (the mutation looks like a boring no-op run),
not a coverage-depth problem. This should be weighed against finding 4
(diameter as accepted-corpus-entries) before treating k>0 as a net win:
diameter improves, but the concrete offset-table failure mode motivating
part of the original interest in k>0 does not manifest as hypothesized,
at least for these two formats.

**Not yet tested:** zip's central directory and a harness that actually
dereferences `stco` sample-table entries deep enough to hit a stale
absolute offset (both ffmpeg runs above may not exercise that path at
all — they didn't crash, but nothing here confirms they reached it,
either). Also untested: `_swap_tuple(m)` at m>2 for real edge yield —
the 100% figure above is m=2 rotation-shaped behavior (`_swap_pair` as it
exists today), not a generalized operator, since `_swap_tuple` has not
been implemented.

**Recommended form, revised:** `_swap_tuple(domain, rng, m)` with m drawn
from a truncated distribution, m=2 kept as a separate bandit arm (unchanged
from the original recommendation). The parity exclusion must cover every
odd m if the cheap rotation form is used, not just m=3 — either exclude
all odd m from rotation-only mode, or fall back to arbitrary-permutation
mode (costs the `!m` factor) specifically for odd m. Cheap alternative
unchanged: rotation of a contiguous window costs O(n) draws instead of
C(n,m), but see the parity caveat above before using it as-is.

### 10b — The exhaustive pool's `allow_bulk` gate is over-conservative

**Where:** `core/exhaustive_pool.py:317-321` (`randbytes`).

**The fix:** a per-call budget — `randbytes(n)` enumerable when `n <= 2`
by default, higher with explicit opt-in. This would let byte-level
exhaustiveness be tested for operators like `spectral_peak` (line 645)
and `de_bruijn_fill` (line 306) that always draw `randbytes` and do
something deterministic with the result.

**Risk:** combinatorially explodes if an operator draws multiple
`randbytes(N)` in one path. The default cap of `n <= 2` keeps the budget
manageable.

### 10c — The coin-flip idiom `rng.random() < p` defeats enumeration

**Where:** every operator that uses it instead of `rng.randint(0, N) < p*N`.

**The fix:** a census (already started in `tests/test_exhaustive_pool.py`
per the module docstring) listing each operator that needs to be
re-expressed as a bounded draw. Each re-expression is a one-line edit.

**Risk:** the cost is in test surface — every operator touched by the
census needs a regression test that asserts `ExhaustivePool.exhausted`.

### 10d — The pairwise Markov chain is first-order only

**Where:** `core/schedulers/monte_carlo.py:103-105`.

**The fix:** a second-order extension `transition_counts[prev2][prev][next]`
costing 150³ ≈ 3.4M counters. Captures "A then B then C" patterns.

**Risk:** combinatorial blow-up; would need a sparse representation
(only observed triples) to stay tractable.

### 10e — The grammar's full derivation space is unreachable

**Where:** `core/grammar.py:256-294` (`generate`).

**The fix:** a precomputed set of "grammar skeletons" — one per
derivation shape. Combinatorially this is the **set of minimum
representative derivations**, one per equivalence class of the grammar's
derivation relation. Requires the grammar to have a finite canonical-
form projection, which not all grammars have.

**Risk:** open rather than closed — some grammars don't have a finite
canonical form, and forcing one would break the existing parser.

### 10f — The cache for de Bruijn constructions is per-process only

**Status: implemented (2026-09-02).**

**Where:** `core/mutations/structured.py:288-298` (`de_bruijn_bytes`)
and `core/mutations/structured.py:390-407` (`de_bruijn_bits`).

**The fix:** persistence to disk so parallel fuzzing processes share the
sequences. Across one process the `@lru_cache(maxsize=32)` already
hits; across N processes each rebuilds.

**Risk:** low — the sequences are pure functions of `(k, n)`. A
content-addressed cache keyed by `(k, n)` would deduplicate across
processes automatically.

**What shipped:** `core/debruijn_cache.py`, following the existing
`cfg_cache.py` convention (XDG-aware `~/.cache/`,
`FUZZER_DISABLE_DEBRUIJN_CACHE` env override, atomic tempfile+replace
writes, best-effort with logged warnings on I/O failure), simplified
since the payload is raw `bytes` rather than a pickled class graph —
invalidation is one sha256 fingerprint over `_de_bruijn_symbols`'
source folded into the filename, so an algorithm edit just orphans the
old artifacts instead of needing runtime validation. Both functions
check the disk cache before falling back to construction and populate
it on a miss; the `@lru_cache` above each still absorbs repeat calls
within one process. Verified end-to-end with two real subprocesses
sharing one cache directory (`tests/test_debruijn_cache.py`) — the
second reads the artifact the first wrote, byte-identical output,
without recomputing.

### 10g — The `byte_shuffle` operator is registered but only the byte version exists

**Where:** `core/operator_registry.py:52` (`byte_shuffle`).

**The fix:** no change. The byte, chunk, and token shuffles (§3) are
already distinct arms. Joint coverage would hide the signal that
distinguishes them.

**Decision: closed** — the current separation is deliberate.

### 10h — `core/markov.py` state transfer across runs

The Markov state is the cumulative structure the corpus has discovered,
and discarding it at run start means every run rediscovers the same
n-gram statistics. Whether cross-run state transfer is currently
implemented is a read-the-source question I haven't verified.

**Where:** `core/markov.py` (not read directly in this analysis).

**The fix:** TBD pending read of `core/markov.py`.

### 10i — The operator pairwise Markov chain's bootstrap problem

**Where:** `core/schedulers/monte_carlo.py:241-264` (`select_op`).

**Revision (2026-09-02, verified by direct computation — see below):**
the fix as originally proposed here **does not change behavior** for any
realistic configuration. Downgraded from "gap to implement" to "verified
no-op, one real edge case identified instead."

The existing pair-score formula (`monte_carlo.py:250`) —
`pair_scores[op] = (count + 1) / (total + len(ops))` — is *already* a
uniform Dirichlet(α=1) prior. When `total == 0` (no transitions recorded
yet from this specific `prev_op`), every op gets the identical score
`1 / len(ops)`. Blended as `w * pair + (1 - w) * thompson` (line 255),
adding the same constant to every op's score under a positive multiplier
`(1 - w)` is a monotonic transform of the Thompson scores — it **cannot
change the argmax** for any blend weight `w < 1`. Verified numerically
across `w ∈ {0, 0.1, 0.5, 0.9, 0.999}`: the early-return path
(`prev_op not in transition_total` → pure Thompson) and the "always
compute the blend" path select the identical operator every time.

So the early-return at line 241 is not a bug suppressing an available
signal — it's a (redundant but harmless) special case of behavior the
formula already produces on its own. Seeding literal uniform prior
counts, as originally proposed, would reproduce the same no-op.

**The one real finding:** the equivalence breaks at the edge case
`pairwise_blend == 1.0` *and* `total == 0`. There, the early-return
correctly falls back to pure Thompson, while an "always blend" version
would collapse every op's score to the same constant and `max()` would
deterministically return `ops[0]` — an arbitrary, order-dependent
selection with zero informational basis. This is a narrower latent
edge case than what this section originally described (a config-only
corner: `pairwise_blend` is user/caller-supplied and 1.0 is an extreme,
probably-unintended value), not the "first ~1000 selections get no
benefit" claim.

**Status: closed as originally stated.** If the `pairwise_blend == 1.0`
edge case is worth guarding, the fix is a one-line clamp
(`min(pairwise_blend, some_max_below_1)` at the config boundary or a
`w == 1.0` special-case in `select_op`), not a transition-matrix prior —
scoped separately from this section since it's a different mechanism
than what was proposed here.

---

## 11. Verification — how to confirm each finding

This is the read-the-source list for any reviewer:

1. **§2 — C(n,2) swap duplication:** grep `_rng.sample(` /
   `rng.sample(` in `core/mutations/` — a plain grep on
   `list(range(` undercounts by 3 (`asf`, `riff`, `adts` skip the
   `list()` wrap) and misses the 2 non-plain-range shapes entirely
   (`sqlite`'s offset range, `arm`'s filtered candidate list). Read each
   of the 13 hits directly rather than trusting the pattern match.
2. **§3 — three shuffle arms:** grep `token_shuffle`/`chunk_shuffle`/
   `byte_shuffle` in `core/operator_registry.py`.
3. **§4 — `perm_lock` and `cycle_lock`:** read `core/mutations/structured.py`
   lines 500-600, confirm the 5 `_PERM_MODES` and 2 `_CYCLE_MODES`.
4. **§5.2 — pairwise Markov chain:** read
   `core/schedulers/monte_carlo.py:103-105, 320-323`.
5. **§5.4 — Cholesky path:** read `core/schedulers/monte_carlo.py:961-1009`
   for `correlated_select`, lines 1023-1162 for the matrix-UCB path.
6. **§6.3 — `SubtreePopulation` reservoir:** read `core/grammar.py:533-577`.
7. **§7 — `ExhaustivePool`:** read `core/exhaustive_pool.py` end-to-end,
   especially `_bounded` (line 195) and `shuffle` (line 266).
8. **§8.2 — `pagerank_credit`:** read `core/lineage.py:302-369`. Confirm
   the `1/n_children` divisor at line 357.
9. **§10 — gaps:** read the cited file:line and verify the gap exists.
   For 10a (swap-pair duplication), grep `core/mutations/` for the
   idiom and count occurrences.

For 10h (`core/markov.py` cross-run persistence) and 10i (pairwise
chain bootstrap), the cited file:line is the entry point but the
verification needs to walk the call graph — both gaps are
**inferences**, not confirmed by direct read in this handover. Confirm
before treating them as facts.

---

## 12. Status summary

- §§1-9 are analysis only, confirmed by direct file reads and the
  targeted grep at the start of the analysis.
- §10 is a list of gaps. Each gap is named with a file:line and the
  combinatorial primitive it would add or centralize. None are
  implementation proposals yet.
- §11 is the verification checklist for the next reviewer.
- §10a.1 (C(n,2+k) generalization) is analysis plus empirical validation
  against two real parsers (`libsqlite3`, `libavformat`) and a real
  coverage-guided fuzzing campaign. The validity-cliff hypothesis it set
  out to test is not supported by that data. `_swap_tuple` itself remains
  unimplemented; the parity caveat found there must be addressed (every
  odd m, not just m=3) before implementing the rotation-only form.
- No code changes proposed. The goal is to surface the combinatorial
  surface area so follow-up implementations can be scoped against §0's
  three-job litmus test.
