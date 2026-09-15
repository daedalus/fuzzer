# Tree mutators: Dyck-path analysis and proposed improvements

**Date:** 2026-09-14
**Scope:** `src/fuzzer_tool/core/tree_mutator.py` (delimiter-based mutator) and
`src/fuzzer_tool/core/grammar.py` (`TreeNode` / `TreeMutator` / `SubtreePopulation`)
**Status:** §3.1, §3.2, §3.3 implemented (see §3/§6). §3.4 verified already
handled elsewhere — no change needed. §3.5 reconsidered and *not*
implemented as a hard cap — see §3/§6 for why. §7 records five further
opportunities found afterward (path-copying in `hierarchical_shrink`,
coarse-to-fine candidate order, canonical subtree hashing, Boltzmann
sampling, cycle-lemma generation) — none implemented yet.

## 1. Mathematical properties currently exploited

Both tree mutators sit on the same mathematical object: an ordered (plane)
tree, which is in bijection with a **Dyck path** — a lattice walk of U/D
steps that never dips below the x-axis — and with a **balanced parenthesis
string**. The properties actually load-bearing in the code:

- **Recursive size/depth invariants.** `TreeNode.size()` and `.depth()`
  (`grammar.py:537-546`) use the standard recursive identities
  (`size = 1 + Σ child sizes`, `depth = 1 + max child depth`). These bound
  output growth without a full byte scan.
- **Unique root-to-node paths.** `TreeNode._find_path()`
  (`grammar.py:572-582`) relies on trees (unlike general graphs/DAGs) having
  exactly one path from root to any node — `hierarchical_shrink()`
  (`grammar.py:956-1002`) uses this to relocate a node by index path after
  cloning.
- **CFG derivation-tree closure under same-nonterminal substitution.**
  `_tree_swap`, `_tree_splice`, `_tree_rule_sub` (`grammar.py:834-921`) filter
  substitution candidates by `node.rule` — any subtree derived from
  nonterminal `R` can replace any other derivation of `R` and the result
  stays grammatically valid. This is the type-closure property that GRIIN
  (ASE '23) and Grammarinator×AFL++ rely on, and it is why grammar-tree
  mutation never needs to re-validate syntax after mutating.
- **Dyck word ↔ tree bijection, executed as a parser.** `partial_parse()`
  (`tree_mutator.py:142-213`) is a stack-based bracket matcher: push a node
  on an opener (U step), pop on a matching closer (D step). The
  `if len(stack) > 1: stack.pop()` guard is exactly the Dyck-path
  nonnegativity constraint — an unmatched closer cannot pop past the root,
  so it falls through to the literal-append path instead of raising. This is
  *why* malformed/partial input degrades gracefully to a raw tail rather
  than needing a separate validity check.
- **Reservoir sampling (Algorithm R), bucketed per rule.**
  `SubtreePopulation.add()`/`.sample()` (`grammar.py:607-635`) keep a
  fixed-size, per-rule uniform sample of interior nodes across an unbounded,
  growing corpus — O(1) memory per rule, every harvested node retains equal
  selection probability.
- **Hierarchical decomposition for shrinking.** `hierarchical_shrink()`
  tries removing whole subtrees before falling back to byte-level reduction,
  collapsing the search space from O(bytes) to O(tree nodes) per round.

## 2. Catalan-number consequence

The number of distinct *shapes* of a tree built from `2n` matched
delimiters is the Catalan number `C_n = C(2n,n)/(n+1) ~ 4^n / n^1.5`
(Stirling asymptotics). This has two direct, verified consequences in the
code:

- `lightweight_tree_mutate()` (`tree_mutator.py:398-443`) refuses to mutate
  inputs shorter than 4 bytes and discards any mutation whose flattened
  result exceeds `max_len`. Given `C_n`'s exponential growth, this cap is
  necessary, not decorative — `mutate_tree_stutter` alone (2-64x
  duplication of an arbitrary subtree) can blow past any reasonable length
  in one call without it.
- Peaks in the Dyck-path view correspond to leaves in the tree
  (`_Node.is_leaf()` / `_collect_leaves()`, `tree_mutator.py:60-61,232-246`)
  — the shallowest, most disposable structure. `mutate_tree_del`/`_dup`
  sample uniformly over **all** nodes rather than biasing toward interior
  nodes, so leaf-heavy trees mutate closer to plain byte-fuzzing than
  structural fuzzing (see §3.2).

## 3. Findings / proposed improvements

Ranked by how directly the Dyck-path/tree math points at them. None of
these are implemented in this patch — this file is the record of the
analysis for follow-up work.

### 3.1 `_Node` has no `size()`/`depth()` — oversized mutations caught too late

`grammar.py`'s `TreeNode` has both methods; `tree_mutator.py`'s `_Node` has
neither. `mutate_tree_stutter` (`tree_mutator.py:296-313`) clones a subtree
and inserts it `n_reps` (2-64) times *before* any length check —
`lightweight_tree_mutate` only discovers the result is oversized after a
full `root.flatten()`. For a large subtree with `n_reps=64` this wastes a
clone+serialize pass on every rejected mutation.

**Proposal:** add a cheap `size()` (node count or cached byte-length) to
`_Node`, and pre-check `subtree_size * n_reps` against `max_len` before
cloning in `mutate_tree_stutter`.

**Status: implemented.** Added `_Node.size()` and `_Node.byte_length()`
(both iterative — `partial_parse`-generated trees can be 2000+ deep per
`TestDeepNesting`, so recursion was never an option here). `byte_length()`
mirrors `flatten()`'s exact accounting. `mutate_tree_stutter` gained an
optional `max_len` parameter that budgets `n_reps` against
`byte_length()` *before* cloning, instead of discovering the overrun after
a full `flatten()`. `lightweight_tree_mutate` now passes its `max_len`
through.

### 3.2 Uniform node selection ignores subtree size (GP sampling bias)

`_collect_nodes` (delimiter mutator) and `TreeNode.collect_interior()`
(grammar mutator) are sampled with plain uniform probability over the full
node list. Under Catalan-distributed random trees, node count concentrates
in small, shallow subtrees, so uniform selection over-samples trivial edits
and under-samples the few large, structurally interesting subtrees near the
root. This is the same bias documented in the genetic-programming
literature (Koza's 90/10 internal-vs-leaf crossover-point split exists
specifically to counteract it).

**Proposal:** weight `rng.choice`/`randrange` selection by `node.size()`
(or bucket by depth) instead of sampling the flat node list uniformly —
cheap once `TreeNode.size()` (already present) or the proposed `_Node.size()`
(§3.1) is available.

**Status: implemented.** Both mutators now sample size-weighted:

- `tree_mutator.py`: new `_collect_nodes_with_sizes()` computes every
  node's subtree size in one O(n) post-order pass (not O(n) per node,
  which is O(n²) worst case on a deep thin chain), and `_weighted_index()`
  does the cumulative-weight pick. `mutate_tree_del`, `_dup`, `_swap`, and
  `_stutter` all use it now.
- `grammar.py`: new module-level `_weighted_choice()` uses the existing
  `TreeNode.size()`. Wired into `_tree_swap`, `_tree_delete`,
  `_tree_duplicate`, `_tree_splice`, and `_tree_rule_sub` in place of the
  previous `self._rng.choice(targets)`.

One compatibility fix during implementation:
`test_subtree_population_crossover.py`'s `_FixedOpRng` test double only
implements `randint`, not `randrange` — `_weighted_choice` was switched to
`rng.randint(0, total - 1)` accordingly. All 503 tests in the
tree/grammar/subtree/tmin/structured slice pass
(`python3 -m pytest tests/ -k "tree or grammar or subtree or tmin or structured"`).

### 3.3 `mutate_tree_swap` has no type/delimiter awareness

Unlike `TreeMutator._tree_swap` in `grammar.py`, which is constrained to
same-`rule` nodes, `tree_mutator.py`'s `_swap_nodes`
(`tree_mutator.py:347-357`) swaps any two nodes regardless of delimiter
kind — a `"`-quoted node can swap with a `{`-braced one. This is plausibly
intentional (Radamsa-style structural chaos for unknown formats), but the
well-formedness guarantee differs from the grammar mutator's and is not
currently documented anywhere near the function.

**Proposal:** either document the intentional lack of type constraint
directly on `mutate_tree_swap`, or add an optional same-delimiter-only mode
for callers that want the stronger guarantee.

**Status: implemented (documented, not constrained).** Added a docstring
to `mutate_tree_swap` explaining explicitly that it is delimiter-type-agnostic
by design, why that's still round-trip safe (each node carries its own
open/close byte), and how it contrasts with `TreeMutator._tree_swap`'s
rule-constrained swap in `grammar.py`. Did not add a same-delimiter-only
mode — no caller currently needs the stronger guarantee, and adding an
unused option would be speculative.

### 3.4 `SubtreePopulation.add` re-walks the whole tree per corpus entry

`collect_interior()` is O(n) per call and runs for every harvested tree.
Fine at current corpus sizes; if entries grow large, harvesting could
become incremental (harvest only nodes touched since the last mutation),
mirroring the "last-harvested index" pattern already used for corpus-level
tracking in `services/operators.py::_op_grammar_tree_mutate`.

**Status: verified already implemented — no change made.** Re-reading
`_op_grammar_tree_mutate` (`services/operators.py:2674-2704`) more
carefully: it already tracks `self._subtree_pop_next_idx` and only calls
`.add()` on corpus entries added since the last call
(`for seed in corpus[next_idx:]`), with a reset guard if the corpus was
externally shrunk/replaced. The O(n)-per-tree cost inside `.add()` itself
is unavoidable and correct — you cannot harvest a tree's nodes without
visiting them once. This finding was based on reading `SubtreePopulation`
in isolation without cross-checking its only call site; the incremental
behavior it asked for already exists there. No code change; corrected
here so this doc doesn't send a future reader looking for a bug that
isn't present.

### 3.5 No explicit depth cap on stutter/duplicate

`mutate_tree_stutter`/`_tree_duplicate` grow breadth (siblings), not depth,
so neither can produce a pathologically deep tree in a single call — but
repeated application across many fuzzing rounds compounds. Catalan-random
tree height scales like `√n`, so an explicit depth cap (checked cheaply,
not via full serialization) would catch degenerate cases earlier than the
current byte-length-only check.

**Status: reconsidered — no hard cap added, added observability instead.**
Two things changed this from the original proposal:

1. **A real depth-growth path exists, but it's `swap`, not
   `stutter`/`duplicate`.** Those two only add siblings, confirmed
   correct. But `mutate_tree_swap` exchanges two nodes' *positions* —
   swapping a shallow, bushy subtree into a deep slot (and vice versa) can
   increase the tree's maximum depth substantially without changing total
   byte length at all, since it only reorders existing bytes. The existing
   `max_len` check in `lightweight_tree_mutate` cannot catch this, because
   `flatten()`'s output length is unchanged by a swap.
2. **Capping depth is not obviously correct for a fuzzer.** Deep nesting
   is a deliberate, valuable test case — recursive-descent parsers
   overflowing their stack on deeply nested input is a real, well-known
   bug class. `TestDeepNesting` (2000 levels) treats this as a supported
   scenario the mutator must handle, not a pathology to suppress. Adding a
   hard depth ceiling to `mutate_tree_swap` risks quietly removing the
   mutator's ability to produce exactly the inputs most likely to find
   that bug class.

Given that tension, this patch adds `_Node.depth()` (iterative, same
pattern as `size()`/`byte_length()`) purely as an observability primitive
— available for a future scheduler-level policy or telemetry — without
wiring it into any rejection logic. No mutation behavior changed.

## 4. Non-findings (things checked and found fine)

- `partial_parse`'s round-trip invariant (`flatten(partial_parse(x)) == x`)
  holds for unmatched delimiters — verified by
  `tests/test_tree_mutator.py::TestPartialParse`.
- `TreeMutator._tree_splice`'s fallback to `_tree_swap` when no donor
  subtree exists yet is correct and avoids a silent no-op
  (`grammar.py:873-905`).
- `hierarchical_shrink`'s path-based node relocation after cloning is sound
  because trees guarantee path uniqueness (§1).

## 5. Empirical check and a correction to the §2 growth claim

§2 asserts (via the classical de Bruijn–Knuth–Rice / Brownian-excursion
result) that average node depth in a **uniformly random** Catalan tree
scales as `Θ(√n)`. This was checked empirically against
`partial_parse()`'s actual output rather than left as an unverified
citation.

The first attempt used a naive generator: a greedy walk that opens a
bracket with fixed probability 0.55 whenever the budget allows, closes
otherwise. That is **not** a uniform sample over the `C_n` Dyck paths of
semilength n — it's a biased-coin walk conditioned to return to zero,
which systematically produces long unbroken runs of opens (0.55 > 0.5,
and the greedy process never redistributes its "budget" evenly across the
walk). Measured average-depth-to-`√n` ratio grew monotonically with n
(0.70 → 1.03 → 1.51 → 2.50 → 6.93 for n = 10/50/200/800/3200) instead of
staying flat — i.e. the generator's trees get disproportionately deeper
as n grows, which is a real fact about that generator, not about
`partial_parse()` or about Catalan-uniform trees.

Replaced it with the standard exact sampler: a Dyck path of size n
decomposes as `U <path of size k> D <path of size n-1-k>`, choosing k with
probability `C_k · C_{n-1-k} / C_n` (recursive Catalan decomposition,
implemented iteratively with an explicit stack — and with big-int
weights, since `C_3200` overflows a float well before that). With this
corrected sampler the ratio stayed in the 0.5-1.0 band across n =
10..3200 — noisy at one sample per n, but flat rather than diverging,
consistent with the `Θ(√n)` claim.

**Takeaway for future readers:** if anyone writes property-based tests or
benchmarks against these mutators using a "random balanced string"
generator, verify it's an exact/uniform Dyck-path sampler (rejection
sampling or the recursive decomposition above) before trusting depth or
size statistics from it — an intuitive greedy generator is very easy to
get subtly, systematically wrong in exactly the way that inflates depth
with n.

## 6. Summary of code changes in this patch

| File | Change |
|---|---|
| `core/tree_mutator.py` | Added `_Node.size()`, `.byte_length()`, `.depth()` (all iterative). Added `_collect_nodes_with_sizes()` and `_weighted_index()`. Rewrote `mutate_tree_del`/`_dup`/`_swap`/`_stutter` to sample size-weighted. `mutate_tree_stutter` gained an optional `max_len` budget check. `lightweight_tree_mutate` passes `max_len` through to `mutate_tree_stutter`. Documented `mutate_tree_swap`'s intentional lack of delimiter-type constraint. |
| `core/grammar.py` | Added module-level `_weighted_choice()`. Wired into `TreeMutator._tree_swap`, `_tree_delete`, `_tree_duplicate`, `_tree_splice`, `_tree_rule_sub` in place of uniform `self._rng.choice(...)`. |

No behavior change to `_op_grammar_tree_mutate` or `SubtreePopulation`
(§3.4 finding was incorrect — see above). No depth cap added (§3.5
reconsidered — see above).

Tests: `pytest tests/ -k "tree or grammar or subtree or tmin or
structured"` → 503 passed, 1 pre-existing/environmental failure
unrelated to this change (`test_regression_vpk_divide_by_zero.py` needs a
vendored FFmpeg checkout not present in this environment; confirmed it
fails identically on `HEAD` before this patch).

## 7. Further opportunities identified (not yet implemented)

While verifying §3.1-3.5's implementation, re-reading `hierarchical_shrink`
(`grammar.py:978-1024`) surfaced a further, concrete inefficiency, plus
some additional tree/combinatorics techniques worth recording for future
work. None of these are implemented in this patch.

### 7.1 `hierarchical_shrink` deep-clones the whole tree per candidate — O(n²) per round

For every candidate node it tries removing, `hierarchical_shrink` calls
`self._clone_tree(tree)` (line 1004) — a full deep clone of the *entire*
tree — then patches one node via `_find_path`. Since trees guarantee a
unique root-to-node path (§1), this doesn't need a full clone: only the
nodes **on the path** from root to the target need copying; every other
subtree can be shared by reference. This is the standard "path copying"
technique from purely functional/persistent data structures (Okasaki,
*Purely Functional Data Structures*). Cost per candidate drops from O(n)
to O(depth), turning a round from O(n²) to O(n·depth) — a large win for
bushy trees (depth ~ `√n`, per §5's empirical check), a smaller but still
real one for the deep chains this fuzzer deliberately produces to hit
stack-overflow bugs (§3.5).

**Proposal:** add a path-copying clone helper (copy nodes along
`_find_path`'s result, share the rest by reference) and use it in place
of `_clone_tree` inside `hierarchical_shrink`'s inner loop.

### 7.2 Candidate order in `hierarchical_shrink` isn't explicitly coarse-to-fine

Classic delta-debugging (ddmin) tries removing large chunks before small
ones, since early large cuts shrink the search space fastest. The
candidate list here (`tree.collect_interior()`, line 997) is tried in
whatever order the traversal returns, not explicitly sorted by size.

**Proposal:** `candidates.sort(key=lambda n: n.size(), reverse=True)`
before the loop — a one-line change now that `size()` already exists,
guaranteeing the biggest cuts are tried first every round.

### 7.3 Canonical subtree hashing (AHU algorithm) for structural dedup

The Aho–Hopcroft–Ullman tree-canonicalization algorithm computes an O(n)
canonical label for a subtree's shape (optionally folding in rule/content),
so structurally-identical subtrees from different parses hash identically.
`SubtreePopulation` currently reservoir-samples every interior node it
sees, including exact duplicates (e.g. the same small JSON object shape
recurring across many corpus entries) — those duplicates waste reservoir
slots on redundant donors. Hashing each subtree canonically and
skipping/down-weighting already-represented shapes would turn the
population into something closer to a *shape-coverage set* than a
size-biased random sample, the same idea Superion/Nautilus-style grammar
fuzzers use as a coverage signal parallel to code coverage.

**Status: implemented.** Added `TreeNode.canonical_hash()` — SHA-1 over
`rule` + (for leaves) raw `data`, or `rule` + the concatenation of
children's canonical hashes (for interior nodes). Deliberately *not*
sorted-child AHU (which hashes unordered trees for isomorphism testing):
these are ordered plane trees where child position is grammar-meaningful,
so swapping two children must change the hash, not collapse to the same
one. Added `TreeNode.collect_interior_hashes()` alongside it — hashes
every node bottom-up in one O(n) traversal and returns each interior
node's hash as a byproduct, avoiding the O(n²) trap of calling
`canonical_hash()` once per node from the outside (each call would
re-walk its own subtree).

`SubtreePopulation` now tracks, per rule, a shape-hash → count map for
whatever currently sits in its pool (parallel `_pool_hashes` list +
`_shape_counts` dict, updated on both insertion and reservoir eviction).
`add()` uses `collect_interior_hashes()` instead of `collect_interior()`
and skips harvesting a node outright if its shape is already represented
in that rule's pool — no reservoir slot spent, `_seen` counter still
advances so later distinct-shape nodes get correct reservoir odds.
Distinct shapes continue to compete via ordinary reservoir sampling.

Added `TestCanonicalHashing` (8 tests) to
`tests/test_subtree_population_crossover.py`: hash equality/inequality
under identical shape, differing leaf content, differing child order,
and differing rule label; the batched hasher agreeing with per-node
`canonical_hash()`; 500 structural duplicates collapsing to a pool of 1;
500 nodes across 20 real distinct shapes still filling the pool to
`max_per_rule`; and internal bookkeeping (`_shape_counts` sums to pool
length, `_pool_hashes` matches actual node hashes) staying consistent
after heavy reservoir churn. Full tree/grammar/subtree/tmin/structured
slice: 510 passed (502 prior + 8 new), same 2 pre-existing/environmental
failures as before this change (`test_tmin_minimizes_crash` needs the
package installed non-editably; `test_fix_applied_to_asan_tree` needs a
vendored FFmpeg checkout not present in this environment).

### 7.4 Boltzmann sampling to replace ad hoc recursive-descent generation in `grammar.generate()`

Directly connects to §5's finding: a naive greedy generator was shown to
sample Dyck paths non-uniformly (biased toward deep, thin shapes).
`grammar.generate()`'s recursive-descent-with-depth-cap is the same kind
of ad hoc process, generalized to a full grammar. Boltzmann samplers
(Duchon–Flajolet–Louchard–Schaeffer) derive per-rule branching
probabilities from the grammar's generating function so that, for a
target size n, every derivation tree *of that size* is equally likely —
a principled fix for the same bias class, grammar-wide rather than just
for balanced-bracket structures. Needs the grammar's generating function
(computed or estimated from the rule set); more machinery than §7.5, but
the right tool if genuinely unbiased size-n structural coverage matters
more than generation speed.

### 7.5 Cycle lemma as a cheaper exact Dyck-path generator

A lighter-weight alternative to the recursive Catalan-decomposition
sampler built for §5's experiment (which needed a full Catalan-number
table and big-int arithmetic to avoid float overflow at n=3200). The
cycle lemma (Dvoretzky–Motzkin) generates a uniformly random Dyck path in
O(n) with neither: take a uniformly random shuffle of n U's and (n+1)
D's; among its cyclic rotations, exactly one leaves the walk non-negative
throughout, and it's found in one linear pass via running minimum. Useful
if the fuzzer ever wants to synthesize new nested seeds (JSON/XML/
expression-like corpus entries) directly from the Dyck-path model rather
than through the grammar.

## 8. References

- GRIIN (ASE '23) and Grammarinator×AFL++ (2026) — subtree-population
  crossover, cited in `docs/DEEP_DIVE.md:46`.
- Koza, *Genetic Programming* — internal/leaf crossover-point bias
  (90/10 split), motivating §3.2.
- Reflection principle for Dyck-path counting (`C_n = C(2n,n) - C(2n,n+1)`)
  — background for the Catalan growth-rate argument in §2.
- Okasaki, *Purely Functional Data Structures* — path-copying / structural
  sharing, motivating §7.1.
- Aho, Hopcroft, Ullman — canonical tree labeling for isomorphism testing,
  motivating §7.3. Superion (Wang et al., S&P '19) and Nautilus (Aschermann
  et al., NDSS '19) — grammar/AST-shape coverage in fuzzing.
- Duchon, Flajolet, Louchard, Schaeffer — Boltzmann samplers for
  combinatorial structures, motivating §7.4.
- Dvoretzky and Motzkin — cycle lemma, motivating §7.5.
