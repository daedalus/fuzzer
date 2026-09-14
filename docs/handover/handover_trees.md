# Tree mutators: Dyck-path analysis and proposed improvements

**Date:** 2026-09-14
**Scope:** `src/fuzzer_tool/core/tree_mutator.py` (delimiter-based mutator) and
`src/fuzzer_tool/core/grammar.py` (`TreeNode` / `TreeMutator` / `SubtreePopulation`)
**Status:** §3.1, §3.2, §3.3 implemented (see §3/§6). §3.4 verified already
handled elsewhere — no change needed. §3.5 reconsidered and *not*
implemented as a hard cap — see §3/§6 for why.

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

## 7. References

- GRIIN (ASE '23) and Grammarinator×AFL++ (2026) — subtree-population
  crossover, cited in `docs/DEEP_DIVE.md:46`.
- Koza, *Genetic Programming* — internal/leaf crossover-point bias
  (90/10 split), motivating §3.2.
- Reflection principle for Dyck-path counting (`C_n = C(2n,n) - C(2n,n+1)`)
  — background for the Catalan growth-rate argument in §2.
