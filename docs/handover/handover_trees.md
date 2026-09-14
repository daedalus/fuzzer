# Tree mutators: Dyck-path analysis and proposed improvements

**Date:** 2026-09-14
**Scope:** `src/fuzzer_tool/core/tree_mutator.py` (delimiter-based mutator) and
`src/fuzzer_tool/core/grammar.py` (`TreeNode` / `TreeMutator` / `SubtreePopulation`)
**Status:** Analysis complete, no code changes made yet — see §3 for proposed
follow-up work.

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

### 3.4 `SubtreePopulation.add` re-walks the whole tree per corpus entry

`collect_interior()` is O(n) per call and runs for every harvested tree.
Fine at current corpus sizes; if entries grow large, harvesting could
become incremental (harvest only nodes touched since the last mutation),
mirroring the "last-harvested index" pattern already used for corpus-level
tracking in `services/operators.py::_op_grammar_tree_mutate`.

### 3.5 No explicit depth cap on stutter/duplicate

`mutate_tree_stutter`/`_tree_duplicate` grow breadth (siblings), not depth,
so neither can produce a pathologically deep tree in a single call — but
repeated application across many fuzzing rounds compounds. Catalan-random
tree height scales like `√n`, so an explicit depth cap (checked cheaply,
not via full serialization) would catch degenerate cases earlier than the
current byte-length-only check.

## 4. Non-findings (things checked and found fine)

- `partial_parse`'s round-trip invariant (`flatten(partial_parse(x)) == x`)
  holds for unmatched delimiters — verified by
  `tests/test_tree_mutator.py::TestPartialParse`.
- `TreeMutator._tree_splice`'s fallback to `_tree_swap` when no donor
  subtree exists yet is correct and avoids a silent no-op
  (`grammar.py:873-905`).
- `hierarchical_shrink`'s path-based node relocation after cloning is sound
  because trees guarantee path uniqueness (§1).

## 5. References

- GRIIN (ASE '23) and Grammarinator×AFL++ (2026) — subtree-population
  crossover, cited in `docs/DEEP_DIVE.md:46`.
- Koza, *Genetic Programming* — internal/leaf crossover-point bias
  (90/10 split), motivating §3.2.
- Reflection principle for Dyck-path counting (`C_n = C(2n,n) - C(2n,n+1)`)
  — background for the Catalan growth-rate argument in §2.
