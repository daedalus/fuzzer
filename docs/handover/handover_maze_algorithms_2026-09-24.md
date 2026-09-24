# Handover: relevance of the jamisbuck.org maze algorithms (2026-09-24)

Source: https://www.jamisbuck.org/mazes/ (Recursive Backtracking, Eller,
Kruskal, Prim, Recursive Division, Blobby Subdivision, Aldous-Broder, Wilson,
Houston, Hunt-and-Kill, Growing Tree, Growing Binary Tree, Binary Tree,
Sidewinder). Clone HEAD at time of writing: `29610365`.

Analysis only. No source changes. Nothing in this document has been measured;
every claim about the fuzzer below was read off the code, every claim about
effect on discovery is a hypothesis.

## Summary

Most of the page is vocabulary and diagnostics, not portable code. Four items
map onto something real. Ranked by cost/benefit:

| # | Item | Kind | Cost | Gate |
|---|------|------|------|------|
| 1 | Lineage shape metrics (leaf fraction, corridor fraction, max depth) | diagnostic | ~40 lines + tests | none |
| 2 | `newest` seed arm (Growing Tree "newest" policy) | new seed arm | small | A/B via `tools/bench_paired.py` |
| 3 | Growing Tree weights as a sweep axis for `bench_paired.py` | benchmark | small | needs 2 |
| 4 | Max-weight spanning tree (Kruskal + union-find) for Ball-Larus / edge_id P1-2 | tool | small | P1-2 verdict positive |

Everything else is either already covered or has no consumer (see "Low / none").

This overlaps `handover_FINDINGS.md` P3-5 (Growing Tree / Growing Forest,
Houston), still OPEN. That entry is vocabulary-level and asks a gating question
("does Growing Tree genuinely subsume two of our schedulers?") to be answered on
paper first. Section 1 answers it.

## 1. Growing Tree <-> seed selection (answers the P3-5 gating question)

Growing Tree is one loop with one knob: keep a frontier list, pick a cell by
policy, carve to an unvisited neighbour, drop the cell from the frontier when it
has none left. Policies: `newest` (= recursive backtracker / DFS), `oldest`
(= BFS-like, minimal corridors), `random` (~ Prim), `middle`, and
weighted mixtures (`random:50,newest:30`).

Mapping: corpus = frontier, seed picker = policy.

| Policy | Existing arm in `_pick_seed_elo` (services/seed_picker.py) |
|--------|-------------------------------------------------------------|
| oldest | `round_robin` (`SeedRoundRobinScheduler`, registration order) |
| random | `weighted` (weighted random), uniform fallbacks |
| newest | **none.** No recency or LIFO arm exists (grep for newest/recency in seed_picker.py returns nothing) |
| middle | none, and no obvious fuzzing meaning |
| mixture | Elo arbitration over arms is already a mixture policy, coarser and outcome-driven |

Answer to the gating question: **partially.** `round_robin` and `weighted`
correspond to two Growing Tree points; the DFS point is missing. Growing Tree
does not subsume the signal-driven arms (katz, tang, entropy_*, residual,
mcts): they use outcome signals, Growing Tree's policies are signal-free
structure rules.

Where the analogy breaks: a maze cell **leaves** the frontier when it has no
unvisited neighbour. Our seeds are only reweighted, never retired on exhaustion.
"Retire when exhausted" is the interesting transfer, but it needs an exhaustion
signal per seed (deterministic stages complete, or fatigue). Not designed here.

Honest expectation: the value is a **benchmark axis** (sweep the mixture in
`bench_paired.py` instead of A/B-ing named classes), not a new scheduler. Elo
already mixes arms; measured Elo arbitration already loses ~13% of discoveries
vs. its best member (documented in `core/schedulers/op_consolidated.py`), so a
`newest` arm is worth adding only if it wins its own A/B.

Proposal:
1. Add `SeedNewestScheduler` (pick the most recently inserted live seed, or
   geometric-decay over insertion order to avoid pure lock-in; use
   `LineageTree._seq` order). Wire like `seed_round_robin.py`
   (`handover_seed_round_robin_2026-09-21.md` lists the wiring points).
2. Watch for lock-in: pure `newest` on a productive seed keeps hitting the same
   lineage. This is the same failure class as `handover_op_katz_lockin_fix` and
   `handover_op_kuramoto_lockin_fix`; read those first.
3. Sweep `p_newest` in {0, 0.3, 0.6, 1.0} against `weighted` and `round_robin`
   with `tools/bench_paired.py`. Off by default, out of `--hail-mary` until
   measured (same criterion as `gate_bonus` / `temperature_control`).

## 2. Lineage shape metrics (the maze characterisation table, applied)

`core/lineage.py::LineageTree` already has `strahler()` (branching complexity).
The maze table's other columns translate to cheap topology queries over the same
parent-pointer forest, using `nodes`, `_children`, `roots()`:

- **leaf fraction** (dead-end %): nodes with no children / live nodes.
- **corridor fraction** (loosely, river factor): nodes with exactly one child /
  live nodes. High = DFS-like long chains.
- **max depth** (diameter proxy): max `LineageNode.depth`.
- **mean maximal-chain length**: mean length of maximal unary chains.

Use: a run's lineage fingerprint says which Growing Tree policy the *effective*
scheduler behaved as (corridor-heavy = newest-like, bushy = random/Prim-like).
That makes item 1 (`newest`) checkable after the fact and gives a scheduler
comparison that does not depend on the edge-count reward. Complements
`subtree_weight` (productivity) and `strahler` (branching).

Falsifier: run `weighted` vs `round_robin` vs `newest` on one target; if the
metrics do not separate the three, they are not a useful fingerprint and the
idea is dropped.

Tests: extend `tests/test_lineage.py` with hand-built chain / star / balanced
trees (chain: corridor ~1, leaf ~1/n; star: leaf ~1).

## 3. Kruskal <-> spanning tree of the CFG (edge_id P1-2)

`handover_edge_id_axis_2026-09-18.md` (rank 108 of 445 edge-count coordinates,
sparse unit-coefficient relations consistent with Kirchhoff conservation) says
Ball-Larus needs only the **complement of a spanning tree** instrumented, and
gates on P1-2 (empirical relations vs. the incidence matrix from
`core/icfg.py`).

If P1-2 is positive, the tool needed is a **maximum**-weight spanning tree with
edge weight = observed hit count (hot edges in the tree = not instrumented,
cold chords instrumented). That is textbook Kruskal + union-find over the
undirected ICFG, with a virtual exit->entry edge so flow conservation closes.
Nothing here is maze-specific and the maze page adds no algorithmic content
beyond the textbook. Union-find already exists at least in `core/lineage.py` (inline DSU in the
batch-LCA walk) and `core/overlap_density.py` (a disjoint-set class, used for
LSH clustering); crash_metadata.py also merges by union-find (not checked for a
separate implementation). If this lands, reuse or factor one shared helper
rather than adding another copy.

Secondary use, same primitive: MST / single-linkage over MinHash-Jaccard
distance between seeds (cut the k-1 longest MST edges for k clusters). Sits next
to `core/mds_local_search.py` (Chan & Har-Peled disk-graph MDS) and
`PoissonDiskAdmission`. No evidence it beats those; do not build without a
concrete diversity question.

Not gated on anything else, but do not start before P1-2 has a verdict.

## 4. Aldous-Broder / Wilson / Houston <-> saturation (conceptual)

- Aldous-Broder is a cover-time random walk; its slow tail (last few cells) is
  the coupon-collector tail of rare-edge discovery. It also explains why
  `core/scaling_exponent.py` uses a random walk as its null model.
- Wilson starts at an unvisited node and loop-erased-walks until it hits the
  built tree. That is exactly the boundary structure of
  `core/horizon.py::HorizonGraph` (unvisited nodes with a visited parent,
  K-Scheduler).
- Houston (Aldous-Broder until N cells visited, then Wilson) names the ad-hoc
  saturation gate: cheap/undirected early, directed late, at the cost of the
  uniformity guarantee.

No code to port. Worth one paragraph in `docs/DEEP_DIVE.md` if that doc is
being maintained. Uniform-spanning-tree sampling itself has no consumer
(uniform trees are not a coverage objective).

## 5. Possible target-side use: graph and grid seed generators (unverified)

Recursive backtracker yields long corridors, i.e. deep DFS. Deep DFS is the
classic trigger for stack exhaustion in recursive graph traversals. Binary Tree
and Sidewinder produce degenerate but valid topologies; Wilson gives uniform
spanning trees; braiding (removing dead ends) adds cycles.

So maze algorithms would make **distinct-topology seed generators** for targets
that parse graphs or grids (DOT, GraphML, DIMACS, adjacency lists, tile maps).

Caveat: a quick grep found no graph-format generator or target in
`src/fuzzer_tool/core/format_generators.py` / `format_seed_generator.py` (the
`adjacency`/`dot` hits are unrelated: dominators, PNG tables, WFC). I did not
audit `targets/`. Only relevant if the user fuzzes graph/map consumers; if so,
fold into the generator work in `handover_generators_2026-09-20.md`
(P2-3 cold-start seed synthesis) rather than a standalone module.

## Low / none

- **Recursive division / Blobby subdivision:** recursive region splitting is
  already `ddmin` (`core/root_cause.py`), `field_map.py`, `fractal_partition.py`.
- **Eller:** row-streaming, O(width) memory generation. No streaming-generator
  consumer exists.
- **Wave Function Collapse:** already `core/wfc.py`. Maze generation is one
  application of the same constraint-propagation idea.
- **Uniform vs biased generation:** already solved for grammars by
  `Grammar.generate_boltzmann()` (uniform over derivation trees of a size). Same
  principle as Wilson (uniform over spanning trees), different structure class.
- **Bias-free vs uniform, "no" vs "never":** already adopted as vocabulary
  (P3-5 / algorithm-catalogue survey). "Never" (unreachable) is the defect class
  behind the Hierarchical scheduler dropping runtime-registered operators.

## What was not verified

- No measurement of any kind. The `newest` arm and the lineage fingerprint are
  hypotheses; both have stated falsifiers above.
- Did not audit `targets/` or every generator for graph formats (section 5).
- Did not read the linked blog posts beyond the index page; algorithm
  descriptions above are the standard ones. The page's own summary text (e.g.
  Houston = Aldous-Broder then Wilson, non-uniform but faster; Growing Tree
  string syntax) is what was quoted.
- "Corridor fraction" is my proxy, not Buck's exact "river factor" definition.

## Suggested order

1. Section 2 (lineage metrics). Small, testable, unblocks judging the rest.
2. Section 1 `newest` arm + `bench_paired` sweep.
3. Section 3 only after edge_id P1-2 has a verdict.
4. Sections 4-5: doc paragraph / defer until a graph-format target exists.
