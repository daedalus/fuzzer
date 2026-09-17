# Betweenness centrality over the ICFG (survey item #3)

**Status:** implemented, standalone diagnostic utility -- same status as
`target_difficulty.py`, `mincut.py`, and (until its own follow-up) the
dominator gate before it. Not wired into any scheduler, `distance.py`, or
the CLI. HEAD at time of writing: `7b7fc10`.

## Trigger

Item #3 from the original graph-theory survey
(`docs/handover/handover_dominator_gate_2026-09-15.md`), picked up as the
next self-contained step per `handover_mincut_2026-09-15.md`'s own
"suggested next steps": unlike dominance (item #1, static/entry-relative)
and min-cut (item #2, frontier/target-relative), betweenness centrality
needs no target set or live coverage frontier at all -- it's a property
of the whole-program ICFG's structure alone. A block with high
betweenness sits on a disproportionate share of shortest paths between
*all* pairs of other blocks, which is the structural signal the
K-Scheduler's Katz centrality (`core/schedulers/seed_katz.py`) is estimating
empirically from execution counts rather than computing directly from
graph structure.

## What was built

### `core/centrality.py` (new module)

`betweenness_centrality(n_nodes, edges, normalized=True) -> list[float]`
-- Brandes' algorithm (Brandes, *A Faster Algorithm for Betweenness
Centrality*, 2001): one BFS per source node plus a backward
dependency-accumulation pass, O(V·E) total instead of the naive
O(V³) all-pairs-shortest-paths approach. Directed, unweighted (matches
the ICFG's own edge model -- no implicit symmetrization; pass edges both
ways for an undirected use case). Normalizes by `(n-1)(n-2)` (the count
of ordered pairs not involving a given node) when `n_nodes > 2`, skipped
below that to avoid a division by zero for `n <= 2`.

10 tests in `test_centrality.py`, every expected score hand-derived from
the definition rather than asserted against the implementation's own
output (same discipline as `test_mincut.py`'s Menger's-theorem
expectations): a directed path (single shortest path per pair), a
diamond (parallel shortest paths splitting credit 0.5/0.5 -- this is the
case that actually exercises Brandes' accumulation step rather than a
simpler "count paths" approach), a star graph (every score exactly 0 --
guards against wrongly crediting a hub for its own outgoing edges), an
isolated node alongside a live component (must score 0 without crashing
its own trivial BFS), and edge cases (empty graph, n=1, n=2's
division-by-zero guard, a self-loop that must not distort anything).

### `core/icfg.py` integration

New method `InterproceduralCFG.centrality_scores(normalized=True) ->
dict[address, float]` -- the address-level wrapper, mirroring
`bottleneck_edges()`'s style but simpler: no hit-set/target-set
parameters at all, since betweenness needs none. Every node in the ICFG
gets an entry (including 0.0 ones) -- unlike `bottleneck_edges`, which
silently drops addresses it has no node for, there's no "unmapped
address" concept here since the caller supplies no addresses to resolve.

4 tests in `test_icfg_centrality.py`, reusing `test_icfg_bottleneck.py`'s
exact diamond-plus-tail graph (`0x10 -> {0x20,0x30} -> 0x40 -> 0x50`) so
the two wrappers are directly comparable on the same structure, plus one
cross-check test: the min-cut bottleneck for `hit={0x10} ->
target={0x50}` on this graph is the single edge `(0x40, 0x50)`, and
`0x40` independently comes out as the highest-betweenness node -- two
different algorithms agreeing on the same structural choke point from
different definitions, which is some evidence (not proof) that both are
computing something meaningful rather than two independently-buggy
outputs that happen to both compile.

Also sanity-checked against a real compiled ELF (gcc, not a hand-built
graph): `target_fn`'s blocks came out with the highest scores in a
5-function/12-symbol binary, ahead of `main` and unrelated glibc startup
functions -- unsurprising for a program whose only interesting logic is
in `target_fn`, but worth having actually run rather than trusting the
synthetic tests alone.

## Design decisions

- **Betweenness only, not closeness.** The mincut handover's suggested
  next step named both ("Betweenness/closeness centrality"). Scoped down
  to betweenness alone to keep this patch a single well-tested
  algorithm rather than two -- closeness (sum of shortest-path distances
  to all other reachable nodes, inverted) is a smaller follow-on if
  wanted, and would reuse the same per-source BFS distance array this
  module already computes internally (just discarded after each source
  today).
- **Directed, no symmetrization.** The ICFG's edges are inherently
  directed (control flow doesn't reverse), so betweenness is computed
  over directed shortest paths, consistent with how `bottleneck_edges`
  and the horizon/Katz machinery already treat this graph.
- **Standalone, not wired into K-Scheduler.** Wiring this into
  `KatzChannel`/`schedulers/seed_katz.py` would mean choosing how to combine a
  purely-structural score with the existing execution-count-based Katz
  signal (replace it? blend it? use it only for cold-start before enough
  executions accumulate?) -- a design question, not a graph-theory one,
  and deliberately left open rather than guessed at here.

## Suggested next steps (not done here)

1. Closeness centrality -- same per-source BFS this module already runs,
   just a different reduction over the distance array.
2. Steiner trees / articulation points -- the two remaining items from
   the original four-item graph-theory survey.
3. Decide whether/how to blend this with K-Scheduler's execution-count
   Katz signal, now that both a structural and an empirical centrality
   measure exist side by side.
4. `gate_bonus` (dominator-gate work) is still unvalidated against a
   real target/campaign -- unrelated to this patch, but still the
   oldest open item across all three rounds of this survey now.
5. `bottleneck_edges`'s `hit_addrs` is still not wired to live coverage
   (`edge_tracker.py`) -- also still open from the mincut round.

## Files changed

- `src/fuzzer_tool/core/centrality.py` (new, 90 lines)
- `src/fuzzer_tool/core/icfg.py` (`centrality_scores()` method, import of
  `betweenness_centrality` -- no change to any existing method's
  behavior)
- `tests/test_centrality.py` (new, 116 lines, 10 tests)
- `tests/test_icfg_centrality.py` (new, 69 lines, 4 tests)
