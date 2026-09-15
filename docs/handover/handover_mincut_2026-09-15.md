# Directed min s-t cut over the ICFG: bottleneck edges

**Status:** implemented, standalone diagnostic utility (matches
`target_difficulty.py`'s existing precedent — no production caller,
confirmed no other core module referenced it before this patch). Not
wired into any scheduler, `distance.py`, or the CLI. HEAD at time of
writing: `fbdf15e`. New commit on top: see the attached patch (base
`fbdf15e`, no push credentials in this environment).

## Trigger

Item #2 from the original graph-theory survey (see
`docs/handover/handover_dominator_gate_2026-09-15.md`), picked up now that
item #1 (dominator trees) landed, was fixed (the BFS-direction bug), and
was wired into the CLI by a follow-up session
(`docs/handover/handover_dominator_gate_followup_2026-09-15.md`).

Dominance answers "which block gates every path to the target **from the
function's entry**." That's a static, a-priori question. Min-cut answers
a different one with more teeth mid-campaign: given the blocks the fuzzer
has *already* exercised (the live coverage frontier) and the target
blocks, what is the smallest set of edges whose removal would sever every
remaining path from frontier to target? Those edges are exactly the
branches worth spending mutation budget on right now — a sharper,
frontier-aware complement to dominance once the frontier has moved past
the entry block.

This is explicitly **not** the same question `target_difficulty.py`
already flagged as intractable. That module's Φ(n) is a min-cut *over
every size-n subset* of the whole graph (an isoperimetric/global
structural question — genuinely hard). This is a min cut between two
*fixed, given* node sets, which is exactly what max-flow solves in
polynomial time (Menger's theorem). Different problem, despite the
overlapping vocabulary — worth being explicit about since it's an easy
mix-up (I checked `target_difficulty.py`'s docstring before starting, to
make sure this wasn't already covered and rejected there).

## What was built

### `core/mincut.py` (new module)

`min_cut(n_nodes, edges, sources, sinks) -> (flow_value, cut_edges)` —
Edmonds-Karp (BFS shortest augmenting path) over a unit-capacity flow
network. O(V·E²) worst case, but bounded by at most E augmentations since
capacities are all 1 (each bounds total flow by
`min(out-degree(sources), in-degree(sinks))`). Chose simplicity over
asymptotic optimality (Dinic's, push-relabel) for the same reason
`dominators.py` chose Cooper-Harvey-Kennedy over Lengauer-Tarjan: this is
a diagnostic query run on demand, not per-iteration, and correctness is
worth far more than constant-factor speed here.

Multi-source/multi-sink via a super-source/super-sink with capacity
`len(edges) + 1` (provably larger than any real max-flow value through
the graph, since each real edge is unit-capacity) — guarantees the
reported cut never contains a virtual edge. Parallel edges between the
same node pair are supported (each contributes +1 capacity;
`tests/test_mincut.py::TestParallelEdges` specifically checks that a
downstream single-edge bottleneck beats a redundant pair of parallel
edges upstream, i.e. the algorithm doesn't naively return "2" just
because there are two edges between some pair).

Raises `ValueError` if `sources`/`sinks` overlap (a node can't be
separated from itself — letting it through would silently report an
infinite-capacity, nonsensical cut). Returns `(0, set())` immediately if
either side is empty, matching `gate_bonus`'s validate-then-short-circuit
style in `distance.py`.

9 tests in `tests/test_mincut.py`: single path, diamond funnelling through
one shared downstream edge (cut size 1), two fully edge-disjoint paths
(cut size 2, per Menger), parallel edges, an already-disconnected graph
(cut size 0 — zero edges needed when there's no path to begin with),
multi-source/multi-sink independence (two unrelated chains must not leak
flow into each other through the super-source/sink), empty-side
short-circuits, and the overlap `ValueError`. Every expected cut is
hand-derived in the test docstring via Menger's theorem, not asserted
against the implementation's own output.

### `core/icfg.py` integration

New method `InterproceduralCFG.bottleneck_edges(hit_addrs, target_addrs) ->
set[tuple[int, int]]` — the address-level convenience wrapper. Translates
addresses to node indices via the existing `node_index` map, silently
drops any address with no matching node (matches this module's existing
tolerance elsewhere for stale/unmapped addresses — e.g.
`probe_key_node_table`'s handling of undecodable call sites), subtracts
any address appearing in both sets before calling `min_cut` (rather than
letting its `ValueError` propagate — a hit block that's also a target
block just means "already there," not an error), and translates the
returned cut back to address pairs.

4 tests in `tests/test_icfg_bottleneck.py`, building an
`InterproceduralCFG` directly (node_addrs/src/dst arrays) rather than via
a real ELF — the same hand-built-structure style already used for
`HorizonGraph` in `test_horizon.py`, since this is a pure graph-structure
question one level above `test_mincut.py`. One bug caught by this test
during development, worth flagging since it's an easy trap for whoever
next hand-builds an `InterproceduralCFG` in a test: **`src`/`dst` are
node *indices* into `node_addrs`, not raw addresses** — the first draft
of `_make_icfg()` in the test file passed addresses directly and every
assertion failed with an empty cut, because `bottleneck_edges` correctly
translated the *query* addresses to indices but then compared them
against edges that were still raw addresses under the hood.

## Design decisions

- **Standalone diagnostic, not wired into anything.** Same status as
  `target_difficulty.py` and (until its own follow-up) the dominator gate
  — grep confirms `target_difficulty.py` has zero production callers
  today, so this isn't a new pattern for this repo. Wiring bottleneck
  edges into actual mutation-energy scheduling needs a mapping from "ICFG
  edge" to "which input bytes/branch condition control taking it," which
  doesn't exist yet (that's closer to taint tracking or symbolic
  execution than graph theory, and a materially bigger project than this
  patch).
- **`hit_addrs` is a parameter, not something this patch sources.**
  Wiring this to the fuzzer's actual live coverage bitmap is deliberately
  left to whoever does the eventual integration — the bitmap's format and
  how it maps to block-start addresses lives in `edge_tracker.py`
  (2978 lines, several existing hashing/coverage-key schemes already in
  there) and picking the *right* one to feed this without understanding
  how it's used elsewhere in that module felt like more risk than this
  patch's scope justified. Left as an explicit "still open" item below
  rather than guessed at.
- **Edge cut, not vertex cut.** A vertex (block) cut would need node
  splitting (in-node/out-node with a unit-capacity edge between them) to
  reduce to the same max-flow machinery — genuinely a five-line change to
  `min_cut` if ever needed, but edges map more directly onto "which branch
  to prioritize mutating" than blocks do, so edge cut was the more
  directly actionable choice for this application.

## Suggested next steps (not done here)

1. Wire `hit_addrs` to the fuzzer's live coverage — the natural
   integration point once someone has traced through `edge_tracker.py`'s
   existing coverage-key scheme far enough to pick the right one.
2. Betweenness/closeness centrality on the raw ICFG (#3 from the original
   survey) — still open, and unlike min-cut, is a property of the graph
   alone (no frontier/target sets needed), so it's a smaller, more
   self-contained next step than actually wiring #2 above.
3. `gate_bonus` (from the dominator-gate work) is still unvalidated
   against a real target/campaign — unrelated to this patch but still the
   oldest open item across both rounds of this survey.

## Files changed

- `src/fuzzer_tool/core/mincut.py` (new)
- `src/fuzzer_tool/core/icfg.py` (`bottleneck_edges()` method, import of
  `min_cut` — no change to any existing method's behavior)
- `tests/test_mincut.py` (new)
- `tests/test_icfg_bottleneck.py` (new)
