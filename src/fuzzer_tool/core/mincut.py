"""Directed s-t minimum edge cut over the ICFG (Menger's theorem).

Dominator trees (``core/dominators.py``) answer "which single block gates
every path to the target." Min-cut answers a different, complementary
question with real teeth for directed fuzzing *in progress*: given the
set of blocks already exercised (the fuzzer's live coverage frontier) and
a set of target blocks, what is the smallest set of edges whose removal
severs every remaining path from frontier to target? Those edges are
exactly the branches worth spending mutation budget on right now —
flipping any single one of them is both necessary and (jointly)
sufficient to make further progress toward the target impossible without
it, which makes them a sharper prioritization signal than either BFS
distance or dominance alone once the frontier has moved past the function
entry (dominance answers "necessary from the very start"; min-cut answers
"necessary from here").

This is a genuinely different, tractable problem from the isoperimetric
question ``core/target_difficulty.py`` already flagged as intractable —
that one asks about the best cut over *every* size-n subset of the whole
graph (a global structural question); this asks for the min cut between
two fixed, given node sets, which is exactly what max-flow solves in
polynomial time.

Algorithm: Edmonds-Karp (BFS shortest augmenting path) over a unit-
capacity flow network — O(V*E^2) worst case but at most E augmentations
(unit capacities bound total flow by min(out-degree(sources),
in-degree(sinks))), and this is a diagnostic/opt-in query intended to run
on demand, not per-iteration. Simplicity over asymptotic optimality
(Dinic's, push-relabel); revisit if adversarial ICFGs make it hot, as
``dominators.py`` did (CHK -> Semi-NCA).

Multi-source/multi-sink is handled via a super-source/super-sink with
capacity effectively infinite (``len(edges) + 1``, provably larger than
any real max-flow value through this graph) so the reported cut can
never contain a virtual edge.

Status: standalone diagnostic utility, like ``target_difficulty.py``
before it — not wired into any scheduler, the CLI, or ``distance.py``.
See ``docs/handover/handover_mincut_2026-09-15.md``.
"""

from __future__ import annotations

from collections import deque

Edge = tuple[int, int]


def min_cut(
    n_nodes: int,
    edges: list[Edge],
    sources: set[int],
    sinks: set[int],
) -> tuple[int, set[Edge]]:
    """Minimum-cardinality edge set separating *sources* from *sinks*.

    Returns ``(max_flow_value, cut_edges)`` — by max-flow/min-cut duality
    ``max_flow_value == len(cut_edges)`` for this unit-capacity network.
    ``cut_edges`` is a subset of *edges* (never a virtual source/sink
    edge). Parallel edges between the same pair are supported — each
    contributes +1 capacity, so a pair with *k* parallel edges needs *k*
    edges removed to actually disconnect it, correctly reflected in both
    the flow value and how many of the *k* copies land in ``cut_edges``.

    Raises ``ValueError`` if *sources* and *sinks* overlap — a node can't
    be separated from itself, and letting an overlapping pair through
    would silently report a nonsensical infinite-capacity cut.
    """
    if not sources.isdisjoint(sinks):
        raise ValueError("sources and sinks must be disjoint")
    if not sources or not sinks:
        return 0, set()

    inf = len(edges) + 1
    super_source, super_sink = n_nodes, n_nodes + 1
    cap: dict[int, dict[int, int]] = {}

    def add(u: int, v: int, c: int) -> None:
        cap.setdefault(u, {})
        cap[u][v] = cap[u].get(v, 0) + c
        cap.setdefault(v, {}).setdefault(u, 0)

    for u, v in edges:
        add(u, v, 1)
    for s in sources:
        add(super_source, s, inf)
    for t in sinks:
        add(t, super_sink, inf)

    flow = 0
    while True:
        parent: dict[int, int] = {super_source: super_source}
        queue = deque([super_source])
        while queue:
            u = queue.popleft()
            if u == super_sink:
                break
            for v, c in cap.get(u, {}).items():
                if c > 0 and v not in parent:
                    parent[v] = u
                    queue.append(v)
        if super_sink not in parent:
            break
        bottleneck = inf
        v = super_sink
        path: list[Edge] = []
        while v != super_source:
            u = parent[v]
            bottleneck = min(bottleneck, cap[u][v])
            path.append((u, v))
            v = u
        for u, v in path:
            cap[u][v] -= bottleneck
            cap[v][u] += bottleneck
        flow += bottleneck

    reachable = {super_source}
    queue = deque([super_source])
    while queue:
        u = queue.popleft()
        for v, c in cap.get(u, {}).items():
            if c > 0 and v not in reachable:
                reachable.add(v)
                queue.append(v)

    cut_edges = {(u, v) for u, v in edges if u in reachable and v not in reachable}
    return flow, cut_edges
