"""Betweenness centrality over a directed, unweighted graph (Brandes 2001).

Item #3 from the graph-theory survey
(``docs/handover/handover_dominator_gate_2026-09-15.md``): unlike dominance
(item #1, static/a-priori: which blocks gate every path from *entry*) and
min-cut (item #2, frontier-aware: which edges sever every remaining path
from the *current coverage frontier*), betweenness is a property of the
whole graph alone -- no target set or live frontier required. A block with
high betweenness sits on a disproportionate share of shortest paths between
*all* pairs of other blocks in the program, which is exactly the "central,
load-bearing branch" signal the K-Scheduler's ICFG-based centrality (see
``core/schedulers/katz.py``) approximates from *execution counts* rather
than pure structure -- this module answers the structural question those
counts are trying to estimate.

Brandes' algorithm: O(V*E) for unweighted graphs, replacing the naive
O(V^3) all-pairs-shortest-paths-then-accumulate approach with a single BFS
per source plus a backward accumulation pass, using the same
dependency-accumulation trick Dijkstra-based variants use for weighted
graphs (Brandes, *A Faster Algorithm for Betweenness Centrality*, Journal
of Mathematical Sociology 25(2), 2001).
"""

from collections import deque


def betweenness_centrality(
    n_nodes: int, edges: list[tuple[int, int]], normalized: bool = True
) -> list[float]:
    """Brandes' algorithm, directed, unweighted.

    ``edges`` are directed (u, v) node-index pairs; only forward traversal
    is used (no implicit symmetrization -- pass both (u, v) and (v, u) for
    an undirected graph).

    Score for node v = sum over all ordered pairs (s, t), s != v != t, of
    (# shortest s->t paths through v) / (# shortest s->t paths), i.e. the
    standard betweenness definition. Endpoints s and t never accumulate
    credit for their own pair. When ``normalized`` (default), divided by
    ``(n-1)(n-2)`` -- the number of ordered (s, t) pairs not involving a
    given v -- so scores land in [0, 1] and are comparable across graphs of
    different size; raw (unnormalized) sums are returned unchanged for
    ``n_nodes <= 2`` since there is no such pair to normalize by.
    """
    if n_nodes <= 0:
        return []
    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].append(v)

    C = [0.0] * n_nodes
    for s in range(n_nodes):
        # Single-source BFS collecting, for every node w: its distance
        # from s, the number of shortest s->w paths (sigma), and the
        # immediate predecessors on some shortest path (P).
        S: list[int] = []
        P: list[list[int]] = [[] for _ in range(n_nodes)]
        sigma = [0.0] * n_nodes
        sigma[s] = 1.0
        dist = [-1] * n_nodes
        dist[s] = 0
        queue = deque([s])
        while queue:
            v = queue.popleft()
            S.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        # Backward accumulation in reverse BFS order: delta[v] is v's
        # total dependency on s as a source, folded in from its
        # successors on shortest paths before v itself is folded into
        # its own predecessors.
        delta = [0.0] * n_nodes
        while S:
            w = S.pop()
            coeff = (1.0 + delta[w]) / sigma[w] if sigma[w] else 0.0
            for v in P[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                C[w] += delta[w]

    if normalized and n_nodes > 2:
        scale = 1.0 / ((n_nodes - 1) * (n_nodes - 2))
        C = [c * scale for c in C]
    return C
