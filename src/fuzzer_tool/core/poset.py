"""Finite posets as successor lists: chain height and Dilworth width.

A poset on vertices ``0..n-1`` is given as ``succ[i]`` = the vertices strictly
above ``i``. Seed coverage under subset is the motivating case
(``tools/edge_matrix_modes.py``): ``succ[i]`` = seeds whose edge set strictly
contains seed i's.

    height  longest chain                 a < b < c          -> 3
    width   largest antichain             {a, b} incomparable -> 2

Dilworth: width = minimum number of chains covering the poset = n - M, M a
maximum matching of the bipartite graph (i on the left, j on the right) with
an edge per ``j in succ[i]``. Each matched pair links i to its successor in
one chain, so every match saves one chain::

    a -> c, b -> c, c -> d       match a-c, c-d      chains {a, c, d}, {b}

Width needs ``succ`` transitively closed (a strict order). On a DAG that is
not, the same count is the minimum *path* cover instead.

Matching is greedy then augmenting paths (Kuhn), not Hopcroft-Karp
(``core/sdr.py``): on seed posets of 200-2000 nodes the greedy pass leaves
few free vertices and Kuhn ran 10-80x faster; HK only won on adversarially
ordered successor lists (2.5 s vs 0.8 s at n = 2000, 500k comparabilities).
"""

from __future__ import annotations

from collections.abc import Sequence

Succ = Sequence[Sequence[int]]

_FREE = -1


def topo_order(succ: Succ) -> list[int]:
    """Vertices with every ``i -> j`` edge pointing forward (Kahn).

    Raises:
        ValueError: on a cycle (a self-loop included): no strict order has one.
    """
    n = len(succ)
    indeg = [0] * n
    for js in succ:
        for j in js:
            indeg[j] += 1

    order = [u for u in range(n) if not indeg[u]]
    for u in order:
        for j in succ[u]:
            indeg[j] -= 1
            if not indeg[j]:
                order.append(j)
    if len(order) != n:
        raise ValueError(f"cycle through {n - len(order)} vertices")
    return order


def chain_height(succ: Succ, order: Sequence[int] | None = None) -> int:
    """Number of elements in the longest chain; 0 for an empty poset.

    Args:
        succ: Strict successors per vertex.
        order: A known linear extension (e.g. seeds by popcount) skips the
            ``topo_order`` pass; it is trusted, not checked.

    Raises:
        ValueError: on a cycle, when *order* is not given.
    """
    if order is None:
        order = topo_order(succ)
    depth = [1] * len(succ)
    for i in order:
        d = depth[i] + 1
        for j in succ[i]:
            if d > depth[j]:
                depth[j] = d
    return max(depth, default=0)


def poset_width(succ: Succ) -> int:
    """Size of the largest antichain (Dilworth: n - maximum matching)."""
    match_r = _match(succ)
    return len(succ) - sum(1 for u in match_r if u != _FREE)


def chain_cover(succ: Succ) -> list[list[int]]:
    """Minimum partition into chains, each listed bottom-up.

    Raises:
        ValueError: on a cycle (its vertices would head no chain).
    """
    match_r = _match(succ)
    nxt = [_FREE] * len(succ)
    for v, u in enumerate(match_r):
        if u != _FREE:
            nxt[u] = v

    # A chain starts at every vertex nothing below is matched to.
    chains = []
    for head, below in enumerate(match_r):
        if below != _FREE:
            continue
        chain = [head]
        while nxt[chain[-1]] != _FREE:
            chain.append(nxt[chain[-1]])
        chains.append(chain)
    if sum(map(len, chains)) != len(succ):
        raise ValueError("cycle: some vertices head no chain")
    return chains


def _match(succ: Succ) -> list[int]:
    """Maximum bipartite matching i -> j in succ[i]; returns match_r (left per right)."""
    n = len(succ)
    match_r = [_FREE] * n

    # Greedy seed: first free successor. Most vertices end matched here.
    free = []
    for u in range(n):
        for v in succ[u]:
            if match_r[v] == _FREE:
                match_r[v] = u
                break
        else:
            if succ[u]:
                free.append(u)

    # Kuhn from each left vertex still free; seen[v] == start marks this search.
    seen = [_FREE] * n
    for start in free:
        _augment(succ, start, match_r, seen)
    return match_r


def _augment(succ: Succ, start: int, match_r: list[int], seen: list[int]) -> None:
    """Iterative DFS for an augmenting path from *start*; flips it when found.

    stack[k] is re-matched to path[k] on success. Iterative: a chain poset
    makes the path n deep.
    """
    stack, path = [start], []
    iters = {start: iter(succ[start])}
    while stack:
        u = stack[-1]
        v = next((w for w in iters[u] if seen[w] != start), None)
        if v is None:
            stack.pop()
            if path:
                path.pop()
            continue

        seen[v] = start
        path.append(v)
        owner = match_r[v]
        if owner == _FREE:
            for lu, rv in zip(stack, path, strict=True):
                match_r[rv] = lu
            return
        stack.append(owner)
        iters[owner] = iter(succ[owner])
