"""Prüfer codes: labeled trees <-> integer sequences, and uniform random trees.

A tree on nodes 0..n-1 has a unique code of n-2 labels (Cayley: n^(n-2)
trees). Uniform codes therefore give uniform labeled trees, which serve as
unbiased nesting skeletons for structured generators (``tlv_reshape``).

Trees are parent arrays rooted at n-1 — the node Prüfer decoding never
removes — so decoding yields the rooting for free::

    code [3, 3, 3, 4]          5
    parent [3,3,3,4,5,ROOT]    |
                               4
                               |
                               3
                             / | \\
                            0  1  2

Both directions run in O(n) with the "pointer + fresh leaf" trick instead
of a heap: after removing a leaf, its neighbour becomes the next leaf iff it
just dropped to degree 1 below the scan pointer.
"""

from __future__ import annotations

from collections.abc import Sequence

ROOT = -1  # parent of the root node


def _check_code(seq: Sequence[int], n: int) -> None:
    for v in seq:
        if not 0 <= v < n:
            raise ValueError(f"label {v} outside [0, {n})")


def decode(seq: Sequence[int]) -> list[int]:
    """Prüfer code -> parent array on ``len(seq) + 2`` nodes, rooted at n-1.

    Raises:
        ValueError: A label is outside ``[0, n)``.
    """
    n = len(seq) + 2
    _check_code(seq, n)

    # Degree = 1 + occurrences in the code.
    degree = [1] * n
    for v in seq:
        degree[v] += 1

    parent = [ROOT] * n
    ptr = degree.index(1)
    leaf = ptr
    for v in seq:
        parent[leaf] = v
        degree[v] -= 1

        # v became a leaf below the pointer: it is the smallest leaf now.
        if degree[v] == 1 and v < ptr:
            leaf = v
            continue

        ptr += 1
        while degree[ptr] != 1:
            ptr += 1
        leaf = ptr

    parent[leaf] = n - 1
    return parent


def _check_parent(parent: Sequence[int]) -> None:
    """Raise unless *parent* is a tree rooted at n-1 (O(n) via path marking)."""
    n = len(parent)
    if parent[n - 1] != ROOT:
        raise ValueError("root must be node n-1")

    # state: 0 unvisited, 1 on current path, 2 reaches root.
    state = bytearray(n)
    state[n - 1] = 2
    for start in range(n):
        path = []
        v = start
        while state[v] == 0:
            state[v] = 1
            path.append(v)
            v = parent[v]
            if not 0 <= v < n:
                raise ValueError(f"node {path[-1]} has bad parent {v}")
        if state[v] == 1:
            raise ValueError(f"cycle through node {v}")
        for u in path:
            state[u] = 2


def encode(parent: Sequence[int]) -> list[int]:
    """Parent array rooted at n-1 -> Prüfer code of length n-2.

    Raises:
        ValueError: *parent* is not a tree rooted at n-1.
    """
    n = len(parent)
    if n == 0:
        raise ValueError("empty tree")
    _check_parent(parent)
    if n <= 2:
        return []

    degree = [1] * n
    degree[n - 1] = 0  # root has no parent edge
    for v in range(n - 1):
        degree[parent[v]] += 1

    seq = []
    ptr = degree.index(1)
    leaf = ptr
    for _ in range(n - 2):
        v = parent[leaf]
        seq.append(v)
        degree[v] -= 1

        if degree[v] == 1 and v < ptr:
            leaf = v
            continue

        ptr += 1
        while degree[ptr] != 1:
            ptr += 1
        leaf = ptr
    return seq


def random_tree(n: int, rng) -> list[int]:
    """Uniform random labeled tree on *n* nodes as a parent array rooted at n-1.

    Args:
        n: Node count (>= 1).
        rng: ``RandPool``-API source (Hard Rule 16).
    """
    if n < 1:
        raise ValueError(f"tree needs >= 1 node, got {n}")
    if n == 1:
        return [ROOT]
    return decode(rng.randint_list(0, n - 1, n - 2))


def children(parent: Sequence[int]) -> list[list[int]]:
    """Child lists, ascending, from a parent array."""
    kids: list[list[int]] = [[] for _ in parent]
    for v, p in enumerate(parent):
        if p != ROOT:
            kids[p].append(v)
    return kids


def edges(parent: Sequence[int]) -> list[tuple[int, int]]:
    """Undirected edge list ``(child, parent)``."""
    return [(v, p) for v, p in enumerate(parent) if p != ROOT]


def root_at(n: int, tree: Sequence[tuple[int, int]], root: int) -> list[int]:
    """Parent array of the tree given by *tree* edges, hung from *root* (BFS).

    Raises:
        ValueError: The edges do not form a spanning tree on n nodes.
    """
    if len(tree) != n - 1:
        raise ValueError(f"{len(tree)} edges cannot span {n} nodes")
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in tree:
        adj[a].append(b)
        adj[b].append(a)

    parent = [ROOT] * n
    seen = bytearray(n)
    seen[root] = 1
    queue = [root]
    for v in queue:  # queue grows while iterating: BFS without deque
        for u in adj[v]:
            if seen[u]:
                continue
            seen[u] = 1
            parent[u] = v
            queue.append(u)
    if len(queue) != n:
        raise ValueError("edges do not connect all nodes")
    return parent
