"""Articulation points with cut damage on the undirected view of a CFG.

One iterative Tarjan low-link pass (soksolv ``artic.go``) gives, per
block v:

- ``pieces[v]``: components v's component splits into when v is removed
  (0 for an isolated block, 1 when v is no cut vertex).
- ``damage[v]``: blocks in the largest part cut off from the entry when v
  is removed; 0 outside the entry's component.

    entry 0 - 1 - 2 - 3        node   0 1 2 3 4
              |                pieces 1 3 2 1 1
              4                damage 4 2 1 0 0

Direction is dropped: a block whose removal disconnects a region in the
undirected view also cuts every directed path into it, so ``damage`` is a
lower bound on the blocks only reachable through v. Iterative, no
recursion limit: ffmpeg's ICFG has 2.7M blocks.
"""

import numpy as np

from fuzzer_tool.core.horizon import _out_csr

# Packed undirected edge key: u << 32 | v (node counts stay below 2**31).
_KEY_SHIFT = 32
_KEY_MASK = (1 << _KEY_SHIFT) - 1


def _undirected_csr(n: int, src: np.ndarray, dst: np.ndarray) -> tuple[list[int], list[int]]:
    """Simple undirected adjacency: no self-loops, no parallel arcs."""
    u = np.concatenate([src, dst]).astype(np.int64, copy=False)
    v = np.concatenate([dst, src]).astype(np.int64, copy=False)
    keep = u != v
    keys = np.unique((u[keep] << _KEY_SHIFT) | v[keep])
    ptr, nbr = _out_csr(n, keys >> _KEY_SHIFT, keys & _KEY_MASK)
    return ptr.tolist(), nbr.tolist()


def cut_damage(
    n: int, src: np.ndarray, dst: np.ndarray, entry: int
) -> tuple[np.ndarray, np.ndarray]:
    """(pieces, damage) per node; see the module docstring.

    Raises:
        ValueError: *entry* is not a node (n > 0).
    """
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if not 0 <= entry < n:
        raise ValueError(f"entry {entry} out of range for {n} nodes")
    ptr, nbr = _undirected_csr(n, src, dst)

    # Python lists: per-element access is ~10x cheaper than on numpy arrays.
    disc = [0] * n  # 0: unvisited
    low = [0] * n
    sub = [1] * n  # DFS subtree size
    pieces = [0] * n
    damage = [0] * n
    parent = [-1] * n
    cur = ptr[:-1]
    end = ptr[1:]
    timer = 0

    # Entry's component first: only its cuts count as damage.
    for root in (entry, *range(n)):
        if disc[root]:
            continue
        in_entry = root == entry
        timer += 1
        disc[root] = low[root] = timer
        stack = [root]
        while stack:
            x = stack[-1]
            i = cur[x]

            # Next neighbour: descend into an unvisited one, else take its low.
            if i < end[x]:
                cur[x] = i + 1
                y = nbr[i]
                if not disc[y]:
                    timer += 1
                    disc[y] = low[y] = timer
                    parent[y] = x
                    pieces[y] = 1  # the part holding its parent
                    stack.append(y)
                elif y != parent[x] and disc[y] < low[x]:
                    low[x] = disc[y]
                continue

            # x finished: report to its parent.
            stack.pop()
            p = parent[x]
            if p < 0:
                continue
            if low[x] < low[p]:
                low[p] = low[x]
            sub[p] += sub[x]
            if low[x] < disc[p]:
                continue

            # x's subtree hangs off p alone: removing p cuts it loose.
            pieces[p] += 1
            if in_entry and sub[x] > damage[p]:
                damage[p] = sub[x]

    return np.array(pieces, dtype=np.int64), np.array(damage, dtype=np.int64)
