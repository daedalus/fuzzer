"""Line graph L(G) of a directed CFG: branch edges as nodes.

Each edge of G is a node of L(G); e -> f when head(e) == tail(f). AFL edge
coverage counts L(G)'s nodes, and edge-pair (``__AFL_NGRAM_K=3``) coverage
its arcs, so ``n_arcs`` is that space's size.

    G: 0 -> 1 -> 3        L(G): (0,1) -> (1,3)
       0 -> 2 -> 3              (0,2) -> (2,3)

L(G) is never built. Successors of e are the out-edges of head(e), so G's
out-CSR indexed by head(e) is L(G)'s adjacency; n_arcs = sum in*out per
node (a hub with 1k in, 1k out edges has 1M arcs, 2k edges).

Distances to target *edges* differ from block distances: (0,2) reaches
block 3 but never takes branch (1,3). Reverse BFS expands each node once:
when the first out-edge of u is reached at level d, every unreached
in-edge of u is at d + 1. O(V + E) instead of O(arcs).
"""

import numpy as np

from fuzzer_tool.core.horizon import _gather, _out_csr

UNREACHED = -1

# Frontiers up to this size take the scalar BFS step. 100k-edge chain:
# 1.75 s vectorized, 0.20 s hybrid; 2.7M-node random graph: parity.
_SCALAR_MAX = 32


class LineGraph:
    """Implicit line graph over parallel ``src``/``dst`` edge arrays."""

    def __init__(self, n_nodes: int, src: np.ndarray, dst: np.ndarray):
        if src.shape != dst.shape:
            raise ValueError("src and dst differ in shape")
        if src.size and (min(src.min(), dst.min()) < 0 or max(src.max(), dst.max()) >= n_nodes):
            raise ValueError("edge endpoint outside [0, n_nodes)")

        self._src = src.astype(np.int64, copy=False)
        self._dst = dst.astype(np.int64, copy=False)
        ids = np.arange(src.size, dtype=np.int64)

        # Edge ids grouped by tail (out) and by head (in).
        self._out_ptr, self._out = _out_csr(n_nodes, self._src, ids)
        self._in_ptr, self._in = _out_csr(n_nodes, self._dst, ids)

    @property
    def n_edges(self) -> int:
        return self._src.size

    @property
    def n_arcs(self) -> int:
        """Arcs of L(G): sum over nodes of in-degree x out-degree."""
        indeg = np.diff(self._in_ptr)
        outdeg = np.diff(self._out_ptr)
        return int(np.dot(indeg, outdeg))

    def successors(self, e: int) -> np.ndarray:
        """Edge ids f with tail(f) == head(e)."""
        h = self._dst[e]
        return self._out[self._out_ptr[h] : self._out_ptr[h + 1]]

    def arcs(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialized arcs (a, b); size n_arcs, so mind hubs."""
        lo, hi = self._out_ptr[self._dst], self._out_ptr[self._dst + 1]
        a = np.repeat(np.arange(self.n_edges, dtype=np.int64), hi - lo)
        return a, self._out[_gather(lo, hi)]

    def distances(self, targets: np.ndarray) -> np.ndarray:
        """Hops in L(G) from every edge to its nearest target edge.

        Raises:
            ValueError: a target is not an edge id.
        """
        dist = np.full(self.n_edges, UNREACHED, dtype=np.int64)
        if targets.size and (targets.min() < 0 or targets.max() >= self.n_edges):
            raise ValueError("target outside [0, n_edges)")

        expanded = np.zeros(self._out_ptr.size - 1, dtype=bool)
        frontier = np.unique(targets.astype(np.int64))
        dist[frontier] = 0
        level = 0
        while frontier.size:
            level += 1
            step = self._step_scalar if frontier.size <= _SCALAR_MAX else self._step
            frontier = step(frontier, level, dist, expanded)
        return dist

    def _step(self, frontier, level, dist, expanded) -> np.ndarray:
        """One BFS level, vectorized: tails of *frontier*, each expanded once."""
        nodes = np.unique(self._src[frontier])
        nodes = nodes[~expanded[nodes]]
        expanded[nodes] = True

        # An edge has one head, so pred holds no repeats.
        pred = self._in[_gather(self._in_ptr[nodes], self._in_ptr[nodes + 1])]
        pred = pred[dist[pred] == UNREACHED]
        dist[pred] = level
        return pred

    def _step_scalar(self, frontier, level, dist, expanded) -> np.ndarray:
        """``_step`` for a small frontier: numpy's ~16 us per level dominates there."""
        pred = []
        for u in {int(self._src[f]) for f in frontier.tolist()}:
            if expanded[u]:
                continue
            expanded[u] = True
            for e in self._in[self._in_ptr[u] : self._in_ptr[u + 1]].tolist():
                if dist[e] == UNREACHED:
                    dist[e] = level
                    pred.append(e)
        return np.array(pred, dtype=np.int64)

    def frontier(self, visited: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Edges leaving the visited blocks, nearest to a target first.

        Args:
            visited: Bool mask per node.
            targets: Target edge ids.

        Returns:
            (edge ids, distances); unreached last, ties by edge id.
        """
        cand = np.flatnonzero(visited[self._src] & ~visited[self._dst])
        d = self.distances(targets)[cand]
        key = np.where(d == UNREACHED, np.iinfo(np.int64).max, d)
        order = np.argsort(key, kind="stable")
        return cand[order], d[order]
