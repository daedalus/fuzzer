"""Seed energy by line-graph distance to the least-tried open branches.

A frontier branch leaves a visited block for a block no seed reached:
the uncovered side of a decision the campaign already sits at. Ranked by
how often their tail block ran (Katz ``hit_counts``), the least-tried
quarter are the targets: hard branches, tried a million times, sink.

    visited 0 1 2 3          targets: 3->5 (block 3 ran once)
    0 -> 1 -> 2 -> 3 -> 5    energy: {3} 1, {2} g, {1} g^2, {0} g^3
         1 -> 4

One reverse BFS over the implicit line graph (``core/line_graph.py``)
gives every edge's hops to a target; a block takes its best out-edge;
energy is ``GAMMA ** hops`` of the seed's nearest covered block. Distances
are to branches, not blocks: a seed reaching block 5 by another branch
does not count as covering 3->5.

Blocks beyond ``MAX_HOPS`` score 0, so energy() reads only the hot
blocks out of a packed node mask (3.16M bits on ffmpeg), nearest hop
level first, and stops at the first level the seed covers: 0.77 ms ->
0.03 ms per seed (2.7M blocks, 155k hot). Refresh: 0.55 s, inside the
Katz recompute budget.
"""

import math

import numpy as np

from fuzzer_tool.core.icfg import InterproceduralCFG
from fuzzer_tool.core.line_graph import UNREACHED

# Energy decay per line-graph hop.
GAMMA = 0.5

# Hops past which a block scores 0 (GAMMA ** 16 ~ 1.5e-5).
MAX_HOPS = 16

# Share of frontier branches, least-tried tails first, taken as targets.
FRESH_FRAC = 0.25


class BranchFrontier:
    """Per-block energy toward the least-tried frontier branches."""

    def __init__(self, icfg: InterproceduralCFG):
        self._lg = icfg.line_graph()
        self._src = icfg.src
        self._dst = icfg.dst
        self._n = icfg.n_nodes
        self.targets = np.zeros(0, dtype=np.int64)

        # Blocks within MAX_HOPS of a target, sorted by hops; level i spans
        # _hot[_bounds[i]:_bounds[i + 1]] at _level_hops[i].
        self._clear()

    def _clear(self) -> None:
        self._hot = np.zeros(0, dtype=np.int64)
        self._bounds = np.zeros(1, dtype=np.int64)
        self._level_hops: list[int] = []

    @property
    def ready(self) -> bool:
        return bool(self.targets.size)

    def refresh(self, visited: np.ndarray, hits: np.ndarray) -> None:
        """Re-target from the visited-block mask and per-block hit counts."""
        frontier = np.flatnonzero(visited[self._src] & ~visited[self._dst])
        if not frontier.size:
            self.targets = frontier
            self._clear()
            return

        # Least-tried tails first; stable, so ties go to the lower edge id.
        k = max(1, math.ceil(FRESH_FRAC * frontier.size))
        order = np.argsort(hits[self._src[frontier]], kind="stable")
        self.targets = np.sort(frontier[order[:k]])

        # Block hops: its best out-edge (an edge leaves its tail).
        dist = self._lg.distances(self.targets)
        near = (dist != UNREACHED) & (dist <= MAX_HOPS)
        hops = np.full(self._n, MAX_HOPS + 1, dtype=np.int64)
        np.minimum.at(hops, self._src[near], dist[near])

        # Group hot blocks by hop level, nearest first.
        hot = np.flatnonzero(hops <= MAX_HOPS)
        lv = hops[hot]
        order = np.argsort(lv, kind="stable")
        self._hot, lv = hot[order], lv[order]
        self._bounds = np.flatnonzero(np.r_[True, lv[1:] != lv[:-1], True])
        self._level_hops = lv[self._bounds[:-1]].tolist()

    def energy(self, mask: bytes) -> float:
        """Best block energy in a packed little-endian node mask.

        Raises:
            ValueError: mask holds fewer than n_nodes bits.
        """
        if len(mask) * 8 < self._n:
            raise ValueError("mask shorter than the node count")
        b = np.frombuffer(mask, dtype=np.uint8)
        for i, h in enumerate(self._level_hops):
            blocks = self._hot[self._bounds[i] : self._bounds[i + 1]]
            if ((b[blocks >> 3] >> (blocks & 7)) & 1).any():
                return GAMMA**h
        return 0.0
