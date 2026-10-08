"""WaveGrid._find_min_entropy scanned every cell in Python per observation.

Each collapse step computed every open cell's entropy with several numpy
calls and drew one tie-break float per cell: O(n^2) numpy calls per 1D row,
0.93 s for a 64x64 BMP (47.8 s per 3k --hail-mary execs). Now one pass:
counts in bulk, entropy once per distinct mask (same _entropy), tie-break
floats from RandPool.random_sequential (the exact random() stream).
"""

import numpy as np

from fuzzer_tool.core.rand_pool import _POOL_ENTRIES, RandPool
from fuzzer_tool.core.wfc import AdjacencyTable, Tile, WaveGrid


def test_random_sequential_matches_scalar_stream():
    """Across pool refills: same values, same stream after."""
    for start, count in ((0, 5), (_POOL_ENTRIES - 3, 10), (17, 3 * _POOL_ENTRIES + 5), (0, 0)):
        a, b = RandPool(3), RandPool(3)
        for _ in range(start):
            a.random()
            b.random()
        assert b.random_sequential(count) == [a.random() for _ in range(count)]
        assert a.random() == b.random()


class _OldGrid(WaveGrid):
    def _find_min_entropy(self):
        """The pre-change per-cell scan, verbatim."""
        min_entropy = float("inf")
        best_idx = None
        for i in range(self.n):
            row = self.superpositions[i]
            count = int(np.count_nonzero(row))
            if count == 0:
                self.contradiction = True
                return None
            if count == 1:
                continue
            entropy = self._entropy(row) + self._rng.random() * 1e-9
            if entropy < min_entropy:
                min_entropy = entropy
                best_idx = i
        return best_idx


def _tiles_adj(seed: int, n_tiles: int):
    rng = RandPool(seed)
    tiles = [Tile(name=bytes([i]), weight=1 + rng.randint(0, 5)) for i in range(n_tiles)]
    adj = AdjacencyTable()
    for i in range(n_tiles):
        for j in range(n_tiles):
            if rng.random() < 0.55:
                adj.add_forward(bytes([i]), bytes([j]))
    return tiles, adj


def _solve(cls, seed: int, width: int, height: int = 1):
    tiles, adj = _tiles_adj(seed, 6 + seed % 9)
    g = cls(tiles, adj, width=width, height=height)
    return g.run(seed=seed, max_restarts=2, ac3_budget=2000), g.contradiction


def test_old_matches_itself():
    """Control (Hard Rule 46)."""
    assert _solve(_OldGrid, 4, 40) == _solve(_OldGrid, 4, 40)


def test_regression_wfc_min_entropy_vectorized():
    for seed in range(40):
        for width in (2, 17, 64):
            assert _solve(WaveGrid, seed, width) == _solve(_OldGrid, seed, width)


def test_contradiction_path_draws_like_old():
    """Adversarial: a dead cell mid-row stops the scan after the same draws."""
    tiles, adj = _tiles_adj(1, 4)
    grids = [cls(tiles, adj, width=12) for cls in (WaveGrid, _OldGrid)]
    for g in grids:
        g._rng.reseed(9)
        g.superpositions[5, :] = False
        assert g._find_min_entropy() is None and g.contradiction
    assert grids[0]._rng.random() == grids[1]._rng.random()
