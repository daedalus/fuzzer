"""Kohonen self-organizing map of seed coverage (report only).

    seed -> edges ──► 0/1 matrix, informative edges only ──► X (seeds x edges)
    X ──► PCA linear init ──► batch SOM, Gaussian neighbourhood shrinking
          sigma: side/2 ──► SIGMA_END over EPOCHS
    map ──► bmu per seed, hits per unit, U-matrix, quantization/topographic error

Seeds on one unit cover alike; a high U-matrix value marks a border between
coverage regions. Example 3x3 hits: ``9 0 1 / 0 0 2 / 1 3 4`` -- nine seeds
share one unit (redundant), the rest spread along a coverage gradient.

Batch training, PCA init and sorted seed keys make the fit deterministic:
no RNG draw. Nothing here feeds scheduling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from fuzzer_tool.core.edge_clusters import REPORT_NNZ_BUDGET

MAX_CELLS = 4_000_000  # dense seeds x edges float32 (16 MB)
MAX_SIDE = 12  # grid rows/cols; keeps the text map readable
MIN_SIDE = 2
UNITS_PER_ROOT = 5  # Vesanto: ~5 * sqrt(n) units for n samples
EPOCHS = 30
SIGMA_END = 0.5
ADJACENT = math.sqrt(2)  # 8-neighbourhood on the grid


def grid_side(n: int) -> int:
    """Square grid side for *n* seeds, clamped to [MIN_SIDE, MAX_SIDE]."""
    side = round(math.sqrt(UNITS_PER_ROOT * math.sqrt(n)))
    return min(MAX_SIDE, max(MIN_SIDE, side))


def coverage_matrix(
    seed_edges: Mapping[str, object],
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """(seed keys, edge ids, 0/1 float32 matrix) over informative edges, or None.

    Edges every seed (or no seed) hits carry no position; dropped. Past
    MAX_CELLS, keeps the edges with the largest owner variance p(1 - p).
    """
    keys = sorted(seed_edges)
    n = len(keys)
    if n < 2:
        return None

    rows = [np.fromiter(seed_edges[k], dtype=np.uint64) for k in keys]
    flat = np.concatenate(rows)
    ids, owners = np.unique(flat, return_counts=True)
    keep = owners < n
    ids, owners = ids[keep], owners[keep]
    if ids.size == 0:
        return None

    cap = max(1, MAX_CELLS // n)
    if ids.size > cap:
        p = owners / n
        top = np.argsort(-(p * (1 - p)), kind="stable")[:cap]
        ids = ids[np.sort(top)]

    x = np.zeros((n, ids.size), dtype=np.float32)
    for i, row in enumerate(rows):
        cols = np.searchsorted(ids, row)
        cols = cols[(cols < ids.size) & (ids[np.minimum(cols, ids.size - 1)] == row)]
        x[i, cols] = 1.0
    return keys, ids, x


def fit_tracker(tracker) -> SeedSom | None:
    """Fresh map of *tracker*'s seed edges; None if empty, degenerate or over budget."""
    seeds = getattr(tracker, "seed_edges", None)
    if not seeds or sum(map(len, seeds.values())) > REPORT_NNZ_BUDGET:
        return None

    built = coverage_matrix(seeds)
    return None if built is None else SeedSom(built[2])


class SeedSom:
    """Batch-trained rectangular SOM over a seeds x edges matrix."""

    def __init__(self, x: np.ndarray, epochs: int = EPOCHS) -> None:
        side = grid_side(x.shape[0])
        self.shape = (side, side)
        self._x = x
        self._grid = np.indices(self.shape).reshape(2, -1).T.astype(np.float32)
        self.weights = self._linear_init()

        self._train(epochs)
        self._summarize()

    def _linear_init(self) -> np.ndarray:
        """Spread units over the plane of the top two principal components."""
        mean = self._x.mean(axis=0)
        _, s, vt = np.linalg.svd(self._x - mean, full_matrices=False)
        sd = np.zeros(2, dtype=np.float32)
        comps = np.zeros((2, self._x.shape[1]), dtype=np.float32)
        k = min(2, s.size)
        sd[:k] = s[:k] / math.sqrt(self._x.shape[0])
        comps[:k] = vt[:k]

        side = self.shape[0]
        coord = self._grid / (side - 1) * 2 - 1  # [-1, 1] per axis
        return mean + (coord * sd) @ comps

    def _sq_dists(self) -> np.ndarray:
        """Squared Euclidean seed-to-unit distances (seeds x units), >= 0."""
        w = self.weights
        d = (w * w).sum(1) - 2 * self._x @ w.T + (self._x * self._x).sum(1)[:, None]
        return np.maximum(d, 0)

    def _train(self, epochs: int) -> None:
        # Batch rule: w_u = sum_s h(u, bmu_s) x_s / sum_s h(u, bmu_s).
        diff = self._grid[:, None, :] - self._grid[None, :, :]
        grid_d2 = (diff * diff).sum(-1)
        sigma0 = self.shape[0] / 2
        units = grid_d2.shape[0]

        for t in range(epochs):
            sigma = sigma0 * (SIGMA_END / sigma0) ** (t / max(1, epochs - 1))
            h = np.exp(-grid_d2 / (2 * sigma * sigma))

            bmu = self._sq_dists().argmin(1)
            onehot = np.zeros((self._x.shape[0], units), dtype=np.float32)
            onehot[np.arange(bmu.size), bmu] = 1.0

            num = h @ (onehot.T @ self._x)
            den = h @ onehot.sum(0)
            live = den > 0
            self.weights[live] = num[live] / den[live, None]

    def _summarize(self) -> None:
        d2 = self._sq_dists()
        order = np.argsort(d2, axis=1, kind="stable")[:, :2]
        self.bmu = order[:, 0]
        self.hits = np.bincount(self.bmu, minlength=d2.shape[1]).reshape(self.shape)
        self.quantization_error = float(np.sqrt(d2[np.arange(self.bmu.size), self.bmu]).mean())

        gap = self._grid[order[:, 0]] - self._grid[order[:, 1]]
        self.topographic_error = float((np.hypot(gap[:, 0], gap[:, 1]) > ADJACENT).mean())
        self.umatrix = self._umatrix()

    def _umatrix(self) -> np.ndarray:
        """Mean weight distance from each unit to its 4-neighbours."""
        w = self.weights.reshape(*self.shape, -1)
        total = np.zeros(self.shape)
        count = np.zeros(self.shape)

        for axis in (0, 1):
            step = np.linalg.norm(np.diff(w, axis=axis), axis=-1)
            lo = [slice(None)] * 2
            hi = [slice(None)] * 2
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            total[tuple(lo)] += step
            total[tuple(hi)] += step
            count[tuple(lo)] += 1
            count[tuple(hi)] += 1
        return total / count
