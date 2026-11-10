"""Exact 1-D k-means by dynamic programming (faiss ``kmeans1d`` port).

Optimal weighted clustering of sorted scalars into ``k`` contiguous buckets,
no Lloyd iteration, no seed dependence. Used to compare the fixed AFL count
ladder (``core/count_class.py``) with data-driven bucket boundaries; report
only, nothing here changes edge ids.

    values  1  2  3 | 7  8 | 20 21 22 | 50      k=4, cost = sum of weighted
    starts  0         3     5            8      squared distance to the
                                                bucket's weighted mean

O(k n^2) time, O(n^2) memory for the cost table, so ``n`` is capped.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

MAX_POINTS = 2048  # cost table is n^2 float64: 32 MB at the cap


class Scale(Enum):
    """Space the squared error is measured in."""

    LINEAR = "linear"
    LOG = "log"


def to_scale(values, scale: Scale) -> np.ndarray:
    """Raw values to the clustering space (log2 matches the AFL ladder)."""
    v = np.asarray(values, dtype=np.float64)
    if scale is Scale.LINEAR:
        return v
    if (v <= 0).any():
        raise ValueError("log scale needs values > 0")
    return np.log2(v)


def _check(values, weights) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.ndim != 1 or v.shape != w.shape or v.size == 0:
        raise ValueError("values and weights must be equal-length, non-empty 1-D")
    if v.size > MAX_POINTS:
        raise ValueError(f"{v.size} points exceeds MAX_POINTS={MAX_POINTS}")
    if not np.isfinite(v).all():
        raise ValueError("values must be finite")
    if not (np.isfinite(w).all() and (w > 0).all()):
        raise ValueError("weights must be finite and > 0")
    if (np.diff(v) <= 0).any():
        raise ValueError("values must be strictly increasing")
    return v, w


def _centered(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Shift by the weighted mean: kills cancellation in Q - S^2/W."""
    return v - (w * v).sum() / w.sum()


def _cost_table(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """cost[i, j] = SSE of points i..j-1 as one bucket; inf unless i < j."""
    c = _centered(v, w)
    zero = np.zeros(1)
    cw = np.concatenate([zero, np.cumsum(w)])
    cs = np.concatenate([zero, np.cumsum(w * c)])
    cq = np.concatenate([zero, np.cumsum(w * c * c)])
    dw = cw[None, :] - cw[:, None]
    ds = cs[None, :] - cs[:, None]
    dq = cq[None, :] - cq[:, None]
    valid = dw > 0
    table = np.full(dw.shape, np.inf)
    table[valid] = np.maximum(dq[valid] - ds[valid] ** 2 / dw[valid], 0.0)
    return table


def optimal_buckets(values, weights, k: int) -> tuple[np.ndarray, float]:
    """Optimal ``k`` buckets: (index of each bucket's first point, cost)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    v, w = _check(values, weights)
    n = v.size
    if k >= n:
        return np.arange(n), 0.0

    table = _cost_table(v, w)
    best = table[0].copy()  # one bucket over the first j points
    back = np.zeros((k, n + 1), dtype=np.intp)
    for m in range(1, k):
        cand = best[:, None] + table
        back[m] = cand.argmin(axis=0)
        best = cand.min(axis=0)

    starts = []
    j = n
    for m in range(k - 1, 0, -1):
        j = int(back[m][j])
        starts.append(j)
    starts.append(0)
    return np.array(starts[::-1]), float(best[n])


def partition_cost(values, weights, starts) -> float:
    """SSE of the partition whose buckets begin at ``starts``."""
    v, w = _check(values, weights)
    s = np.asarray(starts, dtype=np.intp)
    if s.size == 0 or s[0] != 0 or s[-1] >= v.size or (np.diff(s) <= 0).any():
        raise ValueError("starts must be strictly increasing, begin at 0, stay in range")
    c = _centered(v, w)
    sw = np.add.reduceat(w, s)
    ss = np.add.reduceat(w * c, s)
    sq = np.add.reduceat(w * c * c, s)
    return float(np.maximum(sq - ss**2 / sw, 0.0).sum())
