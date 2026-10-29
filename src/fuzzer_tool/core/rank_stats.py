"""Rank statistics with no scipy (Hard Rule 51): tied ranks, Spearman, rank AUC.

One home for what ``edge_matrix``, ``analyzer_coverage_regime`` and several
``tools/`` scripts each spelled out. Ties take their average rank everywhere:
ordinal ranks (``argsort(argsort(x))``) bias Spearman and AUC whenever scores
repeat, which hit counts and clipped scores routinely do.
"""

from __future__ import annotations

import math

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    """0-based ranks with ties given their average rank."""
    _uniq, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts
    return (first + (counts - 1) / 2.0)[inverse]


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho; 0.0 when either side is constant or there are under 3 points."""
    if len(x) < 3 or len(x) != len(y):
        return 0.0
    rx, ry = average_ranks(np.asarray(x, float)), average_ranks(np.asarray(y, float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def rank_auc(scores, labels) -> float:
    """ROC AUC: P(positive outscores negative), ties counted half.

    Mann-Whitney U / (n1 n0), with U the positives' rank sum less its minimum
    n1 (n1 - 1) / 2 (0-based ranks). Scores [0.1, 0.9], labels [0, 1] -> 1.0.
    nan when either class is empty.

    Raises:
        ValueError: on a length mismatch or a NaN score (it has no rank).
    """
    s = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(labels, dtype=bool)
    if s.shape != pos.shape:
        raise ValueError(f"{len(s)} scores for {len(pos)} labels")
    if np.isnan(s).any():
        raise ValueError("NaN score")

    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if not n1 or not n0:
        return float("nan")

    u = float(average_ranks(s)[pos].sum()) - n1 * (n1 - 1) / 2.0
    return u / (n1 * n0)
