"""Edges per bit of seed: coverage bought per bit of input.

Normalizes edge counts by input size so a 40-byte seed reaching 300 edges
(0.94 e/bit) reads as denser than a 4 KiB seed reaching 600 (0.018 e/bit).
"""

from __future__ import annotations

import numpy as np

BITS_PER_BYTE = 8


def edges_per_bit(edges: int, nbytes: int) -> float:
    """Return ``edges / (nbytes * 8)``; 0.0 when either side is non-positive.

    Example: ``edges_per_bit(8, 1) == 1.0``.
    """
    if edges <= 0 or nbytes <= 0:
        return 0.0
    return edges / (nbytes * BITS_PER_BYTE)


#: log2 half-span of the density weight: factors run 2**-1 .. 2**1.
DENSITY_LOG2_SPAN = 1.0


def density_factors(edges: np.ndarray, nbytes: np.ndarray) -> np.ndarray:
    """Seed weight factor ``2 ** (span * (2t - 1))``, t = mid-rank of edges/bit.

    Ranks, not ratios: a 1-byte seed hitting 3 edges (0.375 e/bit) tops the
    order but cannot dwarf the rest. Ties share their mid-rank. Seeds with no
    edges or no bytes get 1.0 and stay out of the ranking.

    Example: densities (0.1, 0.2, 0.4) -> t (0, 0.5, 1) -> (0.5, 1.0, 2.0).
    """
    out = np.ones(len(edges))
    ranked = (edges > 0) & (nbytes > 0)
    m = int(np.count_nonzero(ranked))
    if m < 2:
        return out

    # Mid-rank per distinct density: first sorted slot + half the tie run.
    dens = edges[ranked] / (nbytes[ranked] * BITS_PER_BYTE)
    _, inv, counts = np.unique(dens, return_inverse=True, return_counts=True)
    mid = np.cumsum(counts) - counts + (counts - 1) / 2
    t = mid[inv] / (m - 1)
    out[ranked] = np.exp2(DENSITY_LOG2_SPAN * (2 * t - 1))
    return out
