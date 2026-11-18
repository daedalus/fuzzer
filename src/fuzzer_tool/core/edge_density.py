"""Edges per bit of seed: coverage bought per bit of input.

Normalizes edge counts by input size so a 40-byte seed reaching 300 edges
(0.94 e/bit) reads as denser than a 4 KiB seed reaching 600 (0.018 e/bit).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

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


#: Lowest bloat factor: a discovery's reward is never cut below 10%.
BLOAT_FLOOR = 0.1


def bloat_factor(parent: tuple[int, int], child: tuple[int, int]) -> float | None:
    """Reward scale for a discovery that grew the input; None = neutral.

    *parent* and *child* are ``(trace edges, bytes)``. The scale is child
    edges per bit over parent edges per bit, capped at 1 (a denser child is
    not promoted) and floored at ``BLOAT_FLOOR``. Neutral unless the child is
    longer and both traces are known.

    Example: parent 40 edges / 10 B (0.5 e/bit), child 44 / 40 B
    (0.1375 e/bit) -> 0.275.
    """
    (p_edges, p_bytes), (c_edges, c_bytes) = parent, child
    if p_bytes <= 0 or c_bytes <= p_bytes or p_edges <= 0 or c_edges <= 0:
        return None

    ratio = edges_per_bit(c_edges, c_bytes) / edges_per_bit(p_edges, p_bytes)
    return min(1.0, max(BLOAT_FLOOR, ratio))


#: Density move (relative, first -> last) that counts as a trend.
TREND_DENSITY_SHIFT = 0.25

#: Edge growth at or below which a density drop reads as bloat.
TREND_EDGE_GROWTH = 0.05


class Trend(Enum):
    """Corpus edges-per-bit direction over the coverage timeline."""

    BLOAT = "bloat"  # density down, edges flat: bytes without coverage
    COMPACT = "compacting"  # density up: trimming / pruning is working
    STEADY = "steady"


@dataclass(frozen=True)
class DensityTrend:
    first: float  # edges per bit at the first known snapshot
    last: float  # edges per bit at the last snapshot
    edge_growth: float  # relative edge growth between them
    verdict: Trend


def density_trend(edges: Sequence[int], nbytes: Sequence[int]) -> DensityTrend | None:
    """First -> last corpus edges per bit over aligned snapshots; None under 2 known.

    A snapshot is known when it has edges and bytes (0 bytes: legacy, unrecorded).
    Example: edges 100 -> 102, bytes 1000 -> 4000: density -75%, edges +2% -> BLOAT.
    """
    known = [(e, b) for e, b in zip(edges, nbytes, strict=True) if e > 0 and b > 0]
    if len(known) < 2:
        return None

    (e0, b0), (e1, b1) = known[0], known[-1]
    first, last = edges_per_bit(e0, b0), edges_per_bit(e1, b1)
    shift = last / first - 1
    growth = e1 / e0 - 1

    verdict = Trend.STEADY
    if shift <= -TREND_DENSITY_SHIFT and growth <= TREND_EDGE_GROWTH:
        verdict = Trend.BLOAT
    elif shift >= TREND_DENSITY_SHIFT:
        verdict = Trend.COMPACT
    return DensityTrend(first, last, growth, verdict)
