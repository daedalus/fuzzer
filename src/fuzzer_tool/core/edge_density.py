"""Edges per bit of seed: coverage bought per bit of input.

Normalizes edge counts by input size so a 40-byte seed reaching 300 edges
(0.94 e/bit) reads as denser than a 4 KiB seed reaching 600 (0.018 e/bit).
"""

from __future__ import annotations

BITS_PER_BYTE = 8


def edges_per_bit(edges: int, nbytes: int) -> float:
    """Return ``edges / (nbytes * 8)``; 0.0 when either side is non-positive.

    Example: ``edges_per_bit(8, 1) == 1.0``.
    """
    if edges <= 0 or nbytes <= 0:
        return 0.0
    return edges / (nbytes * BITS_PER_BYTE)
