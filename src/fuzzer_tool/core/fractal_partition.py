"""Fractal jittered Voronoi partition of the seed-content hash space.

Approach C from ``docs/handover/handover_done_2026-09-06.md``, originally
written to hand seeds out to ``-j N`` workers by the fractal Voronoi root
cell of a deterministic hash of their content. That mode is retired; what
survives is the partition itself and the one question the fuzzer still
asks of it.

:func:`root_cell` maps seed content to a layer-0 cell: two seeds with
identical content always land in the same cell, with no shared state or
coordination beyond the seed bytes. :func:`crosses_boundary` flags seeds
whose root cell has a differently-rooted neighbor at the same depth --
the ones sitting structurally *between* clusters of the hash space, where
a small content change would have moved them across. ``seed_picker``'s
``--fractal-diversity`` weight is the consumer: it mildly boosts those
coastline seeds as a cheap counter to mode collapse toward one region of
the corpus's own hash space (see ``_weight_fractal_diversity``).

This module is deliberately independent of
``core/mutations/fractal_voronoi.py``: that module partitions *byte
positions within one input* for mutation; this one partitions *whole
seeds* across a hash space. They share the same fractal jittered Voronoi
construction (Boris the Brave, 2026-08-29) but operate on different
domains, so duplicating the small geometry core here is cheaper than
coupling a mutation operator's internals to the seed picker.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache


def _seed_point(seed: bytes) -> tuple[float, float]:
    """Deterministic map from arbitrary seed bytes to a point in [0, 1)^2."""
    h = hashlib.sha256(seed).digest()
    x = int.from_bytes(h[0:4], "big") / 2**32
    y = int.from_bytes(h[4:8], "big") / 2**32
    return (x, y)


@lru_cache(maxsize=8192)
def _hash2(layer: int, cell: tuple[int, int]) -> tuple[float, float]:
    """Deterministic per-cell jitter offset in [0, 1)^2.

    The ``parallel:`` salt is a fossil of this module's original use for
    worker assignment and is kept verbatim on purpose: it is hashed, so
    renaming it would move every cell site and silently change which
    seeds ``--fractal-diversity`` treats as boundary seeds.
    """
    h = hashlib.sha256(f"parallel:{layer}:{cell[0]}:{cell[1]}".encode()).digest()
    return (h[0] / 256.0, h[1] / 256.0)


@lru_cache(maxsize=8192)
def _site(layer: int, cell: tuple[int, int]) -> tuple[float, float]:
    """The jittered site belonging to integer grid cell ``cell`` at ``layer``."""
    s = 2.0**-layer
    ox, oy = _hash2(layer, cell)
    return (s * (cell[0] + ox), s * (cell[1] + oy))


def _nearest_cell(layer: int, p: tuple[float, float]) -> tuple[int, int]:
    """Nearest layer-``layer`` site to point ``p``, searching a 5x5 neighborhood."""
    s = 2.0**-layer
    cx, cy = int(p[0] / s), int(p[1] / s)
    best_cell = (cx, cy)
    best_d = float("inf")
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            cell = (cx + dx, cy + dy)
            q = _site(layer, cell)
            d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            if d < best_d:
                best_d, best_cell = d, cell
    return best_cell


@lru_cache(maxsize=8192)
def _root(depth: int, cell: tuple[int, int]) -> tuple[int, int]:
    """Trace a cell's parent chain up to the layer-0 root."""
    while depth > 0:
        p = _site(depth, cell)
        cell = _nearest_cell(depth - 1, p)
        depth -= 1
    return cell


def root_cell(seed: bytes, depth: int = 3) -> tuple[int, int]:
    """The layer-0 root cell a seed's content falls into at ``depth``.

    Two seeds with identical content always map to the same root cell, in
    any run -- the partition needs no state beyond the seed bytes
    themselves.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    p = _seed_point(seed)
    cell = _nearest_cell(depth, p)
    return _root(depth, cell)


def crosses_boundary(seed: bytes, depth: int = 3) -> bool:
    """Whether this seed's root cell has a differently-rooted 8-neighbor.

    A seed on a fractal "coastline" sits structurally between clusters of
    the content-hash space: a small change in its hash could have landed
    it in a neighboring cell. That is what ``--fractal-diversity`` boosts,
    mirroring how coastline bytes get blended treatment in the mutation
    operator.
    """
    p = _seed_point(seed)
    cell = _nearest_cell(depth, p)
    root = _root(depth, cell)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        neighbor_root = _root(depth, (cell[0] + dx, cell[1] + dy))
        if neighbor_root != root:
            return True
    return False
