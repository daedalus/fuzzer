"""Fractal jittered Voronoi partitioning of the parallel-worker hash space.

Approach C from ``docs/handover/fractal-voronoi-integration.md``: partition
seeds across ``-j N`` workers by the fractal Voronoi root cell of a
deterministic hash of their content, instead of round-robin or a flat
``hash(seed) % N``. Two things fall out of that for free:

* **Implicit work stealing.** Because root-cell assignment is stable across
  runs and workers, a seed always maps to the same worker regardless of
  which worker happens to discover it first -- ties corpus ownership to
  content, not discovery order.
* **Boundary-aware sync.** ``crosses_boundary()`` flags seeds whose root
  cell has a differently-rooted neighbor at the same depth -- these are
  the ones structurally "between" two partitions, and are the ones worth
  sharing across workers even when a worker doesn't own them (see
  ``accept_for_worker``).

This module is deliberately independent of
``core/mutations/fractal_voronoi.py``: that module partitions *byte
positions within one input* for mutation; this one partitions *whole seeds*
across a hash space for worker assignment. They share the same fractal
jittered Voronoi construction (Boris the Brave, 2026-08-29) but operate on
different domains, so duplicating the small geometry core here is cheaper
than coupling a mutation operator's internals to the parallel-fuzzing
service.
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
    """Deterministic per-cell jitter offset in [0, 1)^2."""
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

    Two seeds with identical content always map to the same root cell,
    on any worker, in any run -- assignment needs no shared state or
    coordination beyond the seed bytes themselves.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    p = _seed_point(seed)
    cell = _nearest_cell(depth, p)
    return _root(depth, cell)


def assign_worker(seed: bytes, n_workers: int, depth: int = 3) -> int:
    """Deterministic worker index (0..n_workers-1) that owns this seed.

    Args:
        seed: The seed's raw content (the same bytes ``hash_data()`` would
            hash for corpus deduplication).
        n_workers: Number of parallel workers (``-j N``).
        depth: Fractal layer depth. Higher values give a finer, more
            "wrinkled" partition boundary and more distinct root cells per
            worker; the default matches the mutation operator's
            recommended starting depth.

    Raises:
        ValueError: if ``n_workers`` is not positive.
    """
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")
    root = root_cell(seed, depth)
    h = int(hashlib.sha256(f"worker:{root}".encode()).hexdigest(), 16)
    return h % n_workers


def crosses_boundary(seed: bytes, depth: int = 3) -> bool:
    """Whether this seed's root cell has a differently-rooted 8-neighbor.

    A seed on a fractal "coastline" sits structurally between partitions:
    a small change in its hash could have landed it in a neighboring
    worker's territory. These are exactly the seeds worth propagating
    across worker boundaries even when a worker doesn't own them outright
    (see ``accept_for_worker``), mirroring how coastline bytes get blended
    treatment in the mutation operator.
    """
    p = _seed_point(seed)
    cell = _nearest_cell(depth, p)
    root = _root(depth, cell)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        neighbor_root = _root(depth, (cell[0] + dx, cell[1] + dy))
        if neighbor_root != root:
            return True
    return False


def accept_for_worker(seed: bytes, worker_id: int, n_workers: int, depth: int = 3) -> bool:
    """Whether ``worker_id`` should hold this seed under fractal partitioning.

    True when the worker owns the seed's root cell outright, or when the
    seed crosses a fractal boundary (shared regardless of ownership, per
    the handover's "sync interval: share seeds that crossed a fractal
    boundary" note). A worker that always says no to everything else is
    the point: it is what turns a flat, fully-shared corpus into a
    partitioned one.
    """
    if assign_worker(seed, n_workers, depth) == worker_id:
        return True
    return crosses_boundary(seed, depth)
