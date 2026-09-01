"""Fractal Jittered Voronoi spatial meta-mutation operator.

Partitions the input buffer using a fractal jittered Voronoi diagram
(Boris the Brave, 2026) and applies different sub-operators to each
cell. Boundary bytes (the "coastlines" between cells) get blended
mutations that often trigger parser state-machine transitions.

This is a class-based mutator implementing the ``MutatorBase`` interface
so it can self-register with ``REGISTRY.register_mutator()``.

Algorithm reference:
    https://www.boristhebrave.com/2026/08/29/fractal-jittered-voronoi-partitions/
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Callable

from fuzzer_tool.core.mutator_interface import MutationContext, MutatorBase


class FractalVoronoiMutator(MutatorBase):
    """Spatial meta-operator using fractal jittered Voronoi partitions.

    The input buffer is mapped to a 2D grid and partitioned into cells
    via a multi-layer Voronoi hierarchy. Each cell is assigned a
    sub-operator deterministically from its root hash. Boundary bytes
    between cells receive blended mutations.

    Args:
        max_depth: Number of fractal layers (default 4). Higher values
            produce finer, more wrinkled boundaries.
        cell_ops: List of callable sub-operators. If None, the mutator
            falls back to simple byte XOR (useful for standalone testing).
    """

    name = "fractal_voronoi"
    category = "structural"

    def __init__(
        self,
        max_depth: int = 4,
        cell_ops: list[Callable[[bytes], bytes]] | None = None,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.max_depth = max_depth
        self.cell_ops = cell_ops or []

    # ------------------------------------------------------------------
    # Voronoi geometry (deterministic, cached)
    # ------------------------------------------------------------------

    def _hash2(self, layer: int, cell: tuple[int, int]) -> tuple[float, float]:
        """Deterministic PRNG for a cell — two floats in [0, 1)."""
        h = hashlib.sha256(f"{layer}:{cell[0]}:{cell[1]}".encode()).digest()
        return (h[0] / 256.0, h[1] / 256.0)

    @lru_cache(maxsize=4096)
    def _site(self, layer: int, cell: tuple[int, int]) -> tuple[float, float]:
        """Return the site belonging to an integer grid cell (x, y)."""
        s = 2.0 ** (-layer)
        ox, oy = self._hash2(layer, cell)
        return (s * (cell[0] + ox), s * (cell[1] + oy))

    def _nearest_site(
        self, layer: int, p: tuple[float, float]
    ) -> tuple[tuple[int, int], tuple[float, float]]:
        """Find the layer-i site nearest to point p.

        Searches the nearest 5×5 cells — sufficient because cells further
        away cannot contain the nearest site for a jittered Voronoi grid.
        """
        s = 2.0 ** (-layer)
        cx = int(p[0] / s)
        cy = int(p[1] / s)

        best_cell: tuple[int, int] | None = None
        best_site: tuple[float, float] | None = None
        best_distance = float("inf")

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                cell = (cx + dx, cy + dy)
                q = self._site(layer, cell)
                d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
                if d < best_distance:
                    best_distance = d
                    best_cell = cell
                    best_site = q

        # best_cell is never None because the loop always runs at least once
        return best_cell, best_site  # type: ignore[return-value]

    @lru_cache(maxsize=4096)
    def _root(self, depth: int, cell: tuple[int, int]) -> tuple[int, int]:
        """Trace parent chain up to layer 0."""
        while depth > 0:
            p = self._site(depth, cell)
            cell, _ = self._nearest_site(depth - 1, p)
            depth -= 1
        return cell

    def _is_boundary(
        self, depth: int, px: float, py: float
    ) -> bool:
        """Check if point (px, py) lies on a fractal boundary at given depth."""
        cell, _ = self._nearest_site(depth, (px, py))
        root = self._root(depth, cell)
        # Check 8 neighbors
        for dx, dy in (
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ):
            nc = (cell[0] + dx, cell[1] + dy)
            nr = self._root(depth, nc)
            if nr != root:
                return True
        return False

    # ------------------------------------------------------------------
    # MutatorBase contract
    # ------------------------------------------------------------------

    def is_available(self, context: MutationContext, data: bytes) -> bool:
        """Always available — no external dependencies."""
        return True

    def mutate(
        self,
        data: bytes,
        rng,
        max_len: int = 0,
        *,
        context: MutationContext | None = None,
        **ctx,
    ) -> bytes | None:
        """Apply fractal Voronoi spatial mutation.

        Returns ``None`` for inputs too small to partition meaningfully.
        Returns a new ``bytes`` object (never mutates in place).
        """
        if len(data) < 16:
            return None

        # Map 1D buffer to roughly-square 2D grid
        side = max(1, int(math.sqrt(len(data))))
        out = bytearray(data)

        for idx in range(len(data)):
            y, x = divmod(idx, side)
            # Normalize to [0, 1) with slight oversample for toroidal feel
            px = (x + 0.5) / side
            py = (y + 0.5) / side

            # Find the root cell at max_depth
            cell, _ = self._nearest_site(self.max_depth, (px, py))
            root = self._root(self.max_depth, cell)

            # Deterministic hash from root cell
            root_hash = int(hashlib.sha256(f"root:{root}".encode()).hexdigest(), 16)

            if self.cell_ops:
                # Select sub-operator deterministically
                op_idx = root_hash % len(self.cell_ops)
                op = self.cell_ops[op_idx]

                # Apply operator stochastically based on hash — not every
                # byte every time, to avoid over-mutation
                if (root_hash + idx) % 7 == 0:
                    # For simplicity in the meta-operator, we apply a
                    # lightweight transformation. A full integration would
                    # pass a sub-region to the sub-operator.
                    mutated_byte = op(bytes([data[idx]]))
                    if mutated_byte and len(mutated_byte) == 1:
                        out[idx] = mutated_byte[0]
            else:
                # Fallback: simple XOR when no sub-operators configured
                if (root_hash + idx) % 5 == 0:
                    out[idx] ^= (root_hash & 0xFF)

            # Boundary bonus: if on a fractal coastline, add extra jitter
            if self._is_boundary(self.max_depth, px, py):
                boundary_hash = int(
                    hashlib.sha256(f"boundary:{root}:{px:.6f}:{py:.6f}".encode()).hexdigest(),
                    16,
                )
                if (boundary_hash + idx) % 3 == 0:
                    out[idx] ^= ((boundary_hash >> 8) & 0xFF)

        result = bytes(out)
        if max_len and len(result) > max_len:
            result = result[:max_len]
        return result


# ------------------------------------------------------------------
# Self-registration on module import
# ------------------------------------------------------------------

def _register() -> None:
    from fuzzer_tool.core.operator_registry import REGISTRY

    m = FractalVoronoiMutator()
    if m.name not in REGISTRY.names():
        REGISTRY.register_mutator(m)


_register()
