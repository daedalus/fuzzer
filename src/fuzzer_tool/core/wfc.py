"""Wave Function Collapse (WFC) for structural fuzzing.

WFC is a constraint-satisfaction solver: cells with a superposition of
possible tiles, adjacency constraints, min-entropy collapse, AC-3
arc-consistency propagation, and bounded backtrack on contradiction.

This module provides:
  - Tile: atomic building block with name and weight
  - AdjacencyTable: directional compatibility rules between tiles
  - WaveGrid: 1D/2D constraint-satisfaction engine with collapse,
    AC-3 propagation, and seeded determinism
  - ConstraintSet: predefined adjacency tables for known formats

The value-add over the existing causal (markov.py) and top-down recursive
(grammar.py) generators is non-causal global consistency: WFC can enforce
constraints between non-adjacent positions that neither of the existing
approaches can express.

Reference: Gumin, "Wave Function Collapse" 2016 (github.com/mxgmn/WaveFunctionCollapse)
"""

from __future__ import annotations

import collections
import math
import random
from dataclasses import dataclass
from typing import Literal

# ── Direction constants ───────────────────────────────────────────────

Direction = Literal["left", "right", "up", "down"]
OPPOSITE: dict[Direction, Direction] = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}
DIRECTIONS_1D: list[Direction] = ["left", "right"]
DIRECTIONS_2D: list[Direction] = ["up", "down", "left", "right"]

# Default iteration and backtrack budgets
DEFAULT_AC3_BUDGET = 5000
DEFAULT_MAX_RESTARTS = 3


# ── Tile ──────────────────────────────────────────────────────────────


@dataclass
class Tile:
    """An atomic building block for WFC.

    Attributes:
        name: Tile identifier (e.g. b"IHDR", b"IDAT", or small pixel block).
        weight: Relative selection probability during collapse (default 1.0).
    """

    name: bytes
    weight: float = 1.0


# ── AdjacencyTable ────────────────────────────────────────────────────


class AdjacencyTable:
    """Directional adjacency constraints between tiles.

    Stores compatibility rules as a dict:
      rules[tile_name][direction] = set of compatible tile names

    ``add_forward(A, B)`` means "A can be immediately followed by B":
      - A allows B to its RIGHT:  rules[A]["right"] += B
      - B allows A to its LEFT:   rules[B]["left"] += A

    The ``compatible(A, B, direction)`` check answers:
      "can A have B in the given direction from A?"
    It checks rules[A][direction] when rules exist. If A has NO rules
    for that direction, the check is *closed-world*: returns False.
    """

    def __init__(self):
        self._rules: dict[bytes, dict[str, set[bytes]]] = {}

    def add_forward(self, a: bytes, b: bytes):
        """Add a forward-only rule: A can be immediately followed by B."""
        self._ensure(a)
        self._ensure(b)
        self._rules[a]["right"].add(b)
        self._rules[b]["left"].add(a)

    def add_undirected(self, a: bytes, b: bytes):
        """A and B can be adjacent in either order."""
        self._ensure(a)
        self._ensure(b)
        self._rules[a]["right"].add(b)
        self._rules[a]["left"].add(b)
        self._rules[b]["right"].add(a)
        self._rules[b]["left"].add(a)

    def compatible(self, a: bytes, b: bytes, direction: Direction) -> bool:
        """Check if tile *a* is compatible with having *b* in *direction*.

        Uses a closed-world interpretation: if *a* has rules for this
        direction, *b* must be among them. If *a* has no rules for this
        direction, compatibility fails (the tile is not allowed there).
        """
        dir_rules = self._rules.get(a, {}).get(direction, None)
        if dir_rules is None:
            return False
        return b in dir_rules

    def has_tile(self, tile_name: bytes) -> bool:
        """Check if any adjacency rules exist for this tile."""
        return tile_name in self._rules

    @classmethod
    def from_pairs(cls, pairs: list[tuple[bytes, bytes]]) -> AdjacencyTable:
        """Build from ordered pairs: each pair (a, b) means a → b."""
        table = cls()
        for a, b in pairs:
            table.add_forward(a, b)
        return table

    @classmethod
    def from_corpus(
        cls,
        tile_names: list[bytes],
        sequences: list[list[bytes]],
    ) -> AdjacencyTable:
        """Learn adjacency from observed sequences."""
        table = cls()
        for name in tile_names:
            table._ensure(name)
        for seq in sequences:
            for i in range(len(seq) - 1):
                table.add_forward(seq[i], seq[i + 1])
        return table

    def _ensure(self, name: bytes):
        if name not in self._rules:
            self._rules[name] = {d: set() for d in ["left", "right", "up", "down"]}


# ── WaveGrid ──────────────────────────────────────────────────────────


class WaveGrid:
    """Constraint-satisfaction grid using Wave Function Collapse.

    Maintains a superposition of possible tiles per cell, collapses the
    most-constrained cell (min-entropy) first, propagates constraints via
    AC-3, and restarts on contradiction with a bounded backtrack budget.

    Two modes:
    - 1D: cells arranged in a line, left/right adjacency
    - 2D: cells arranged in a w×h grid, up/down/left/right adjacency
    """

    def __init__(
        self,
        tiles: list[Tile],
        adjacency: AdjacencyTable,
        width: int,
        height: int = 1,
    ):
        self.tiles = tiles
        self.adjacency = adjacency
        self.w = width
        self.h = height
        self.n = width * height

        # superpositions[cell][tile_id] = True if tile is still possible
        self.superpositions: list[list[bool]] = [[True] * len(tiles) for _ in range(self.n)]

        self.contradiction = False

    # ── Public API ──────────────────────────────────────────────────

    def run(
        self,
        seed: int | None = None,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
        ac3_budget: int = DEFAULT_AC3_BUDGET,
    ) -> list[list[bytes | None]]:
        """Run WFC collapse loop.

        Args:
            seed: Random seed for deterministic output. None = unseeded.
            max_restarts: Max restarts on contradiction.
            ac3_budget: Max AC-3 propagation iterations before greedy fallback.

        Returns:
            2D grid: ``grid[y][x]`` = tile name at (x, y), or None if
            the cell couldn't be collapsed (budget exhausted).
        """
        if seed is not None:
            random.seed(seed)

        for attempt in range(max_restarts + 1):
            if attempt > 0:
                self._reset()
                if seed is not None:
                    random.seed(seed + attempt * 7919)

            self.contradiction = False
            self._run_loop(ac3_budget)
            if not self.contradiction:
                break

        return self._to_grid()

    def to_1d(self) -> list[bytes | None]:
        """Return 1D output for height=1 grid."""
        grid = self._to_grid()
        return grid[0] if grid else []

    # ── Internal collapse loop ──────────────────────────────────────

    def _run_loop(self, ac3_budget: int):
        """Collapse cells one by one until done or contradiction."""
        while not self.contradiction:
            idx = self._find_min_entropy()
            if idx is None:
                break  # all collapsed
            self._observe(idx)
            if not self.contradiction:
                self._propagate(ac3_budget)

    def _find_min_entropy(self) -> int | None:
        """Find cell with smallest non-zero entropy.

        Returns:
            Cell index, or None if all cells collapsed.
        """
        min_entropy = float("inf")
        best_idx = None
        for i in range(self.n):
            count = self.superpositions[i].count(True)
            if count == 0:
                self.contradiction = True
                return None
            if count == 1:
                continue
            entropy = self._entropy(i) + random.random() * 1e-9
            if entropy < min_entropy:
                min_entropy = entropy
                best_idx = i
        return best_idx

    def _entropy(self, idx: int) -> float:
        """Shannon entropy of a cell's superposition."""
        total = sum(self.tiles[tid].weight for tid, ok in enumerate(self.superpositions[idx]) if ok)
        if total <= 0:
            return 0.0
        h = 0.0
        for tid, ok in enumerate(self.superpositions[idx]):
            if not ok:
                continue
            p = self.tiles[tid].weight / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    def _observe(self, idx: int):
        """Collapse cell *idx*: pick a tile weighted by probability."""
        possible = [tid for tid, ok in enumerate(self.superpositions[idx]) if ok]
        if not possible:
            self.contradiction = True
            return

        weights = [self.tiles[tid].weight for tid in possible]
        total = sum(weights)
        if total <= 0:
            chosen = random.choice(possible)
        else:
            r = random.random() * total
            cumulative = 0.0
            chosen = possible[-1]
            for tid, w in zip(possible, weights, strict=True):
                cumulative += w
                if r <= cumulative:
                    chosen = tid
                    break

        for tid in range(len(self.tiles)):
            self.superpositions[idx][tid] = tid == chosen

    def _propagate(self, budget: int = DEFAULT_AC3_BUDGET):
        """AC-3 arc-consistency propagation.

        Returns when stable, budget exhausted (falls back to greedy),
        or contradiction detected.
        """
        # Initial full consistency pass
        queue: collections.deque[int] = collections.deque()
        changed_any = False
        for i in range(self.n):
            if self.superpositions[i].count(True) <= 1:
                continue
            result = self._prune_cell(i)
            if result is None:
                return  # contradiction
            if result:
                queue.append(i)
                changed_any = True

        if not changed_any:
            return  # already stable

        iterations = 0
        while queue and iterations < budget:
            iterations += 1
            idx = queue.popleft()

            for nidx in self._neighbors(idx):
                if self.superpositions[nidx].count(True) <= 1:
                    continue
                result = self._prune_cell(nidx)
                if result is None:
                    return  # contradiction
                if result:
                    queue.append(nidx)

        if iterations >= budget:
            self._fallback_greedy()

    def _prune_cell(self, idx: int) -> bool | None:
        """Remove tile options from cell *idx* that have no compatible neighbor.

        For each tile option T at cell idx, checks all neighbors:
          T at idx must be compatible with SOME tile option at each neighbor.
        T is removed if it has NO compatible tile in ANY neighbor.

        Returns:
            True if any tile was removed.
            False if nothing changed.
            None if contradiction (cell has 0 remaining tiles).
        """
        removed_any = False
        for tid in range(len(self.tiles)):
            if not self.superpositions[idx][tid]:
                continue

            tile_name = self.tiles[tid].name
            all_neighbors_ok = True

            for nidx in self._neighbors(idx):
                dir_from_idx = self._direction_to(idx, nidx)
                # Check if tile_name at idx is compatible with ANY tile at nidx
                neighbor_ok = False
                for ntid, ok in enumerate(self.superpositions[nidx]):
                    if not ok:
                        continue
                    if self.adjacency.compatible(tile_name, self.tiles[ntid].name, dir_from_idx):
                        neighbor_ok = True
                        break

                if not neighbor_ok:
                    all_neighbors_ok = False
                    break

            if not all_neighbors_ok:
                self.superpositions[idx][tid] = False
                removed_any = True

        # Check contradiction
        if self.superpositions[idx].count(True) == 0:
            self.contradiction = True
            return None
        return removed_any

    def _fallback_greedy(self):
        """Greedy fallback when AC-3 budget exhausted.

        Collapses remaining uncollapsed cells without propagation.
        May produce invalid adjacency but guarantees completion.
        """
        for i in range(self.n):
            if self.superpositions[i].count(True) > 1 and not self.contradiction:
                self._observe(i)

    # ── Grid/coordinate helpers ─────────────────────────────────────

    def _neighbors(self, idx: int) -> list[int]:
        """Return neighbor cell indices (up to 4 in 2D, 2 in 1D)."""
        x = idx % self.w
        y = idx // self.w
        result = []
        if x > 0:
            result.append(idx - 1)
        if x < self.w - 1:
            result.append(idx + 1)
        if y > 0:
            result.append(idx - self.w)
        if y < self.h - 1:
            result.append(idx + self.w)
        return result

    @staticmethod
    def _direction_to(from_idx: int, to_idx: int) -> Direction:
        """Direction from *from_idx* to *to_idx*.

        Returns the direction in which *to_idx* lies relative to *from_idx*.
        E.g., if to_idx is left of from_idx, return "left".
        """
        diff = to_idx - from_idx
        if diff == -1:
            return "left"
        if diff == 1:
            return "right"
        if diff < 0:
            return "up"
        return "down"

    # ── Reset ───────────────────────────────────────────────────────

    def _reset(self):
        """Reset superposition to all tiles possible."""
        for i in range(self.n):
            for tid in range(len(self.tiles)):
                self.superpositions[i][tid] = True
        self.contradiction = False

    # ── Output ──────────────────────────────────────────────────────

    def _to_grid(self) -> list[list[bytes | None]]:
        """Convert collapsed superpositions to a 2D grid of tile names.

        If a cell has a single tile, outputs that tile name.
        If multiple tiles remain, outputs the most-likely tile.
        If zero tiles remain (contradiction), outputs None.
        """
        grid: list[list[bytes | None]] = []
        for y in range(self.h):
            row: list[bytes | None] = []
            for x in range(self.w):
                idx = y * self.w + x
                choices = [tid for tid, ok in enumerate(self.superpositions[idx]) if ok]
                if len(choices) == 1:
                    row.append(self.tiles[choices[0]].name)
                elif len(choices) > 1:
                    best = max(choices, key=lambda t: self.tiles[t].weight)
                    row.append(self.tiles[best].name)
                else:
                    row.append(None)
            grid.append(row)
        return grid

    def tile_at(self, x: int, y: int = 0) -> bytes | None:
        """Return the tile name at position (x, y), or None if uncollapsed."""
        return self._to_grid()[y][x]


# ── Predefined constraint sets ────────────────────────────────────────


class ConstraintSet:
    """Predefined adjacency tables for known formats."""

    @staticmethod
    def png_chunks() -> AdjacencyTable:
        """PNG chunk ordering rules (PNG spec §4.3, §4.4)."""
        table = AdjacencyTable()
        # IHDR can be followed by critical or ancillary chunks
        table.add_forward(b"IHDR", b"PLTE")
        ancillary = [
            b"tRNS",
            b"gAMA",
            b"pHYs",
            b"cHRM",
            b"sBIT",
            b"iCCP",
            b"tEXt",
            b"zTXt",
            b"iTXt",
            b"bKGD",
            b"hIST",
            b"sPLT",
        ]
        for a in ancillary:
            table.add_forward(b"IHDR", a)
        table.add_forward(b"IHDR", b"IDAT")
        table.add_forward(b"IHDR", b"IEND")

        # PLTE → ancillary or IDAT
        table.add_forward(b"PLTE", b"tRNS")
        table.add_forward(b"PLTE", b"bKGD")
        table.add_forward(b"PLTE", b"hIST")
        for a in ancillary:
            table.add_forward(b"PLTE", a)
        table.add_forward(b"PLTE", b"IDAT")
        table.add_forward(b"PLTE", b"IEND")

        # Ancillary ↔ Ancillary (cross-compatible)
        for a in ancillary:
            for b in ancillary:
                if a != b:
                    table.add_undirected(a, b)
            table.add_forward(a, b"IDAT")
            table.add_forward(a, b"IEND")

        # IDAT → IDAT or IEND
        table.add_forward(b"IDAT", b"IDAT")
        table.add_forward(b"IDAT", b"IEND")

        return table

    @staticmethod
    def jpeg_markers() -> AdjacencyTable:
        """JPEG marker ordering rules."""
        table = AdjacencyTable()
        markers = [
            b"APP0",
            b"APP1",
            b"DHT",
            b"DQT",
            b"SOF0",
            b"SOF2",
            b"COM",
        ]
        table.add_forward(b"SOI", b"SOS")
        for m in markers:
            table.add_forward(b"SOI", m)
            for n in markers:
                if m != n:
                    table.add_undirected(m, n)
            table.add_forward(m, b"SOS")
            table.add_forward(m, b"EOI")
        table.add_forward(b"SOS", b"EOI")
        return table
