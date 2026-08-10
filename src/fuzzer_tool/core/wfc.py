"""Wave Function Collapse (WFC) for structural fuzzing.

WFC is a constraint-satisfaction solver: cells with a superposition of
possible tiles, adjacency constraints, min-entropy collapse, AC-3
arc-consistency propagation, and bounded backtrack on contradiction.

Superpositions stored as ``numpy.ndarray`` of shape ``(n_cells, n_tiles)``
with dtype ``bool`` — ~15× less memory than ``list[list[bool]]`` (1 byte
per bool vs 28 bytes per Python bool object).

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
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np

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
# Hard cap on total _prune_cell calls for one run(). Propagation cost is
# O(n * n_tiles**2) per observation, so a wide or high-alphabet input can
# otherwise stall the caller for minutes (~9 s at 512 cells / 128 tiles,
# minutes at BMP row widths). Exhausting the budget leaves cells
# uncollapsed instead of hanging; callers map those to a fallback.
DEFAULT_WORK_BUDGET = 50_000


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

        Closed-world: if *a* has rules for this direction, *b* must be
        among them. If *a* has no rules, compatibility fails.
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

    Superpositions stored as 2D numpy bool array:
      superpositions[cell, tile_id] == True  → tile still possible
      superpositions[cell, tile_id] == False → tile ruled out

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
        self.n_tiles = len(tiles)

        # Local RNG. Every stochastic choice in this class draws from here
        # rather than the ``random`` module, so a seeded run is reproducible
        # without disturbing any other consumer of the global stream.
        self._rng = random.Random()

        # Precompute tile weights as array for vectorized operations
        self._weights = np.array([t.weight for t in tiles], dtype=np.float64)

        # superpositions[cell, tile_id] = True if tile is still possible
        self.superpositions: np.ndarray = np.ones((self.n, self.n_tiles), dtype=bool)

        self.contradiction = False

        # Hard-work budget accounting for one run(). _budget_exhausted is a
        # soft stop: run() returns whatever is collapsed so far (callers map
        # multi/zero-entropy cells to a fallback) rather than hanging.
        self._budget_exhausted = False
        self._work_used = 0

        # Snapshot of the caller-constrained wave, captured at run() entry so
        # restarts can restore it rather than clearing every constraint.
        self._initial: np.ndarray | None = None

        # Precompute adjacency matrix for vectorized _prune_cell:
        # adj_matrix[i, j, d] = compatible(tiles[i], tiles[j], direction[d])
        self._dir_names: list[Direction] = ["left", "right", "up", "down"]
        self._adj_matrix: np.ndarray | None = None
        mat = np.zeros((self.n_tiles, self.n_tiles, 4), dtype=bool)
        for i, t_i in enumerate(tiles):
            for j, t_j in enumerate(tiles):
                for d, dname in enumerate(self._dir_names):
                    mat[i, j, d] = adjacency.compatible(t_i.name, t_j.name, dname)
        self._adj_matrix = mat

    # ── Public API ──────────────────────────────────────────────────

    def run(
        self,
        seed: int | None = None,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
        ac3_budget: int = DEFAULT_AC3_BUDGET,
        work_budget: int = DEFAULT_WORK_BUDGET,
    ) -> list[list[bytes | None]]:
        """Run WFC collapse loop.

        Args:
            seed: Random seed for deterministic output. None = unseeded.
                Seeds this grid's own RNG only. Seeding the ``random``
                module here instead would reset the process-wide stream
                that the fuzzer's operators and schedulers draw from,
                collapsing their entropy to a point determined solely by
                this value — and WFC is invoked per PNG reorder and per
                BMP row, so that reset would happen constantly.
            max_restarts: Max restarts on contradiction.
            ac3_budget: Max AC-3 propagation iterations before greedy fallback.
            work_budget: Hard cap on total propagation work for this run.
                Exhausting it stops the collapse and returns the partial
                grid (cells left uncollapsed) instead of hanging on a wide
                or high-alphabet input.

        Returns:
            2D grid: ``grid[y][x]`` = tile name at (x, y), or None if
            the cell couldn't be collapsed (budget exhausted).
        """
        if seed is not None:
            self._rng = random.Random(seed)

        # Capture whatever constraints the caller applied before running, so
        # _reset() can restore them instead of wiping them.
        self._initial = self.superpositions.copy()

        for attempt in range(max_restarts + 1):
            if attempt > 0:
                self._reset()
                if seed is not None:
                    self._rng = random.Random(seed + attempt * 7919)

            self.contradiction = False
            self._budget_exhausted = False
            self._run_loop(ac3_budget, work_budget)
            if not self.contradiction and not self._budget_exhausted:
                break

        return self._to_grid()

    def to_1d(self) -> list[bytes | None]:
        """Return 1D output for height=1 grid."""
        grid = self._to_grid()
        return grid[0] if grid else []

    # ── Internal collapse loop ──────────────────────────────────────

    def _run_loop(self, ac3_budget: int, work_budget: int = DEFAULT_WORK_BUDGET):
        """Collapse cells one by one until done, contradiction, or budget out.

        The first propagation sweeps the whole grid so caller-applied pins
        are enforced; afterwards only the observed cell's neighbours are
        re-pruned (plus the AC-3 queue cascade). Sweeping the full grid on
        every observation costs O(n * n_tiles**2) per step — the dominant
        cost on wide rows, and the reason a BMP-width wave used to stall
        for minutes.
        """
        self._work_used = 0
        self._propagate(ac3_budget, sources=None, work_budget=work_budget)
        while not self.contradiction and not self._budget_exhausted:
            idx = self._find_min_entropy()
            if idx is None:
                break
            self._observe(idx)
            if not self.contradiction and not self._budget_exhausted:
                self._propagate(ac3_budget, sources=self._neighbors(idx), work_budget=work_budget)

    def _find_min_entropy(self) -> int | None:
        """Find cell with smallest non-zero entropy.

        Returns:
            Cell index, or None if all cells collapsed.
        """
        min_entropy = float("inf")
        best_idx = None
        for i in range(self.n):
            row = self.superpositions[i]
            count = int(row.sum())
            if count == 0:
                self.contradiction = True
                return None
            if count == 1:
                continue
            entropy = self._entropy(row) + self._rng.random() * 1e-9
            if entropy < min_entropy:
                min_entropy = entropy
                best_idx = i
        return best_idx

    def _entropy(self, row: np.ndarray) -> float:
        """Shannon entropy of a cell's superposition."""
        available_mask = row.astype(bool, copy=False)
        if not available_mask.any():
            return 0.0
        w = self._weights[available_mask]
        total = w.sum()
        if total <= 0:
            return 0.0
        p = w / total
        return float(-(p * np.log2(p)).sum())

    def _observe(self, idx: int):
        """Collapse cell *idx*: pick a tile weighted by probability."""
        row = self.superpositions[idx]
        possible = np.flatnonzero(row)
        if len(possible) == 0:
            self.contradiction = True
            return

        weights = self._weights[possible]
        total = weights.sum()
        if total <= 0:
            chosen = self._rng.choice(possible)
        else:
            r = self._rng.random() * total
            cumulative = 0.0
            chosen = possible[-1]
            for tid, w in zip(possible.tolist(), weights.tolist(), strict=True):
                cumulative += w
                if r <= cumulative:
                    chosen = tid
                    break

        row[:] = False
        row[chosen] = True

    def _propagate(
        self,
        budget: int = DEFAULT_AC3_BUDGET,
        sources: list[int] | range | None = None,
        work_budget: int = DEFAULT_WORK_BUDGET,
    ):
        """AC-3 arc-consistency propagation.

        Args:
            budget: Max queue iterations before greedy fallback.
            sources: Cells to re-prune. None = full sweep (every cell), used
                once per collapse attempt to enforce caller-applied pins;
                later calls prune only the observed cell's neighbours, which
                is complete for AC-3 (the queue cascade re-checks everything
                adjacent to a change).
            work_budget: Hard cap on total ``_prune_cell`` calls across the
                run. On exhaustion ``_budget_exhausted`` is set: the caller
                stops collapsing and returns the partial grid instead of
                stalling on an expensive wave.
        """
        queue: collections.deque[int] = collections.deque()
        changed_any = False
        if sources is None:
            sources = range(self.n)
        for i in sources:
            if self._budget_exhausted:
                return
            if self.superpositions[i].sum() <= 1:
                continue
            self._work_used += 1
            if self._work_used > work_budget:
                self._budget_exhausted = True
                return
            result = self._prune_cell(i)
            if result is None:
                return
            if result:
                queue.append(i)
                changed_any = True

        if not changed_any:
            return

        iterations = 0
        while queue and iterations < budget:
            iterations += 1
            idx = queue.popleft()

            for nidx in self._neighbors(idx):
                if self._budget_exhausted:
                    return
                if self.superpositions[nidx].sum() <= 1:
                    continue
                self._work_used += 1
                if self._work_used > work_budget:
                    self._budget_exhausted = True
                    return
                result = self._prune_cell(nidx)
                if result is None:
                    return
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
        row = self.superpositions[idx]
        removed_any = False

        if self._adj_matrix is not None:
            for nidx in self._neighbors(idx):
                d = self._dir_names.index(self._direction_to(idx, nidx))
                nbr_row = self.superpositions[nidx]
                # compat[t] = any of tile t's compatible neighbors still possible?
                compat = np.any(self._adj_matrix[:, :, d] & nbr_row[None, :], axis=1)
                removed = ~compat & row
                if removed.any():
                    row[removed] = False
                    removed_any = True
                if not row.any():
                    self.contradiction = True
                    return None
            return removed_any

        # Pure-Python fallback
        for tid in range(self.n_tiles):
            if not row[tid]:
                continue

            tile_name = self.tiles[tid].name
            all_neighbors_ok = True

            for nidx in self._neighbors(idx):
                dir_from_idx = self._direction_to(idx, nidx)
                nbr_row = self.superpositions[nidx]

                # Build compatibility mask for this (tile_name, direction) pair
                compat_mask = np.array(
                    [
                        self.adjacency.compatible(tile_name, self.tiles[t].name, dir_from_idx)
                        for t in range(self.n_tiles)
                    ],
                    dtype=bool,
                )

                if not np.any(nbr_row & compat_mask):
                    all_neighbors_ok = False
                    break

            if not all_neighbors_ok:
                row[tid] = False
                removed_any = True

        if row.sum() == 0:
            self.contradiction = True
            return None
        return removed_any

    def _fallback_greedy(self):
        """Greedy fallback when AC-3 budget exhausted."""
        for i in range(self.n):
            if self.superpositions[i].sum() > 1 and not self.contradiction:
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

    def _direction_to(self, from_idx: int, to_idx: int) -> Direction:
        """Direction from *from_idx* to *to_idx*.

        Vertical steps are ``±self.w`` and horizontal steps are ``±1``. Those
        coincide when ``w == 1``, so the row index must be compared rather
        than inferring the axis from the raw difference — otherwise a
        single-column grid resolves its vertical neighbours as left/right and
        checks them against horizontal adjacency rules.
        """
        if from_idx // self.w != to_idx // self.w:
            return "down" if to_idx > from_idx else "up"
        return "right" if to_idx > from_idx else "left"

    # ── Reset ───────────────────────────────────────────────────────

    def _reset(self):
        """Restore the superposition to its state at the start of ``run()``.

        Callers constrain cells before running — pinning IHDR to the first
        cell and IEND to the last, for example — by writing directly into
        ``superpositions``. Resetting to all-True would discard those pins on
        every restart, so a wave that contradicts once would come back
        unconstrained and happily place the anchor mid-sequence.
        """
        if self._initial is not None:
            self.superpositions[:, :] = self._initial
        else:
            self.superpositions[:, :] = True
        self.contradiction = False

    # ── Output ──────────────────────────────────────────────────────

    def _to_grid(self) -> list[list[bytes | None]]:
        """Convert collapsed superpositions to a 2D grid of tile names."""
        grid: list[list[bytes | None]] = []
        for y in range(self.h):
            row_out: list[bytes | None] = []
            for x in range(self.w):
                idx = y * self.w + x
                row = self.superpositions[idx]
                count = int(row.sum())
                if count == 1:
                    row_out.append(self.tiles[int(row.argmax())].name)
                elif count > 1:
                    # Most-likely tile by weight
                    best = int((row * self._weights).argmax())
                    row_out.append(self.tiles[best].name)
                else:
                    row_out.append(None)
            grid.append(row_out)
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

        table.add_forward(b"PLTE", b"tRNS")
        table.add_forward(b"PLTE", b"bKGD")
        table.add_forward(b"PLTE", b"hIST")
        for a in ancillary:
            table.add_forward(b"PLTE", a)
        table.add_forward(b"PLTE", b"IDAT")
        table.add_forward(b"PLTE", b"IEND")

        for a in ancillary:
            for b in ancillary:
                if a != b:
                    table.add_undirected(a, b)
            table.add_forward(a, b"IDAT")
            table.add_forward(a, b"IEND")

        table.add_forward(b"IDAT", b"IDAT")
        table.add_forward(b"IDAT", b"IEND")
        return table

    @staticmethod
    def jpeg_markers() -> AdjacencyTable:
        """JPEG marker ordering rules.

        Covers the marker set ``parse_jpeg_markers`` actually emits, so every
        parsed segment has a tile with rules. A marker missing from this table
        has no adjacency entries at all, and ``compatible()`` is closed-world
        — it would be pruned from every cell and the wave would contradict.
        """
        table = AdjacencyTable()
        markers = [
            b"APP0",
            b"APP1",
            b"DHT",
            b"DQT",
            b"DRI",
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
