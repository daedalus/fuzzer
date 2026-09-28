"""PositionFractalScheduler: adaptive-resolution offset selection.

``core/schedulers/pos_burn_front.py`` bins a seed at one fixed resolution,
chosen up front from its length (``width = ceil(len/MAX_BINS)``): a large
seed (the newly-vendored multi-MB FFmpeg binaries, say) gets a coarse bin
whether or not the format needs byte-level precision (PNG's CRC-32 makes
every byte edge-relevant -- see ``docs/handover/handover_*_sensitivity*``).
This scheduler instead starts at one bin covering the whole seed and only
refines -- splits a bin into two half-width children -- once evidence
(accumulated coverage-gain heat) justifies the extra resolution. The same
split rule applies at every depth (self-similar, hence "fractal"), so
resolution grows exactly where gains cluster and nowhere else, with no
grid materialized for the rest of the buffer.

This is deliberately independent of the two other fractal constructions
already in the repo -- ``core/fractal_partition.py`` (jittered Voronoi over
the corpus's seed-hash space, for ``--fractal-diversity``) and
``core/mutations/fractal_voronoi.py`` (jittered Voronoi over one buffer,
picking a *sub-operator* per cell) -- neither of which proposes an offset
for the position arena. Reusing their 2D jittered-site geometry here would
buy nothing: a byte offset is already a 1D coordinate, and jitter exists
in those modules to avoid grid-square artifacts in a Voronoi diagram, not
to make a binary interval split fairer. A plain, unjittered binary tree is
simpler and sufficient.

Cell tree, per parent seed
---------------------------
A cell is ``(depth, index)``, spanning ``[index*span, (index+1)*span)``
of the *parent seed's* length (mirroring burn-front: keyed by seed
content, tolerant of a shrunk live buffer via clamping, not the live
buffer itself -- see ``pos_fibonacci.py``'s docstring for why that would
be wrong for a stateless sweep; it is not wrong here because cells are
clamped the same way burn-front's are). ``span(depth) = ceil(width0 /
2**depth)``. A cell starts as a leaf; once its heat crosses
``SPLIT_THRESHOLD`` (and depth/span allow it), it forks and every later
visit to its span descends one level further, into whichever of its two
half-width children owns the offset.

Cells are created lazily, only along paths a gain has actually visited
(``_deposit``), and ``propose`` degrades gracefully to a coarser bin when
it meets a gap in the tree (an ancestor was trimmed, or a branch was
never visited) rather than erroring. Fuel/cooling/trim/spark mirror
burn-front's steel-wool metaphor; nothing here integrates over time, so
nothing can diverge (the field-model rule from
``core/analyzers/analyzer_navier_stokes.py`` applies equally to a tree).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import xxhash

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome

log = logging.getLogger(__name__)

STATE_VERSION = 1
MAX_SEEDS = 256  # LRU bound on per-seed trees
MAX_CELLS = 512  # sparse cap on cells per seed (root always kept)
MAX_DEPTH = 12  # finest span is width0 / 2**12 (or MIN_SPAN, whichever binds)
MIN_SPAN = 1  # a 1-byte span cannot split further
SPLIT_THRESHOLD = 3.0  # accumulated heat before a leaf forks
FUEL_BURN = 0.15  # fuel fraction one proposal consumes
FUEL_FLOOR = 1e-3  # keeps every heated cell selectable
SPARK_RATE = 0.10  # uniform escapes from the tree
COOL_EVERY = 32  # proposals between cooling steps
COOL_FACTOR = 0.9  # heat multiplier per cooling step
HEAT_FLOOR = 1e-4  # cells cooler than this are dropped


@dataclass
class _Cell:
    heat: float = 0.0
    fuel: float = 1.0
    forked: bool = False


@dataclass
class _Tree:
    width0: int  # parent seed length (>= 1)
    cells: dict[tuple[int, int], _Cell]
    proposals: int = 0


def _span(width0: int, depth: int) -> int:
    """Ceil-divided span of a cell at *depth*, floored at ``MIN_SPAN``."""
    return max(MIN_SPAN, -(-width0 // (1 << depth)))


def _num_bins(width0: int, span: int) -> int:
    return max(1, -(-width0 // span))


class PositionFractalScheduler:
    name = "fractal"

    def __init__(self, rng: RandPool) -> None:
        self._rng = rng
        self._trees: OrderedDict[int, _Tree] = OrderedDict()

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """Descend the seed's tree toward its hottest frontier; None when unknown."""
        tree = self._trees.get(self._key(data))
        if buf_len <= 0 or tree is None or not tree.cells:
            return None

        self._tick(tree)
        if not tree.cells:
            return None

        last = buf_len - 1
        if self._rng.randint(0, 9) < 1:  # SPARK_RATE = 0.10
            return self._rng.randint(0, last)

        depth, idx = 0, 0
        cell = tree.cells.get((0, 0))
        if cell is None:
            return None
        while cell.forked:
            left = tree.cells.get((depth + 1, idx * 2))
            right = tree.cells.get((depth + 1, idx * 2 + 1))
            lw = left.heat * left.fuel if left else 0.0
            rw = right.heat * right.fuel if right else 0.0
            if left is None and right is None:
                break
            go_right = self._rng.weighted_choice([0, 1], [lw + 1e-9, rw + 1e-9])
            nxt = right if go_right else left
            if nxt is None:
                break
            depth, idx, cell = depth + 1, idx * 2 + go_right, nxt

        cell.fuel = max(FUEL_FLOOR, cell.fuel * (1.0 - FUEL_BURN))
        span = _span(tree.width0, depth)
        lo = min(idx * span, last)
        hi = min(lo + span - 1, last)
        return self._rng.randint(lo, hi)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """On GAIN, deposit heat along the path to each offset's frontier cell."""
        offsets = [o for o in offsets if o >= 0]
        if outcome is not Outcome.GAIN or not offsets:
            return

        tree = self._tree_for(data)
        share = weight / len(offsets)
        for off in offsets:
            self._deposit(tree, off, share)
        self._trim(tree)

    def _deposit(self, tree: _Tree, offset: int, share: float) -> None:
        depth = 0
        while True:
            span = _span(tree.width0, depth)
            idx = min(offset // span, _num_bins(tree.width0, span) - 1)
            key = (depth, idx)
            cell = tree.cells.get(key)
            if cell is None:
                cell = tree.cells[key] = _Cell()
            cell.heat += share
            cell.fuel = 1.0
            if not cell.forked:
                if cell.heat >= SPLIT_THRESHOLD and span > MIN_SPAN and depth < MAX_DEPTH:
                    cell.forked = True
                return
            depth += 1

    def cell_count(self, data: bytes) -> int:
        tree = self._trees.get(self._key(data))
        return len(tree.cells) if tree else 0

    def cell_state(
        self, data: bytes, depth: int = 0, idx: int = 0
    ) -> tuple[float, float, bool] | None:
        """``(heat, fuel, forked)`` of one cell (a copy); None when unknown."""
        tree = self._trees.get(self._key(data))
        cell = tree.cells.get((depth, idx)) if tree else None
        return (cell.heat, cell.fuel, cell.forked) if cell else None

    def seed_count(self) -> int:
        return len(self._trees)

    def to_dict(self) -> dict:
        """Trees oldest-first, so a restore keeps the LRU order."""
        return {
            "version": STATE_VERSION,
            "trees": {
                k: (
                    t.width0,
                    {f"{d}:{i}": (c.heat, c.fuel, c.forked) for (d, i), c in t.cells.items()},
                    t.proposals,
                )
                for k, t in self._trees.items()
            },
        }

    def from_dict(self, data) -> None:
        """Replace every tree with *data*'s; a malformed payload clears them."""
        self._trees = OrderedDict()
        if not data:
            return
        try:
            if data.get("version") != STATE_VERSION:
                raise ValueError(f"version {data.get('version')!r}")
            trees = OrderedDict()
            for k, (width0, cells, proposals) in data["trees"].items():
                width0 = int(width0)
                if width0 < 1:
                    raise ValueError("tree width0 < 1")
                parsed: dict[tuple[int, int], _Cell] = {}
                for ck, (heat, fuel, forked) in cells.items():
                    d_s, i_s = ck.split(":")
                    parsed[(int(d_s), int(i_s))] = _Cell(float(heat), float(fuel), bool(forked))
                trees[int(k)] = _Tree(width0, parsed, int(proposals))
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            log.warning("fractal position state unreadable, starting fresh: %s", e)
            return

        # Oldest entries are dropped first, as _tree_for would have.
        while len(trees) > MAX_SEEDS:
            trees.popitem(last=False)
        self._trees = trees

    @staticmethod
    def _key(data: bytes) -> int:
        return xxhash.xxh3_64_intdigest(data)

    def _tree_for(self, data: bytes) -> _Tree:
        key = self._key(data)
        tree = self._trees.get(key)
        if tree is None:
            tree = self._trees[key] = _Tree(width0=max(1, len(data)), cells={})
            while len(self._trees) > MAX_SEEDS:
                self._trees.popitem(last=False)
        self._trees.move_to_end(key)
        return tree

    @staticmethod
    def _tick(tree: _Tree) -> None:
        """Cool every COOL_EVERY proposals; drop cells below HEAT_FLOOR.

        Dropping a forked cell only loses that branch's routing precision
        (a later ``propose`` treats the gap as an unvisited leaf and stops
        one level short) -- never a crash, since ``_deposit`` recreates
        cells lazily and ``propose`` bails to the last cell it still holds.
        """
        tree.proposals += 1
        if tree.proposals % COOL_EVERY:
            return

        for cell in tree.cells.values():
            cell.heat *= COOL_FACTOR
        tree.cells = {k: c for k, c in tree.cells.items() if k == (0, 0) or c.heat >= HEAT_FLOOR}

    @staticmethod
    def _trim(tree: _Tree) -> None:
        """Keep the MAX_CELLS hottest cells (root always kept)."""
        if len(tree.cells) <= MAX_CELLS:
            return

        root = tree.cells.get((0, 0))
        keep = sorted(
            (k for k in tree.cells if k != (0, 0)),
            key=lambda k: tree.cells[k].heat,
            reverse=True,
        )[: MAX_CELLS - 1]
        tree.cells = {k: tree.cells[k] for k in keep}
        if root is not None:
            tree.cells[(0, 0)] = root
