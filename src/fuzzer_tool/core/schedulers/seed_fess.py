"""SeedFessScheduler: FESS (Feature Space Search, Shoham) as a seed arm.

FESS projects search states into cells of a small feature space and expands
from the least-explored cell, so no region of the space starves while the
heuristic is wrong for many states in a row (healeycodes.com Sokoban solver,
Shoham & Schaeffer FESS). Here a seed's features are edges, rare edges,
lineage depth and size, each log2-bucketed::

    seed --features--> (edges, rare, depth, size) --log2--> cell (1, 2, 3, 4)

    pick:  occupied cell with least effort  ->  its least-fuzzed seed
    record: effort[cell at pick time] += 1

Effort stays with the cell a seed was in when picked, so a seed migrating
cells (rarity drifts as the corpus grows) does not move its history. The
grid is capped at ``(FESS_BUCKET_CAP + 1) ** 4`` cells, which bounds the
effort ledger. Off by default and out of ``--hail-mary`` until it wins its A/B.
"""

from __future__ import annotations

from collections.abc import Callable

from fuzzer_tool.core.schedulers._arm_counts import ArmCounts

#: Highest log2 bucket per axis; 8 ** 4 cells bounds the effort ledger.
FESS_BUCKET_CAP = 7
#: Feature axes: edges, rare edges, lineage depth, size.
FESS_AXES = 4

Cell = tuple[int, int, int, int]


def _bucket(n: int, shift: int = 0) -> int:
    """Log2 bucket of *n*, floored at 0 and capped: 1 -> 1, 3 -> 2, 4 -> 3."""
    return min(max(0, n).bit_length() >> shift, FESS_BUCKET_CAP)


def fess_cell(edges: int, rare: int, depth: int, size: int) -> Cell:
    """Grid cell of one seed; size uses half-width buckets (bytes span more octaves)."""
    return (_bucket(edges), _bucket(rare), _bucket(depth), _bucket(size, 1))


def _valid_cell(cell) -> bool:
    if not isinstance(cell, tuple) or len(cell) != FESS_AXES:
        return False
    return all(isinstance(v, int) and 0 <= v <= FESS_BUCKET_CAP for v in cell)


class SeedFessScheduler(ArmCounts):
    """Least-explored feature cell first, then its least-fuzzed seed."""

    #: Effort counts, not Beta priors, drive selection.
    supports_priors = False

    def __init__(self, rng) -> None:
        if rng is None:
            raise ValueError("SeedFessScheduler requires a RandPool (Hard Rule 16)")
        super().__init__()
        self._rng = rng
        self._effort: dict[Cell, int] = {}  # cell -> execs charged at pick time
        self._cell_of: dict[str, Cell] = {}  # key -> cell at its last select

    def select_seed(self, seed_ids: list[str], cell: Callable[[str], Cell]) -> str:
        if not seed_ids:
            return ""
        members = self._group(seed_ids, cell)
        if len(seed_ids) == 1:
            return seed_ids[0]

        # Least-explored occupied cell (never charged = 0), then its
        # least-fuzzed seed (s + f is one per recorded exec).
        chosen = self._argmin({c: self._effort.get(c, 0) for c in members})
        return self._argmin({k: sum(self._counts[k]) for k in members[chosen]})

    def _group(self, seed_ids: list[str], cell: Callable[[str], Cell]) -> dict[Cell, list[str]]:
        """Register and trim live seeds, then bucket them by cell for record()."""
        for key in seed_ids:
            self.init_arm(key)
        self._trim(seed_ids)
        if len(self._cell_of) > len(self._counts):
            self._cell_of = {k: c for k, c in self._cell_of.items() if k in self._counts}

        members: dict[Cell, list[str]] = {}
        for key in seed_ids:
            c = cell(key)
            self._cell_of[key] = c
            members.setdefault(c, []).append(key)
        return members

    def _argmin(self, scores: dict):
        """Key with the lowest score; ties drawn from the pool, a unique one draws nothing."""
        low = min(scores.values())
        ties = [k for k, v in scores.items() if v == low]
        return ties[0] if len(ties) == 1 else self._rng.choice(ties)

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Count the exec on the seed and charge its pick-time cell."""
        super().record(name, success, weight)
        c = self._cell_of.get(name)
        if c is None:
            return
        self._effort[c] = self._effort.get(c, 0) + 1

    def to_dict(self) -> dict:
        """Effort ledger only: seed keys are corpus-local, cells are not."""
        return {"effort": dict(self._effort)}

    def from_dict(self, data) -> None:
        """Restore effort in place (the fuzzer's arm tuple holds this instance).

        Malformed entries are dropped, never trusted.
        """
        self._effort = {}
        effort = data.get("effort") if isinstance(data, dict) else None
        if not isinstance(effort, dict):
            return

        for c, n in effort.items():
            if not _valid_cell(c) or not isinstance(n, int) or n < 0:
                continue
            self._effort[c] = n
