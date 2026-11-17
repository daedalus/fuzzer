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

``jump`` (``--fess-inherit``, soksolv fess_search.go NewCellJump): a cell
occupied for the first time after the first select is picked next, once.

``mode`` (``--seed-fess-mode``, soksolv fess_search.go selectMove / fa2.go
pickCell): ``LEAST_EFFORT`` above, or ``TRIPLE``::

    turn % FESS_RANDOM_EVERY == 0  uniform random occupied cell
    even turn                      best cell: max s / (s + f) over its seeds
    odd turn                       round robin over cells in creation order
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from enum import Enum

from fuzzer_tool.core.run_mode import Switch
from fuzzer_tool.core.schedulers._arm_counts import ArmCounts

#: Highest log2 bucket per axis; 8 ** 4 cells bounds the effort ledger.
FESS_BUCKET_CAP = 7
#: Feature axes: edges, rare edges, lineage depth, size.
FESS_AXES = 4

#: TRIPLE: one pick in this many is a uniform random occupied cell (soksolv PFTypes).
FESS_RANDOM_EVERY = 8

Cell = tuple[int, int, int, int]


class FessMode(Enum):
    """Cell schedule of the FESS arm."""

    LEAST_EFFORT = "least-effort"
    TRIPLE = "triple"


def _bucket(n: int, shift: int = 0) -> int:
    """Log2 bucket of *n*, floored at 0 and capped: 1 -> 1, 3 -> 2, 4 -> 3."""
    return min(max(0, n).bit_length() >> shift, FESS_BUCKET_CAP)


def fess_cell(edges: int, rare: int, depth: int, size: int) -> Cell:
    """Grid cell of one seed; size uses half-width buckets (bytes span more octaves)."""
    return (_bucket(edges), _bucket(rare), _bucket(depth), _bucket(size, 1))


def cell_better(a: Cell, b: Cell) -> bool:
    """*a* strictly better than *b* (soksolv cellBetter): more edges, then more
    rare edges, then smaller size, then shallower lineage. Equal is not better."""
    return (a[0], a[1], -a[3], -a[2]) > (b[0], b[1], -b[3], -b[2])


def _valid_cell(cell) -> bool:
    if not isinstance(cell, tuple) or len(cell) != FESS_AXES:
        return False
    return all(isinstance(v, int) and 0 <= v <= FESS_BUCKET_CAP for v in cell)


def _valid_order(order) -> bool:
    """Creation order: a list of distinct valid cells."""
    if not isinstance(order, list) or not all(_valid_cell(c) for c in order):
        return False
    return len(set(order)) == len(order)


def _is_count(n) -> bool:
    """Non-negative int, bool excluded."""
    return type(n) is int and n >= 0


def _ignore(_members) -> None:
    """No-op cell observer (jump off)."""


class SeedFessScheduler(ArmCounts):
    """Least-explored feature cell first, then its least-fuzzed seed."""

    #: Effort counts, not Beta priors, drive selection.
    supports_priors = False

    def __init__(
        self, rng, jump: Switch = Switch.OFF, mode: FessMode = FessMode.LEAST_EFFORT
    ) -> None:
        if rng is None:
            raise ValueError("SeedFessScheduler requires a RandPool (Hard Rule 16)")
        if not isinstance(mode, FessMode):
            raise ValueError(f"mode must be a FessMode, got {mode!r}")
        super().__init__()
        self._rng = rng
        self._effort: dict[Cell, int] = {}  # cell -> execs charged at pick time
        self._cell_of: dict[str, Cell] = {}  # key -> cell at its last select

        # New-cell jump: cells ever occupied (<= grid size) and the queue of
        # first-time cells still owed a pick. Bound once: off costs nothing.
        self._jump_on = jump
        self._seen_cells: set[Cell] = set()
        self._jumps: deque[Cell] = deque()

        # TRIPLE: turn counter, round-robin cursor into creation order.
        self._mode = mode
        self._turn = 0
        self._cursor = 0
        self._order: list[Cell] = []

        triple = mode is FessMode.TRIPLE
        self._mode_pick = self._triple if triple else self._least_effort
        tracked = triple or jump is Switch.ON
        self._note_cells = self._note_new if tracked else _ignore
        self._pick_cell = self._jump_cell if jump is Switch.ON else self._mode_pick

    def select_seed(self, seed_ids: list[str], cell: Callable[[str], Cell]) -> str:
        if not seed_ids:
            return ""
        members = self._group(seed_ids, cell)
        self._note_cells(members)
        if len(seed_ids) == 1:
            return seed_ids[0]

        # Chosen cell, then its least-fuzzed seed (s + f is one per recorded exec).
        chosen = self._pick_cell(members)
        return self._argmin({k: sum(self._counts[k]) for k in members[chosen]})

    def _least_effort(self, members: dict[Cell, list[str]]) -> Cell:
        """Least-explored occupied cell (never charged = 0)."""
        return self._argmin({c: self._effort.get(c, 0) for c in members})

    def _note_new(self, members: dict[Cell, list[str]]) -> None:
        """Append first-time cells to creation order; queue them for a jump
        unless this is the priming select."""
        fresh = [c for c in members if c not in self._seen_cells]
        if not fresh:
            return
        if self._seen_cells and self._jump_on is Switch.ON:
            self._jumps.extend(fresh)
        self._seen_cells.update(fresh)
        self._order.extend(fresh)

    def _triple(self, members: dict[Cell, list[str]]) -> Cell:
        """Random every FESS_RANDOM_EVERY turns, else best (even) / round robin (odd)."""
        self._turn += 1
        if len(members) == 1:
            return next(iter(members))
        if self._turn % FESS_RANDOM_EVERY == 0:
            return self._rng.choice([c for c in self._order if c in members])
        if self._turn % 2 == 0:
            return self._best_cell(members)
        return self._round_robin(members)

    def _best_cell(self, members: dict[Cell, list[str]]) -> Cell:
        """Highest s / (s + f) summed over the cell's seeds; untried = 0."""
        rates = {}
        for c, keys in members.items():
            s = f = 0.0
            for k in keys:
                ks, kf = self._counts[k]
                s += ks
                f += kf
            rates[c] = -s / (s + f) if s + f > 0.0 else 0.0
        return self._argmin(rates)

    def _round_robin(self, members: dict[Cell, list[str]]) -> Cell:
        """Next occupied cell in creation order from the cursor."""
        order = self._order
        n = len(order)
        for step in range(n):
            i = (self._cursor + step) % n
            if order[i] in members:
                self._cursor = i + 1
                return order[i]
        return next(iter(members))  # unreachable: every member cell is in order

    def _jump_cell(self, members: dict[Cell, list[str]]) -> Cell:
        """Oldest owed new cell still occupied, else the normal pick."""
        while self._jumps:
            c = self._jumps.popleft()
            if c in members:
                return c
        return self._mode_pick(members)

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
        """Effort ledger and TRIPLE schedule: seed keys are corpus-local, cells are not."""
        return {
            "effort": dict(self._effort),
            "turn": self._turn,
            "cursor": self._cursor,
            "order": list(self._order),
        }

    def from_dict(self, data) -> None:
        """Restore effort and schedule in place (the fuzzer's arm tuple holds
        this instance). Malformed entries are dropped, never trusted.
        """
        data = data if isinstance(data, dict) else {}
        self._load_schedule(data)
        self._load_effort(data.get("effort"))

    def _load_schedule(self, data: dict) -> None:
        """Turn, cursor and creation order restore together or not at all."""
        self._turn, self._cursor, self._order = 0, 0, []
        self._seen_cells = set()
        turn, cursor, order = data.get("turn", 0), data.get("cursor", 0), data.get("order", [])
        if not (_is_count(turn) and _valid_order(order) and _is_count(cursor)):
            return
        if cursor > len(order):
            return
        self._turn, self._cursor, self._order = turn, cursor, list(order)
        self._seen_cells = set(order)

    def _load_effort(self, effort) -> None:
        self._effort = {}
        if not isinstance(effort, dict):
            return

        for c, n in effort.items():
            if not _valid_cell(c) or not isinstance(n, int) or n < 0:
                continue
            self._effort[c] = n
