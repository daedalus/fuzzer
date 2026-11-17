"""PositionHotspotScheduler: the byte that blocks the most open comparisons.

Port of soksolv's hotspot table (``hotspots.go``, ``hotsides.go``): for each
blocker, the set of boxes it blocks; the "opener" moves the hottest one.
Here a blocker is an input offset and a box is an UNSOLVED ``CondStmt``
(``core/cond_stmt.py``) whose operand bytes cover it::

    data    . . . . A B C D E F . .
    cmp 0         [A B C D]           ABCD vs WXYZ, unsolved
    cmp 1             [C D E F]       CDEF vs QRST, unsolved
    table   4:{0} 5:{0} 6:{0,1} 7:{0,1} 8:{1} 9:{1}
    propose 6 or 7 (count 2), ties via RandPool

Operand matches stand in for taint, as in ``CondStmt.update_from_input``;
operands shorter than ``MIN_OPERAND`` match everywhere and are skipped.

Incremental: a table is built once per seed from the first ``MAX_SCAN``
unsolved conds of the current ``LazyConds`` (keeping ``MAX_CONDS`` that
occur in the seed). Each propose drops conds that left UNSOLVED
(``condstmt_solve`` marks them) from their offsets' sets, no rescan. A
table that empties is rebuilt from the next unsolved conds; a new cmplog
drain (new ``LazyConds``) clears every table.

Differs from ``cmplog`` (redqueen offsets + Weizz spans, weighted by tag
kind, blind to solved state) and from ``_wall_head`` (picks one comparison
for ``condstmt_solve``, not an offset for every operator): the rank is the
multiplicity of open comparisons per byte.

Passive: ``record()`` is a no-op; tables are a derived LRU cache, not
persisted. Declines with no table and on ``EPSILON`` of draws.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from fuzzer_tool.core.cond_stmt import CondState, CondStmt
from fuzzer_tool.core.hashing import hash64_len
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome

MAX_SEEDS = 256  # LRU bound on per-seed tables
MAX_SCAN = 256  # unsolved conds scanned per build
MAX_CONDS = 64  # conds kept per table (those that occur in the seed)
MAX_HITS = 8  # occurrences kept per operand
MIN_OPERAND = 2  # shorter operands match everywhere: noise
SCAN_BYTES = 1 << 12  # prefix searched: the default -m 4096 (build ~2 ms there)
EPSILON = 0.1  # uniform escape (decline) probability


@dataclass(slots=True)
class _Table:
    """Inverted table of one seed: offset -> cmpids it blocks."""

    members: list[tuple[CondStmt, tuple[int, ...]]]
    blocks: dict[int, set[int]]
    exhaustive: bool  # the scan saw every unsolved cond
    hot: list[int] | None = field(default=None)  # cached hottest tier

    def prune(self) -> bool:
        """Drop conds that left UNSOLVED (solved -> remove); True if it emptied."""
        if all(c.state is CondState.UNSOLVED for c, _ in self.members):
            return False
        keep = []
        for c, offs in self.members:
            if c.state is CondState.UNSOLVED:
                keep.append((c, offs))
                continue
            cid = c.base.cmpid
            for o in offs:
                ids = self.blocks[o]
                ids.discard(cid)
                if not ids:
                    del self.blocks[o]
        self.members = keep
        self.hot = None
        return not keep

    def hottest(self, buf_len: int) -> list[int]:
        """Offsets below *buf_len* blocking the most conds, ascending."""
        if self.hot is None:
            self.hot = _top(self.blocks, len(self.blocks) and max(self.blocks) + 1)
        if self.hot and self.hot[-1] < buf_len:
            return self.hot
        return _top(self.blocks, buf_len)


def _top(blocks: dict[int, set[int]], limit: int) -> list[int]:
    best, out = 0, []
    for o, ids in blocks.items():
        if o >= limit:
            continue
        n = len(ids)
        if n > best:
            best, out = n, [o]
        elif n == best:
            out.append(o)
    out.sort()
    return out


def _spots(data: bytes, cond: CondStmt) -> tuple[int, ...]:
    """Offsets covered by the cond's operands in *data* (capped)."""
    offs: set[int] = set()
    for op in (cond.base.op_a, cond.base.op_b):
        w = len(op)
        if w < MIN_OPERAND:
            continue
        idx = data.find(op)
        hits = 0
        while idx != -1 and hits < MAX_HITS:
            offs.update(range(idx, idx + w))
            hits += 1
            idx = data.find(op, idx + 1)
    return tuple(sorted(offs))


class PositionHotspotScheduler:
    """Propose the offset that blocks the most unsolved comparisons."""

    name = "hotspot"

    def __init__(self, rng: RandPool, conds_of: Callable[[], Any]) -> None:
        self._rng = rng
        self._conds_of = conds_of
        self._conds: Any = None  # the LazyConds the tables were built from
        self._tables: OrderedDict[int, _Table] = OrderedDict()

    def seed_count(self) -> int:
        return len(self._tables)

    def blocked(self, data: bytes) -> dict[int, int]:
        """Offset -> number of unsolved conds it blocks (diagnostics)."""
        table = self._table(data)
        if table is None:
            return {}
        return {o: len(ids) for o, ids in table.blocks.items()}

    # -- table maintenance --------------------------------------------------

    def _build(self, data: bytes, conds: Any) -> _Table:
        view = conds.unsolved_view()
        n = len(view)
        scan = min(n, MAX_SCAN)
        prefix = data[:SCAN_BYTES]

        members: list[tuple[CondStmt, tuple[int, ...]]] = []
        blocks: dict[int, set[int]] = {}
        for k in range(scan):
            c = view[k]
            offs = _spots(prefix, c)
            if not offs:
                continue
            members.append((c, offs))
            cid = c.base.cmpid
            for o in offs:
                blocks.setdefault(o, set()).add(cid)
            if len(members) >= MAX_CONDS:
                break
        return _Table(members, blocks, exhaustive=scan == n and len(members) < MAX_CONDS)

    def _table(self, data: bytes) -> _Table | None:
        conds = self._conds_of()
        if conds is None or not len(conds):
            return None
        if conds is not self._conds:  # new drain: every table is stale
            self._conds = conds
            self._tables.clear()

        key = hash64_len(data)
        table = self._tables.get(key)
        if table is not None:
            self._tables.move_to_end(key)
            # Rebuilt only when solving emptied it: an empty build stays put.
            if not table.prune() or table.exhaustive:
                return table

        table = self._build(data, conds)
        self._tables[key] = table
        self._tables.move_to_end(key)
        while len(self._tables) > MAX_SEEDS:
            self._tables.popitem(last=False)
        return table

    # -- protocol -----------------------------------------------------------

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if buf_len < 1 or not data:
            return None
        table = self._table(data)
        if table is None or not table.blocks:
            return None
        if self._rng.random() < EPSILON:
            return None

        hot = table.hottest(buf_len)
        if not hot:
            return None
        return self._rng.choice(hot)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """No-op: the arm is passive (the table comes from cmplog state)."""
