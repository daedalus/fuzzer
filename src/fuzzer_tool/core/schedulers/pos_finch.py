"""PositionFinchScheduler: quantitative hot-byte proposals (Finch).

``PositionEffectorScheduler`` knows only LIVE or INERT per byte: it picks
uniformly among the bytes whose flip moved the trace. Finch weights a byte by
*how much* its flip moved it::

    seed bytes   H E A D E R p a d p a d L E N ...
    byteflip     3 1 1 9 . . . . . . . . 5 2 1       (edges moved, . = 0)
    proposals    ^ ^ ^ ^^^^^^^^           ^ ^ ^       share ~ min(moved, HEAT_CAP)

The magnitudes come free from the deterministic byteflip pass
(``OperatorEngine.effector_heat``). On top of that static map the arm keeps a
small per-seed adaptive bonus: offsets that were in a round which found new
coverage are heated by ``GAIN_BONUS`` (capped at ``HEAT_CAP``) and cooled by
the same amount on a miss. A miss alone never adds heat, and a gain can make a
byte proposable that the static map never covered (or covered as inert).

Bounded everywhere: ``HEAT_CAP`` stops one huge byte starving the rest,
``MAX_BONUS_OFFSETS`` bounds the bonus per seed, ``MAX_SEEDS`` bounds the seeds
tracked (LRU). Nothing is persisted.

Declines (``None``, which the arena turns into a uniform offset charged to this
arm) when there is no heat and no bonus inside the live buffer, and with
probability ``EPSILON`` otherwise. Tracker-style arm: it joins the pool only
while ``active()`` holds, i.e. once the engine has a finished map.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable, Sequence
from itertools import accumulate
from typing import Any

from fuzzer_tool.core.lru import LRUCache
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome

EPSILON = 0.1  # uniform escape (decline) probability
HEAT_CAP = 64  # per-byte weight ceiling, for static heat and for the bonus
GAIN_BONUS = 4  # added on a gain round, removed on a miss round
MAX_BONUS_OFFSETS = 32  # bonus entries kept per seed (oldest evicted)
MAX_SEEDS = 512  # seeds with a bonus table (LRU)

# (static heat offsets object, buf_len, offsets, cumulative weights)
_Table = tuple[Any, int, list[int], list[int]]


class PositionFinchScheduler:
    """Propose offsets with probability proportional to flip magnitude."""

    name = "finch"

    def __init__(
        self,
        rng: RandPool,
        heat_of: Callable[[bytes], tuple[Sequence[int], Sequence[int]] | None],
        ready: Callable[[], bool],
        key_of: Callable[[bytes], str],
    ) -> None:
        self._rng = rng
        self._heat_of = heat_of
        self._ready = ready
        self._key_of = key_of
        self._bonus: LRUCache = LRUCache(MAX_SEEDS)
        # One-entry cache: the arena proposes for the same parent many times.
        self._table: tuple[str, _Table] | None = None

    def active(self) -> bool:
        """Arena gate: at least one finished effector map exists."""
        return bool(self._ready())

    def bonus_size(self, data: bytes) -> int:
        """Number of bonus offsets held for *data*'s seed."""
        table = self._bonus.get(self._key_of(data))
        return len(table) if table else 0

    def seeds_tracked(self) -> int:
        """Number of seeds that currently hold a bonus table."""
        return len(self._bonus)

    def _build(self, key: str, buf_len: int, heat: Any) -> _Table | None:
        weights: dict[int, int] = {}
        offs = None
        if heat is not None:
            offs, mags = heat
            for off, mag in zip(offs, mags, strict=False):
                if mag > 0 and 0 <= off < buf_len:
                    weights[int(off)] = min(int(mag), HEAT_CAP)
        bonus = self._bonus.get(key)
        if bonus:
            for off, b in bonus.items():
                if off < buf_len:
                    weights[off] = weights.get(off, 0) + b
        if not weights:
            return None
        order = sorted(weights)
        return offs, buf_len, order, list(accumulate(weights[o] for o in order))

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if buf_len < 1 or not data:
            return None

        key = self._key_of(data)
        heat = self._heat_of(data)
        offs = heat[0] if heat is not None else None

        cached = self._table
        if (
            cached is not None
            and cached[0] == key
            and cached[1][0] is offs
            and cached[1][1] == buf_len
        ):
            table: _Table | None = cached[1]
        else:
            table = self._build(key, buf_len, heat)
            self._table = (key, table) if table is not None else None
        if table is None:
            return None

        if self._rng.random() < EPSILON:
            return None

        _, _, order, cum = table
        target = self._rng.random() * cum[-1]
        # bisect_right: a draw on a boundary belongs to the next byte.
        idx = min(bisect_right(cum, target), len(order) - 1)
        return order[idx]

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Heat the offsets behind a gain, cool them on a miss.

        *weight* is ignored: the bonus is a fixed step so a single round cannot
        dominate the static magnitudes.
        """
        n = len(data)
        if n == 0 or not offsets:
            return
        gain = outcome is Outcome.GAIN
        key = self._key_of(data)
        table = self._bonus.get(key)
        if table is None:
            if not gain:
                return  # a miss alone adds nothing
            table = OrderedDict()
            self._bonus[key] = table
            self._table = None  # an insert may have evicted the cached seed
        changed = False
        for raw in offsets:
            off = int(raw)
            if not 0 <= off < n:
                continue
            if gain:
                table[off] = min(table.get(off, 0) + GAIN_BONUS, HEAT_CAP)
                table.move_to_end(off)
                while len(table) > MAX_BONUS_OFFSETS:
                    table.popitem(last=False)
                changed = True
            elif off in table:
                left = table[off] - GAIN_BONUS
                if left > 0:
                    table[off] = left
                else:
                    del table[off]
                changed = True
        if not table:
            self._bonus.pop(key, None)
        if changed:
            self._table = None
