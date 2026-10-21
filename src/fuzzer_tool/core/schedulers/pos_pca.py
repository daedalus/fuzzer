"""PositionPCAScheduler: offsets drawn by PCA behaviour group (opt-in).

``PositionFinchScheduler`` weights a byte by how many edges its flip moved, so
its heat piles onto the bytes that trip the biggest path -- usually the shared
reject path, which is volume, not a field. This arm groups bytes by *which*
edges they move instead (``core/byte_pca.py``), drops the clutter component,
and spreads proposals across the remaining groups::

    byteflip rows --PCA--> groups   g0: bytes 4-7   (length-field branches)
                                    g1: byte  8     (checksum branches)
                                    g2: bytes 12-15 (tag dispatch)
    propose:  pick a group by its smoothed gain rate, then a byte in it
              with probability proportional to its squared score

A group is the unit of exploration: a seed with one hot byte and three quiet
fields gets the quiet fields a third of the draws each, where finch gives them
almost none. Per-seed gain/try counts per group bias the pick toward groups
that paid.

Declines (``None``: the arena turns it into a uniform offset charged to this
arm) when the seed has no groups, no group member lies inside the live buffer,
and with probability ``EPSILON`` otherwise. Tracker-style arm: it joins the
pool only once a drained effector map produced groups. Nothing is persisted.

Unmeasured: opt-in, not implied by ``--position-arena``, excluded from
``--hail-mary`` until ``pos-arena-pca`` beats ``pos-arena-uniform``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from fuzzer_tool.core.byte_pca import ByteGroups
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome

EPSILON = 0.1  # uniform escape (decline) probability
MAX_SEEDS = 512  # seeds with group statistics (LRU)
RATE_PRIOR_GAIN = 1.0  # Laplace smoothing of the per-group gain rate
RATE_PRIOR_TRIES = 2.0


@dataclass
class _Table:
    """Draw table of one seed at one buffer length."""

    groups: ByteGroups
    buf_len: int
    ids: np.ndarray  # group ids that have a member inside the buffer
    members: list[tuple[np.ndarray, np.ndarray]]  # per id: offsets, cumulative weights


@dataclass
class _Stats:
    gains: dict[int, int]
    tries: dict[int, int]


class PositionPCAScheduler:
    """Propose offsets by PCA behaviour group of the byteflip trace matrix."""

    name = "pca"

    def __init__(
        self,
        rng: RandPool,
        groups_of: Callable[[bytes], ByteGroups | None],
        ready: Callable[[], bool],
        key_of: Callable[[bytes], str],
    ) -> None:
        self._rng = rng
        self._groups_of = groups_of
        self._ready = ready
        self._key_of = key_of
        self._stats: OrderedDict[str, _Stats] = OrderedDict()
        # One-entry cache: the arena proposes for the same parent many times.
        self._table: tuple[str, _Table] | None = None

    def active(self) -> bool:
        """Arena gate: at least one seed has PCA groups."""
        return bool(self._ready())

    def seeds_tracked(self) -> int:
        """Number of seeds holding group statistics."""
        return len(self._stats)

    def group_rate(self, data: bytes, group: int) -> float:
        """Smoothed gain rate of *group* for *data*'s seed."""
        stats = self._stats.get(self._key_of(data))
        gains = stats.gains.get(group, 0) if stats else 0
        tries = stats.tries.get(group, 0) if stats else 0
        return (RATE_PRIOR_GAIN + gains) / (RATE_PRIOR_TRIES + tries)

    def _build(self, groups: ByteGroups, buf_len: int) -> _Table | None:
        inside = groups.offsets < buf_len
        if not inside.any():
            return None
        offs, comp, wts = groups.offsets[inside], groups.comp[inside], groups.weight[inside]
        ids = np.unique(comp)
        members = []
        for gid in ids:
            sel = comp == gid
            members.append((offs[sel], np.cumsum(wts[sel], dtype=np.float64)))
        return _Table(groups=groups, buf_len=buf_len, ids=ids, members=members)

    def _table_for(self, data: bytes, key: str, buf_len: int) -> _Table | None:
        groups = self._groups_of(data)
        if groups is None:
            return None
        cached = self._table
        if (
            cached is not None
            and cached[0] == key
            and cached[1].groups is groups
            and cached[1].buf_len == buf_len
        ):
            return cached[1]
        table = self._build(groups, buf_len)
        self._table = (key, table) if table is not None else None
        return table

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if buf_len < 1 or not data:
            return None

        key = self._key_of(data)
        table = self._table_for(data, key, buf_len)
        if table is None:
            return None
        if self._rng.random() < EPSILON:
            return None

        rates = np.array([self.group_rate(data, int(g)) for g in table.ids])
        cum = np.cumsum(rates)
        pick = min(
            int(np.searchsorted(cum, self._rng.random() * cum[-1], side="right")), len(cum) - 1
        )

        offs, wcum = table.members[pick]
        at = min(
            int(np.searchsorted(wcum, self._rng.random() * wcum[-1], side="right")), len(offs) - 1
        )
        return int(offs[at])

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Credit each group an offset fell in, once per round.

        *weight* is ignored: a group is credited a try and, on a gain, a gain,
        so one heavy round cannot swamp the rates. Offsets outside every group
        (clutter bytes, or the arm never served them) credit nothing.
        """
        offsets = [o for o in offsets if o >= 0]
        if not offsets:
            return
        groups = self._groups_of(data)
        if groups is None or groups.offsets.size == 0:
            return

        probe = np.asarray(offsets, dtype=np.int64)
        at = np.searchsorted(groups.offsets, probe)
        at = np.minimum(at, groups.offsets.size - 1)
        hit = groups.offsets[at] == probe
        if not hit.any():
            return

        key = self._key_of(data)
        stats = self._stats.get(key)
        if stats is None:
            stats = self._stats[key] = _Stats(gains={}, tries={})
            while len(self._stats) > MAX_SEEDS:
                self._stats.popitem(last=False)
        self._stats.move_to_end(key)

        won = outcome is Outcome.GAIN
        for gid in np.unique(groups.comp[at[hit]]):
            g = int(gid)
            stats.tries[g] = stats.tries.get(g, 0) + 1
            if won:
                stats.gains[g] = stats.gains.get(g, 0) + 1
