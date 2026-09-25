"""PositionRoundRobinScheduler: deterministic offset cycling.

The position-arena counterpart of ``core/schedulers/op_round_robin.py`` and
``core/schedulers/seed_round_robin.py``. ``services/position_arena.py``
runs one tournament over position proposers (uniform, sensitivity, te,
phase, mi, crash_mi, region, burn_front, canary, ...) under ``pos_<n>``
Elo keys; this module gives that tournament the same deterministic,
signal-free baseline the operator and seed arenas already have.

Unlike ``PositionCanaryScheduler``, round-robin is not a deliberately-bad
floor: it is a real, if simple, position policy -- bin a seed's offsets
the same way ``pos_burn_front.py`` does (``width = ceil(len(data) /
MAX_BINS)``, LRU-bounded over ``MAX_SEEDS`` seeds) and cycle through the
bins in order, ignoring the outcome signal entirely. That is also why,
mirroring ``op_round_robin``'s standalone reach into ``select_position``'s
non-arena candidate list alongside burn-front, it needs no arbiter to be
worth running: it is one more candidate in ``OperatorEngine.
select_position``'s uniform pick even without ``--position-arena``.

``record()`` is kept for interface parity with the other position
schedulers only (a rated strategy needs an outcome signal for its Elo
match to resolve, same as ``RoundRobinScheduler.record`` on the operator
side); the cycling in ``propose`` never reads it.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import xxhash

from fuzzer_tool.core.schedulers.pos_base import Outcome

MAX_BINS = 4096  # offsets per seed are binned down to this, as pos_burn_front
MAX_SEEDS = 256  # LRU bound on per-seed cycles


@dataclass
class _Cycle:
    width: int
    index: int = 0


class PositionRoundRobinScheduler:
    """Simple round-robin scheduler for position selection.

    Cycles through a seed's offset bins in fixed order. Provides a
    deterministic baseline for the position-arena Elo tournament, and
    needs no arbiter to run standalone (see the module docstring).
    """

    name = "round_robin"

    #: No meaningful priors for round-robin, mirrors RoundRobinScheduler /
    #: SeedRoundRobinScheduler.
    supports_priors = False

    def __init__(self) -> None:
        self._seeds: OrderedDict[int, _Cycle] = OrderedDict()

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """First byte of the next bin in cycle order; never declines on a live buffer."""
        if buf_len <= 0:
            return None
        cyc = self._cycle_for(data)
        num_bins = max(1, -(-len(data) // cyc.width)) if data else 1

        b = cyc.index % num_bins
        cyc.index += 1

        last = buf_len - 1
        return min(b * cyc.width, last)

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """No-op: cycling ignores the outcome signal entirely, same as
        RoundRobinScheduler.record / SeedRoundRobinScheduler.record.
        """

    def seed_count(self) -> int:
        return len(self._seeds)

    @staticmethod
    def _key(data: bytes) -> int:
        return xxhash.xxh3_64_intdigest(data)

    def _cycle_for(self, data: bytes) -> _Cycle:
        key = self._key(data)
        cyc = self._seeds.get(key)
        if cyc is None:
            width = max(1, -(-len(data) // MAX_BINS)) if data else 1
            cyc = self._seeds[key] = _Cycle(width=width)
            while len(self._seeds) > MAX_SEEDS:
                self._seeds.popitem(last=False)
        self._seeds.move_to_end(key)
        return cyc
