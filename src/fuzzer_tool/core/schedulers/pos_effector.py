"""PositionEffectorScheduler: land mutations on bytes the target reads.

The deterministic stage's byteflip 8/8 pass already executes one mutant per
byte and compares its trace with the seed's (``services/operators.py``,
``DeterministicEffectorMap``). That verdict -- LIVE (the trace moved) or
INERT (it did not) -- used to gate only the arithmetic and interesting-value
passes, and was dropped with the queue. The engine now keeps the LIVE offsets
(``OperatorEngine.effector_live``) and this arm hands them to every operator::

    seed bytes   H E A D E R p a d p a d L E N ...
    byteflip     L L L L L L I I I I I I L L L      (L live, I inert)
    proposals    ^ ^ ^ ^ ^ ^             ^ ^ ^      uniform over L

Zero extra executions: the evidence was already paid for.

Only bytes the pass covered are known. A long seed's deterministic stage
covers ``MAX_DET_MUTATIONS // 33`` bytes (rotated by ``fuzz_count``), so the
rest of it is never proposed except through the ``EPSILON`` escape.

Passive: ``record()`` is a no-op and nothing is persisted. Declines (``None``,
which the arena turns into a uniform offset charged to this arm) when the
seed has no finished map, no LIVE byte, or no LIVE byte inside a shrunk live
buffer, and with probability ``EPSILON`` otherwise.

Tracker-style arm (``PositionArena._add_trackers``): it joins the pool only
while ``active()`` holds, i.e. once the engine has a finished map.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Sequence

from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome

EPSILON = 0.1  # uniform escape (decline) probability


class PositionEffectorScheduler:
    """Propose uniformly among a seed's byteflip-LIVE offsets."""

    name = "effector"

    def __init__(
        self,
        rng: RandPool,
        live_of: Callable[[bytes], Sequence[int] | None],
        ready: Callable[[], bool],
    ) -> None:
        self._rng = rng
        self._live_of = live_of
        self._ready = ready

    def active(self) -> bool:
        """Arena gate: at least one finished effector map exists."""
        return bool(self._ready())

    def propose(self, data: bytes, buf_len: int) -> int | None:
        if buf_len < 1 or not data:
            return None

        live = self._live_of(data)
        if not live:
            return None

        if self._rng.random() < EPSILON:
            return None

        # Sorted offsets: the ones inside a shrunk buffer are a prefix.
        k = bisect_left(live, buf_len)
        if k == 0:
            return None

        return int(live[self._rng.randint(0, k - 1)])

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """No-op: the map comes from the deterministic stage, not outcomes."""
