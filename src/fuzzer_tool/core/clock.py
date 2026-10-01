"""Decision clock: wall time, or virtual time for reproducible seeded runs.

Every time read that can change what the fuzzer does (rewards scaled by
cost, seed deadlines, cadences, EPS-sized caps) goes through a ``Clock``.
Safety deadlines (subprocess timeouts, watchdogs) and display stay on the
host clock: a virtual clock must never decide when to kill a hung target.

Virtual time advances only on ``tick()`` -- once per target execution -- so
the same seed and the same inputs replay the same decisions::

    clock = Clock(ClockMode.VIRTUAL)
    clock.monotonic()   # 0.0
    clock.tick()
    clock.monotonic()   # VIRTUAL_EXEC_S
"""

import time
from enum import Enum

# Simulated cost of one target execution, seconds. 1 ms keeps EPS-derived
# quantities (1000 eps) inside the ranges the wall-clock heuristics expect.
VIRTUAL_EXEC_S = 1e-3

# Fixed epoch for virtual ``time()``: 2023-11-14T22:13:20Z.
VIRTUAL_EPOCH = 1_700_000_000.0


class ClockMode(Enum):
    """Where decision-relevant time comes from."""

    WALL = "wall"  # host clock
    VIRTUAL = "virtual"  # execution count × VIRTUAL_EXEC_S


class Clock:
    """``monotonic()`` / ``time()`` behind a mode switch.

    Wall mode binds the host functions directly, so the hot path pays one
    attribute lookup and no branch.
    """

    __slots__ = ("_mode", "_ticks", "monotonic", "time")

    def __init__(self, mode: ClockMode = ClockMode.WALL):
        self._mode = mode
        self._ticks = 0
        if mode is ClockMode.WALL:
            self.monotonic = time.monotonic
            self.time = time.time
            return
        self.monotonic = self._virtual_monotonic
        self.time = self._virtual_time

    @property
    def mode(self) -> ClockMode:
        return self._mode

    def tick(self) -> None:
        """Advance virtual time by one execution; no-op on the wall clock."""
        self._ticks += 1

    def _virtual_monotonic(self) -> float:
        return self._ticks * VIRTUAL_EXEC_S

    def _virtual_time(self) -> float:
        return VIRTUAL_EPOCH + self._ticks * VIRTUAL_EXEC_S


# Shared host clock: the default for objects built without one.
WALL_CLOCK = Clock()


def clock_of(owner) -> Clock:
    """*owner*'s decision clock, or the host clock when it has none.

    Services reach the clock through the Fuzzer they hold. Partial fuzzers
    (``__new__``-built, ``SimpleNamespace``, mocks) carry no real ``Clock``;
    they get wall time, which is what they had before the clock existed.
    """
    clock = getattr(owner, "_clock", None)
    return clock if isinstance(clock, Clock) else WALL_CLOCK
