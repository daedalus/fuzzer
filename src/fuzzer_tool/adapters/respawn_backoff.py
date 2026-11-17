"""Respawn backoff: pace restarts of a target that keeps dying at once.

Port of soksolv ``restart_backoff.go``. Restarting a target that dies the
moment it starts spins fork/exec as fast as the kernel allows and starves
the run. A phase that lived ``QUICK_LIFE_S`` or longer restarts at once as
before; each quick failure in a row sleeps longer::

    started() ── runs ── dies ── failed(DEATH)
                                   │ life >= 250 ms → quick = 0, no sleep
                                   │ quick + 1 >= breaker → RespawnLoopError
                                   └ sleep min(20 ms << min(quick, 6), 1 s), quick += 1
                                     (never past deadline - 1 s)

A start that never came up (``failed(START)``) is always quick. The circuit
breaker turns an endless respawn loop into one clear error, which the fuzz
loop logs and turns into a persisted, aborted run.

Safety pacing, not a decision: it runs on the host clock, never the
virtual decision clock (``core/clock.py``).
"""

import time
from collections.abc import Callable
from enum import Enum

QUICK_LIFE_S = 0.25  # phases shorter than this are quick failures
BASE_SLEEP_S = 0.02  # first quick failure's pause
CAP_SLEEP_S = 1.0  # longest pause
MAX_SHIFT = 6  # 20 ms << 6 already exceeds the cap
DEADLINE_MARGIN_S = 1.0  # soksolv: stop sleeping 1 s before the deadline
# Quick failures in a row before giving up: ~11 s of pauses at the cap.
BREAKER_QUICK = 16


class Failure(Enum):
    """How an instance failed."""

    START = "start"  # never came up: always quick
    DEATH = "death"  # came up, then died: quick when it lived < QUICK_LIFE_S


class RespawnLoopError(RuntimeError):
    """The target died within QUICK_LIFE_S of every start, BREAKER_QUICK times."""


class RespawnBackoff:
    """Consecutive quick-failure count and the pause it buys.

    Args:
        now: Monotonic clock, seconds.
        sleep: Pause function, seconds.
        deadline: ``now()`` value past which no pause may run; None = none.
        breaker: Quick failures in a row that raise ``RespawnLoopError``.
    """

    __slots__ = ("_born", "_breaker", "_deadline", "_now", "_quick", "_sleep")

    def __init__(
        self,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        deadline: float | None = None,
        breaker: int = BREAKER_QUICK,
    ):
        self._now = now
        self._sleep = sleep
        self._deadline = deadline
        self._breaker = breaker
        self._quick = 0
        self._born = now()

    def started(self) -> None:
        """An instance came up: its lifetime starts now."""
        self._born = self._now()

    def failed(self, how: Failure) -> float:
        """Record a failure; pause when quick. Returns the seconds slept."""
        now = self._now()
        if how is Failure.DEATH and now - self._born >= QUICK_LIFE_S:
            self._quick = 0
            return 0.0

        if self._quick + 1 >= self._breaker:
            raise RespawnLoopError(
                f"target respawn loop: {self._quick + 1} failures in a row within "
                f"{QUICK_LIFE_S:g}s of start; giving up (--respawn-backoff)"
            )

        d = min(BASE_SLEEP_S * (1 << min(self._quick, MAX_SHIFT)), CAP_SLEEP_S)
        self._quick += 1
        if self._deadline is not None:
            d = max(min(d, self._deadline - now - DEADLINE_MARGIN_S), 0.0)
        if d > 0.0:
            self._sleep(d)
        return d


class _NullBackoff:
    """Flag off: restarts run immediately, exactly as without the feature."""

    __slots__ = ()

    def started(self) -> None:
        return None

    def failed(self, how: Failure) -> float:
        return 0.0


# Shared default for every runner; stateless.
NO_BACKOFF = _NullBackoff()
