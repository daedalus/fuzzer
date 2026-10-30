"""Adaptive write batch size: hill-climb on observed end-to-end seeds/s.

Each flush closes a window (replay + write since the previous flush)::

    rate = seeds in window / window seconds
    rate < previous rate * (1 - tol)  -> reverse direction (tol absorbs noise)
    size *= FACTOR (up) or size //= FACTOR (down), clamped to [lo, hi]

Bigger batches amortize per-write cost (SQLite commit, pickle frame); past
the point where writes are noise next to replay, the rate stops improving
and the climb oscillates around the best size. Memory is the caller's to
cap (edge rows buffered), not this controller's.
"""

from __future__ import annotations

import time
from collections.abc import Callable

FACTOR = 2


class AdaptiveBatch:
    """Batch size that follows the throughput gradient, one step per window."""

    def __init__(
        self,
        start: int,
        lo: int,
        hi: int,
        tol: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not 1 <= lo <= start <= hi:
            raise ValueError(f"need 1 <= lo <= start <= hi, got {lo}, {start}, {hi}")

        self.size = start
        self._lo = lo
        self._hi = hi
        self._keep = 1.0 - tol  # rate must fall below prev * keep to reverse
        self._clock = clock
        self._last = clock()
        self._prev_rate: float | None = None
        self._up = True

    def observe(self, seeds: int) -> None:
        """Close the window that produced *seeds* and step the size."""
        now = self._clock()
        dt = now - self._last
        self._last = now
        if dt <= 0 or seeds <= 0:
            return

        rate = seeds / dt
        if self._prev_rate is not None and rate < self._prev_rate * self._keep:
            self._up = not self._up
        self._prev_rate = rate

        step = self.size * FACTOR if self._up else self.size // FACTOR
        self.size = min(self._hi, max(self._lo, step))
