"""Wall-time accumulators: target execution vs whole fuzz cycle.

Display only (host clock, see ``core/clock.py``). The stats tick drains a
window, so each stats line reports averages since the previous one::

    t.add_target(500_000); t.add_cycle(2_000_000)
    t.drain()  # (0.5, 2.0) ms
"""

NS_PER_MS = 1_000_000


class ExecTimer:
    """Integer ns sums per window: no float drift, O(1) per sample."""

    __slots__ = ("_target_ns", "_target_n", "_cycle_ns", "_cycle_n")

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._target_ns = 0
        self._target_n = 0
        self._cycle_ns = 0
        self._cycle_n = 0

    def add_target(self, ns: int) -> None:
        """One target execution took *ns*."""
        self._target_ns += ns
        self._target_n += 1

    def add_cycle(self, ns: int) -> None:
        """One fuzz cycle (``fuzz_one``) took *ns*."""
        self._cycle_ns += ns
        self._cycle_n += 1

    def drain(self) -> tuple[float | None, float | None]:
        """Average ms per target exec and per cycle; None when no samples."""
        tgt = self._target_ns / self._target_n / NS_PER_MS if self._target_n else None
        cyc = self._cycle_ns / self._cycle_n / NS_PER_MS if self._cycle_n else None
        self._reset()
        return tgt, cyc
