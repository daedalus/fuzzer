"""Wall-time accumulators: target execution vs whole fuzz cycle.

Display only (host clock, see ``core/clock.py``). The stats tick drains a
window, so each stats line reports averages since the previous one::

    t.add_target(500_000); t.add_cycle(2_000_000)
    t.drain()  # (0.5, 2.0) ms
"""

NS_PER_MS = 1_000_000


class ExecTimer:
    """Integer ns sums per window: no float drift, O(1) per sample."""

    __slots__ = ("_target_ns", "_target_n", "_cycle_ns", "_cycle_n", "_life")

    def __init__(self) -> None:
        self._reset()

        # Run totals [target_ns, target_n, cycle_ns, cycle_n], kept across drains.
        self._life = [0, 0, 0, 0]

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
        win = (self._target_ns, self._target_n, self._cycle_ns, self._cycle_n)
        self._life = [a + b for a, b in zip(self._life, win, strict=True)]
        self._reset()
        return _avg_ms(*win)

    def lifetime(self) -> tuple[float | None, float | None]:
        """Run-long averages, weighted per sample (unlike a mean of windows)."""
        return _avg_ms(*self._life)


def _avg_ms(tgt_ns: int, tgt_n: int, cyc_ns: int, cyc_n: int) -> tuple[float | None, float | None]:
    """Average ms per target exec and per cycle; None when no samples."""
    tgt = tgt_ns / tgt_n / NS_PER_MS if tgt_n else None
    cyc = cyc_ns / cyc_n / NS_PER_MS if cyc_n else None
    return tgt, cyc
