"""Wall-time accumulators: target execution vs whole fuzz cycle.

Display only (host clock, see ``core/clock.py``). The stats tick drains a
window, so each stats line reports averages since the previous one::

    m = t.cycle_mark(); t.add_target(500_000); t.add_cycle(2_000_000, m)
    t.drain()  # (0.5, 2.0, 1.5) ms: tgt, cyc, overhead

Overhead is ``cycle_ns - target ns inside that cycle``, per cycle. Not
``cyc - tgt``: those divide by different counts, and ``tgt`` also holds
execs outside ``fuzz_one`` (calibration), which drove the gap negative.
"""

NS_PER_MS = 1_000_000


class ExecTimer:
    """Integer ns sums per window: no float drift, O(1) per sample."""

    __slots__ = (
        "_target_ns",
        "_target_n",
        "_cycle_ns",
        "_cycle_n",
        "_over_ns",
        "_mark_ns",
        "_life",
    )

    def __init__(self) -> None:
        self._reset()

        # Run-long target ns, never drained: cycle marks index into it.
        self._mark_ns = 0

        # Run totals [target_ns, target_n, cycle_ns, cycle_n, over_ns], kept across drains.
        self._life = [0, 0, 0, 0, 0]

    def _reset(self) -> None:
        self._target_ns = 0
        self._target_n = 0
        self._cycle_ns = 0
        self._cycle_n = 0
        self._over_ns = 0

    def add_target(self, ns: int) -> None:
        """One target execution took *ns*."""
        self._target_ns += ns
        self._target_n += 1
        self._mark_ns += ns

    def cycle_mark(self) -> int:
        """Token for a cycle start; pass it to ``add_cycle``."""
        return self._mark_ns

    def add_cycle(self, ns: int, mark: int) -> None:
        """One fuzz cycle (``fuzz_one``) took *ns*, started at *mark*."""
        self._cycle_ns += ns
        self._cycle_n += 1

        # Sequential in-cycle execs nest in the cycle interval, so this is
        # >= 0; the clamp guards overlapping (threaded) execs.
        self._over_ns += max(0, ns - (self._mark_ns - mark))

    def drain(self) -> tuple[float | None, float | None, float | None]:
        """Average ms per target exec, per cycle, and overhead per cycle; None when no samples."""
        win = (self._target_ns, self._target_n, self._cycle_ns, self._cycle_n, self._over_ns)
        self._life = [a + b for a, b in zip(self._life, win, strict=True)]
        self._reset()
        return _avg_ms(*win)

    def lifetime(self) -> tuple[float | None, float | None, float | None]:
        """Run-long averages, weighted per sample (unlike a mean of windows)."""
        return _avg_ms(*self._life)


def _avg_ms(
    tgt_ns: int, tgt_n: int, cyc_ns: int, cyc_n: int, over_ns: int
) -> tuple[float | None, float | None, float | None]:
    """Average ms per target exec, per cycle, and overhead per cycle; None when no samples."""
    tgt = tgt_ns / tgt_n / NS_PER_MS if tgt_n else None
    if not cyc_n:
        return tgt, None, None
    return tgt, cyc_ns / cyc_n / NS_PER_MS, over_ns / cyc_n / NS_PER_MS
