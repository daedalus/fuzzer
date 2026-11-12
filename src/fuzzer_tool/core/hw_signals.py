"""Signals derived from per-exec perf counter deltas.

    exec deltas ─┬─> HwMax.observe()      new campaign max x GROWTH -> admit
                 └─> FaultBudget.spend()  pages faulted since start -> restart

HwMax is honggfuzz's perfFeedback: an input that runs more instructions or
branches than any before it is kept (loops spun harder, deeper parse paths).
Growth, not strict ``>``: counts jitter, and each report admits an input, so
reports per counter stay bounded by log_GROWTH(max / first).

FaultBudget restarts a long-lived persistent target once the minor faults it
took since start pass a size budget. A minor fault is (roughly) a page first
touched, so a leaking target faults in fresh heap every iteration while a
steady one stops faulting after warm-up.
"""

# Smallest step that counts as a new maximum; inside it is jitter.
HW_MAX_GROWTH = 1.05

# Deterministic enough to compare across execs. CPU time and cache events
# move with load and are not tracked.
HW_MAX_COUNTERS = ("instructions", "branches")

PAGE_BYTES = 4096

# Fault budget when --rss-limit-mb is unset.
DEFAULT_FAULT_BUDGET_MB = 2048
_MIB = 1 << 20


class HwMax:
    """Campaign maximum per tracked counter; reports growth past the factor."""

    def __init__(
        self, counters: tuple[str, ...] = HW_MAX_COUNTERS, growth: float = HW_MAX_GROWTH
    ) -> None:
        self._counters = counters
        self._growth = growth
        self._max: dict[str, int] = {}
        self.hits = 0

    def observe(self, deltas: dict[str, int]) -> bool:
        """Whether *deltas* sets a new maximum on any tracked counter.

        Example: max 1000, growth 1.05 -> 1050 is jitter, 1051 reports.
        The first sighting is a baseline, not novelty.
        """
        new = False
        for name in self._counters:
            value = deltas.get(name, 0)
            if value <= 0:
                continue

            top = self._max.get(name)
            if top is None:
                self._max[name] = value
                continue

            if value > top * self._growth:
                self._max[name] = value
                new = True

        self.hits += new
        return new


class FaultBudget:
    """Page faults a persistent target may take before it is restarted."""

    def __init__(self, limit_mb: int) -> None:
        self._limit = limit_mb * _MIB // PAGE_BYTES
        self._used = 0

    def spend(self, faults: int) -> bool:
        """Charge *faults*; True once the budget is exceeded."""
        self._used += faults
        return self._used > self._limit

    def reset(self) -> None:
        """New target process: nothing spent."""
        self._used = 0
