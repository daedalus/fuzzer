"""Calibrated execution-time anomaly detector.

Observes a stream of per-execution wall-clock times and, once a baseline
is established, returns a median-based threshold for flagging unusually
slow executions.  The detector is intentionally narrow: it does not
replace the hard ``f.timeout`` hang ceiling, it does not adjust timeouts
itself, and it does not speculate about *why* an execution is slow.  It
only answers the question "is this execution unusual *for this target*,
given the last ``window`` executions?"

Why median, not mean
--------------------
Latency distributions are heavily right-skewed: occasional huge stalls
(regex backtracking, hash-flooding, quadratic blowup) drag the mean
upward, so a mean-based threshold gets inflated by the very spikes it
is trying to detect.  The median stays anchored to the typical case
regardless of tail severity, making ``mult * median`` a robust
anomaly threshold.
"""

from __future__ import annotations

import bisect
import collections

DEFAULT_MIN_SAMPLES: int = 200
DEFAULT_THRESH_MULT: float = 2.0
# Retained exec times. Unbounded, threshold() sorted every exec ever seen on
# every exec: 1.1 ms/call at 10k execs, 235 ms at 1M.
DEFAULT_WINDOW: int = 4096


class ExecTimeCalibrator:
    """Flag unusually slow executions using a median-based threshold.

    Accumulates observed execution times and, once a minimum sample count
    is reached, returns a threshold of ``mult * median``.  Before that
    point :meth:`threshold` returns ``None`` so call sites fail closed
    rather than acting on an under-sampled baseline.

    Args:
        min_samples: Minimum observations before :meth:`threshold`
            returns a value.  Defaults to :data:`DEFAULT_MIN_SAMPLES`.
        window: Most recent observations the median is taken over.
            Defaults to :data:`DEFAULT_WINDOW`.
    """

    def __init__(
        self, min_samples: int = DEFAULT_MIN_SAMPLES, window: int = DEFAULT_WINDOW
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._min_samples = min_samples
        self._count = 0

        # Arrival order (eviction) mirrored by a sorted copy (O(1) median).
        self._times: collections.deque[float] = collections.deque(maxlen=window)
        self._sorted: list[float] = []

    def observe(self, elapsed: float) -> None:
        """Record one execution time.

        Args:
            elapsed: Wall-clock seconds for a completed execution.
        """
        # Read the victim before append() evicts it (see ExecutionTimeTracker).
        evicted = self._times[0] if len(self._times) == self._times.maxlen else None
        self._times.append(elapsed)
        self._count += 1

        bisect.insort(self._sorted, elapsed)
        if evicted is not None:
            self._sorted.pop(bisect.bisect_left(self._sorted, evicted))

    def threshold(self, mult: float = DEFAULT_THRESH_MULT) -> float | None:
        """Return the anomaly threshold, or ``None`` if the baseline is too small.

        The threshold is ``mult * median(observed_times)`` once at least
        :attr:`_min_samples` observations have been recorded, over the
        last ``window`` of them.

        Args:
            mult: Multiplier applied to the median.  Larger values reduce
                false positives at the cost of more false negatives.

        Returns:
            Threshold in seconds, or ``None`` if fewer than
            ``min_samples`` observations have been recorded.
        """
        if self._count < self._min_samples:
            return None
        return mult * self._median()

    @staticmethod
    def is_anomalous(elapsed: float, threshold: float) -> bool:
        """True when *elapsed* exceeds *threshold*.

        Named so call sites read as intent rather than a bare ``>``.

        Args:
            elapsed: Observed execution time.
            threshold: Calibrated anomaly threshold.

        Returns:
            True when the execution is anomalous.
        """
        return elapsed > threshold

    @property
    def count(self) -> int:
        """Number of observations recorded so far."""
        return self._count

    @property
    def median(self) -> float | None:
        """Median of recorded times, or ``None`` if no observations."""
        if not self._sorted:
            return None
        return self._median()

    def _median(self) -> float:
        # Same even-length rule as statistics.median: mean of the middle two.
        s = self._sorted
        mid = len(s) // 2
        if len(s) % 2:
            return s[mid]
        return (s[mid - 1] + s[mid]) / 2
