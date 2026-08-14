"""Calibrated execution-time anomaly detector.

Observes a stream of per-execution wall-clock times and, once a baseline
is established, returns a median-based threshold for flagging unusually
slow executions.  The detector is intentionally narrow: it does not
replace the hard ``f.timeout`` hang ceiling, it does not adjust timeouts
itself, and it does not speculate about *why* an execution is slow.  It
only answers the question "is this execution unusual *for this target*,
given the executions seen so far?"

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

import statistics

DEFAULT_MIN_SAMPLES: int = 200
DEFAULT_THRESH_MULT: float = 2.0


class ExecTimeCalibrator:
    """Flag unusually slow executions using a median-based threshold.

    Accumulates observed execution times and, once a minimum sample count
    is reached, returns a threshold of ``mult * median``.  Before that
    point :meth:`threshold` returns ``None`` so call sites fail closed
    rather than acting on an under-sampled baseline.

    Args:
        min_samples: Minimum observations before :meth:`threshold`
            returns a value.  Defaults to :data:`DEFAULT_MIN_SAMPLES`.
    """

    def __init__(self, min_samples: int = DEFAULT_MIN_SAMPLES) -> None:
        self._min_samples = min_samples
        self._times: list[float] = []

    def observe(self, elapsed: float) -> None:
        """Record one execution time.

        Args:
            elapsed: Wall-clock seconds for a completed execution.
        """
        self._times.append(elapsed)

    def threshold(self, mult: float = DEFAULT_THRESH_MULT) -> float | None:
        """Return the anomaly threshold, or ``None`` if the baseline is too small.

        The threshold is ``mult * median(observed_times)`` once at least
        :attr:`_min_samples` observations have been recorded.

        Args:
            mult: Multiplier applied to the median.  Larger values reduce
                false positives at the cost of more false negatives.

        Returns:
            Threshold in seconds, or ``None`` if fewer than
            ``min_samples`` observations have been recorded.
        """
        if len(self._times) < self._min_samples:
            return None
        return mult * statistics.median(self._times)

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
        return len(self._times)

    @property
    def median(self) -> float | None:
        """Median of recorded times, or ``None`` if no observations."""
        if not self._times:
            return None
        return statistics.median(self._times)
