"""Execution time tracker for adaptive timeout calibration.

Maintains a running empirical CDF of observed execution times and uses
CRPS (Continuous Ranked Probability Score) to detect drift in the
target's runtime behavior. Enables moving from static "median * factor"
timeout tuning to calibrated percentile-based timeout selection with
honest uncertainty quantification.
"""

import bisect
import collections

import numpy as _np

from fuzzer_tool.core.running_stats import RunningMoments

# Skewness above this value flags the input family as "tail-risk":
# occasional large excursions (regex backtracking, hash-flood) against
# a modest mean — distinct from "generally slow" (high mean) or "noisy"
# (high stddev).
TAIL_RISK_SKEWNESS_THRESHOLD = 2.0


class ExecutionTimeTracker:
    """Track execution times with CRPS-based calibration.

    Maintains a bounded sliding window of observed execution times,
    supports percentile-based timeout selection, and computes CRPS
    to measure how well the empirical CDF predicts future observations.

    Args:
        window_size: Max number of recent observations to retain.
        timeout_factor: Multiply the selected percentile by this to get timeout.
    """

    def __init__(
        self, window_size: int = 200, timeout_factor: float = 1.0, correction_factor: float = 1.5
    ):
        self.window_size = window_size
        self.timeout_factor = timeout_factor
        self.correction_factor = correction_factor
        self._times: collections.deque = collections.deque(maxlen=window_size)
        self._sorted: list[float] = []
        self._crps_history: collections.deque = collections.deque(maxlen=100)
        self._total_observations = 0
        self._crps_sum = 0.0
        self._moments: RunningMoments = RunningMoments(window=window_size)
        # n -> arange(1, n+1)/n, reused across executions (see _compute_crps)
        self._crps_ramp_cache: dict = {}

    def record(self, elapsed: float) -> float:
        """Record an execution time and return the CRPS score.

        The CRPS score measures how well the existing empirical CDF
        predicted this new observation. Lower = better calibrated.

        Args:
            elapsed: Wall-clock seconds for this execution.

        Returns:
            CRPS score for this observation against the running forecast.
        """
        crps = self._compute_crps(elapsed)
        self._crps_history.append(crps)
        self._crps_sum += crps
        self._total_observations += 1

        self._times.append(elapsed)
        self._moments.update(elapsed)
        bisect.insort(self._sorted, elapsed)
        if len(self._sorted) > self.window_size:
            # Remove oldest observation from sorted list
            oldest = self._times[0] if len(self._times) > 1 else None
            if oldest is not None:
                idx = bisect.bisect_left(self._sorted, oldest)
                if idx < len(self._sorted):
                    self._sorted.pop(idx)

        return crps

    def _compute_crps(self, observation: float) -> float:
        """CRPS of a point observation against the running empirical CDF.

        CRPS(F, x) = ∫(F(y) - 𝟙[y ≥ x])² dy

        Vectorized: for a sorted walk the loop is the prefix recurrence
        crps = Σᵢ (i/n - 𝟙[vᵢ≥x])² · (vᵢ₊₁ - vᵢ), computed with numpy in one
        shot. The legacy `gap > 0` guard is dead code — a sorted walk has
        gap ≥ 0 always, and a zero gap contributes 0 regardless.
        """
        if not self._sorted:
            return 0.0

        arr = _np.asarray(self._sorted, dtype=_np.float64)
        n = len(arr)
        # i/n is constant for a given n, and n is pinned at window_size for
        # nearly the whole run — cache it instead of reallocating an arange
        # on every execution (this runs once per exec, on the hot path).
        ramp = self._crps_ramp_cache.get(n)
        if ramp is None:
            ramp = _np.arange(1, n + 1) / n
            self._crps_ramp_cache[n] = ramp
        cd = ramp - (arr >= observation)
        # np.dot over the elementwise square avoids materializing a third
        # temp array (np.sum(cd²·diff) does) — this runs once per exec.
        crps = float(_np.dot(cd[:-1] * cd[:-1], _np.diff(arr)))
        # Region from last observation to observation (if obs > max):
        # F(y) = 1 for y ≥ max_val, 𝟙[y ≥ obs] = 0 for max_val ≤ y < obs
        max_val = arr[-1]
        if observation > max_val:
            crps += observation - max_val
        return crps

    def suggested_timeout(self, percentile: float = 99.0) -> float:
        """Suggest a timeout based on the empirical CDF percentile + std dev.

        Args:
            percentile: Which percentile to use (0-100). Default 99th.

        Returns:
            Timeout in seconds.
        """
        if not self._sorted:
            return 5.0  # fallback
        idx = min(
            int(len(self._sorted) * percentile / 100),
            len(self._sorted) - 1,
        )
        p99 = self._sorted[idx]
        # Add one standard deviation for headroom instead of a flat multiplier.
        # This adapts to the actual variance: tight distributions get small
        # headroom, high-variance targets get more.
        return p99 + self._moments.stddev * self.correction_factor

    def mean_crps(self) -> float:
        """Mean CRPS over recent observations — lower is better calibrated."""
        if not self._crps_history:
            return 0.0
        return sum(self._crps_history) / len(self._crps_history)

    def crps_trend(self) -> float:
        """Slope of CRPS over last 20 observations — positive = degrading calibration."""
        if len(self._crps_history) < 10:
            return 0.0
        recent = list(self._crps_history)[-20:]
        n = len(recent)
        mean_x = (n - 1) / 2
        mean_y = sum(recent) / n
        num = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(recent))
        den = sum((i - mean_x) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

    @property
    def count(self) -> int:
        return self._total_observations

    @property
    def p50(self) -> float:
        if not self._sorted:
            return 0.0
        return self._sorted[len(self._sorted) // 2]

    @property
    def p99(self) -> float:
        if not self._sorted:
            return 0.0
        return self._sorted[min(int(len(self._sorted) * 0.99), len(self._sorted) - 1)]

    @property
    def variance(self) -> float:
        """Variance of observed execution times."""
        return self._moments.variance

    @property
    def std(self) -> float:
        """Standard deviation of observed execution times."""
        return self._moments.stddev

    @property
    def skewness(self) -> float:
        """Skewness of observed execution times."""
        return self._moments.skewness

    @property
    def tail_risk(self) -> bool:
        """True when execution times show heavy right skew.

        Heavy right skew with a modest mean is the profile of
        algorithmic-complexity inputs (regex backtracking, hash-flood)
        that occasionally trigger a big execution-time excursion.
        """
        return self._moments.skewness > TAIL_RISK_SKEWNESS_THRESHOLD
