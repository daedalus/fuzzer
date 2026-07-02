"""Execution time tracker for adaptive timeout calibration.

Maintains a running empirical CDF of observed execution times and uses
CRPS (Continuous Ranked Probability Score) to detect drift in the
target's runtime behavior. Enables moving from static "median * factor"
timeout tuning to calibrated percentile-based timeout selection with
honest uncertainty quantification.
"""

import bisect
import collections


class ExecutionTimeTracker:
    """Track execution times with CRPS-based calibration.

    Maintains a bounded sliding window of observed execution times,
    supports percentile-based timeout selection, and computes CRPS
    to measure how well the empirical CDF predicts future observations.

    Args:
        window_size: Max number of recent observations to retain.
        timeout_factor: Multiply the selected percentile by this to get timeout.
    """

    def __init__(self, window_size: int = 200, timeout_factor: float = 5.0):
        self.window_size = window_size
        self.timeout_factor = timeout_factor
        self._times: collections.deque = collections.deque(maxlen=window_size)
        self._sorted: list[float] = []
        self._crps_history: collections.deque = collections.deque(maxlen=100)
        self._total_observations = 0
        self._crps_sum = 0.0

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

        Uses incremental cdf_diff tracking: walk sorted observation points,
        maintain signed difference (F(y) - indicator), accumulate cdf_diff² * gap.
        Same pattern as EdgeTracker._cdf_norms for Wasserstein.
        """
        if not self._sorted:
            return 0.0
        n = len(self._sorted)
        crps = 0.0
        cdf_diff = 0.0
        prev = self._sorted[0]

        for i, val in enumerate(self._sorted):
            gap = val - prev
            if gap > 0:
                crps += cdf_diff * cdf_diff * gap
            # CDF at this point: fraction of observations ≤ val
            f_val = (i + 1) / n
            # 𝟙[y ≥ x]: 1 when val ≥ observation, 0 when val < observation
            indicator = 1.0 if val >= observation else 0.0
            cdf_diff = f_val - indicator
            prev = val

        # Region from last observation to observation (if obs > max):
        # F(y) = 1 for y ≥ max_val, 𝟙[y ≥ obs] = 0 for max_val ≤ y < obs
        # → cdf_diff = 1, contribution = 1² × gap
        max_val = self._sorted[-1]
        if observation > max_val:
            gap = observation - max_val
            crps += 1.0 * gap  # (1 - 0)² * gap

        return crps

    def suggested_timeout(self, percentile: float = 99.0) -> float:
        """Suggest a timeout based on the empirical CDF percentile.

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
        return self._sorted[idx] * self.timeout_factor

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
