"""Execution time tracker for adaptive timeout calibration.

Maintains a running empirical CDF of observed execution times and uses
CRPS (Continuous Ranked Probability Score) to detect drift in the
target's runtime behavior. Enables moving from static "median * factor"
timeout tuning to calibrated percentile-based timeout selection with
honest uncertainty quantification.
"""

import bisect
import collections
import math

from fuzzer_tool.core.gaussian import norm_cdf
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
        # Moments of log(elapsed), the parameters of the lognormal forecast
        # _compute_crps scores against. Separate from _moments, which stays
        # on the raw-second axis because its consumers (suggested_timeout's
        # headroom term, `variance`, the tail-risk skewness gate) all want
        # seconds, and a log would flatten exactly the right tail that gate
        # is looking for.
        self._log_moments: RunningMoments = RunningMoments(window=window_size)

    # _compute_crps used to rebuild a numpy array from self._sorted and run
    # a dot + diff over it on every call -- measured at ~2% of total
    # fuzz-loop wall-clock on a cProfile run against nop_target, and
    # mitigated by scoring only every 8th observation once the window was
    # full. The closed form below is O(1) in the window size (5.78us ->
    # 0.63us at n=200, 9.2x), so there is nothing left to subsample and the
    # sampling branch is gone: every observation is scored again, which
    # also takes `execution_time.crps` out of the periodic-cadence
    # co-firing set that core/cadence.py tracks.

    def record(self, elapsed: float) -> float:
        """Record an execution time and return a CRPS score.

        The CRPS score measures how well the running forecast predicted
        this new observation. Lower = better calibrated.

        Args:
            elapsed: Wall-clock seconds for this execution.

        Returns:
            CRPS score against the running forecast, in seconds.
        """
        self._total_observations += 1
        crps = self._compute_crps(elapsed)
        self._crps_history.append(crps)
        self._crps_sum += crps

        # _times is a deque(maxlen=window_size): append() evicts the oldest
        # value itself, so it must be read BEFORE the append. Reading
        # _times[0] afterwards removed the second-oldest from _sorted and
        # left the evicted value there for good -- an early outlier then
        # held the percentiles (and suggested_timeout()) up indefinitely.
        evicted = self._times[0] if len(self._times) == self._times.maxlen else None
        self._times.append(elapsed)
        self._moments.update(elapsed)
        if elapsed > 0.0:
            self._log_moments.update(math.log(elapsed))
        bisect.insort(self._sorted, elapsed)
        if evicted is not None:
            self._sorted.pop(bisect.bisect_left(self._sorted, evicted))

        return crps

    def _compute_crps(self, observation: float) -> float:
        """CRPS of a point observation against the running lognormal forecast.

        CRPS(F, x) = ∫(F(y) - 𝟙[y ≥ x])² dy, which for F = LogN(mu, sigma)
        integrates in closed form (Baran & Lerch 2015, eq. 4):

            CRPS = x·(2Φ(ω) - 1) - 2·e^(mu + sigma²/2)·(Φ(ω - sigma)
                                                        + Φ(sigma/√2) - 1)

        with ω = (ln x - mu)/sigma. Three Φ evaluations, i.e. three
        ``math.erf`` calls, and no dependence on the window size at all.

        Why lognormal rather than the Gaussian closed form, which is one Φ
        cheaper: execution times are positive and right-skewed -- this very
        class carries ``TAIL_RISK_SKEWNESS_THRESHOLD = 2.0`` to detect that
        tail -- so a normal forecast is misspecified on its own terms and
        would put forecast mass below zero. On the log axis the multiplicative
        noise that generates timing spread (cache state, scheduler, branch
        counts) is additive, which is where the CLT actually applies.

        This replaces an O(n) walk over the sorted window. The two are not
        the same estimator: the old one scored the *empirical* CDF of the
        window, this scores a two-parameter fit to it. On a lognormal sample
        the two agree closely (mean CRPS 2.90e-4 vs 2.83e-4 over 2000 draws
        at n=200), and the parametric form extrapolates into the tail where
        the empirical CDF is flat by construction. The empirical estimator
        is still available exactly where it is needed: ``suggested_timeout``,
        ``p50`` and ``p99`` read ``_sorted`` and are deliberately untouched,
        since a hang threshold should come from observed times, not from a
        model's extrapolation.

        Degenerate cases return the CRPS of the point forecast they really
        are -- |x - m|, the correct limit as sigma -> 0 -- rather than
        dividing by zero: a target with genuinely constant timing, or a
        window not yet holding two distinct values, reaches this.
        """
        n = self._log_moments.count
        if n < 1:
            return 0.0

        mu = self._log_moments.mean
        sigma = self._log_moments.stddev
        median = math.exp(mu)
        if n < 2 or sigma <= 1e-12 or observation <= 0.0:
            # Dirac forecast at the running median: CRPS(δ_m, x) = |x - m|.
            return abs(observation - median)

        omega = (math.log(observation) - mu) / sigma
        mean_ln = math.exp(mu + 0.5 * sigma * sigma)
        crps = observation * (2.0 * norm_cdf(omega) - 1.0) - 2.0 * mean_ln * (
            norm_cdf(omega - sigma) + norm_cdf(sigma / math.sqrt(2.0)) - 1.0
        )
        # CRPS is an integral of a square and cannot be negative; only
        # floating-point cancellation between the two large terms can push
        # it below zero, and only when it is already ~0.
        return max(crps, 0.0)

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
