"""Critical slowing down detection for fuzzing discovery rates.

Complex-systems science has a well-established signature that precedes
a bifurcation/phase transition: rising variance and rising autocorrelation
in the system's state just before it shifts regimes (used in climate
science, ecology, epidemiology for predicting regime shifts).

Applied to fuzzing: watch the discovery-rate time series for rising
variance/autocorrelation. A rise is the precursor signature of an
approaching coverage jump — the fuzzer is "near a phase transition,"
not actually plateaued. That's the moment to intensify effort on the
current corpus region rather than concluding it's stuck.

The detector uses three signals:
1. Rolling variance of discovery rate — increases as the system
   approaches a transition
2. Lag-1 autocorrelation — increases as the system "slows down"
   near a bifurcation point
3. Rolling skewness — rising right skew indicates occasional large
   discovery-rate spikes against a flat baseline, a stronger signal
   of a productive transition region

When variance + autocorrelation are rising, the detector signals
"approaching transition."  When skewness also rises, the verdict
upgrades to "approaching transition, and it looks productive."
Uses :class:`RunningMoments` (with O(1) sliding-window updates) for
variance and skewness instead of hand-rolling the same calculations.
"""

import collections
import logging

from fuzzer_tool.core.running_stats import RunningMoments

log = logging.getLogger(__name__)


class CriticalSlowingDown:
    """Detect critical slowing down in discovery-rate time series.

    Args:
        window_size: Number of recent observations to track.
        rise_threshold: How much variance/autocorrelation must rise
            above baseline to trigger (multiplier). 1.5 = 50% increase.
        skew_rise_threshold: How much skewness must rise above baseline
            to trigger the "productive" verdict tier. 1.5 = 50% increase.
        min_observations: Minimum observations before detection is active.
    """

    def __init__(
        self,
        window_size: int = 50,
        rise_threshold: float = 1.5,
        skew_rise_threshold: float = 1.5,
        min_observations: int = 20,
    ):
        self.window_size = window_size
        self.rise_threshold = rise_threshold
        self.skew_rise_threshold = skew_rise_threshold
        self.min_observations = min_observations
        self._history: collections.deque = collections.deque(maxlen=window_size)
        self._moments: RunningMoments = RunningMoments(window=window_size)
        self._variance_baseline: float | None = None
        self._autocorr_baseline: float | None = None
        self._skewness_baseline: float | None = None

    def observe(self, value: float) -> None:
        """Record a discovery-rate observation.

        Args:
            value: Discovery rate (edges per 1000 execs).
        """
        self._history.append(value)
        self._moments.update(value)

    def _compute_variance(self) -> float:
        """Variance of the current window (via RunningMoments)."""
        return self._moments.variance

    def _compute_skewness(self) -> float:
        """Skewness of the current window (via RunningMoments)."""
        return self._moments.skewness

    def _compute_autocorrelation(self) -> float:
        """Compute lag-1 autocorrelation of the current window."""
        n = len(self._history)
        if n < 3:
            return 0.0
        data = list(self._history)
        mean = sum(data) / n
        var = sum((x - mean) ** 2 for x in data) / n
        if var < 1e-10:
            return 0.0
        cov = sum((data[i] - mean) * (data[i + 1] - mean) for i in range(n - 1)) / (n - 1)
        return cov / var

    def is_approaching_transition(self) -> tuple[bool, str]:
        """Check if the system shows critical slowing down.

        Returns:
            Tuple of (detected, reason_string).
        """
        n = len(self._history)
        if n < self.min_observations:
            return False, f"need {self.min_observations} obs (have {n})"

        variance = self._compute_variance()
        autocorr = self._compute_autocorrelation()
        skewness = self._compute_skewness()

        if self._variance_baseline is None:
            self._variance_baseline = variance
            self._autocorr_baseline = autocorr
            self._skewness_baseline = skewness
            return False, "establishing baseline"

        var_ratio = variance / max(self._variance_baseline, 1e-10)
        acf_ratio = autocorr / max(self._autocorr_baseline, 1e-10)
        # Skewness ratio requires the baseline to be meaningfully nonzero
        # to avoid infinite ratios from constant-series baselines.
        skew_rise = (
            abs(skewness) / max(abs(self._skewness_baseline), 0.1)
            if abs(self._skewness_baseline) > 0.01
            else 0.0
        )

        self._variance_baseline = 0.9 * self._variance_baseline + 0.1 * variance
        self._autocorr_baseline = 0.9 * self._autocorr_baseline + 0.1 * autocorr
        self._skewness_baseline = 0.9 * self._skewness_baseline + 0.1 * skewness

        if var_ratio > self.rise_threshold and acf_ratio > self.rise_threshold:
            if skew_rise > self.skew_rise_threshold:
                return True, (
                    f"variance {var_ratio:.2f}x, autocorrelation {acf_ratio:.2f}x, "
                    f"skewness {skew_rise:.2f}x above baseline — "
                    f"approaching transition, and it looks productive"
                )
            return True, (
                f"variance {var_ratio:.2f}x, autocorrelation {acf_ratio:.2f}x "
                f"above baseline — approaching transition"
            )

        return False, (
            f"variance {var_ratio:.2f}x, autocorrelation {acf_ratio:.2f}x, "
            f"skewness {skew_rise:.2f}x"
        )

    def reset(self) -> None:
        """Reset detector state."""
        self._history.clear()
        self._variance_baseline = None
        self._autocorr_baseline = None
        self._skewness_baseline = None

    def save(self) -> dict:
        """Serialize state."""
        return {
            "history": list(self._history),
            "moments": self._moments.save(),
            "variance_baseline": self._variance_baseline,
            "autocorr_baseline": self._autocorr_baseline,
            "skewness_baseline": self._skewness_baseline,
            "window_size": self.window_size,
            "rise_threshold": self.rise_threshold,
            "skew_rise_threshold": self.skew_rise_threshold,
            "min_observations": self.min_observations,
        }

    def load(self, data: dict) -> None:
        """Restore state."""
        self._history = collections.deque(
            data.get("history", []), maxlen=data.get("window_size", self.window_size)
        )
        if "moments" in data:
            self._moments.load(data["moments"])
        self._variance_baseline = data.get("variance_baseline")
        self._autocorr_baseline = data.get("autocorr_baseline")
        self._skewness_baseline = data.get("skewness_baseline")
        self.window_size = data.get("window_size", self.window_size)
        self.rise_threshold = data.get("rise_threshold", self.rise_threshold)
        self.skew_rise_threshold = data.get("skew_rise_threshold", self.skew_rise_threshold)
        self.min_observations = data.get("min_observations", self.min_observations)
