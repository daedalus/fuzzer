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

The detector uses two signals:
1. Rolling variance of discovery rate — increases as the system
   approaches a transition
2. Lag-1 autocorrelation — increases as the system "slows down"
   near a bifurcation point

When both are rising above their recent baselines, the detector
signals "approaching transition."
"""

import collections
import logging

log = logging.getLogger(__name__)


class CriticalSlowingDown:
    """Detect critical slowing down in discovery-rate time series.

    Args:
        window_size: Number of recent observations to track.
        rise_threshold: How much variance/autocorrelation must rise
            above baseline to trigger (multiplier). 1.5 = 50% increase.
        min_observations: Minimum observations before detection is active.
    """

    def __init__(
        self,
        window_size: int = 50,
        rise_threshold: float = 1.5,
        min_observations: int = 20,
    ):
        self.window_size = window_size
        self.rise_threshold = rise_threshold
        self.min_observations = min_observations
        self._history: collections.deque = collections.deque(maxlen=window_size)
        self._variance_baseline: float | None = None
        self._autocorr_baseline: float | None = None

    def observe(self, value: float) -> None:
        """Record a discovery-rate observation.

        Args:
            value: Discovery rate (edges per 1000 execs).
        """
        self._history.append(value)

    def _compute_variance(self) -> float:
        """Compute variance of the current window."""
        n = len(self._history)
        if n < 2:
            return 0.0
        mean = sum(self._history) / n
        return sum((x - mean) ** 2 for x in self._history) / (n - 1)

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

        if self._variance_baseline is None:
            self._variance_baseline = variance
            self._autocorr_baseline = autocorr
            return False, "establishing baseline"

        var_ratio = variance / max(self._variance_baseline, 1e-10)
        acf_ratio = autocorr / max(self._autocorr_baseline, 1e-10)

        self._variance_baseline = 0.9 * self._variance_baseline + 0.1 * variance
        self._autocorr_baseline = 0.9 * self._autocorr_baseline + 0.1 * autocorr

        if var_ratio > self.rise_threshold and acf_ratio > self.rise_threshold:
            return True, (
                f"variance {var_ratio:.2f}x, autocorrelation {acf_ratio:.2f}x "
                f"above baseline — approaching transition"
            )

        return False, (f"variance {var_ratio:.2f}x, autocorrelation {acf_ratio:.2f}x")

    def reset(self) -> None:
        """Reset detector state."""
        self._history.clear()
        self._variance_baseline = None
        self._autocorr_baseline = None

    def save(self) -> dict:
        """Serialize state."""
        return {
            "history": list(self._history),
            "variance_baseline": self._variance_baseline,
            "autocorr_baseline": self._autocorr_baseline,
            "window_size": self.window_size,
            "rise_threshold": self.rise_threshold,
            "min_observations": self.min_observations,
        }

    def load(self, data: dict) -> None:
        """Restore state."""
        self._history = collections.deque(
            data.get("history", []), maxlen=data.get("window_size", self.window_size)
        )
        self._variance_baseline = data.get("variance_baseline")
        self._autocorr_baseline = data.get("autocorr_baseline")
        self.window_size = data.get("window_size", self.window_size)
        self.rise_threshold = data.get("rise_threshold", self.rise_threshold)
        self.min_observations = data.get("min_observations", self.min_observations)
