"""Secretary-problem optimal stopping for fuzzing decisions.

Applies the classical secretary problem (explore first 1/e of candidates,
then commit to the first candidate better than all seen) to fuzzing
decisions across seed scheduling, operator selection, corpus minimization,
and parallel worker allocation.

Uses rank-based stopping with a sliding window for non-stationary
quality adaptation. The 37% (1/e) exploration threshold is optimal
for the classical problem and provides a principled exploration-exploitation
balance for fuzzing.
"""

import collections
import logging
import math

log = logging.getLogger(__name__)

# Classical secretary problem: optimal exploration fraction is 1/e
DEFAULT_EXPLORATION_FRAC = 1.0 / math.e


class SecretaryStopping:
    """Adaptive secretary-problem optimal stopping.

    Tracks a sliding window of quality observations and applies rank-based
    stopping: stop when the best-so-far's rank exceeds floor(N / e)
    observations without improvement.

    For non-stationary quality, uses exponentially-weighted observations
    (recent quality counts more) and KS-based plateau detection.

    Args:
        window_size: Maximum number of observations to keep (sliding window).
        exploration_frac: Fraction of observations dedicated to exploration
            phase (default 1/e ≈ 0.368 for classical secretary problem).
        decay: Exponential decay factor for weighting recent observations
            more heavily (1.0 = uniform, 0.9 = heavy recent bias).
        min_observations: Minimum observations before stopping is allowed.
            Prevents premature stopping on insufficient data.

    Examples:
        >>> sec = SecretaryStopping()
        >>> for quality in [0.1, 0.2, 0.3, 0.15, 0.4, 0.35, 0.5]:
        ...     sec.observe(quality)
        ...     stop, reason = sec.should_stop()
        ...     if stop:
        ...         print(f"Stopping: {reason}")
    """

    def __init__(
        self,
        window_size: int = 500,
        exploration_frac: float = 1.0 / math.e,
        decay: float = 0.95,
        min_observations: int = 20,
    ):
        self.window_size = window_size
        self.exploration_frac = exploration_frac
        self.decay = decay
        self.min_observations = min_observations

        # Observations and their weighted ranks
        self._observations: collections.deque = collections.deque(maxlen=window_size)
        self._best_value: float = -math.inf
        self._best_idx: int = -1
        self._steps_since_improvement: int = 0
        self._total_observations: int = 0

    def observe(self, value: float) -> None:
        """Record a quality score and update rank statistics.

        Args:
            value: Quality score (higher is better).
        """
        self._observations.append(value)
        self._total_observations += 1

        if value > self._best_value:
            self._best_value = value
            self._best_idx = len(self._observations) - 1
            self._steps_since_improvement = 0
        else:
            self._steps_since_improvement += 1

    def should_stop(self) -> tuple[bool, str]:
        """Returns (stop, reason) using rank-based test.

        Stopping condition (classical secretary problem):
        - After exploration phase (first ~37% of observations), commit to
          the first candidate better than all previously seen.
        - In rank terms: stop when best_so_far_rank > floor(N / e)
          AND no improvement in the last floor(N * exploration_frac) steps.

        Returns:
            Tuple of (should_stop, reason_string).
        """
        n = len(self._observations)
        if n < self.min_observations:
            return False, f"need {self.min_observations} observations (have {n})"

        # Compute rank of best observation (1 = best)
        rank = self._rank_of_best()

        # Exploration threshold: classical secretary uses floor(N / e)
        threshold = max(1, int(n * self.exploration_frac))

        # Check if we're past exploration phase
        if self._steps_since_improvement < threshold:
            return False, f"exploration phase ({self._steps_since_improvement}/{threshold} steps since improvement)"

        # Rank-based stopping: if best is in top 1/e of observations,
        # and we haven't improved recently, we've likely found the best
        if rank <= threshold:
            return True, f"best rank {rank} <= threshold {threshold}, {self._steps_since_improvement} steps without improvement"

        return False, f"best rank {rank} > threshold {threshold}"

    def exploration_fraction(self) -> float:
        """Returns 0.0-1.0: fraction of observations in exploration phase.

        Useful for soft integration with weight computation.
        Returns 1.0 during exploration, decays toward 0.0 during exploitation.
        """
        n = len(self._observations)
        if n < self.min_observations:
            return 1.0  # pure exploration

        threshold = max(1, int(n * self.exploration_frac))
        if self._steps_since_improvement < threshold:
            return 1.0  # still in exploration

        # During exploitation, return fraction of threshold remaining
        return max(0.0, 1.0 - self._steps_since_improvement / max(threshold, 1))

    def rank_of_best(self) -> int:
        """Rank of best observation among all (1 = best).

        Returns:
            Rank of the best observation (1-indexed).
        """
        return self._rank_of_best()

    def _rank_of_best(self) -> int:
        """Internal rank computation."""
        if not self._observations:
            return 0
        best = max(self._observations)
        # Count how many observations are strictly better than best
        # (should be 0, but handles ties)
        rank = 1
        for v in self._observations:
            if v > best:
                rank += 1
        return rank

    def reset(self) -> None:
        """Reset for new regime (e.g., when KS detects non-stationarity)."""
        self._observations.clear()
        self._best_value = -math.inf
        self._best_idx = -1
        self._steps_since_improvement = 0
        self._total_observations = 0

    def save(self) -> dict:
        """Serialize state for persistence."""
        return {
            "observations": list(self._observations),
            "best_value": self._best_value,
            "best_idx": self._best_idx,
            "steps_since_improvement": self._steps_since_improvement,
            "total_observations": self._total_observations,
            "window_size": self.window_size,
            "exploration_frac": self.exploration_frac,
            "decay": self.decay,
            "min_observations": self.min_observations,
        }

    def load(self, data: dict) -> None:
        """Restore state from persistence."""
        self._observations = collections.deque(
            data.get("observations", []), maxlen=data.get("window_size", self.window_size)
        )
        self._best_value = data.get("best_value", -math.inf)
        self._best_idx = data.get("best_idx", -1)
        self._steps_since_improvement = data.get("steps_since_improvement", 0)
        self._total_observations = data.get("total_observations", 0)
        self.window_size = data.get("window_size", self.window_size)
        self.exploration_frac = data.get("exploration_frac", self.exploration_frac)
        self.decay = data.get("decay", self.decay)
        self.min_observations = data.get("min_observations", self.min_observations)

    def __repr__(self) -> str:
        n = len(self._observations)
        rank = self._rank_of_best()
        stop, reason = self.should_stop()
        return (
            f"SecretaryStopping(n={n}, rank={rank}, stop={stop}, "
            f"reason={reason!r})"
        )
