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

DEFAULT_EXPLORATION_FRAC = 1.0 / math.e


class SecretaryStopping:
    """Adaptive secretary-problem optimal stopping.

    Tracks a sliding window of quality observations and applies rank-based
    stopping. The "rank" is the decay-weighted count of record-setting
    observations (new all-time bests) currently in the window. A high rank
    means many recent improvements; a low rank means quality has plateaued.

    After the exploration phase (~1/e of observations), stops when
    rank <= floor(N / e) — meaning few records remain in the window,
    indicating diminishing returns.

    Args:
        window_size: Maximum number of observations to keep (sliding window).
        exploration_frac: Fraction of observations dedicated to exploration
            phase (default 1/e ≈ 0.368 for classical secretary problem).
        decay: Exponential decay factor for weighting recent records
            more heavily (1.0 = uniform, 0.9 = heavy recent bias).
            Older records contribute less to the rank metric.
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

        self._observations: collections.deque = collections.deque(maxlen=window_size)
        self._record_flags: collections.deque = collections.deque(maxlen=window_size)
        self._record_count: int = 0
        self._best_value: float = -math.inf
        self._best_idx: int = -1
        self._steps_since_improvement: int = 0
        self._total_observations: int = 0

    def observe(self, value: float) -> None:
        """Record a quality score and update rank statistics.

        Args:
            value: Quality score (higher is better).
        """
        is_new_record = value > self._best_value

        if len(self._observations) == self.window_size and self._record_flags[0]:
            self._record_count -= 1

        self._observations.append(value)
        self._record_flags.append(is_new_record)
        self._total_observations += 1

        if is_new_record:
            self._best_value = value
            self._best_idx = len(self._observations) - 1
            self._steps_since_improvement = 0
            self._record_count += 1
        else:
            self._steps_since_improvement += 1

    def should_stop(self) -> tuple[bool, str]:
        """Returns (stop, reason) using rank-based test.

        Stopping condition (classical secretary problem):
        - After exploration phase (first ~37% of observations), commit to
          the first candidate better than all previously seen.
        - In rank terms: stop when the decay-weighted record count
          (rank) <= floor(N / e) AND no improvement in the last
          floor(N * exploration_frac) steps.

        Returns:
            Tuple of (should_stop, reason_string).
        """
        n = len(self._observations)
        if n < self.min_observations:
            return False, f"need {self.min_observations} observations (have {n})"

        rank = self._rank_of_best()

        threshold = max(1, int(n * self.exploration_frac))

        if self._steps_since_improvement < threshold:
            return (
                False,
                f"exploration phase ({self._steps_since_improvement}/{threshold} steps since improvement)",
            )

        if rank <= threshold:
            return (
                True,
                f"best rank {rank:.2f} <= threshold {threshold}, {self._steps_since_improvement} steps without improvement",
            )

        return False, f"best rank {rank:.2f} > threshold {threshold}"

    def exploration_fraction(self) -> float:
        """Returns 0.0-1.0: fraction of observations in exploration phase.

        Returns 1.0 during exploration, decays toward 0.0 during exploitation.
        """
        n = len(self._observations)
        if n < self.min_observations:
            return 1.0

        threshold = max(1, int(n * self.exploration_frac))
        if self._steps_since_improvement < threshold:
            return 1.0

        return max(0.0, 1.0 - self._steps_since_improvement / max(threshold, 1))

    def rank_of_best(self) -> float:
        """Decay-weighted count of record-setting observations in the window.

        Returns:
            Weighted rank metric (float). Higher means more recent records.
        """
        return self._rank_of_best()

    def _rank_of_best(self) -> float:
        """Decay-weighted count of records in the sliding window.

        Each observation that set a new all-time best contributes to the rank,
        weighted by recency: more recent records count more. Decay factor
        controls how quickly old records fade.

        High rank = many recent improvements (keep going).
        Low rank = few/no recent improvements (time to stop).
        """
        if not self._observations:
            return 0.0
        n = len(self._observations)
        rank = 0.0
        for i, is_record in enumerate(self._record_flags):
            if is_record:
                age = n - 1 - i
                rank += self.decay**age
        return rank

    def reset(self) -> None:
        """Reset for new regime (e.g., when KS detects non-stationarity)."""
        self._observations.clear()
        self._record_flags.clear()
        self._record_count = 0
        self._best_value = -math.inf
        self._best_idx = -1
        self._steps_since_improvement = 0
        self._total_observations = 0

    def save(self) -> dict:
        """Serialize state for persistence."""
        return {
            "observations": list(self._observations),
            "record_flags": list(self._record_flags),
            "record_count": self._record_count,
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
        self.window_size = data.get("window_size", self.window_size)
        self._observations = collections.deque(
            data.get("observations", []), maxlen=self.window_size
        )
        self._record_flags = collections.deque(
            data.get("record_flags", []), maxlen=self.window_size
        )
        self._record_count = data.get("record_count", 0)
        self._best_value = data.get("best_value", -math.inf)
        self._best_idx = data.get("best_idx", -1)
        self._steps_since_improvement = data.get("steps_since_improvement", 0)
        self._total_observations = data.get("total_observations", 0)
        self.exploration_frac = data.get("exploration_frac", self.exploration_frac)
        self.decay = data.get("decay", self.decay)
        self.min_observations = data.get("min_observations", self.min_observations)

    def __repr__(self) -> str:
        n = len(self._observations)
        rank = self._rank_of_best()
        stop, reason = self.should_stop()
        return f"SecretaryStopping(n={n}, rank={rank:.2f}, stop={stop}, reason={reason!r})"
