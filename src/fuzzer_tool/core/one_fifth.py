"""Rechenberg's 1/5 success rule over the per-round mutation count.

Evolution strategies tune the mutation step from the share of offspring that
beat their parent: above the target rate, step harder; below it, step softer.
Here the "step" is how many operators a round stacks (``-M`` after the perf
score), and a round "beats its parent" when it finds new coverage.

Continuous form (Kern et al. 2004), applied every round::

    log s += (hit - target) / (damping * (1 - target))

At hit rate == target the expected step is zero, so ``s`` hovers there.
``log s`` is clamped to [log SCALE_MIN, log SCALE_MAX]; clamping the state
itself (not just the output) is what stops windup: after a long dry spell
the first hit moves ``s`` off the floor at once.

Rechenberg's 1/5 is wrong for fuzzing: new coverage is rare and bursty, so
``DEFAULT_TARGET`` is calibrated on fuzzgoat instead (docs/DEEP_DIVE.md).
"""

from __future__ import annotations

import enum
import math

#: Calibrated target hit rate (fuzzgoat sweep, docs/DEEP_DIVE.md §1/5 rule).
DEFAULT_TARGET = 0.2

#: Rounds of averaging per e-fold: a hit at target 0.2 multiplies s by e^(1/(d*0.8)).
DEFAULT_DAMPING = 4.0

#: Bounds on the multiplier over -M: 8 x 1/8 = 1, 8 x 8 = 64 operators.
SCALE_MIN = 0.125
SCALE_MAX = 8.0

_LOG_MIN = math.log(SCALE_MIN)
_LOG_MAX = math.log(SCALE_MAX)


class Outcome(enum.IntEnum):
    """Round outcome; the value is the indicator in the update."""

    MISS = 0
    HIT = 1


class OneFifthRule:
    """Success-rate controller for the round's mutation count.

    Args:
        target: Hit rate the controller steers toward, in (0, 1).
        damping: Larger is smoother; must be > 0.
    """

    def __init__(self, target: float = DEFAULT_TARGET, damping: float = DEFAULT_DAMPING) -> None:
        if not 0.0 < target < 1.0:  # NaN fails both comparisons
            raise ValueError(f"target must be in (0, 1), got {target}")
        if not damping > 0.0:
            raise ValueError(f"damping must be > 0, got {damping}")

        self._target = target
        self._denom = damping * (1.0 - target)
        self._log_s = 0.0
        self._rounds = 0
        self._hits = 0

    def record(self, outcome: Outcome) -> None:
        """Fold one round's outcome into the scale."""
        self._rounds += 1
        self._hits += outcome

        log_s = self._log_s + (outcome - self._target) / self._denom
        self._log_s = min(_LOG_MAX, max(_LOG_MIN, log_s))

    def scale(self) -> float:
        """Multiplier for the round's mutation count."""
        return math.exp(self._log_s)

    def stats(self) -> dict:
        """Diagnostics for the report and tests."""
        return {
            "one_fifth_target": self._target,
            "one_fifth_scale": self.scale(),
            "one_fifth_rounds": self._rounds,
            "one_fifth_hits": self._hits,
        }
