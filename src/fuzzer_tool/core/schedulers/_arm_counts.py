"""Per-arm [successes, failures] ledger shared by the OS / network ports.

Not a scheduler (like ``_bin_rates.py``): every Elo-rated arm needs
``init_arm`` / ``record`` / ``bandit_stats`` for its match to resolve, and
ten ports (``seed_mlfq`` ... ``op_p2c``) share one copy instead of ten.
``mean`` is the Beta(1, 1) posterior mean, the score the adaptive ones rank on::

    untried 0.50    3 wins 0.80    3 losses 0.20
"""

from __future__ import annotations


def reward(success: bool, weight: float) -> float:
    """Clamp a weighted success into [0, 1]; NaN and failures are 0."""
    if not success:
        return 0.0
    return min(1.0, max(0.0, float(weight)))


class ArmCounts:
    """Success / failure counts per arm key."""

    def __init__(self) -> None:
        self._counts: dict[str, list[float]] = {}  # key -> [successes, failures]

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register *name*; priors are ignored and re-registering never resets."""
        if name not in self._counts:
            self._counts[name] = [0.0, 0.0]

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        self.init_arm(name)
        r = reward(success, weight)
        self._counts[name][0] += r
        self._counts[name][1] += 1.0 - r

    def mean(self, name: str) -> float:
        """Beta(1, 1) posterior mean; 0.5 for an unseen arm."""
        s, f = self._counts.get(name, (0.0, 0.0))
        return (s + 1.0) / (s + f + 2.0)

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        return {k: (s, f) for k, (s, f) in sorted(self._counts.items())}
