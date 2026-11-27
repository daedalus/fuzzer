"""Uniform operator scheduler: picks an operator uniformly at random.

The operator-arena counterpart of ``seed_uniform`` and the position-arena's
``UniformPosition``. Provides a signal-free baseline for the operator Elo
tournament and, like ``op_round_robin``, is reachable both as an Elo arm
and in the no-``--elo`` fallback chain (via ``_FALLBACK_PRECEDENCE`` and
``_PLAIN_STRATEGIES``).
"""

from __future__ import annotations

from typing import cast

from fuzzer_tool.core.rand_pool import RandPool


class UniformScheduler:
    """Uniform operator scheduler.

    Selects an operator uniformly at random from the available set. Does not
    learn from outcomes; ``record`` is kept for Elo-arena compatibility only.
    """

    #: No meaningful priors for uniform random selection.
    supports_priors = False

    def __init__(self, rng: RandPool | None = None) -> None:
        self._rng = rng or RandPool()
        self._operator_counts: dict[str, list[float]] = {}

    def init_arm(self, name: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register an operator with zero initial counts.

        ``prior_alpha``/``prior_beta`` are accepted for interface parity with
        other schedulers' ``init_arm`` but ignored (see ``supports_priors``).
        """
        if name not in self._operator_counts:
            self._operator_counts[name] = [0.0, 0.0]  # [successes, failures]

    def select_op(self, ops: list[str]) -> str:
        """Select an operator uniformly at random."""
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]
        # Register any new ops on the fly
        for op in ops:
            self.init_arm(op)
        return cast("str", self._rng.choice(ops))

    def record(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome for Elo compatibility; ignored by selection."""
        self.init_arm(name)
        r = min(1.0, max(0.0, float(weight))) if success else 0.0
        self._operator_counts[name][0] += r
        self._operator_counts[name][1] += 1.0 - r

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        """Return success/failure counts for each registered arm."""
        return {
            name: (successes, failures)
            for name, (successes, failures) in sorted(self._operator_counts.items())
        }
