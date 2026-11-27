"""Uniform seed scheduler: picks a seed key uniformly at random.

The seed-arena counterpart of ``op_uniform`` and the position-arena's
``UniformPosition``. Provides a signal-free baseline for the seed-arena Elo
tournament and, like ``seed_round_robin``, is reachable both as an Elo arm
and in the no-``--elo`` fallback chain (via ``_FALLBACK_ARMS`` and
``_pick_seed_uniform_seed``).
"""

from __future__ import annotations

from fuzzer_tool.core.rand_pool import RandPool


class SeedUniformScheduler:
    """Uniform seed scheduler.

    Selects a seed key uniformly at random from the available set. Does not
    learn from outcomes; ``record`` is kept for Elo-arena compatibility only.
    """

    #: No meaningful priors for uniform random seed selection.
    supports_priors = False

    def __init__(self, rng: RandPool | None = None) -> None:
        self._rng = rng or RandPool()
        self._seed_counts: dict[str, list[float]] = {}
        self._seed_order: list[str] = []

    def init_arm(self, seed_id: str, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        """Register a seed key with zero initial counts.

        ``prior_alpha``/``prior_beta`` are accepted for interface parity
        with other schedulers' ``init_arm`` but ignored (see ``supports_priors``).
        """
        if seed_id not in self._seed_counts:
            self._seed_counts[seed_id] = [0.0, 0.0]  # [successes, failures]
            self._seed_order.append(seed_id)

    def select_seed(self, seed_ids: list[str]) -> str:
        """Select a seed key uniformly at random."""
        if not seed_ids:
            return ""
        if len(seed_ids) == 1:
            return seed_ids[0]
        for seed_id in seed_ids:
            self.init_arm(seed_id)
        return self._rng.choice(seed_ids)

    def record(self, seed_id: str, success: bool, weight: float = 1.0) -> None:
        """Record outcome for Elo compatibility; ignored by selection."""
        self.init_arm(seed_id)
        r = min(1.0, max(0.0, float(weight))) if success else 0.0
        self._seed_counts[seed_id][0] += r
        self._seed_counts[seed_id][1] += 1.0 - r

    def bandit_stats(self) -> dict[str, tuple[float, float]]:
        """Return success/failure counts for each registered arm."""
        return {
            seed_id: (successes, failures)
            for seed_id, (successes, failures) in sorted(self._seed_counts.items())
        }
