"""SeedP2CScheduler: power of two choices (Mitzenmacher 2001).

Load balancers pick two servers at random and send to the less loaded one;
the max load drops from O(log n / log log n) to O(log log n). Here: draw two
seeds uniformly, keep the one with the higher Beta(1, 1) posterior mean of
new coverage. O(1) per pick, no full-corpus scoring pass::

    draws (a, b), mean a=0.5 b=0.8  ->  b
    draws (a, b), equal means       ->  a    (the first draw: uniform)

With no signal it is uniform random selection -- the falsification
condition. The worst seed is only chosen when drawn twice (1/n^2).
"""

from __future__ import annotations

from fuzzer_tool.core.schedulers._arm_counts import ArmCounts


class SeedP2CScheduler(ArmCounts):
    """Two uniform draws, higher posterior mean wins."""

    #: Posterior starts at Beta(1, 1) for every seed; no prior override.
    supports_priors = False

    def __init__(self, rng) -> None:
        if rng is None:
            raise ValueError("SeedP2CScheduler requires a RandPool (Hard Rule 16)")
        super().__init__()
        self._rng = rng

    def select_seed(self, seed_ids: list[str]) -> str:
        if not seed_ids:
            return ""
        if len(seed_ids) == 1:
            return seed_ids[0]

        first = self._rng.choice(seed_ids)
        second = self._rng.choice(seed_ids)
        return second if self.mean(second) > self.mean(first) else first
