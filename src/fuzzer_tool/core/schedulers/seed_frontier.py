"""FrontierSeedScheduler: fuzz seeds that execute rarely reached open branch conditions.

A frontier condition is a branch the corpus has taken one way only
(adapters/frontier_cov.py). A seed that executes it sits one input change
away from the missing side; if few seeds get there, little else in the
corpus will try. Weight per seed::

    w(s) = sum over open conditions c that s executes of 1 / reachers(c)

    s1 executes A (1 reacher)          -> 1.0
    s2 executes B (2) and C (4)        -> 0.75
    s3 not measured yet                -> 0       (never picked)

Roulette over w; all zero declines (empty key) so another arm serves.
Weights are recomputed when the oracle's ``epoch`` moves (a seed measured)
or the key list changes. ``record`` keeps the ArmCounts ledger for the
Elo match; selection does not read it.
"""

from __future__ import annotations

import bisect
import itertools

from fuzzer_tool.core.schedulers._arm_counts import ArmCounts


class FrontierSeedScheduler(ArmCounts):
    """Roulette over rare-reach of open frontier conditions."""

    #: Weights come from coverage, not a Beta posterior: no prior override.
    supports_priors = False

    def __init__(self, rng, oracle) -> None:
        if rng is None:
            raise ValueError("FrontierSeedScheduler requires a RandPool (Hard Rule 16)")
        super().__init__()
        self._rng = rng
        self._oracle = oracle
        self._epoch = -1
        self._keys: list[str] = []
        self._weights: list[float] = []
        self._cum: list[float] = []

    def weights(self, seed_ids: list[str]) -> list[float]:
        """Per-key weight, cached per (oracle epoch, key list)."""
        epoch = self._oracle.epoch
        if epoch != self._epoch or seed_ids != self._keys:
            self._epoch = epoch
            self._keys = list(seed_ids)
            self._weights = self._oracle.rare_reach(self._keys)
            self._cum = list(itertools.accumulate(self._weights))
        return self._weights

    def select_seed(self, seed_ids: list[str]) -> str:
        if not seed_ids:
            return ""
        self._trim(seed_ids)
        self.weights(seed_ids)
        total = self._cum[-1]
        if total <= 0.0:
            return ""

        # First cumulative weight above the draw; clamp guards float round-off.
        i = bisect.bisect_right(self._cum, self._rng.random() * total)
        return self._keys[min(i, len(self._keys) - 1)]
