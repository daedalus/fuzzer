"""Auction target scheduler: max-weight seed -> target matching per epoch.

Sibling of ``gale_shapley``: same epoch, yield rows, seat quotas, plan and
seed hint (inherited). Only the match differs. Gale-Shapley finds a stable
match between two tastes; this one maximizes the summed Thompson draw:

    w[s][t] ~ Beta(gains + 1, misses + 1)      per (seed, target) row
    match   = argmax sum w  s.t. seats(t) <= quota(t)   (core/assignment.py)

Exploration comes from the posterior draw, not from a target-side taste.
Cost: one ``auction`` per epoch, ~0.45 ms at 32 seeds; ~27 us per arena
round served vs ~13 us for gale_shapley (4 targets, 1/edges quotas).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable

import numpy as np

from fuzzer_tool.core.assignment import UNMATCHED, auction
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.tgt_base import Weights
from fuzzer_tool.core.schedulers.tgt_gale_shapley import (
    DEFAULT_BATCH,
    DEFAULT_MAX_SEEDS,
    GaleShapleyTarget,
    seat_quotas,
)


class AuctionTarget(GaleShapleyTarget):
    """Target arm that picks the seed too, by max-weight matching."""

    name = "auction"

    def __init__(
        self,
        rng: RandPool,
        n_targets: int,
        seeds: Callable[[int], list[bytes]],
        weights: Weights,
        key: Callable[[bytes], Hashable],
        batch: int = DEFAULT_BATCH,
        max_seeds: int = DEFAULT_MAX_SEEDS,
    ) -> None:
        super().__init__(n_targets, seeds, weights, key, batch, max_seeds)
        self._rng = rng

    def _rematch(self, n: int) -> None:
        seeds = list(dict.fromkeys(self._seeds(self._batch)))
        if not seeds:
            return

        # One Thompson draw per (seed, target): a single vectorized call.
        gains, tries = self._counts(seeds, n)
        draws = self._rng.betavariate_array(gains + 1, tries - gains + 1)

        quotas = seat_quotas(len(seeds), self._weights()[:n])
        match = auction(np.asarray(draws).tolist(), quotas)
        self._plan.extend((s, t) for s, t in zip(seeds, match, strict=True) if t != UNMATCHED)

    def _counts(self, seeds: list[bytes], n: int) -> tuple[np.ndarray, np.ndarray]:
        """(gains, tries) matrices, seeds x targets; unseen seeds are zeros."""
        gains = np.zeros((len(seeds), n), dtype=np.int64)
        tries = np.zeros((len(seeds), n), dtype=np.int64)
        for i, seed in enumerate(seeds):
            row = self._rows.get(self._key(seed))
            if row is None:
                continue
            tries[i] = row[:n]
            gains[i] = row[self._n : self._n + n]
        return gains, tries
