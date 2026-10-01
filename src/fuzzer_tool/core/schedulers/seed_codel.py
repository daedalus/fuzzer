"""SeedCoDelScheduler: controlled delay (Nichols & Jacobson 2012, RFC 8289).

CoDel drops packets whose queue *sojourn* stayed above ``target`` for one
``interval``, then drops ever faster (next drop at ``interval / sqrt(n)``)
until the sojourn falls back. Here the sojourn is a seed's run of fruitless
visits; a "drop" is skipping the seed when the round-robin cursor lands on
it. After ``target + interval`` fruitless visits the seed enters the
dropping state and is served only every ``1 + isqrt(count)`` landings
(capped at ``max_gap``), so stale seeds fade without ever starving::

    target=interval=1, a always fruitless, b always finds:
    a b a b | b a b b a b b a ...   (gap 2, 2, 3 ...)

A single find leaves the dropping state. Seeds are never removed from the
corpus. An unreachable target is exactly ``seed_round_robin``.
"""

from __future__ import annotations

import math

from fuzzer_tool.core.schedulers._arm_counts import ArmCounts

#: Fruitless visits tolerated before the sojourn counts as high.
CODEL_TARGET = 8
#: Further fruitless visits before dropping starts.
CODEL_INTERVAL = 8
#: Longest landing gap between two services of a dropping seed.
MAX_GAP = 16
PRUNE_FACTOR = 2
PRUNE_SLACK = 8
# _drop entry fields.
_LANDINGS, _NEXT, _COUNT = 0, 1, 2


class SeedCoDelScheduler(ArmCounts):
    """Round robin that sheds stale seeds on CoDel's sqrt control law."""

    #: No informative priors: staleness comes from observed visits.
    supports_priors = False

    def __init__(
        self, target: int = CODEL_TARGET, interval: int = CODEL_INTERVAL, max_gap: int = MAX_GAP
    ) -> None:
        super().__init__()
        self._limit = max(1, int(target) + int(interval))
        self._max_gap = max(1, int(max_gap))
        self._stale: dict[str, int] = {}
        self._drop: dict[str, list[int]] = {}  # seed -> [landings, next_serve, count]
        self._order: list[str] = []
        self._cursor = 0

    @property
    def max_gap(self) -> int:
        return self._max_gap

    def select_seed(self, seed_ids: list[str]) -> str:
        if not seed_ids:
            return ""
        if len(seed_ids) == 1:
            return seed_ids[0]
        if seed_ids != self._order:
            self._order = list(seed_ids)
            self._prune(set(seed_ids))

        # Each dropping seed is served within max_gap landings: bounded.
        n = len(self._order)
        seed = self._order[self._cursor % n]
        for _ in range(n * self._max_gap):
            seed = self._order[self._cursor % n]
            self._cursor += 1
            if self._serves(seed):
                return seed
        return seed

    def record(self, seed_id: str, success: bool, weight: float = 1.0) -> None:
        """A find clears the sojourn; a long fruitless run starts dropping."""
        super().record(seed_id, success, weight)
        if success:
            self._stale.pop(seed_id, None)
            self._drop.pop(seed_id, None)
            return

        stale = self._stale.get(seed_id, 0) + 1
        self._stale[seed_id] = stale
        if stale >= self._limit and seed_id not in self._drop:
            self._drop[seed_id] = [0, self._gap(1), 1]

    def _gap(self, count: int) -> int:
        return min(self._max_gap, 1 + math.isqrt(count))

    def _serves(self, seed: str) -> bool:
        """Landing on *seed*: serve it, or drop (skip) it per the schedule."""
        state = self._drop.get(seed)
        if state is None:
            return True

        state[_LANDINGS] += 1
        if state[_LANDINGS] < state[_NEXT]:
            return False
        state[_COUNT] += 1
        state[_NEXT] = state[_LANDINGS] + self._gap(state[_COUNT])
        return True

    def _prune(self, live: set[str]) -> None:
        if len(self._stale) <= PRUNE_FACTOR * len(live) + PRUNE_SLACK:
            return
        self._stale = {k: v for k, v in self._stale.items() if k in live}
        self._drop = {k: v for k, v in self._drop.items() if k in live}
