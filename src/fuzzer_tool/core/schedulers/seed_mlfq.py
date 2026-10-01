"""SeedMLFQScheduler: multi-level feedback queue (Corbato 1962, 4.4BSD).

New seeds enter the top queue. A seed that burns its allotment of
fruitless visits drops one level; a find resets the allotment. Only the
highest non-empty level is served, round robin inside it. Every
``boost_period`` picks all seeds return to the top, so none starves::

    level 0  [c d]     <- fresh / productive, served first
    level 1  [a]       <- 4 fruitless visits
    level 2  []        <- 8 more
    level 3  [b]       <- bottom: never demoted further

Allotment doubles per level (``BASE_ALLOTMENT << level``). Fruitless visits
come from ``record``, fed for every parent whatever arm picked it. One
level is exactly ``seed_round_robin`` -- the falsification condition.
"""

from __future__ import annotations

from collections import deque

from fuzzer_tool.core.schedulers._arm_counts import ArmCounts

MLFQ_LEVELS = 4
#: Fruitless visits a level-0 seed may burn before demotion; doubles per level.
BASE_ALLOTMENT = 4
#: Picks between priority boosts (anti-starvation).
BOOST_PERIOD = 1024
#: Queue state is pruned when it tracks this many times the live corpus.
PRUNE_FACTOR = 2
PRUNE_SLACK = 8


class SeedMLFQScheduler(ArmCounts):
    """Multi-level feedback queue for seed selection."""

    #: No informative priors: levels come from observed fruitless visits.
    supports_priors = False

    def __init__(self, levels: int = MLFQ_LEVELS, boost_period: int = BOOST_PERIOD) -> None:
        super().__init__()
        self._levels = max(1, int(levels))
        self._boost_period = max(1, int(boost_period))
        self._level: dict[str, int] = {}
        self._used: dict[str, int] = {}
        self._queues: list[deque[str]] = [deque() for _ in range(self._levels)]
        self._live: set[str] = set()
        self._seen: list[str] | None = None
        self._picks = 0

    def select_seed(self, seed_ids: list[str]) -> str:
        if not seed_ids:
            return ""
        if len(seed_ids) == 1:
            return seed_ids[0]
        if self._seen is None or seed_ids != self._seen:
            self._sync(seed_ids)

        self._picks += 1
        if self._picks % self._boost_period == 0:
            self._boost()

        queue = next(q for q in self._queues if q)
        seed = queue[0]
        queue.rotate(-1)
        return seed

    def record(self, seed_id: str, success: bool, weight: float = 1.0) -> None:
        """Count a fruitless visit, demoting on an exhausted allotment."""
        super().record(seed_id, success, weight)
        self._join(seed_id)
        if success:
            self._used[seed_id] = 0
            return

        self._used[seed_id] += 1
        level = self._level[seed_id]
        if level == self._levels - 1 or self._used[seed_id] < BASE_ALLOTMENT << level:
            return

        # Demote: one level down, fresh allotment.
        self._used[seed_id] = 0
        self._level[seed_id] = level + 1
        if seed_id in self._live:
            self._queues[level].remove(seed_id)
            self._queues[level + 1].append(seed_id)

    def _join(self, seed_id: str) -> None:
        if seed_id not in self._level:
            self._level[seed_id] = 0
            self._used[seed_id] = 0

    def _sync(self, seed_ids: list[str]) -> None:
        """Drop departed seeds from the queues; enqueue arrivals at their level."""
        live = set(seed_ids)
        if self._live - live:
            self._queues = [deque(k for k in q if k in live) for q in self._queues]

        for seed_id in seed_ids:
            if seed_id in self._live:
                continue
            self._join(seed_id)
            self._queues[self._level[seed_id]].append(seed_id)

        self._live = live
        self._seen = list(seed_ids)
        self._prune(live)

    def _boost(self) -> None:
        """Every seed back to level 0, keeping queue order top to bottom."""
        merged = deque(k for q in self._queues for k in q)
        self._queues = [merged] + [deque() for _ in range(self._levels - 1)]
        self._level = dict.fromkeys(self._level, 0)
        self._used = dict.fromkeys(self._used, 0)

    def _prune(self, live: set[str]) -> None:
        if len(self._level) <= PRUNE_FACTOR * len(live) + PRUNE_SLACK:
            return
        self._level = {k: v for k, v in self._level.items() if k in live}
        self._used = {k: v for k, v in self._used.items() if k in live}
