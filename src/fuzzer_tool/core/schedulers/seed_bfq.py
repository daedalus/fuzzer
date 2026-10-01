"""SeedBFQScheduler: budget fair queueing (Linux BFQ I/O scheduler).

A picked seed keeps the service for a *budget* of consecutive picks. When
the budget runs out it is re-sized from how the seed used it: doubled if the
seed found new coverage during it, halved otherwise (``[min, max]``). The
next seed is the one with the lowest virtual finish tag; every served pick
advances its owner's tag by ``1 / weight``, so budgets change burst length,
never long-run share::

    budget a=4 b=4:  a a a a b b b b     (a found coverage, b did not)
    budget a=8 b=2:  a a a a a a a a b b b b ...

Bursts keep one seed's mutations back to back (warm caches, deterministic
stages). ``min = max = init = 1`` is exactly ``seed_round_robin``.
"""

from __future__ import annotations

from collections.abc import Callable

from fuzzer_tool.core.fair_queue import neutral
from fuzzer_tool.core.schedulers._arm_counts import ArmCounts

MIN_BUDGET = 1
MAX_BUDGET = 64
INIT_BUDGET = 4
PRUNE_FACTOR = 2
PRUNE_SLACK = 8


def _unit(_key: str) -> float:
    return 1.0


class SeedBFQScheduler(ArmCounts):
    """Budget fair queueing for seed selection; weight per pick."""

    #: No informative priors: budgets come from observed finds.
    supports_priors = False

    def __init__(
        self,
        min_budget: int = MIN_BUDGET,
        max_budget: int = MAX_BUDGET,
        init_budget: int = INIT_BUDGET,
    ) -> None:
        super().__init__()
        self._lo = max(1, int(min_budget))
        self._hi = max(self._lo, int(max_budget))
        self._init = min(self._hi, max(self._lo, int(init_budget)))
        self._budget: dict[str, int] = {}
        self._finish: dict[str, float] = {}
        self._vt = 0.0
        self._active: str | None = None
        self._left = 0
        self._used = 0
        self._productive = False
        self._live: set[str] = set()
        self._seen: list[str] | None = None

    @property
    def max_budget(self) -> int:
        return self._hi

    def select_seed(self, seed_ids: list[str], weight_fn: Callable[[str], float] = _unit) -> str:
        if not seed_ids:
            return ""
        if len(seed_ids) == 1:
            return seed_ids[0]
        if self._seen is None or seed_ids != self._seen:
            self._sync(seed_ids)

        # Continue the running budget.
        if self._active in self._live and self._left > 0:
            self._left -= 1
            self._used += 1
            return self._active

        # Budget spent: settle it, then serve the lowest finish tag.
        self._expire(weight_fn)
        seed = min(seed_ids, key=self._finish.__getitem__)
        self._active = seed
        self._left = self._budget[seed] - 1
        self._used = 1
        self._productive = False
        self._vt = self._finish[seed]
        return seed

    def record(self, seed_id: str, success: bool, weight: float = 1.0) -> None:
        """A find during the running budget marks it productive."""
        super().record(seed_id, success, weight)
        if success and seed_id == self._active:
            self._productive = True

    def _expire(self, weight_fn: Callable[[str], float]) -> None:
        """Re-size the finished budget and charge the picks it used."""
        seed = self._active
        self._active = None
        if seed is None or seed not in self._finish:
            return

        budget = self._budget[seed]
        self._budget[seed] = (
            min(self._hi, budget * 2) if self._productive else max(self._lo, budget // 2)
        )
        self._finish[seed] += self._used / neutral(weight_fn(seed))

    def _sync(self, seed_ids: list[str]) -> None:
        """Arrivals join at the virtual clock: no banked credit across absence."""
        live = set(seed_ids)
        for seed_id in seed_ids:
            if seed_id in self._live:
                continue
            self._finish[seed_id] = max(self._finish.get(seed_id, 0.0), self._vt)
            self._budget.setdefault(seed_id, self._init)

        self._live = live
        self._seen = list(seed_ids)
        self._prune(live)

    def _prune(self, live: set[str]) -> None:
        if len(self._finish) <= PRUNE_FACTOR * len(live) + PRUNE_SLACK:
            return
        self._finish = {k: v for k, v in self._finish.items() if k in live}
        self._budget = {k: v for k, v in self._budget.items() if k in live}
