"""SeedConsolidatedScheduler: the seed-arena OS/network arms in one picker.

Each OS / network seed arm owns one idea; this one keeps the idea that each
contributes, in one O(1)-per-pick pass::

    feature                     from            here
    two random candidates       seed_p2c        draw 2, keep the higher score
    posterior yield             seed_p2c        Beta(1, 1) mean of new coverage
    cost per target time        seed_eevdf/drr  score / cost
    favored share               seed_stride/bfq score * weight
    yield decays when stale     seed_aimd/codel LOSS_RUN misses: successes *= LOSS_DECAY
    no starvation               round_robin/mlfq every SWEEP_EVERY-th pick: next in order
    flow fairness               seed_sfq        candidate = uniform flow, then member

    score(k) = mean(k) * weight(k) / cost(k)

A flow is the lineage parent (``--lineage``), else the seed itself, so one
prolific parent's children share one candidate slot. Flows are grouped
exactly (no hash buckets), only when the corpus changes.

No signal (flat cost and weight, no records, singleton flows) is uniform
random selection plus the sweep -- the falsification condition.
``record`` is fed off-policy for every corpus parent (``_record_seed_os_arms``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable

from fuzzer_tool.core.schedulers._arm_counts import PRUNE_FACTOR, PRUNE_SLACK, ArmCounts

#: Consecutive fruitless visits that make one loss event (seed_aimd).
LOSS_RUN = 8
#: Success evidence kept after a loss event (seed_aimd's beta).
LOSS_DECAY = 0.5
#: Every this-many-th pick is a round-robin sweep step (anti-starvation).
SWEEP_EVERY = 16
#: Cheapest relative cost a seed is credited with (bounds the cost tilt).
COST_FLOOR = 1.0 / 16

# ArmCounts ledger fields.
_SUCC = 0


def _unit(_key: str) -> float:
    return 1.0


def _self(_key: str) -> Hashable | None:
    return None


def _cost(raw: float) -> float:
    """Relative cost; zero floors at COST_FLOOR, NaN/inf/negative are neutral."""
    if not math.isfinite(raw) or raw < 0.0:
        return 1.0
    return max(COST_FLOOR, raw)


def _weight(raw: float) -> float:
    """Share weight; NaN/inf/non-positive are neutral."""
    return raw if math.isfinite(raw) and raw > 0.0 else 1.0


class SeedConsolidatedScheduler(ArmCounts):
    """P2C over flows, scored by decayed yield per cost, with a sweep."""

    #: Posterior starts at Beta(1, 1) for every seed; no prior override.
    supports_priors = False

    def __init__(self, rng) -> None:
        if rng is None:
            raise ValueError("SeedConsolidatedScheduler requires a RandPool (Hard Rule 16)")
        super().__init__()
        self._rng = rng
        self._misses: dict[str, int] = {}
        self._flows: list[list[str]] = []
        self._seen: list[str] | None = None
        self._ref: list[str] | None = None
        self._picks = 0
        self._cursor = 0

    def select_seed(
        self,
        seed_ids: list[str],
        cost_fn: Callable[[str], float] = _unit,
        weight_fn: Callable[[str], float] = _unit,
        flow_fn: Callable[[str], Hashable | None] = _self,
    ) -> str:
        if not seed_ids:
            return ""
        self._trim(seed_ids)
        if len(seed_ids) == 1:
            return seed_ids[0]

        # Sweep: a seed that never wins a duel is still served.
        self._picks += 1
        if self._picks % SWEEP_EVERY == 0:
            seed = seed_ids[self._cursor % len(seed_ids)]
            self._cursor += 1
            return seed

        # Identity first: SeedPicker._corpus_keys hands back the same list
        # until the corpus changes (a fresh list then), so a hit skips the
        # O(n) compare (~6 us at 5000 seeds). Equality covers other callers.
        if seed_ids is not self._ref and seed_ids != self._seen:
            self._sync(seed_ids, flow_fn)
        self._ref = seed_ids

        first = self._draw()
        second = self._draw()
        if self._score(second, cost_fn, weight_fn) > self._score(first, cost_fn, weight_fn):
            return second
        return first

    def record(self, seed_id: str, success: bool, weight: float = 1.0) -> None:
        """Count the outcome; a LOSS_RUN of misses decays past successes."""
        super().record(seed_id, success, weight)
        if success:
            self._misses.pop(seed_id, None)
            return

        misses = self._misses.get(seed_id, 0) + 1
        if misses < LOSS_RUN:
            self._misses[seed_id] = misses
            return

        # Loss event: the neighbourhood is drying up, as AIMD's window cut.
        self._counts[seed_id][_SUCC] *= LOSS_DECAY
        self._misses.pop(seed_id, None)

    def _score(
        self, seed_id: str, cost_fn: Callable[[str], float], weight_fn: Callable[[str], float]
    ) -> float:
        return self.mean(seed_id) * _weight(weight_fn(seed_id)) / _cost(cost_fn(seed_id))

    def _draw(self) -> str:
        """Uniform flow, then uniform member: one draw for a singleton flow."""
        flows = self._flows
        group = flows[self._rng.randint(0, len(flows) - 1)]
        if len(group) == 1:
            return group[0]
        return group[self._rng.randint(0, len(group) - 1)]

    def _sync(self, seed_ids: list[str], flow_fn: Callable[[str], Hashable | None]) -> None:
        """Group the corpus by flow, flows in first-appearance order."""
        groups: dict[Hashable, list[str]] = {}
        for seed_id in seed_ids:
            flow = flow_fn(seed_id)
            groups.setdefault(seed_id if flow is None else ("flow", flow), []).append(seed_id)
        self._flows = list(groups.values())
        self._seen = list(seed_ids)

        if len(self._misses) > PRUNE_FACTOR * len(seed_ids) + PRUNE_SLACK:
            live = set(seed_ids)
            self._misses = {k: v for k, v in self._misses.items() if k in live}
