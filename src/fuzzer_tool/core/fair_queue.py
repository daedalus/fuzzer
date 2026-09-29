"""Fair-queue primitives: smooth WRR, DRR, WFQ.

Sequencers, not bandits: no arm to reward, so they live here and not in
``core/schedulers/`` (same split as ``core/job_scheduling.py``). A flow is
any hashable key (a target index, a seed key); a "packet" is one visit.

All three share one contract for garbage input: a non-finite or
non-positive *weight* excludes a flow (WRR, WFQ) or counts as neutral
(DRR); a non-finite or non-positive *cost* is neutral, never free and never
infinite. Nothing here raises on bad numbers, hangs, or divides by zero.

    SmoothWRR          counts   O(n)/pick   deterministic, no cost signal
    DeficitRR          cost     O(n)/pick   carries unspent credit, seeds
    WeightedFairQueue  cost     O(n)/pick   tightest fairness, targets

Picks are O(n) because callers hand over the live flow set every call; the
flow sets here (targets, corpus) are rebuilt per pick by the caller anyway.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence

__all__ = ["DeficitRR", "SmoothWRR", "WeightedFairQueue"]

NEUTRAL_WEIGHT = 1.0
NEUTRAL_COST = 1.0
# Visits one DRR pick may burn before forcing service; bounds cost >> quantum.
DRR_ROUND_CAP = 64
# Prune DRR registry when it holds this many times the live flow count.
DRR_PRUNE_FACTOR = 2
DRR_PRUNE_SLACK = 8


def _valid(x: float) -> bool:
    return math.isfinite(x) and x > 0.0


def _positive(weights: Mapping[Hashable, float]) -> dict[Hashable, float]:
    """Weights that can win. Excluded flows keep no state anywhere."""
    return {k: float(w) for k, w in weights.items() if _valid(w)}


class SmoothWRR:
    """Smooth weighted round robin (nginx variant).

    Each pick: ``cur[k] += w[k]``, serve the max, ``cur[best] -= total``.
    Over one period each flow is served exactly ``w[k]`` times, interleaved
    instead of bursted (5:1:1 -> a a b a c a a, never a a a a a b c).
    Weights may change between picks; absent flows lose their state.
    Counts picks, not time: a slow flow eats wall clock. Use WFQ for that.
    """

    def __init__(self) -> None:
        self._cur: dict[Hashable, float] = {}
        self._turn = 0

    def pick(self, weights: Mapping[Hashable, float]) -> Hashable:
        if not weights:
            raise ValueError("SmoothWRR.pick needs at least one flow")
        live = _positive(weights)
        if not live:
            return self._rotate(list(weights))

        # Absent flows restart at 0: no credit hoarded across an absence.
        self._cur = {k: self._cur.get(k, 0.0) for k in live}
        total = sum(live.values())
        for k, w in live.items():
            self._cur[k] += w

        # max() keeps the first of equal values: ties go to insertion order.
        best = max(live, key=self._cur.__getitem__)
        self._cur[best] -= total
        return best

    def _rotate(self, keys: list[Hashable]) -> Hashable:
        """All weights unusable: plain rotation, still deterministic."""
        key = keys[self._turn % len(keys)]
        self._turn += 1
        return key


class DeficitRR:
    """Deficit round robin over hashable flows.

    Visiting a flow credits ``quantum * weight``; it is served while its
    deficit covers ``cost``. Unspent credit carries to the next round, so a
    flow costlier than one quantum is delayed, never starved, and served
    *time* (not pick count) follows the weights::

        quantum=4  fast(cost 1)  ->  4 picks / round
                   slow(cost 4)  ->  1 pick  / round     (equal time)

    With flat cost and weight this is plain round robin. A pick returns the
    flow the cursor sits on while credit lasts, then advances.

    ``cost`` and ``weight`` are callables so the caller keeps its own ledger.
    """

    def __init__(self, quantum: float = NEUTRAL_COST) -> None:
        self._quantum = float(quantum) if _valid(quantum) else NEUTRAL_COST
        self._deficit: dict[Hashable, float] = {}
        self._order: list[Hashable] = []
        self._cursor = 0
        self._credited = False  # cursor flow already got this visit's quantum
        self._seen: list[Hashable] | None = None  # flow list of the last pick
        self._active: list[Hashable] = []

    def pick(
        self,
        flows: Sequence[Hashable],
        cost: Callable[[Hashable], float],
        weight: Callable[[Hashable], float],
    ):
        if not flows:
            return ""
        if len(flows) == 1:
            return flows[0]

        active = self._active_flows(flows)

        cap = len(active) * DRR_ROUND_CAP
        for _ in range(cap):
            flow = active[self._cursor % len(active)]
            if not self._credited:
                self._deficit[flow] += self._quantum * self._share(weight(flow))
                self._credited = True

            price = self._price(cost(flow))
            if self._deficit[flow] >= price:
                self._deficit[flow] -= price
                return flow
            self._advance()

        # cost >> quantum: force service so one pick stays O(cap).
        # Advancing after forcing keeps forced service rotating, not sticky.
        flow = active[self._cursor % len(active)]
        self._deficit[flow] = 0.0
        self._advance()
        return flow

    def _active_flows(self, flows: Sequence[Hashable]) -> list[Hashable]:
        """Registered flows present in ``flows``, in registration order.

        The flow set rarely changes between picks (a corpus grows by one seed
        in hundreds of picks), and comparing two lists is a C-speed identity
        scan, so the O(n) rebuild runs only when the set actually changed.
        """
        if self._seen is not None and flows == self._seen:
            return self._active

        for flow in flows:
            self._join(flow)
        live = set(flows)
        self._prune(live)
        self._active = [k for k in self._order if k in live]
        self._seen = list(flows)
        return self._active

    def _join(self, flow: Hashable) -> None:
        if flow in self._deficit:
            return
        self._deficit[flow] = 0.0
        self._order.append(flow)

    def _advance(self) -> None:
        self._cursor += 1
        self._credited = False

    def _share(self, weight: float) -> float:
        return weight if _valid(weight) else NEUTRAL_WEIGHT

    def _price(self, cost: float) -> float:
        # Garbage cost costs exactly one quantum: neutral, like plain RR.
        return cost if _valid(cost) else self._quantum

    def _prune(self, live: set) -> None:
        """Forget flows that left the corpus so the registry cannot grow forever."""
        if len(self._order) <= DRR_PRUNE_FACTOR * len(live) + DRR_PRUNE_SLACK:
            return

        self._order = [k for k in self._order if k in live]
        self._deficit = {k: self._deficit[k] for k in self._order}
        self._cursor = 0
        self._credited = False
        self._seen = None


class WeightedFairQueue:
    """Weighted fair queuing on a self-clocked virtual time (SCFQ).

    Each flow keeps a virtual finish tag ``F``. A pick serves the flow whose
    *projected* finish ``F + est_cost / weight`` is smallest; ``charge``
    then advances ``F`` by the *measured* ``cost / weight`` and moves the
    virtual clock ``V`` to that tag. Served time follows the weights.

    Exact WFQ needs the GPS virtual clock, which costs O(n) events per
    departure. SCFQ replaces it with "V = tag of the flow just served":
    same ordering rule, delay bound weaker by at most one max-cost packet.

    A flow absent from the previous pick re-enters at ``V``: an idle flow
    banks no credit and cannot burst when it returns. A continuously
    present flow starts each packet at its own last finish tag.

    ``est_cost`` is the flow's mean charged cost, else the global mean,
    else 1.0: the cost is unknown until the packet has run.
    """

    def __init__(self) -> None:
        self._finish: dict[Hashable, float] = {}
        self._backlog: frozenset = frozenset()
        self._vt = 0.0
        self._sum: dict[Hashable, float] = {}
        self._n: dict[Hashable, int] = {}
        self._turn = 0

    @property
    def virtual_time(self) -> float:
        return self._vt

    def pick(self, weights: Mapping[Hashable, float]) -> Hashable:
        if not weights:
            raise ValueError("WeightedFairQueue.pick needs at least one flow")
        live = _positive(weights)
        if not live:
            keys = list(weights)
            key = keys[self._turn % len(keys)]
            self._turn += 1
            return key

        # Returning or new flows join at the current virtual time.
        for k in live:
            if k not in self._backlog:
                self._finish[k] = max(self._finish.get(k, 0.0), self._vt)
        self._backlog = frozenset(live)

        # min() keeps the first of equal tags: ties go to insertion order.
        return min(live, key=lambda k: self._finish[k] + self._estimate(k) / live[k])

    def charge(self, flow: Hashable, cost: float, weight: float) -> None:
        """Account one served packet of measured ``cost`` to ``flow``."""
        if _valid(cost):
            self._sum[flow] = self._sum.get(flow, 0.0) + cost
            self._n[flow] = self._n.get(flow, 0) + 1
        price = cost if _valid(cost) else self._estimate(flow)
        share = weight if _valid(weight) else NEUTRAL_WEIGHT

        self._finish[flow] = self._start(flow) + price / share
        self._vt = self._finish[flow]

    def _start(self, flow: Hashable) -> float:
        """Start tag: own last finish while present, else the virtual clock."""
        last = self._finish.get(flow, 0.0)
        return last if flow in self._backlog else max(last, self._vt)

    def _estimate(self, flow: Hashable) -> float:
        n = self._n.get(flow, 0)
        if n:
            return self._sum[flow] / n
        total_n = sum(self._n.values())
        if total_n:
            return sum(self._sum.values()) / total_n
        return NEUTRAL_COST
