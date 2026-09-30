"""Target schedulers: which binary runs next in multi-target mode.

Fourth scheduling axis beside seed, operator and position selection.
``services/target_arena.py`` arbitrates between them under ``tgt_<name>``
Elo keys (``--target-arena``). Without the arena the same policies run
inline in ``Fuzzer._select_next_target`` (``--target-schedule``); the arms
here are their ports, each with its own state so arms serving interleaved
rounds do not share a queue.

Contract::

    name: str
    pick(n) -> int            # target index in [0, n)
    record(TargetRound)       # every settled round, whoever served

    weighted     round robin for WARMUP_EXECS, then a 1/edges draw
    round_robin  exec i -> target i mod n
    wrr / phi    smooth WRR on 1/edges / on first-passage weights
    wfq          WFQ on 1/edges, charged each round's wall time
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from fuzzer_tool.core.fair_queue import SmoothWRR, WeightedFairQueue
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.pos_base import Outcome

# Same gate as Fuzzer._select_next_target's weighted default.
WARMUP_EXECS = 100

Weights = Callable[[], list[float]]


@dataclass(frozen=True, slots=True)
class TargetRound:
    """One settled round: ``seed`` ran on target ``idx``; ``cost`` is its wall time."""

    seed: bytes
    idx: int
    outcome: Outcome
    weight: float
    cost: float


@runtime_checkable
class TargetScheduler(Protocol):
    name: str

    def pick(self, n: int) -> int: ...

    def record(self, rnd: TargetRound) -> None: ...


class RoundRobinTarget:
    """Exactly equal shares, no signal."""

    name = "round_robin"

    def __init__(self) -> None:
        self._turn = 0

    def pick(self, n: int) -> int:
        idx = self._turn % n
        self._turn += 1
        return idx

    def record(self, rnd: TargetRound) -> None:
        """Stateless."""


class WeightedTarget:
    """``--target-schedule weighted``: least-covered target drawn most often."""

    name = "weighted"

    def __init__(self, rng: RandPool, weights: Weights, execs: Callable[[], int]) -> None:
        self._rng = rng
        self._weights = weights
        self._execs = execs
        self._turn = 0

    def pick(self, n: int) -> int:
        # Warm-up: cycle, so every target shows edges before 1/edges means anything.
        if n < 2 or self._execs() <= WARMUP_EXECS:
            idx = self._turn % n
            self._turn += 1
            return idx

        # CDF walk; float drift can leave r just past the sum -> last target.
        weights = self._weights()
        r = self._rng.random() * sum(weights)
        cumulative = 0.0
        for idx, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return idx
        return n - 1

    def record(self, rnd: TargetRound) -> None:
        """Reads coverage live; nothing to learn."""


class WrrTarget:
    """Smooth WRR over ``weights``: ``wrr`` (1/edges) and ``phi`` (first passage)."""

    def __init__(self, name: str, weights: Weights) -> None:
        self.name = name
        self._weights = weights
        self._wrr = SmoothWRR()

    def pick(self, n: int) -> int:
        return self._wrr.pick(dict(enumerate(self._weights())))

    def record(self, rnd: TargetRound) -> None:
        """Deterministic credit counter; no feedback."""


class WfqTarget:
    """WFQ on 1/edges: shares of wall time, not of executions."""

    name = "wfq"

    def __init__(self, weights: Weights) -> None:
        self._weights = weights
        self._wfq = WeightedFairQueue()

    def pick(self, n: int) -> int:
        return self._wfq.pick(dict(enumerate(self._weights())))

    def record(self, rnd: TargetRound) -> None:
        # Charged every round, so time spent under other arms counts too.
        weights = self._weights()
        if 0 <= rnd.idx < len(weights):
            self._wfq.charge(rnd.idx, rnd.cost, weights[rnd.idx])
