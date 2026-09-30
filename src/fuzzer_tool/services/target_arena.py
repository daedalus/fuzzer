"""Elo arena over target schedulers (``tgt_<name>`` keys, ``--target-arena``).

Fourth tournament beside operator, seed and position selection. In
multi-target mode ``Fuzzer._select_next_target`` normally runs the one
``--target-schedule`` policy. With the arena on (needs ``--elo``), Elo
(Thompson over the ``tgt_`` posteriors) picks which scheduler chooses the
target for each exec::

    select()          Elo -> arm -> target index      (_select_next_target)
    seed_hint()       gale_shapley's matched seed     (SeedPicker.pick_seed)
    settle(...)       every arm records the round,    (Fuzzer.fuzz_one)
                      served arm plays the rest

Arms: every ``TargetSchedule`` policy plus ``gale_shapley``
(``core/schedulers/tgt_gale_shapley.py``). ``weighted`` (the
``--target-schedule`` default) is first, so it is Elo's cold-start pick.

Score: surprisal weight on a gain, 0 on a miss (same as the other arenas).
A round's cost is select -> settle wall time, charged to the target that
ran; every arm records every round, so WFQ sees all spent time and
Gale-Shapley learns yields off-policy.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from fuzzer_tool.core.analyzers.analyzer_elo import TGT_STRATEGY_PREFIX
from fuzzer_tool.core.schedulers.pos_base import Outcome
from fuzzer_tool.core.schedulers.tgt_base import (
    RoundRobinTarget,
    TargetRound,
    TargetScheduler,
    WeightedTarget,
    WfqTarget,
    WrrTarget,
)
from fuzzer_tool.core.schedulers.tgt_gale_shapley import GaleShapleyTarget
from fuzzer_tool.core.target_schedule import TargetSchedule

GALE_SHAPLEY = GaleShapleyTarget.name
TARGET_STRATEGY_NAMES = tuple(s.value.replace("-", "_") for s in TargetSchedule) + (GALE_SHAPLEY,)


class TargetArena:
    def __init__(self, f, clock: Callable[[], float] = time.monotonic) -> None:
        self._f = f
        self._clock = clock
        self._arms: dict[str, TargetScheduler] = {a.name: a for a in self._build_arms()}
        self._gs = self._arms[GALE_SHAPLEY]

        # Round in flight: arm that served and when it was selected.
        self._served: str | None = None
        self._t0 = 0.0

    def _build_arms(self) -> list[TargetScheduler]:
        f = self._f
        inv = f._inv_edge_weights

        def seeds(k: int) -> list[bytes]:
            return f._rng.sample(f.corpus, min(k, len(f.corpus)))

        by_schedule: dict[TargetSchedule, TargetScheduler] = {
            TargetSchedule.WEIGHTED: WeightedTarget(f._rng, inv, lambda: f.exec_count),
            TargetSchedule.ROUND_ROBIN: RoundRobinTarget(),
            TargetSchedule.WRR: WrrTarget("wrr", inv),
            TargetSchedule.WFQ: WfqTarget(inv),
            TargetSchedule.PHI: WrrTarget("phi", f._phi_weights),
        }
        gs = GaleShapleyTarget(len(f.multi_targets), seeds, inv, f._seed_key)

        # Enum order; a TargetSchedule without an arm fails loudly here.
        return [by_schedule[s] for s in TargetSchedule] + [gs]

    def pool(self) -> list[str]:
        """Arm names; ``weighted`` first."""
        return list(self._arms)

    def select(self) -> int:
        """Elo picks the arm, the arm picks the target. Drops an unsettled round."""
        name = self._arbitrate(self.pool())
        n = len(self._f.multi_targets)
        idx = self._arms[name].pick(n)

        self._served = name
        self._t0 = self._clock()
        return min(max(idx, 0), n - 1)

    def seed_hint(self) -> bytes | None:
        """The seed Gale-Shapley matched to this round's target, if it served."""
        if self._served != GALE_SHAPLEY:
            return None
        return self._gs.take_hint()

    def settle(self, seed: bytes, idx: int, outcome: Outcome, weight: float) -> None:
        """End of round: every arm records it, then the served arm plays the rest."""
        served, self._served = self._served, None
        if served is None:
            return

        rnd = TargetRound(seed, idx, outcome, weight, self._clock() - self._t0)
        for arm in self._arms.values():
            arm.record(rnd)

        if not self._elo_on():
            return
        score = weight if outcome is Outcome.GAIN else 0.0
        for other in self._arms:
            if other != served:
                self._f._elo.record_strategy_match(
                    TGT_STRATEGY_PREFIX + served, TGT_STRATEGY_PREFIX + other, score
                )

    def _elo_on(self) -> bool:
        return bool(getattr(self._f, "_use_elo", False) and getattr(self._f, "_elo", None))

    def _arbitrate(self, pool: list[str]) -> str:
        if len(pool) == 1 or not self._elo_on():
            return pool[0]

        picked = self._f._elo.select_strategy([TGT_STRATEGY_PREFIX + n for n in pool])
        return picked.removeprefix(TGT_STRATEGY_PREFIX)
