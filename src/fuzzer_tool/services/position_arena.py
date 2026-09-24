"""Elo arena over position schedulers (``pos_<name>`` keys).

Third tournament beside operator and seed selection (see
``core/schedulers/pos_base.py``). ``OperatorEngine.select_position`` used
to pick uniformly among whatever tracker proposals existed; with the arena
on, Elo (Thompson over the ``pos_`` posteriors) picks which proposer
speaks, and uniform is a member: a proposer rated at or below it is
flagged (``Fuzzer._check_canary_inspection``).

Arms::

    uniform      baseline, always in the pool, first (Elo's cold-start pick)
    sensitivity  per-byte Lyapunov sensitivity
    te / phase   transfer-entropy map / record-stride phase lock
    mi           mutual-information map
    crash_mi     crash mutual-information map (after min_observations)
    region       statistical region profile
    burn_front   BurnFrontPositionScheduler (opt-in, --burn-front)

Only arms whose feature is on join the pool, so nobody accrues phantom
matches. An arm that declines is replaced by uniform and *charged as
uniform*: a decline must not earn or lose rating for the arm.

Matches: a round's operators may land several positions. Every arm that
served one plays each pool member that did not, with the round score. Arms
that shared a round do not play each other.

``burn_front`` is credited off-policy on every settled round, whoever
served the positions, like ``seed_canary`` on the seed side.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from fuzzer_tool.core.analyzers.analyzer_elo import POS_STRATEGY_PREFIX
from fuzzer_tool.core.schedulers.pos_base import (
    CallablePosition,
    Outcome,
    PositionScheduler,
    UniformPosition,
)

UNIFORM = "uniform"
POSITION_STRATEGY_NAMES = (
    UNIFORM,
    "sensitivity",
    "te",
    "phase",
    "mi",
    "crash_mi",
    "region",
    "burn_front",
)

Gate = Callable[[], bool]
Arm = tuple[PositionScheduler, Gate]


class PositionArena:
    def __init__(self, f, region_fn, burn_front: PositionScheduler | None = None) -> None:
        self._f = f
        self._uniform = UniformPosition(f._rng)
        self._burn_front = burn_front
        self._arms: dict[str, Arm] = {UNIFORM: (self._uniform, lambda: True)}
        self._add_trackers(region_fn)
        if burn_front is not None:
            self._arms[burn_front.name] = (burn_front, lambda: True)
        self._used: list[str] = []
        self._seen_pool: list[str] = []

    def _add_trackers(self, region_fn: Callable[[bytes, int], int | None]) -> None:
        f = self._f

        def on(flag: str, tracker: str) -> Gate:
            return lambda: bool(getattr(f, flag, False) and getattr(f, tracker, None))

        def crash_ready() -> bool:
            cm = getattr(f, "_crash_mi", None)
            return bool(cm and cm.total_execs >= cm.min_observations)

        def phase(data: bytes, n: int) -> int | None:
            meta = f.seed_meta.get(data)
            return f._get_phase_weighted_position(n, meta.get("record_stride") if meta else None)

        te_on = on("_use_transfer_entropy", "_te")
        specs: list[tuple[str, Callable[[bytes, int], int | None], Gate]] = [
            ("sensitivity", lambda d, n: f._sensitivity.get_weighted_position(d, n),
             on("_use_sensitivity", "_sensitivity")),
            ("te", lambda d, n: f._get_te_weighted_position(n), te_on),
            ("phase", phase, te_on),
            ("mi", lambda d, n: f._mi.weighted_position(n), on("_use_mi", "_mi")),
            ("crash_mi", lambda d, n: f._crash_mi.weighted_position(n), crash_ready),
            ("region", region_fn, lambda: bool(getattr(f, "_use_region_profile", False))),
        ]  # fmt: skip
        for name, fn, gate in specs:
            self._arms[name] = (CallablePosition(name, fn), gate)

    def pool(self) -> list[str]:
        """Names of arms whose feature is on now; uniform first."""
        return [name for name, (_, gate) in self._arms.items() if gate()]

    def used(self) -> list[str]:
        """Arms that served a position this round, in order."""
        return list(self._used)

    def select(self, data: bytes, buf_len: int) -> int:
        """Elo picks the arm; a declining arm falls back to uniform."""
        pool = self.pool()
        self._seen_pool.extend(n for n in pool if n not in self._seen_pool)
        name = self._arbitrate(pool)

        pos = self._arms[name][0].propose(data, buf_len)
        if pos is None:
            name = UNIFORM
            pos = self._uniform.propose(data, buf_len)

        self._used.append(name)
        return min(max(pos, 0), buf_len - 1)

    def _arbitrate(self, pool: list[str]) -> str:
        if len(pool) == 1:
            return pool[0]

        picked = self._f._elo.select_strategy([POS_STRATEGY_PREFIX + n for n in pool])
        return picked.removeprefix(POS_STRATEGY_PREFIX)

    def settle(
        self,
        data: bytes,
        offsets: Sequence[int],
        outcome: Outcome,
        weight: float,
        score: float,
    ) -> None:
        """End of round: feed burn_front, then play the Elo matches."""
        if self._burn_front is not None:
            self._burn_front.record(data, offsets, outcome, weight)

        served = list(dict.fromkeys(self._used))
        pool = self._seen_pool
        self._used, self._seen_pool = [], []

        elo = getattr(self._f, "_elo", None)
        if not (getattr(self._f, "_use_elo", False) and elo):
            return

        for name in served:
            for other in pool:
                if other not in served:
                    elo.record_strategy_match(
                        POS_STRATEGY_PREFIX + name, POS_STRATEGY_PREFIX + other, score
                    )
