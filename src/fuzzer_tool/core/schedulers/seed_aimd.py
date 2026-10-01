"""SeedAIMDScheduler: additive increase, multiplicative decrease (TCP Reno).

Each seed holds a window, its stride tickets. A find grows it by ``alpha``
(capped at ``w_max``); a *loss event* -- ``loss_run`` fruitless visits in a
row -- cuts it by ``beta`` (floored at ``w_min``). Seeds are then served by
stride scheduling in proportion to their windows::

    finds:     1 -> 2 -> 3 -> 4          (+alpha)
    8 misses:  4 -> 2                    (*beta)

AIMD converges to a fair share among competing flows and tracks a moving
optimum, which is the fuzzing case: a seed's yield decays as its
neighbourhood is exhausted. ``alpha = 0, beta = 1`` freezes every window:
exactly ``seed_round_robin`` (falsification).
"""

from __future__ import annotations

from fuzzer_tool.core.fair_queue import Stride
from fuzzer_tool.core.schedulers._arm_counts import ArmCounts

AIMD_ALPHA = 1.0
AIMD_BETA = 0.5
#: Consecutive fruitless visits that make one loss event.
LOSS_RUN = 8
W_INIT = 1.0
W_MIN = 1.0 / 16
W_MAX = 64.0
PRUNE_FACTOR = 2
PRUNE_SLACK = 8


class SeedAIMDScheduler(ArmCounts):
    """AIMD windows as stride tickets for seed selection."""

    #: No informative priors: windows start equal at W_INIT.
    supports_priors = False

    def __init__(
        self,
        alpha: float = AIMD_ALPHA,
        beta: float = AIMD_BETA,
        loss_run: int = LOSS_RUN,
        w_min: float = W_MIN,
        w_max: float = W_MAX,
    ) -> None:
        super().__init__()
        self._alpha = max(0.0, float(alpha))
        self._beta = min(1.0, max(0.0, float(beta)))
        self._loss_run = max(1, int(loss_run))
        self.w_min = float(w_min)
        self.w_max = max(self.w_min, float(w_max))
        self._window: dict[str, float] = {}
        self._losses: dict[str, int] = {}
        self._stride = Stride()

    def select_seed(self, seed_ids: list[str]) -> str:
        if len(self._window) > PRUNE_FACTOR * len(seed_ids) + PRUNE_SLACK:
            self._prune(set(seed_ids))
        self._trim(seed_ids)
        return self._stride.pick(seed_ids, self._tickets)

    def record(self, seed_id: str, success: bool, weight: float = 1.0) -> None:
        super().record(seed_id, success, weight)
        window = self._window.get(seed_id, W_INIT)
        if success:
            self._window[seed_id] = min(self.w_max, window + self._alpha)
            self._losses[seed_id] = 0
            return

        losses = self._losses.get(seed_id, 0) + 1
        if losses < self._loss_run:
            self._losses[seed_id] = losses
            return
        self._window[seed_id] = max(self.w_min, window * self._beta)
        self._losses[seed_id] = 0

    def _tickets(self, seed_id: str) -> float:
        return self._window.get(seed_id, W_INIT)

    def _prune(self, live: set[str]) -> None:
        self._window = {k: v for k, v in self._window.items() if k in live}
        self._losses = {k: v for k, v in self._losses.items() if k in live}
