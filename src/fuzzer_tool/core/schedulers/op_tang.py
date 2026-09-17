"""OpTangScheduler: Tang's low-rank recommender over operator x edge hits.

Thin composition, not a fork: ``core/schedulers/seed_tang.py``'s
``TangRecommendationScheduler`` is reused verbatim (see
``core/op_edge_tracker.py``'s module docstring for why no changes to
``seed_tang.py`` were needed), wrapped here so the fuzzer can hold one object
exposing the same ``select_op(ops)`` / ``record(op, success, ...)``-shaped
interface as every other operator scheduler in this package, rather than
threading a raw ``TangRecommendationScheduler`` plus an ``OperatorEdgeTracker``
through ``services/operators.py`` by hand.

Empirical note: see ``core/op_edge_tracker.py``'s docstring. Landed off by
default; two independent synthetic/seed-side negatives is not evidence
this helps, it's the reason to check before switching it on.
"""

from __future__ import annotations

import numpy as np

from fuzzer_tool.core.op_edge_tracker import OperatorEdgeTracker
from fuzzer_tool.core.rand_pool import RandPool
from fuzzer_tool.core.schedulers.seed_tang import (
    DEFAULT_RANK,
    DEFAULT_REFIT_INTERVAL,
    TangRecommendationScheduler,
)


class OpTangScheduler:
    """Elo-arm operator scheduler: Tang recommender over op x edge hits.

    Args:
        rng: Shared ``RandPool`` (Hard Rule 16); passed through to the
            wrapped ``TangRecommendationScheduler``.
        rank: Low-rank approximation rank (Tang's ``k``).
        refit_interval: Executions between refits of the low-rank basis.
    """

    supports_priors = False

    def __init__(
        self,
        rng: RandPool | None = None,
        rank: int = DEFAULT_RANK,
        refit_interval: int = DEFAULT_REFIT_INTERVAL,
    ):
        if rng is None:
            raise ValueError("OpTangScheduler requires a RandPool (Hard Rule 16)")
        self._rng = rng
        self._tracker = OperatorEdgeTracker()
        self._tang = TangRecommendationScheduler(rng, rank=rank, refit_interval=refit_interval)
        self._last_refit_exec = -(1 << 60)
        self.refit_interval = max(1, int(refit_interval))

    def observe_new_edges(self, op: str, new_edge_ids) -> None:
        """Feed the actual new edge ids an operator contributed this round.

        Called from the edge-discovery path (where the fuzzer already
        knows both the new edge ids and which operators ran), separately
        from :meth:`record` -- Tang needs edge identities, not a
        success/failure flag.
        """
        self._tracker.record(op, new_edge_ids)

    def maybe_refit(self, exec_count: int) -> bool:
        if exec_count - self._last_refit_exec < self.refit_interval:
            return False
        self._last_refit_exec = exec_count
        return self._tang.refit(self._tracker)

    def record(self, op: str, success: bool, weight: float = 1.0) -> None:
        """No-op: this arm learns from :meth:`observe_new_edges` instead.

        Present only so this object satisfies the shared
        ``record(op, success, weight=...)`` contract other schedulers use
        in the generic reward fan-out loop, without that loop needing a
        special case for the one arm that ignores it.
        """

    @property
    def fitted(self) -> bool:
        return self._tang.fitted

    def select_op(self, ops: list[str]) -> str:
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]
        if not self._tang.fitted:
            return self._rng.choice(ops)
        energies = np.array([max(self._tang.seed_energy(op), 1e-4) for op in ops])
        total = float(energies.sum())
        probs = energies / total if total > 0 else np.full(len(ops), 1.0 / len(ops))
        r = self._rng.random()
        cumulative = 0.0
        for op, p in zip(ops, probs.tolist(), strict=True):
            cumulative += p
            if r <= cumulative:
                return op
        return ops[-1]
