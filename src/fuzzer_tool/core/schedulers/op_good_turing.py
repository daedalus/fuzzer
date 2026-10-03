"""OpGoodTuringScheduler: per-operator Good-Turing discovery probability.

Operator-arena twin of ``seed_good_turing`` (entropy handover 7.1, "per arm").
Edges are species, an operator's executions are sampling units::

    T   executions in which the operator ran
    Q1  edges hit by exactly one of those executions

    M_op = (Q1_op + K * M_global) / (T_op + K)

``M_op`` is the chance the operator's next execution hits an edge none of its
earlier executions hit. An operator that keeps re-treading the same edges has
``Q1 -> 0`` and decays; one still turning up singletons keeps paying. An
operator never run scores the campaign rate ``M_global``::

    havoc  edges {1} {2} {3}  -> Q1=3, T=3   scores high
    flip   edges {7} {7} {7}  -> Q1=0, T=3   scores low

Credit smear: every operator stacked in one execution is credited with all of
that execution's edges (the same fan-out ``_record_schedulers`` applies to
success flags). A lone productive operator and its stack-mates therefore
score alike; there is no counterfactual.

Experimental, Elo-only (absent from ``_FALLBACK_PRECEDENCE``). Not persisted:
a resumed run re-earns it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fuzzer_tool.core.lru import LRUCache
from fuzzer_tool.core.schedulers._arm_counts import ArmCounts
from fuzzer_tool.core.schedulers.seed_good_turing import (
    PRIOR_STRENGTH,
    _Incidence,
    good_turing_m0,
    shrunk_m0,
)

#: Executions (campaign-wide) before the arm will pick; below this the global
#: rate is noise and every score would be a coin flip.
MIN_OBSERVATIONS = 50

#: Floor so the cumulative draw never sees an all-zero vector (a saturated
#: campaign has Q1 = 0 everywhere).
MIN_WEIGHT = 1e-9

#: Per-operator tables kept (LRU). Far above the registry size; bounds only
#: pathological callers.
OP_CAP = 512


class OpGoodTuringScheduler(ArmCounts):
    """Elo-arbitrated ``op_good_turing`` arm: pick by operator discovery probability."""

    supports_priors = False

    def __init__(
        self,
        rng: Any,
        prior_strength: float = PRIOR_STRENGTH,
        op_cap: int = OP_CAP,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        if rng is None:
            raise ValueError("OpGoodTuringScheduler requires a RandPool (Hard Rule 16)")
        if prior_strength < 0:
            raise ValueError("prior_strength must be >= 0")
        super().__init__()
        self._rng = rng
        self._k = float(prior_strength)
        self._min_obs = min_observations
        self._global = _Incidence()
        self._tables: LRUCache = LRUCache(op_cap)
        self._selected = 0

    # ── Observation (every execution) ──────────────────────────────────

    def observe(self, ops: Iterable[str], edges: Iterable[int]) -> None:
        """Credit one execution to each distinct operator in *ops*."""
        used = list(dict.fromkeys(ops))
        if not used:
            return

        hit = edges if isinstance(edges, set | frozenset) else set(edges)
        for op in used:
            table = self._tables.get(op)
            if table is None:
                table = _Incidence()
                self._tables[op] = table
            table.add(hit)
        self._global.add(hit)

    # ── Estimates ──────────────────────────────────────────────────────

    def residual_risk(self) -> float:
        """Campaign-wide ``M0``: P(next execution hits an unseen edge)."""
        return good_turing_m0(self._global.q1, self._global.t)

    def discovery_probability(self, op: str) -> float:
        """Shrunk ``M0`` for *op*; the global rate while it is unseen."""
        prior = self.residual_risk()
        table = self._tables.get(op)
        if table is None or table.t == 0:
            return prior
        return shrunk_m0(table.q1, table.t, prior, self._k)

    @property
    def ready(self) -> bool:
        return self._global.t >= self._min_obs

    def scores(self, ops: list[str]) -> list[float]:
        """Weight per op, aligned 1:1 with ``ops``."""
        return [max(self.discovery_probability(op), MIN_WEIGHT) for op in ops]

    def select_op(self, ops: list[str]) -> str:
        """Draw an op in proportion to discovery probability; uniform until ready."""
        if not ops:
            return ""
        if len(ops) == 1:
            return ops[0]
        if not self.ready:
            return str(self._rng.choice(ops))

        self._selected += 1
        weights = self.scores(ops)
        r = self._rng.random() * sum(weights)
        cumulative = 0.0
        for op, w in zip(ops, weights, strict=True):
            cumulative += w
            if r <= cumulative:
                return op
        return ops[-1]

    # ── Introspection ──────────────────────────────────────────────────

    def executions(self, op: str) -> int:
        table = self._tables.get(op)
        return table.t if table is not None else 0

    def singletons(self, op: str) -> int:
        table = self._tables.get(op)
        return table.q1 if table is not None else 0

    def stats(self) -> dict[str, Any]:
        g = self._global
        return {
            "observed": g.t,
            "selected": self._selected,
            "ops": len(self._tables),
            "q1": g.q1,
            "q2": g.q2,
            "residual_risk": self.residual_risk(),
        }
