"""Good-Turing seed arm (``--good-turing-seed``): discovery probability per seed.

Chao2 (``EdgeTracker``) estimates *how many* edges exist. Nothing estimated
*the probability that the next execution hits an edge nobody has hit*; that
is the Good-Turing missing mass. Treat edges as species and a seed's mutants
as sampling units (incidence data)::

    T   executions of the seed's mutants
    Q1  edges hit by exactly one of those T executions
    Q2  edges hit by exactly two

    M0 = Q1 / T                          Good-Turing
    M0 = Q1/T * (T-1)Q1 / ((T-1)Q1 + 2Q2) Chao-Jost bias-corrected (``chao``)

``M0`` is the chance that the next mutant of this seed hits something no
earlier mutant of it hit. A seed whose mutants keep re-treading the same
edges has Q1 -> 0 and decays; one still turning up singletons keeps paying.
The same estimator over every execution is the campaign's STADS-style
residual risk (``residual_risk``).

Small samples: with T = 1 every edge is a singleton and ``Q1/T`` is the
whole path, so a raw estimate is ~1.0 by construction (the failure
``docs/TODO.md`` records for Chao2 at execs 1-8). Each seed's estimate is
therefore shrunk toward the campaign-wide ``M0`` with ``prior_strength``
pseudo-executions::

    M_s = (Q1_s + K * M_global) / (T_s + K)

A seed never fuzzed scores exactly the global rate; evidence moves it away
at rate ``T / (T + K)``.

Not persisted across ``--resume`` (nothing here is a corpus property: it is
a record of what mutants did, and a resumed run re-earns it).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from fuzzer_tool.core.lru import LRUCache

#: Pseudo-executions of prior evidence pulling a seed toward the global rate.
PRIOR_STRENGTH = 20.0

#: Executions (campaign-wide) before the arm will pick. Below this the
#: global rate is noise and every weight would be a coin flip.
MIN_OBSERVATIONS = 50

#: Floor so ``weighted_choice`` never sees an all-zero or negative vector
#: (a saturated campaign has Q1 = 0 everywhere).
MIN_WEIGHT = 1e-9

#: Per-seed tables kept (LRU). A table holds every edge the seed's mutants
#: hit, so it is bounded by the coverage map, not by the rare set.
SEED_CAP = 2048

ESTIMATORS = ("gt", "chao")


# ── Pure estimators ────────────────────────────────────────────────────


def good_turing_m0(q1: int, t: int) -> float:
    """Good-Turing missing mass ``Q1 / T`` (0.0 for an empty sample)."""
    if t <= 0:
        return 0.0
    return min(1.0, q1 / t)


def chao_jost_m0(q1: int, q2: int, t: int) -> float:
    """Chao-Jost bias-corrected incidence missing mass.

    ``Q1/T * (T-1)Q1 / ((T-1)Q1 + 2Q2)``; collapses to ``Q1/T`` when there
    are no doubletons to temper it and to 0 when there are no singletons.
    """
    if t <= 0 or q1 <= 0:
        return 0.0
    if t == 1:
        return 1.0
    core = (t - 1) * q1
    return min(1.0, (q1 / t) * core / (core + 2 * q2))


def m0_variance(q1: int, q2: int, t: int) -> float:
    """Esty-style variance of ``Q1/T``: ``(Q1 + 2*Q2) / T^2``."""
    if t <= 0:
        return 0.0
    return (q1 + 2 * q2) / (t * t)


def shrunk_m0(q1: int, t: int, prior_m0: float, prior_strength: float) -> float:
    """Posterior-mean style blend of a seed's ``Q1`` with the global rate."""
    denom = t + prior_strength
    if denom <= 0:
        return prior_m0
    return min(1.0, (q1 + prior_strength * prior_m0) / denom)


# ── Incidence table ────────────────────────────────────────────────────


class _Incidence:
    """Per-edge hit counts with Q1/Q2 maintained incrementally.

    Updating costs O(|hit|); reading ``q1``/``q2`` is O(1), so scoring a
    seed never rescans its table.
    """

    __slots__ = ("counts", "t", "q1", "q2")

    def __init__(self) -> None:
        self.counts: dict[int, int] = {}
        self.t = 0
        self.q1 = 0
        self.q2 = 0

    def add(self, hit: Iterable[int]) -> None:
        counts = self.counts
        q1 = self.q1
        q2 = self.q2
        for edge in hit:
            c = counts.get(edge, 0)
            if c == 0:
                q1 += 1
            elif c == 1:
                q1 -= 1
                q2 += 1
            elif c == 2:
                q2 -= 1
            counts[edge] = c + 1
        self.q1 = q1
        self.q2 = q2
        self.t += 1


class GoodTuringSeedStrategy:
    """Elo-arbitrated ``good_turing`` seed arm: pick by discovery probability."""

    def __init__(
        self,
        rng: Any,
        prior_strength: float = PRIOR_STRENGTH,
        seed_cap: int = SEED_CAP,
        estimator: str = "gt",
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
        if prior_strength < 0:
            raise ValueError("prior_strength must be >= 0")
        self._rng = rng
        self._k = float(prior_strength)
        self._estimator = estimator
        self._min_obs = min_observations

        self._global = _Incidence()
        self._tables: LRUCache = LRUCache(seed_cap)

        self._selected = 0

    # ── Observation (every execution) ──────────────────────────────────

    def observe(self, seed: bytes, edges: Iterable[int]) -> None:
        """Credit one executed mutant of *seed* with the edges it hit."""
        hit = edges if isinstance(edges, (set, frozenset)) else set(edges)
        table = self._tables.get(seed)
        if table is None:
            table = _Incidence()
            self._tables[seed] = table
        table.add(hit)
        self._global.add(hit)

    # ── Estimates ──────────────────────────────────────────────────────

    def _m0(self, inc: _Incidence) -> float:
        if self._estimator == "chao":
            return chao_jost_m0(inc.q1, inc.q2, inc.t)
        return good_turing_m0(inc.q1, inc.t)

    def residual_risk(self) -> float:
        """Campaign-wide ``M0``: P(next exec of any seed hits an unseen edge)."""
        return self._m0(self._global)

    def discovery_probability(self, seed: bytes) -> float:
        """Shrunk ``M0`` for *seed*; the global rate while it is unfuzzed."""
        prior = self.residual_risk()
        table = self._tables.get(seed)
        if table is None or table.t == 0:
            return prior
        if self._estimator == "chao":
            # Chao-Jost is not linear in Q1; shrink the point estimate by
            # the same evidence weight instead of the Q1 numerator.
            w = table.t / (table.t + self._k)
            return min(1.0, w * self._m0(table) + (1.0 - w) * prior)
        return shrunk_m0(table.q1, table.t, prior, self._k)

    def variance(self, seed: bytes) -> float:
        """Esty variance of the seed's raw ``Q1/T`` (0.0 while unfuzzed)."""
        table = self._tables.get(seed)
        if table is None:
            return 0.0
        return m0_variance(table.q1, table.q2, table.t)

    @property
    def ready(self) -> bool:
        return self._global.t >= self._min_obs

    def scores(self, seeds: list[bytes]) -> list[float]:
        """Weight per seed, aligned 1:1 with ``seeds``."""
        return [max(self.discovery_probability(s), MIN_WEIGHT) for s in seeds]

    def select(self, seeds: list[bytes]) -> bytes | None:
        """Draw a seed in proportion to discovery probability; None until ready."""
        if not seeds or not self.ready:
            return None
        self._selected += 1
        chosen: bytes = self._rng.weighted_choice(seeds, self.scores(seeds))
        return chosen

    # ── Introspection ──────────────────────────────────────────────────

    def executions(self, seed: bytes) -> int:
        table = self._tables.get(seed)
        return table.t if table is not None else 0

    def singletons(self, seed: bytes) -> int:
        table = self._tables.get(seed)
        return table.q1 if table is not None else 0

    def stats(self) -> dict[str, Any]:
        g = self._global
        return {
            "observed": g.t,
            "selected": self._selected,
            "seeds": len(self._tables),
            "q1": g.q1,
            "q2": g.q2,
            "residual_risk": self.residual_risk(),
            "stderr": math.sqrt(m0_variance(g.q1, g.q2, g.t)),
        }
