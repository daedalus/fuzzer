"""Discrete finite-time occupation measures from per-run edge counts.

Implements the classical finite-time occupation construction of Du
("A Systematic Solution to the Arrow of Time Problem", Sec. 3) using the
visit counts already exported by the coverage map, without any timing
instrumentation or changes to afl_shim.c.

For one execution history the microscopic occupation is the normalised
(or raw) edge-count vector:

    µ_I(e) = count(e) / Σ_e count(e)     (normalised)
    µ_I(e) = count(e)                    (raw visits)

A macroscopic push-forward is obtained by mapping edge ids through an
optional coarse partition M (function / module / identity).

Longitudinal rarity is derived from occupation mass across histories that
hit an edge; it is deliberately distinct from horizontal hit-frequency
rarity (how many seeds ever hit the edge).

This module is a pure primitive: it does not touch EdgeTracker, seed
metadata, schedulers, or the CLI. Callers that want to wire it later can
snapshot OccupationMeasure.from_counts(...) after each interesting run.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return 0.0
    return num / den


@dataclass(frozen=True)
class OccupationMeasure:
    """Finite-time occupation of edges (or macroregions) for one history.

    Attributes:
        counts: Sparse map edge_id -> visit count (or macro_id -> mass).
        total_visits: Σ counts.values(); 0 for an empty run.
        normalised: If True, values in ``mass`` sum to 1 (when total > 0).
        mass: The occupation used for scoring — normalised probabilities
            when ``normalised`` is True, else the raw counts as floats.
    """

    counts: dict[int, int]
    total_visits: int
    normalised: bool = True
    mass: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_counts(
        cls,
        counts: Mapping[int, int] | Iterable[tuple[int, int]],
        *,
        normalised: bool = True,
        drop_zero: bool = True,
    ) -> OccupationMeasure:
        """Build an occupation measure from a sparse count map.

        Args:
            counts: edge_id -> non-negative visit count, or an iterable of
                (edge_id, count) pairs (e.g. from a SHM sparse table).
            normalised: If True, store µ_I as probabilities; else raw visits.
            drop_zero: If True, omit zero-count entries from the sparse maps.
        """
        if isinstance(counts, Mapping):
            items = counts.items()
        else:
            items = counts

        raw: dict[int, int] = {}
        total = 0
        for eid, c in items:
            c_int = int(c)
            if c_int < 0:
                raise ValueError(f"occupation counts must be non-negative, got {c_int} for {eid}")
            if drop_zero and c_int == 0:
                continue
            eid_int = int(eid)
            raw[eid_int] = raw.get(eid_int, 0) + c_int
            total += c_int

        if normalised and total > 0:
            mass = {e: c / total for e, c in raw.items()}
        else:
            mass = {e: float(c) for e, c in raw.items()}

        return cls(counts=raw, total_visits=total, normalised=normalised, mass=mass)

    def is_empty(self) -> bool:
        return self.total_visits == 0 or not self.counts

    def support_size(self) -> int:
        """Number of distinct edges (or macros) with positive occupation."""
        return len(self.counts)

    def mass_of(self, edge_id: int) -> float:
        return self.mass.get(int(edge_id), 0.0)

    def count_of(self, edge_id: int) -> int:
        return self.counts.get(int(edge_id), 0)

    def dominant(self) -> int | None:
        """Arg-max edge/macro under occupation mass (paper Sec. 3.2 style)."""
        if not self.mass:
            return None
        return max(self.mass.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    def top_k(self, k: int) -> list[tuple[int, float]]:
        """Return up to k (id, mass) pairs sorted by mass descending."""
        if k <= 0:
            return []
        items = sorted(self.mass.items(), key=lambda kv: (-kv[1], kv[0]))
        return items[:k]

    def shannon_entropy(self) -> float:
        """Shannon entropy of the occupation distribution in bits."""
        if self.total_visits == 0:
            return 0.0
        if self.normalised:
            probs = [p for p in self.mass.values() if p > 0.0]
        else:
            total = float(self.total_visits)
            probs = [c / total for c in self.counts.values() if c > 0]
        if not probs:
            return 0.0
        if _HAS_NUMPY:
            p = np.asarray(probs, dtype=np.float64)
            return float(-np.sum(p * np.log2(p)))
        return -sum(p * math.log2(p) for p in probs)

    def renyi_entropy(self, alpha: float) -> float:
        """Rényi-α entropy of the occupation distribution in bits."""
        if self.total_visits == 0:
            return 0.0
        if self.normalised:
            probs = [p for p in self.mass.values() if p > 0.0]
        else:
            total = float(self.total_visits)
            probs = [c / total for c in self.counts.values() if c > 0]
        if not probs:
            return 0.0
        if alpha == 0:
            return math.log2(max(1, len(probs)))
        if abs(alpha - 1.0) < 1e-12:
            return self.shannon_entropy()
        if _HAS_NUMPY:
            p = np.asarray(probs, dtype=np.float64)
            s = float(np.sum(p**alpha))
        else:
            s = sum(p**alpha for p in probs)
        if s <= 0.0:
            return 0.0
        return math.log2(s) / (1.0 - alpha)

    def push_forward(self, partition: Mapping[int, int]) -> OccupationMeasure:
        """Macroscopic occupation under a coarse map M: edge_id -> macro_id.

        Edges missing from ``partition`` are dropped (treated as outside the
        macroscopic descriptive classes of interest).
        """
        macro_counts: dict[int, int] = {}
        for eid, c in self.counts.items():
            mid = partition.get(eid)
            if mid is None:
                continue
            mid_int = int(mid)
            macro_counts[mid_int] = macro_counts.get(mid_int, 0) + c
        return OccupationMeasure.from_counts(macro_counts, normalised=self.normalised)

    def sparse_snapshot(self, max_edges: int | None = None) -> dict[int, int]:
        """Bounded sparse count dict suitable for seed metadata storage.

        If ``max_edges`` is set, keep only the top-mass edges (by count).
        """
        if max_edges is None or max_edges >= len(self.counts):
            return dict(self.counts)
        if max_edges <= 0:
            return {}
        top = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_edges]
        return dict(top)

    def total_variation(self, other: OccupationMeasure) -> float:
        """Total variation distance between two normalised occupations."""
        if self.total_visits == 0 and other.total_visits == 0:
            return 0.0
        keys = set(self.mass) | set(other.mass)
        # Ensure both are on the probability simplex for comparison
        a = self._as_probs()
        b = other._as_probs()
        return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)

    def _as_probs(self) -> dict[int, float]:
        if self.normalised:
            return dict(self.mass)
        if self.total_visits == 0:
            return {}
        total = float(self.total_visits)
        return {e: c / total for e, c in self.counts.items()}


class LongitudinalRarity:
    """Accumulate per-edge longitudinal occupation mass across histories.

    Horizontal rarity asks "how many seeds ever hit this edge?".
    Longitudinal rarity asks "how little relative occupation does a typical
    hitting history give this edge?".

    Call ``observe(occupation)`` once per history; query ``rarity(edge_id)``
    or ``rarity_map()`` for scores in [0, 1] (higher = rarer longitudinally).
    """

    def __init__(self, *, epsilon: float = 1e-12):
        self._sum_mass: dict[int, float] = {}
        self._hit_histories: dict[int, int] = {}
        self._n_histories = 0
        self._epsilon = epsilon

    def observe(self, occupation: OccupationMeasure) -> None:
        """Record one history's occupation (uses normalised mass)."""
        self._n_histories += 1
        probs = occupation._as_probs()
        for eid, p in probs.items():
            if p <= 0.0:
                continue
            self._sum_mass[eid] = self._sum_mass.get(eid, 0.0) + p
            self._hit_histories[eid] = self._hit_histories.get(eid, 0) + 1

    @property
    def n_histories(self) -> int:
        return self._n_histories

    def mean_mass(self, edge_id: int) -> float:
        """Mean normalised occupation among histories that hit the edge."""
        hits = self._hit_histories.get(int(edge_id), 0)
        if hits == 0:
            return 0.0
        return self._sum_mass.get(int(edge_id), 0.0) / hits

    def rarity(self, edge_id: int) -> float:
        """Longitudinal rarity in [0, 1]: high when mean occupation is small.

        Edges never observed return 1.0 (maximally rare longitudinally).
        """
        eid = int(edge_id)
        if eid not in self._hit_histories:
            return 1.0
        mean = self.mean_mass(eid)
        # Map small mean mass -> high rarity. Bound into [0, 1].
        return 1.0 / (1.0 + mean / self._epsilon) if mean <= 0.0 else 1.0 / (1.0 + mean)

    def rarity_map(self) -> dict[int, float]:
        return {eid: self.rarity(eid) for eid in self._hit_histories}

    def clear(self) -> None:
        self._sum_mass.clear()
        self._hit_histories.clear()
        self._n_histories = 0
