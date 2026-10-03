"""PositionGoodTuringScheduler: per-offset-bin Good-Turing discovery probability.

Position-arena twin of ``seed_good_turing`` / ``op_good_turing`` (entropy
handover 7.1, "per-position bin"). Edges are species; a round whose mutation
landed in an offset bin is one sampling unit of that bin::

    T   rounds that mutated the bin
    Q1  edges hit by exactly one of those rounds

    M_seed = (Q1_seed + K * M_global) / (T_seed + K)
    M_bin  = (Q1_bin  + K * M_seed)   / (T_bin  + K)

``M_bin`` is the chance the next mutation in that bin hits an edge no earlier
mutation of it hit. A bin whose rounds keep re-treading the same edges decays;
one still turning up singletons keeps paying. A bin never mutated scores its
seed's rate. Unlike the Beta/heat proposers, the signal is edge *identity*
(incidence), not a gain flag, so it keeps working once gains are rare::

    seed (100 B, width 1)   bin 5: rounds {1} {2} {3} -> Q1=3, T=3   scores high
                            bin 9: rounds {7} {7} {7} -> Q1=0, T=3   scores low

Feed: ``observe(edges)`` every execution (``FuzzRound._feed_pos_good_turing``),
then ``record`` at settle consumes those edges once. Credit smears across every
bin a round touched (no counterfactual), as ``op_good_turing`` does across
stacked operators. ``record`` without a prior ``observe`` is a no-op.

Bounded: ``MAX_BINS`` bins per seed, ``MAX_SEEDS`` seeds (LRU). Not persisted.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import xxhash

from fuzzer_tool.core.lru import LRUCache
from fuzzer_tool.core.schedulers.pos_base import Outcome
from fuzzer_tool.core.schedulers.seed_good_turing import (
    PRIOR_STRENGTH,
    _Incidence,
    good_turing_m0,
    shrunk_m0,
)

#: Rounds (campaign-wide) before the arm proposes; below this the global rate is noise.
MIN_OBSERVATIONS = 50

#: Floor so the cumulative draw never sees an all-zero vector (saturated: Q1 = 0).
MIN_WEIGHT = 1e-9

#: Offset bins per seed (width = ceil(len / MAX_BINS)); caps per-seed tables.
MAX_BINS = 64

#: Per-seed states kept (LRU). Each holds <= MAX_BINS edge tables.
MAX_SEEDS = 64


class _SeedState:
    """One seed's bin width, seed-level incidence and touched-bin incidences."""

    __slots__ = ("width", "seed", "bins")

    def __init__(self, width: int) -> None:
        self.width = width
        self.seed = _Incidence()
        self.bins: dict[int, _Incidence] = {}


class PositionGoodTuringScheduler:
    """Position arm: pick the offset bin most likely to hit a new edge."""

    name = "good_turing"

    #: Position proposers take no Beta priors (mirrors PositionCanaryScheduler).
    supports_priors = False

    def __init__(
        self,
        rng: Any,
        prior_strength: float = PRIOR_STRENGTH,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        if rng is None:
            raise ValueError("PositionGoodTuringScheduler requires a RandPool (Hard Rule 16)")
        if prior_strength < 0:
            raise ValueError("prior_strength must be >= 0")
        self._rng = rng
        self._k = float(prior_strength)
        self._min_obs = min_observations
        self._global = _Incidence()
        self._seeds: LRUCache = LRUCache(MAX_SEEDS)
        self._pending: frozenset[int] | None = None

    # ── Feed ───────────────────────────────────────────────────────────

    def observe(self, edges: Iterable[int]) -> None:
        """One execution's edges: campaign incidence now, bin credit at ``record``."""
        hit = edges if isinstance(edges, frozenset) else frozenset(edges)
        self._global.add(hit)
        self._pending = hit

    def record(
        self, data: bytes, offsets: Sequence[int], outcome: Outcome, weight: float = 1.0
    ) -> None:
        """Credit the pending edges to every bin the round mutated (outcome unused)."""
        hit, self._pending = self._pending, None
        if hit is None:
            return
        state = self._state_for(data, offsets)
        if state is None:
            return

        state.seed.add(hit)
        for b in {o // state.width for o in offsets if o >= 0}:
            table = state.bins.get(b)
            if table is None:
                table = state.bins[b] = _Incidence()
            table.add(hit)

    # ── Estimates ──────────────────────────────────────────────────────

    def residual_risk(self) -> float:
        """Campaign-wide ``M0``: P(next execution hits an unseen edge)."""
        return good_turing_m0(self._global.q1, self._global.t)

    def discovery_probability(self, data: bytes, bin_idx: int) -> float:
        """Shrunk ``M0`` of one bin; the seed's rate while the bin is unseen."""
        state = self._seeds.get(self._key(data))
        if state is None:
            return self.residual_risk()
        return self._bin_m0(state, bin_idx)

    def scores(self, data: bytes) -> list[float]:
        """Weight per bin, aligned 1:1 with the seed's bins."""
        state = self._seeds.get(self._key(data))
        if state is None:
            return []
        # Seed rate hoisted: it is the same prior for every bin.
        seed_m = self._seed_m0(state)
        bins = state.bins
        out = []
        for b in range(self._num_bins(data, state)):
            table = bins.get(b)
            m = seed_m if table is None else shrunk_m0(table.q1, table.t, seed_m, self._k)
            out.append(max(m, MIN_WEIGHT))
        return out

    # ── Proposal ───────────────────────────────────────────────────────

    def propose(self, data: bytes, buf_len: int) -> int | None:
        """Offset in a bin drawn by discovery probability; None until warm."""
        if buf_len <= 0 or self._global.t < self._min_obs:
            return None
        state = self._seeds.get(self._key(data))
        if state is None:
            return None

        weights = self.scores(data)
        b = self._draw(weights)
        lo = b * state.width
        hi = min(lo + state.width - 1, max(len(data) - 1, lo))
        return int(min(self._rng.randint(lo, hi), buf_len - 1))

    # ── Introspection ──────────────────────────────────────────────────

    def executions(self, data: bytes, bin_idx: int) -> int:
        state = self._seeds.get(self._key(data))
        table = None if state is None else state.bins.get(bin_idx)
        return 0 if table is None else table.t

    def seed_count(self) -> int:
        return len(self._seeds)

    def stats(self) -> dict[str, Any]:
        g = self._global
        return {
            "observed": g.t,
            "seeds": len(self._seeds),
            "q1": g.q1,
            "q2": g.q2,
            "residual_risk": self.residual_risk(),
        }

    # ── Internals ──────────────────────────────────────────────────────

    def _seed_m0(self, state: _SeedState) -> float:
        return shrunk_m0(state.seed.q1, state.seed.t, self.residual_risk(), self._k)

    def _bin_m0(self, state: _SeedState, bin_idx: int) -> float:
        seed_m = self._seed_m0(state)
        table = state.bins.get(bin_idx)
        if table is None or table.t == 0:
            return seed_m
        return shrunk_m0(table.q1, table.t, seed_m, self._k)

    def _draw(self, weights: list[float]) -> int:
        r = self._rng.random() * sum(weights)
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return len(weights) - 1

    @staticmethod
    def _num_bins(data: bytes, state: _SeedState) -> int:
        return max(1, -(-len(data) // state.width))

    @staticmethod
    def _key(data: bytes) -> int:
        return xxhash.xxh3_64_intdigest(data)

    def _state_for(self, data: bytes, offsets: Sequence[int]) -> _SeedState | None:
        """The seed's state, created on first use; None when no offset is usable."""
        if not any(o >= 0 for o in offsets):
            return None
        key = self._key(data)
        state: _SeedState | None = self._seeds.get(key)
        if state is None:
            width = max(1, -(-len(data) // MAX_BINS)) if data else 1
            state = self._seeds[key] = _SeedState(width)
        return state

    def _seed_table(self, data: bytes) -> _Incidence:
        """Seed-level incidence (test hook for the closed-form oracle)."""
        state: _SeedState = self._seeds[self._key(data)]
        return state.seed
