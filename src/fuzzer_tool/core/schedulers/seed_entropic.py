"""Entropic seed arm (``--entropic-seed``): libFuzzer's ``-entropic``.

Böhme, Manès, Cha, "Boosting Fuzzer Efficiency: An Information Theoretic
Perspective" (FSE'20). A seed's energy is the information its next mutant
is expected to reveal: the Shannon entropy of the rare edges its own
mutants have hit, add-one smoothed so unseen rare edges count as possible::

    mutants of A hit:  e1 e1 e1 e2      ->  low entropy, A is exhausted
    mutants of B hit:  e3 e4 e5 e6      ->  high entropy, B keeps paying

Only *rare* edges are tracked (libFuzzer: the 100 rarest, plus any hit
<= 0xFF times): abundant edges carry no information about a seed, and the
cap bounds every per-seed table. Not ``--schedule entropic``: that one is a
log(rare-count) mutation-depth factor, not a pick weight.

Local counts are not persisted across ``--resume`` (libFuzzer parity):
every seed restarts at full energy and re-earns its rank.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from fuzzer_tool.core.lru import LRUCache

#: libFuzzer ``-entropic_number_of_rarest_features`` default.
RARE_CAP = 100

#: libFuzzer ``-entropic_feature_frequency_threshold`` default.
FREQ_THRESHOLD = 0xFF

#: Global frequencies are uint16 in libFuzzer; saturate the same way.
FREQ_SATURATE = 0xFFFF

#: Per-seed tables kept (LRU). A table holds at most the rare set, which
#: is bounded by the edges discovered (the coverage map size).
SEED_CAP = 4096

#: Table size where numpy beats the scalar loop (measured: 20 -> 2.7 vs
#: 5.3 us, 50 -> 12.3 vs 7.4 us, 1000 -> 121 vs 36 us).
_NUMPY_MIN = 40


class _SeedTable:
    """One seed's mutant statistics: rare-edge hit counts, executions."""

    __slots__ = ("counts", "execs", "energy", "gen")

    def __init__(self) -> None:
        self.counts: dict[int, int] = {}
        self.execs = 0
        self.energy = 0.0
        self.gen = -1


class EntropicSeedStrategy:
    """Elo-arbitrated ``entropic`` seed arm: pick seeds by local entropy."""

    def __init__(
        self,
        rng: Any,
        rare_cap: int = RARE_CAP,
        freq_threshold: int = FREQ_THRESHOLD,
        seed_cap: int = SEED_CAP,
    ) -> None:
        self._rng = rng
        self._rare_cap = rare_cap
        self._threshold = freq_threshold

        # Global frequency of each rare edge; `_seen` holds every edge ever
        # hit, so a first sighting is a set difference, not a dict probe.
        self._rare: dict[int, int] = {}
        self._seen: set[int] = set()
        self._max_freq = 0

        self._tables: LRUCache = LRUCache(seed_cap)
        # Bumped whenever the rare set changes: every cached energy depends
        # on its size, so one counter invalidates them all in O(1).
        self._gen = 0

        self._observed = 0
        self._selected = 0

    # ── Observation (every execution) ──────────────────────────────────

    def observe(self, seed: bytes, edges: Iterable[int]) -> None:
        """Credit one executed mutant of *seed* with the edges it hit."""
        hit = edges if isinstance(edges, (set, frozenset)) else set(edges)
        table = self._tables.get(seed)
        if table is None:
            table = _SeedTable()
            self._tables[seed] = table

        # New edges become rare first (libFuzzer AddRareFeature), then
        # every rare edge hit is counted (UpdateFeatureFrequency).
        for edge in hit - self._seen:
            self._seen.add(edge)
            self._add_rare(edge)

        table.execs += 1
        table.gen = -1
        self._observed += 1

        # Hot path: locals, no per-edge method call.
        rare = self._rare
        counts = table.counts
        max_freq = self._max_freq
        for edge in hit & rare.keys():
            freq = rare[edge]
            if freq >= FREQ_SATURATE:
                continue
            rare[edge] = freq + 1
            if freq == max_freq:
                max_freq += 1
            counts[edge] = counts.get(edge, 0) + 1
        self._max_freq = max_freq

    def _add_rare(self, edge: int) -> None:
        # Keep at least rare_cap edges plus every edge at or under the
        # threshold; evict the most abundant past that.
        while len(self._rare) > self._rare_cap and self._max_freq > self._threshold:
            self._evict_most_abundant()

        self._rare[edge] = 0
        self._gen += 1

    def _evict_most_abundant(self) -> None:
        victim = max(self._rare, key=self._rare.__getitem__)
        del self._rare[victim]
        for table in self._tables.values():
            table.counts.pop(victim, None)

        self._max_freq = max(self._rare.values(), default=0)
        self._gen += 1

    # ── Energy ─────────────────────────────────────────────────────────

    def energy(self, seed: bytes) -> float:
        """Smoothed local entropy (libFuzzer InputInfo::UpdateEnergy).

        Each rare edge seen locally contributes incidence f+1, each rare
        edge not seen contributes 1, and the seed's own executions form one
        abundant species of incidence execs+1. A seed never fuzzed scores
        log(n_rare + 1), the maximum.
        """
        table = self._tables.get(seed)
        if table is None:
            return math.log(len(self._rare) + 1)
        if table.gen == self._gen:
            return table.energy

        table.energy = _entropy(table.counts.values(), table.execs, len(self._rare))
        table.gen = self._gen
        return table.energy

    def scores(self, seeds: list[bytes]) -> list[float]:
        """Energy per seed, aligned 1:1 with ``seeds``."""
        return [self.energy(s) for s in seeds]

    def select(self, seeds: list[bytes]) -> bytes | None:
        """Draw a seed in proportion to energy; None while nothing is rare."""
        if not seeds or not self._rare:
            return None

        weights = self.scores(seeds)
        if sum(weights) <= 0:
            return None

        self._selected += 1
        chosen: bytes = self._rng.weighted_choice(seeds, weights)
        return chosen

    # ── Introspection ──────────────────────────────────────────────────

    @property
    def rare_count(self) -> int:
        return len(self._rare)

    def rare_edges(self) -> set[int]:
        return set(self._rare)

    def freq(self, edge: int) -> int:
        """Global hit count of a rare edge (0 when not rare)."""
        return self._rare.get(edge, 0)

    def local_counts(self, seed: bytes) -> dict[int, int]:
        table = self._tables.get(seed)
        return dict(table.counts) if table is not None else {}

    def executions(self, seed: bytes) -> int:
        table = self._tables.get(seed)
        return table.execs if table is not None else 0

    def stats(self) -> dict[str, Any]:
        return {
            "observed": self._observed,
            "selected": self._selected,
            "rare": len(self._rare),
            "seeds": len(self._tables),
        }


def _entropy(counts: Iterable[int], execs: int, n_rare: int) -> float:
    """Add-one smoothed entropy estimate over a seed's species counts."""
    values = list(counts)
    if len(values) >= _NUMPY_MIN:
        return _entropy_np(values, execs, n_rare)
    return _entropy_py(values, execs, n_rare)


def _entropy_np(values: list[int], execs: int, n_rare: int) -> float:
    li = np.asarray(values, dtype=np.float64) + 1.0
    energy = -float((li * np.log(li)).sum())
    incidence = float(li.sum()) + n_rare - len(values)
    return _close(energy, incidence, execs)


def _entropy_py(values: list[int], execs: int, n_rare: int) -> float:
    energy = 0.0
    incidence = 0.0
    for freq in values:
        li = freq + 1
        energy -= li * math.log(li)
        incidence += li

    # Locally unseen rare edges: incidence 1 each, log(1) = 0 energy.
    incidence += n_rare - len(values)
    return _close(energy, incidence, execs)


def _close(energy: float, incidence: float, execs: int) -> float:
    """Add the seed's executions as one abundant species, then normalize."""
    abundant = execs + 1
    energy -= abundant * math.log(abundant)
    incidence += abundant
    return energy / incidence + math.log(incidence)
